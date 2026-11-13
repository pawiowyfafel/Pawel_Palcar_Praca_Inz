"""Trening jednego eksperymentu.

    python -m train.train --config configs/base.yaml --set model=effl0 head=multi seed=0 ai_train_n=500 name=effl0_ai500_s0

Procedura (transfer learning, 2 etapy): zamrożony backbone (uczy się tylko klasyfikator) -> pełny fine-tuning,
AdamW + cosine z rozgrzewką, mixed precision na GPU, early stopping po val ROC-AUC (wynik p_violation).
Wynik w <out_dir>/<name>/: config.json, epochs.csv, best.pt, preds.parquet (predykcje dla val/test/test_ood/test_real),
summary.json. Ocena (progi, decyzja trójstopniowa) jest osobno: python -m train.evaluate.
"""
import argparse, hashlib, json, math, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from train.data import CLASSES, CachedDataset, TrailDataset, class_weights, load_manifest, transforms
from train.gpu_aug import GPUAugment
from train.models import build, complexity, freeze_backbone_bn, set_backbone_trainable
from train import wb

EVAL_SPLITS = ["val", "test", "test_ood", "test_real"]


def parse_value(v):
    try:
        v = yaml.safe_load(v)
    except yaml.YAMLError:
        return v
    if isinstance(v, str):                   # YAML 1.1 czyta "1e-3" jako tekst
        try:
            return float(v)
        except ValueError:
            pass
    return v


def load_config():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="nadpisania klucz=wartość")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = parse_value(v)
    return cfg


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def weighted_ce(logits, y, w, present, eps):
    """CrossEntropy z wagami klas i JEDNOLITYM wygładzaniem etykiet na klasy obecne w treningu.
    (nn.CrossEntropyLoss(weight=, label_smoothing=) waży też składnik wygładzania wagami klas – przy rzadkiej
    klasie AI pcha to model do przypisywania "AI" każdemu zdjęciu; przy 0 obrazach AI – drastycznie.)"""
    logp = logits.float().log_softmax(1)
    nll = -logp.gather(1, y[:, None]).squeeze(1)
    smooth = -logp[:, present].mean(1)
    wy = w[y]
    return (wy * ((1 - eps) * nll + eps * smooth)).sum() / wy.sum()


def violation_prob(logits, head):
    if head == "binary":
        return torch.sigmoid(logits.squeeze(1)), None
    p = logits.float().softmax(1)
    return 1 - p[:, 0], p                                   # klasa 0 = nature (autentyczne)


@torch.no_grad()
def predict(model, loader, head, dev, amp, aug=None):
    model.eval()
    pv, pc, idx = [], [], []
    for x, _, i in loader:
        x = x.to(dev, non_blocking=True)
        if aug is not None:
            x = aug(x)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
            out = model(x)
        v, p = violation_prob(out.float(), head)
        v = torch.nan_to_num(v, nan=0.5)                    # zabezpieczenie: NaN nie wywraca oceny
        pv.append(v.cpu())
        idx.append(i)
        if p is not None:
            pc.append(p.cpu())
    return torch.cat(idx).numpy(), torch.cat(pv).numpy(), (torch.cat(pc).numpy() if pc else None)


def make_loader(df, root, tf, head, cfg, shuffle):
    ds = (CachedDataset(df, Path(cfg["cache_dir"]) / "images.npy", head) if cfg.get("cache_dir")
          else TrailDataset(df, root, tf, head))
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=cfg["workers"], pin_memory=torch.cuda.is_available(), drop_last=shuffle,
                      persistent_workers=cfg["workers"] > 0)


def main():
    cfg = load_config()
    seed_all(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = dev.type == "cuda"
    out = Path(cfg["out_dir"]) / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["data_root"])
    manifest = root / cfg["manifest"]
    df = load_manifest(manifest, cfg.get("ai_train_n"))
    if cfg.get("limit"):                                    # test techniczny: mały podzbiór
        df = pd.concat([g.sample(min(len(g), cfg["limit"]), random_state=0) for _, g in df.groupby("split")])
        df = df.reset_index(drop=True)
    head, nc = cfg["head"], (1 if cfg["head"] == "binary" else len(CLASSES))

    model, size, mean, std = build(cfg["model"], nc, pretrained=cfg["pretrained"])
    size = cfg.get("img_size") or size
    model.to(dev)
    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    va_stop = va[~va.subclass.isin(cfg.get("val_exclude_subclasses") or [])]   # kryterium early stoppingu
    tl = make_loader(tr, root, transforms(size, mean, std, True), head, cfg, True)
    vl = make_loader(va_stop, root, transforms(size, mean, std, False), head, cfg, False)
    # z cache'em: augmentacje i przygotowanie wejścia na GPU (dane w loaderze to surowe uint8 256×256)
    aug_tr = GPUAugment(size, mean, std, True) if cfg.get("cache_dir") else None
    aug_ev = GPUAugment(size, mean, std, False) if cfg.get("cache_dir") else None

    w, present = class_weights(df, head)
    w = w.to(dev)
    if head == "binary":
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=w)
        crit = lambda logits, y: bce(logits.float().squeeze(1), y)
    else:
        present = present.to(dev)
        crit = lambda logits, y: weighted_ce(logits, y, w, present, cfg["label_smoothing"])
    scaler = torch.amp.GradScaler(enabled=amp)
    meta = {"manifest_md5": hashlib.md5(manifest.read_bytes()).hexdigest(), "img_size": size,
            "n_train": len(tr), "n_train_ai": int((tr.subclass == "ai").sum()), "device": str(dev),
            **complexity(build(cfg["model"], nc, pretrained=False)[0], size)}
    (out / "config.json").write_text(json.dumps({**cfg, **meta}, indent=1, ensure_ascii=False))
    run = wb.init(cfg, meta, out)

    def optimizer(stage):
        lr = cfg["lr_head"] if stage == 1 else cfg["lr"]
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])

    best, bad, hist, t0 = -1.0, 0, [], time.time()
    stage, epochs_total = 1, cfg["freeze_epochs"] + cfg["epochs"]
    set_backbone_trainable(model, cfg["freeze_epochs"] == 0 or cfg["model"] == "cnn")
    if cfg["freeze_epochs"] == 0 or cfg["model"] == "cnn":             # CNN bez pretreningu: od razu cały
        stage = 2
    opt = optimizer(stage)
    steps_stage = (cfg["epochs"] if stage == 2 else cfg["freeze_epochs"]) * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, steps_stage, cfg["warmup_frac"]))

    for ep in range(epochs_total):
        if stage == 1 and ep == cfg["freeze_epochs"]:                  # etap 2: odmrożenie backbone
            stage = 2
            set_backbone_trainable(model, True)
            opt = optimizer(2)
            steps_stage = cfg["epochs"] * len(tl)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine(s, steps_stage, cfg["warmup_frac"]))
        model.train()
        if stage == 1:
            freeze_backbone_bn(model)
        tot, n, skipped = 0.0, 0, 0
        for x, y, _ in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if aug_tr is not None:
                x = aug_tr(x)
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):                    # przepełnienie FP16 itp.: pomiń paczkę
                skipped += 1
                if skipped > 0.2 * len(tl):
                    raise RuntimeError(f"niestabilny trening: {skipped} paczek z NaN/inf w epoce {ep}")
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, n = tot + loss.item() * len(x), n + len(x)
        idx, pv, _ = predict(model, vl, head, dev, amp, aug_ev)
        yv = (va_stop.label.values[idx] == 0).astype(int)
        auc = roc_auc_score(yv, pv) if 0 < yv.sum() < len(yv) else float("nan")
        hist.append({"epoch": ep, "stage": stage, "train_loss": tot / max(1, n), "val_auc": auc, "skipped": skipped,
                     "lr": opt.param_groups[0]["lr"], "time_s": round(time.time() - t0, 1)})
        pd.DataFrame(hist).to_csv(out / "epochs.csv", index=False)
        print(f"[{cfg['name']}] ep {ep:2d} st{stage} loss {tot / max(1, n):.4f} val_auc {auc:.4f}", flush=True)
        if run:
            h = hist[-1]
            run.log({"epoch": ep, "stage": stage, "train/loss": h["train_loss"], "val/auc_bez_ai": auc,
                     "lr": h["lr"], "time_min": h["time_s"] / 60}, step=ep)
        if stage == 2 or cfg["freeze_epochs"] == 0:
            if auc > best:
                best, bad = auc, 0
                torch.save(model.state_dict(), out / "best.pt")
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    break

    # predykcje najlepszego modelu dla wszystkich zbiorów ewaluacyjnych (cała ocena offline z tego pliku)
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    ev = df[df.split.isin(EVAL_SPLITS)].reset_index(drop=True)
    idx, pv, pc = predict(model, make_loader(ev, root, transforms(size, mean, std, False), head, cfg, False),
                          head, dev, amp, aug_ev)
    keep = [c for c in ["id", "split", "label", "subclass", "source", "gen", "manip", "meme_style", "region"]
            if c in ev]
    preds = ev.iloc[idx][keep].reset_index(drop=True).assign(p_violation=pv)
    if pc is not None:
        for k, c in enumerate(CLASSES):
            preds[f"p_{c}"] = pc[:, k]
    preds.to_parquet(out / "preds.parquet")
    summary = {"name": cfg["name"], "best_val_auc": best, "epochs_run": len(hist),
               "train_min": round((time.time() - t0) / 60, 1), **meta}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if run:
        run.summary.update({"best_val_auc_bez_ai": best, "epochs_run": len(hist), "train_min": summary["train_min"]})
        run.finish()
    print("gotowe:", summary)


def cosine(step, total, warmup_frac):
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm))))


if __name__ == "__main__":
    main()
