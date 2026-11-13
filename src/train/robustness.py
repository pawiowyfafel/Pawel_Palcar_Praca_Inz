"""Test odporności: predykcje na zdjęciach testowych po degradacji (jak po drodze z telefonu do aplikacji).

    python -m train.robustness runs/effl0_multi_s0 [--split test] [--n 3000]

Degradacje: JPEG q=90/70/50, zmniejszenie ×0.5/×0.25 (i powrót do rozmiaru), "komunikator"
(dłuższy bok 1280 + dwukrotna kompresja q=75/80), zrzut ekranu (lekkie przeskalowanie + PNG->JPEG q=90).
Wynik: <run>/robustness.csv – ROC-AUC i decyzje trójstopniowe (progi z eval.json) dla każdej degradacji.
"""
import argparse, io, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from train.data import CLASSES
from train.evaluate import decide
from train.models import build
from train import wb
from torchvision import transforms as T


def jpeg(im, q):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


def scale(im, s):
    w, h = im.size
    return im.resize((max(16, int(w * s)), max(16, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def messenger(im):
    s = 1280 / max(im.size)
    if s < 1:
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    return jpeg(jpeg(im, 75), 80)


DEGR = {"oryginal": lambda im: im, "jpeg90": lambda im: jpeg(im, 90), "jpeg70": lambda im: jpeg(im, 70),
        "jpeg50": lambda im: jpeg(im, 50), "skala0.5": lambda im: scale(im, 0.5), "skala0.25": lambda im: scale(im, 0.25),
        "komunikator": messenger, "zrzut_ekranu": lambda im: jpeg(scale(im, 0.9), 90)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--data-root", default=None, help="inna sesja = inna ścieżka montowania datasetu")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    ev = json.loads((run / "eval.json").read_text(encoding="utf-8"))
    head, size = cfg["head"], cfg["img_size"]
    model, _, mean, std = build(cfg["model"], 1 if head == "binary" else len(CLASSES), pretrained=False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    model.to(dev).eval()
    tf = T.Compose([T.Resize(int(round(size / 0.875))), T.CenterCrop(size), T.ToTensor(), T.Normalize(mean, std)])
    root = Path(a.data_root or cfg["data_root"])
    df = pd.read_parquet(root / cfg["manifest"])
    df = df[df.split == a.split].sample(min(a.n, (df.split == a.split).sum()), random_state=0).reset_index(drop=True)
    y = (df.label == 0).astype(int).values
    rows = []
    for name, f in DEGR.items():
        pv, pc = [], []
        for i0 in range(0, len(df), 64):
            x = torch.stack([tf(f(Image.open(root / p).convert("RGB"))) for p in df.path[i0:i0 + 64]]).to(dev)
            with torch.no_grad():
                out = model(x).float()
            if head == "binary":
                pv.append(torch.sigmoid(out[:, 0]).cpu())
            else:
                p = out.softmax(1).cpu()
                pv.append(1 - p[:, 0])
                pc.append(p)
        g = df.assign(p_violation=torch.cat(pv).numpy())
        if pc:
            for k, c in enumerate(CLASSES):
                g[f"p_{c}"] = torch.cat(pc)[:, k].numpy()
        dec = decide(g, ev["t_przepusc"], ev["t_blokuj"])
        auth = g.label == 1
        rows.append({"degradacja": name, "roc_auc": roc_auc_score(y, g.p_violation),
                     "bledne_blokady_autentycznych": float((dec[auth] == "zablokuj").mean()),
                     "przepuszczone_naruszenia": float((dec[~auth] == "przepusc").mean()),
                     "obciazenie_moderacji": float((dec == "flaguj").mean())})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(run / "robustness.csv", index=False)
    w = wb.resume(run, "robustness")
    if w:
        import wandb
        w.log({"odpornosc": wandb.Table(dataframe=pd.DataFrame(rows))})
        w.summary.update({f"odpornosc/{r['degradacja']}/roc_auc": r["roc_auc"] for r in rows})
        w.finish()


if __name__ == "__main__":
    main()
