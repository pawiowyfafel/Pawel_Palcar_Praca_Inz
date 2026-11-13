"""Weights & Biases: logowanie treningu i oceny. Brak klucza / brak sieci -> trening działa dalej bez W&B.

Każdy run treningowy = run W&B (nazwa jak katalog runu), grupa = eksperyment (E1_modele…), tagi = model,
głowica, liczba obrazów AI, ziarno. evaluate.py i robustness.py dopisują wyniki do TEGO SAMEGO runu
(id zapisane w <run>/wandb_id.txt). Klucz: zmienna WANDB_API_KEY (Kaggle: sekret albo komórka notebooka).
"""
import json, os
from pathlib import Path


def _enabled(cfg):
    if not cfg.get("wandb", True) or os.environ.get("WANDB_MODE") == "disabled":
        return False
    return bool(os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline")


def init(cfg, meta, out):
    """Start runu W&B dla treningu. Zwraca obiekt run albo None."""
    if not _enabled(cfg):
        print("W&B: wyłączone (brak WANDB_API_KEY)")
        return None
    try:
        import wandb
        tags = [cfg["model"], cfg["head"], f"ai{cfg.get('ai_train_n') if cfg.get('ai_train_n') is not None else 'all'}",
                f"seed{cfg['seed']}", f"r{meta['img_size']}"]
        run = wandb.init(project=cfg.get("wandb_project", "praca-inz-trail"), entity=cfg.get("wandb_entity"),
                         name=cfg["name"], group=cfg.get("group", "debug"), job_type="train", tags=tags,
                         config={**cfg, **meta}, dir=str(out), reinit="finish_previous")
        (Path(out) / "wandb_id.txt").write_text(run.id)
        return run
    except Exception as e:                      # brak sieci, zły klucz itp. – nie przerywamy treningu
        print("W&B: nie udało się uruchomić:", e)
        return None


def resume(run_dir, job="eval"):
    """Dołączenie do runu treningu (ocena, odporność). Zwraca run albo None."""
    f = Path(run_dir) / "wandb_id.txt"
    cfg_f = Path(run_dir) / "config.json"
    if not f.exists() or not cfg_f.exists():
        return None
    cfg = json.loads(cfg_f.read_text(encoding="utf-8"))
    if not _enabled(cfg):
        return None
    try:
        import wandb
        return wandb.init(project=cfg.get("wandb_project", "praca-inz-trail"), entity=cfg.get("wandb_entity"),
                          id=f.read_text().strip(), resume="allow", dir=str(run_dir), reinit="finish_previous")
    except Exception as e:
        print(f"W&B ({job}): nie udało się dołączyć:", e)
        return None


def flat(prefix, d, out=None):
    """{'test': {'score': {'x': 1}}} -> {'test/score/x': 1} (tylko liczby)."""
    out = {} if out is None else out
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat(key, v, out)
        elif isinstance(v, (int, float)) and v == v:
            out[key] = v
        elif isinstance(v, list) and len(v) == 2 and all(isinstance(x, float) for x in v):
            out[key + "_low"], out[key + "_high"] = v
    return out
