"""Ocena z predykcji (preds.parquet): metryki + decyzja trójstopniowa asystenta moderatora.

    python -m train.evaluate runs/effl0_multi_s0 [runs/effl0_multi_s1 ...] [--fbr 0.005 --miss 0.02]

Reguła decyzyjna (progi wyznaczane WYŁĄCZNIE na val):
  p_violation >= t_blokuj  -> ZABLOKUJ   (t_blokuj: błędne blokady autentycznych zdjęć <= fbr, np. 0,5%)
  p_violation <  t_przepusc -> PRZEPUŚĆ  (t_przepusc: przepuszczone naruszenia oczywiste <= miss, np. 2%)
  pomiędzy                 -> FLAGUJ (do administracji)
Wariant "podklasowy" (tylko głowica 5-klasowa): blokować wolno tylko gdy dominuje naruszenie oczywiste
(urban/meme); AI i manipulacje co najwyżej flagowane.
Wynik: <run>/eval.json, <run>/decisions.csv; przy wielu runach także średnia ± sd (results/models/<nazwa>.csv).
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from train.data import OBVIOUS

DECISIONS = ["przepusc", "flaguj", "zablokuj"]


def thresholds(val, fbr, miss):
    auth = np.sort(val.loc[val.label == 1, "p_violation"].values)
    obv = np.sort(val.loc[val.subclass.isin(OBVIOUS), "p_violation"].values)
    t_block = float(np.quantile(auth, 1 - fbr, method="higher")) + 1e-6 if len(auth) else 1.0
    t_accept = float(np.quantile(obv, miss, method="lower")) if len(obv) else 0.0
    return min(t_accept, t_block), t_block


def decide(df, t_accept, t_block, per_subclass=False):
    p = df.p_violation.values
    d = np.where(p >= t_block, 2, np.where(p < t_accept, 0, 1))
    if per_subclass and "p_urban" in df:                     # blokada tylko dla naruszeń oczywistych
        obvious = df[[f"p_{c}" for c in OBVIOUS]].sum(axis=1).values
        forensic = df[["p_manipulated", "p_ai"]].sum(axis=1).values
        d = np.where((d == 2) & (forensic > obvious), 1, d)
    return pd.Series(np.array(DECISIONS)[d], index=df.index)


def safe(f, y, s):
    return float(f(y, s)) if 0 < y.sum() < len(y) else float("nan")


def bootstrap_ci(y, s, n=1000, seed=0):
    """95% przedział ufności ROC-AUC (bootstrap po zdjęciach) – do porównań modeli w rozdz. 6."""
    rng, y, s = np.random.default_rng(seed), np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            vals.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def evaluate_run(run, fbr, miss, boot=True):
    pr = pd.read_parquet(Path(run) / "preds.parquet")
    pr["y"] = (pr.label == 0).astype(int)
    t_acc, t_blk = thresholds(pr[pr.split == "val"], fbr, miss)
    res = {"run": str(run), "t_przepusc": t_acc, "t_blokuj": t_blk, "fbr": fbr, "miss": miss, "splits": {}}
    rows = []
    for rule in (["score", "subclass"] if "p_urban" in pr else ["score"]):
        pr[f"dec_{rule}"] = decide(pr, t_acc, t_blk, rule == "subclass")
    for sp, g in pr.groupby("split"):
        s = {"n": len(g), "roc_auc": safe(roc_auc_score, g.y, g.p_violation),
             "roc_auc_ci95": bootstrap_ci(g.y, g.p_violation) if boot else None,
             "pr_auc": safe(average_precision_score, g.y, g.p_violation)}
        for rule in [c[4:] for c in pr if c.startswith("dec_")]:
            dec = g[f"dec_{rule}"]
            auth = g.label == 1
            s[rule] = {"pokrycie_auto": float((dec != "flaguj").mean()),            # rozstrzygnięte bez człowieka
                       "obciazenie_moderacji": float((dec == "flaguj").mean()),
                       "bledne_blokady_autentycznych": float((dec[auth] == "zablokuj").mean()) if auth.any() else None,
                       "przepuszczone_naruszenia": float((dec[~auth] == "przepusc").mean()) if (~auth).any() else None}
            for (sub, src), gg in g.groupby(["subclass", "source"]):
                cnt = gg[f"dec_{rule}"].value_counts(normalize=True)
                rows.append({"split": sp, "rule": rule, "subclass": sub, "source": src, "n": len(gg),
                             **{k: float(cnt.get(k, 0)) for k in DECISIONS}})
        res["splits"][sp] = s
    Path(run, "eval.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    dec = pd.DataFrame(rows)
    dec.to_csv(Path(run, "decisions.csv"), index=False)
    return res, dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--fbr", type=float, default=0.005, help="dopuszczalne błędne blokady autentycznych (val)")
    ap.add_argument("--miss", type=float, default=0.02, help="dopuszczalne przepuszczone naruszenia oczywiste (val)")
    ap.add_argument("--out", default=None, help="CSV ze średnią ± sd po runach (np. results/models/effl0.csv)")
    ap.add_argument("--no-bootstrap", action="store_true", help="bez przedziałów ufności (szybciej)")
    a = ap.parse_args()
    flat = []
    for run in a.runs:
        res, _ = evaluate_run(run, a.fbr, a.miss, not a.no_bootstrap)
        for sp, s in res["splits"].items():
            flat.append({"run": run, "split": sp, "roc_auc": s["roc_auc"], "pr_auc": s["pr_auc"],
                         **{f"{rule}_{k}": v for rule in ("score", "subclass") if rule in s for k, v in s[rule].items()}})
        print(f"{run}: t_przepusc={res['t_przepusc']:.3f} t_blokuj={res['t_blokuj']:.3f}")
    df = pd.DataFrame(flat)
    agg = df.drop(columns="run").groupby("split").agg(["mean", "std"]).round(4)
    print(agg.to_string())
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(a.out)


if __name__ == "__main__":
    main()
