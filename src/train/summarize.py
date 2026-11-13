"""Tabele wyników z wielu runów (średnia ± sd po ziarnach) -> results/models/*.csv (materiał do rozdziału 6).

    python -m train.summarize [--runs runs] [--out results/models]

Tabele:
  modele.csv            – E1: złożoność, ROC-AUC test/test_ood, decyzja trójstopniowa na test (reguła score)
  decyzje_podklasy.csv  – E1: odsetek przepuść/flaguj/zablokuj dla każdej podklasy (test i test_ood)
  krzywa_ai.csv         – E3 (+ punkt "wszystkie" z E1): ROC-AUC i wykrywalność AI vs liczba obrazów AI w treningu
  ai_generatory.csv     – wykrywalność AI (nie przepuszczone) dla każdego generatora, model × liczba AI
"""
import argparse, json
from pathlib import Path
import pandas as pd

ORDER = ["cnn", "mnv3s", "mnv3l", "effl0", "effb3"]


def load(runs):
    rows, decs = [], []
    for f in sorted(Path(runs).glob("*/eval.json")):
        d = f.parent
        e, s = json.loads(f.read_text(encoding="utf-8")), json.loads((d / "summary.json").read_text())
        c = json.loads((d / "config.json").read_text(encoding="utf-8"))
        ai = c.get("ai_train_n")
        key = {"run": d.name, "group": c.get("group"), "model": c["model"], "head": c["head"],
               "ai_train": "wszystkie" if ai is None else int(ai), "img": c["img_size"], "seed": c["seed"]}
        r = {**key, "params_M": c["params_M"], "MMAC": c["MMAC"], "fp32_MB": c["fp32_MB"],
             "train_min": s["train_min"], "epoki": s["epochs_run"]}
        for sp, v in e["splits"].items():
            r[f"{sp}_auc"] = v["roc_auc"]
            r[f"{sp}_prauc"] = v["pr_auc"]
            for k, val in v.get("score", {}).items():
                r[f"{sp}_{k}"] = val
        rows.append(r)
        dc = pd.read_csv(d / "decisions.csv")
        decs.append(dc.assign(**key))
        p = pd.read_parquet(d / "preds.parquet")
        ai_rows = p[p.subclass == "ai"].copy()
        if len(ai_rows):
            t_acc = e["t_przepusc"]
            ai_rows["wykryte"] = ai_rows.p_violation >= t_acc            # nie przepuszczone automatycznie
            g = ai_rows.groupby(["split", ai_rows.gen.fillna("?")]).wykryte.agg(["mean", "size"]).reset_index()
            decs.append(g.assign(**key, kind="ai_gen"))
    return pd.DataFrame(rows), pd.concat(decs, ignore_index=True)


def agg(df, by, cols):
    m = df.groupby(by)[cols].mean()
    sd = df.groupby(by)[cols].std()
    out = m.round(4).astype(str) + " ± " + sd.round(4).astype(str)
    out.insert(0, "n_ziaren", df.groupby(by).size())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/models")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, dec = load(a.runs)
    df["model"] = pd.Categorical(df.model, ORDER, ordered=True)

    e1 = df[df.group == "E1_modele"]
    cols = ["test_auc", "test_ood_auc", "test_pokrycie_auto", "test_obciazenie_moderacji",
            "test_bledne_blokady_autentycznych", "test_przepuszczone_naruszenia", "test_ood_bledne_blokady_autentycznych",
            "test_ood_przepuszczone_naruszenia", "train_min"]
    t = agg(e1, "model", cols)
    t.insert(1, "params_M", e1.groupby("model").params_M.first())
    t.insert(2, "MMAC", e1.groupby("model").MMAC.first())
    t.insert(3, "fp32_MB", e1.groupby("model").fp32_MB.first())
    t.to_csv(out / "modele.csv")
    print("== E1: modele\n", t.to_string())

    d = dec[(dec.get("kind").isna()) & (dec.group == "E1_modele") & (dec.rule == "score")]
    d = d.groupby(["split", "subclass", "model"])[["przepusc", "flaguj", "zablokuj"]].mean().round(3)
    d.to_csv(out / "decyzje_podklasy.csv")
    print("\n== E1: decyzje per podklasa (test)\n", d.loc["test"].unstack("model").round(2).to_string())

    curve = df[(df.group == "E3_krzywa_ai") | ((df.group == "E1_modele") & df.model.isin(["effl0", "mnv3l", "mnv3s", "cnn"]))]
    gen = dec[dec.get("kind") == "ai_gen"]
    known = gen[gen.gen.isin(["sd15", "sdxl"]) & (gen.split == "test")].groupby("run")["mean"].mean()
    unknown = gen[(gen.split == "test_ood")].groupby("run").apply(lambda g: (g["mean"] * g["size"]).sum() / g["size"].sum())
    curve = curve.assign(ai_wykryte_znane=curve.run.map(known), ai_wykryte_nieznane=curve.run.map(unknown))
    c = agg(curve.assign(ai_train=curve.ai_train.astype(str)), ["model", "ai_train"],
            ["test_auc", "test_ood_auc", "ai_wykryte_znane", "ai_wykryte_nieznane", "test_bledne_blokady_autentycznych"])
    c.to_csv(out / "krzywa_ai.csv")
    print("\n== E3: krzywa uczenia AI\n", c.to_string())

    # E2: głowica binarna vs 5-klasowa (reguła "score"; dla 5 klas także reguła podklasowa)
    sub = dec[(dec.get("kind").isna()) & (dec.split == "test")]
    def rule_stats(run, rule):
        s = sub[(sub.run == run) & (sub.rule == rule)]
        w = s.n / s.n.sum()
        auth = s.subclass == "nature"
        return {"auto": float(((1 - s.flaguj) * w).sum()),
                "bledne_blokady": float(s[auth].zablokuj.mean()) if auth.any() else None,
                "ai_zablokowane": float(s[s.subclass == "ai"].zablokuj.mean()),
                "manip_zablokowane": float(s[s.subclass == "manipulated"].zablokuj.mean())}
    e2 = df[(df.group == "E2_glowica") | ((df.group == "E1_modele") & df.model.isin(["effl0", "mnv3l"]))].copy()
    for rule in ("score", "subclass"):
        st = e2.run.map(lambda r: rule_stats(r, rule) if ((sub.run == r) & (sub.rule == rule)).any() else {})
        for k in ("auto", "bledne_blokady", "ai_zablokowane", "manip_zablokowane"):
            e2[f"{rule}_{k}"] = st.map(lambda d: d.get(k))
    cols2 = ["test_auc", "test_ood_auc"] + [c for c in e2.columns if c.startswith(("score_", "subclass_"))]
    t2 = agg(e2, ["model", "head"], [c for c in cols2 if e2[c].notna().any()])
    t2.to_csv(out / "glowica.csv")
    print("\n== E2: głowica\n", t2.to_string())

    # E4: rozdzielczość (effl0, ziarno 0) + punkt 224 z E1
    e4 = df[((df.group == "E4_rozdzielczosc") | ((df.group == "E1_modele") & (df.model == "effl0"))) & (df.seed == 0)]
    e4 = e4.assign(ai_wykryte=e4.run.map(known))[["img", "test_auc", "test_ood_auc", "ai_wykryte", "MMAC",
                                                    "test_pokrycie_auto", "train_min"]].sort_values("img")
    e4.to_csv(out / "rozdzielczosc.csv", index=False)
    print("\n== E4: rozdzielczość (effl0, s0)\n", e4.round(4).to_string(index=False))

    # odporność i eksport (runy z robustness.csv / export.json)
    rob = [pd.read_csv(f).assign(model=f.parent.name.split("_")[1]) for f in Path(a.runs).glob("*/robustness.csv")]
    if rob:
        r = pd.concat(rob).pivot_table(index="degradacja", columns="model", values="roc_auc").round(4)
        r.to_csv(out / "odpornosc.csv")
        print("\n== Odporność: ROC-AUC (test, 3000 zdjęć)\n", r.to_string())
    ex = []
    for f in Path(a.runs).glob("*/export/export.json"):
        j = json.loads(f.read_text())
        for k, v in j.items():
            ex.append({"model": f.parent.parent.name.split("_")[1], "format": k, **v})
    if ex:
        x = pd.DataFrame(ex).set_index(["model", "format"]).round(4)
        x.to_csv(out / "eksport.csv")
        print("\n== Eksport ONNX (zgodność na 1000 zdjęć testowych)\n", x.to_string())

    g = gen.groupby(["model", "ai_train", "split", "gen"])[["mean"]].mean().round(3).rename(columns={"mean": "wykryte"})
    g = g.join(gen.groupby(["model", "ai_train", "split", "gen"])[["size"]].first())
    g.to_csv(out / "ai_generatory.csv")
    print("\n== AI per generator (effl0)\n", g.loc["effl0"].unstack("ai_train")["wykryte"].to_string())


if __name__ == "__main__":
    main()
