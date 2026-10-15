"""Podział obu klas: komponent (grupa + klaster duplikatów) w całości do jednego splitu.

Klucz komponentu = najmniejsza nazwa grupy -> deterministyczny i stabilny, gdy do grupy dochodzą
przeróbki (memy, manipulacje, img2img dziedziczą group po oryginale).
Kolejność reguł: force_split (role zadań z Kaggle) > own -> test_real > holdout -> test_ood > hash.
"""
import hashlib, pandas as pd
from common import INTERIM, MANIF, UF

TRAIN, VAL = 70, 85            # train <70, val <85, test reszta (w %)
OOD = ["geo", "source", "generator"]
OOD_REGIONS = {"bieszczady", "lublana"}   # góry / miasto niewidziane w treningu (Mapillary)


def main():
    df = pd.read_parquet(INTERIM / "dedup.parquet")
    rej_f = INTERIM / "rejected.txt"
    rej = set(rej_f.read_text().split()) if rej_f.exists() else set()
    df = df[~df.is_dup & ~df.id.isin(rej)].reset_index(drop=True)

    uf = UF(len(df))
    for col in ("group", "dup_cluster"):
        for idx in df.groupby(col).indices.values():
            for j in idx[1:]:
                uf.union(idx[0], j)
    df["comp"] = [uf.find(i) for i in range(len(df))]
    key = df.groupby("comp").group.transform("min")
    bucket = key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 100)
    df["split"] = pd.cut(bucket, [-1, TRAIN - 1, VAL - 1, 99], labels=["train", "val", "test"]).astype(str)

    h = df.get("holdout", pd.Series(None, index=df.index))
    reg = df.get("region", pd.Series(None, index=df.index))
    ood = (h.isin(OOD) | reg.isin(OOD_REGIONS)).groupby(df.comp).transform("any")
    real = h.eq("real").groupby(df.comp).transform("any")
    df.loc[ood, "split"] = "test_ood"
    df.loc[real, "split"] = "test_real"
    if "force_split" in df:
        fs = df.groupby("comp").force_split.transform("first")
        df.loc[fs.notna(), "split"] = fs[fs.notna()]

    cols = [c for c in df.columns if c not in ("raw_path", "comp", "is_dup")]
    df[cols].to_parquet(MANIF / "dataset.parquet")
    df[cols].to_csv(MANIF / "dataset.csv", index=False)
    df[df.label == 1][cols].to_parquet(MANIF / "positive.parquet")   # zgodność ze skryptami synth/kaggle
    print(pd.crosstab([df.label, df.subclass], df.split, margins=True))


if __name__ == "__main__":
    main()
