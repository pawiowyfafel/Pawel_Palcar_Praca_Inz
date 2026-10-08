"""Podział: komponent (grupa + klaster duplikatów) w całości do jednego splitu, deterministycznie po hashu.

Specjalne: own -> test_real, region Bieszczady i całe Wikimedia -> test_ood.
"""
import hashlib, pandas as pd
from common import INTERIM, MANIF, UF

TRAIN, VAL = 70, 85            # train <70, val <85, test reszta (w %)


def main():
    df = pd.read_parquet(INTERIM / "positive_dedup.parquet")
    rej_f = INTERIM / "rejected.txt"
    rej = set(rej_f.read_text().split()) if rej_f.exists() else set()
    df = df[~df.is_dup & ~df.id.isin(rej)].reset_index(drop=True)

    uf = UF(len(df))
    for col in ("group", "dup_cluster"):
        for idx in df.groupby(col).indices.values():
            for j in idx[1:]:
                uf.union(idx[0], j)
    df["comp"] = [uf.find(i) for i in range(len(df))]
    key = df.groupby("comp").id.transform("min")
    bucket = key.map(lambda k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 100)
    df["split"] = pd.cut(bucket, [-1, TRAIN - 1, VAL - 1, 99], labels=["train", "val", "test"]).astype(str)

    h = df.get("holdout")
    if h is not None:
        real = h.eq("real").groupby(df.comp).transform("any")
        ood = h.isin(["geo", "source"]).groupby(df.comp).transform("any")
        df.loc[ood, "split"] = "test_ood"
        df.loc[real, "split"] = "test_real"

    cols = [c for c in df.columns if c not in ("raw_path", "comp", "is_dup")]
    df[cols].to_parquet(MANIF / "positive.parquet")
    df[cols].to_csv(MANIF / "positive.csv", index=False)
    print(pd.crosstab(df.source, df.split, margins=True))


if __name__ == "__main__":
    main()
