"""Statystyki zbioru (obie klasy) -> results/dataset/*.csv (materiał do rozdziału 3)."""
import pandas as pd
from common import ROOT, INTERIM, MANIF

OUT = ROOT / "results" / "dataset"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(MANIF / "dataset.parquet")

pd.crosstab([df.label, df.subclass, df.source], df.split, margins=True).to_csv(OUT / "source_split.csv")
pd.crosstab([df.label, df.subclass], df.split, margins=True).to_csv(OUT / "subclass_split.csv")
tags = [c for c in df.columns if c.startswith("tag_")]
pos = df[df.label == 1]
(pos[tags] > 0.5).groupby(pos.split).mean().round(3).to_csv(OUT / "pos_tags_by_split.csv")
df.assign(short=df[["orig_w", "orig_h"]].min(axis=1)).groupby(["label", "source"]).short.describe() \
  .round(0).to_csv(OUT / "resolution.csv")
df.groupby(["label", "subclass"]).jpeg_q.describe().round(1).to_csv(OUT / "jpeg_q.csv")   # kontrola skrótów
if "captured_at" in df:
    mon = pd.to_datetime(pos.captured_at, errors="coerce", utc=True).dt.month
    pd.crosstab(pos.source, mon).to_csv(OUT / "pos_months.csv")
pd.read_csv(INTERIM / "funnel.csv", index_col=0).to_csv(OUT / "funnel.csv")

print(pd.crosstab([df.label, df.subclass], df.split, margins=True))
print((pos[tags] > 0.5).mean().round(3).sort_values())
