"""Statystyki zbioru pozytywnego -> results/dataset/*.csv (materiał do rozdziału 3)."""
import pandas as pd
from common import ROOT, INTERIM, MANIF

OUT = ROOT / "results" / "dataset"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(MANIF / "positive.parquet")

pd.crosstab(df.source, df.split, margins=True).to_csv(OUT / "pos_source_split.csv")
tags = [c for c in df.columns if c.startswith("tag_")]
(df[tags] > 0.5).groupby(df.split).mean().round(3).to_csv(OUT / "pos_tags_by_split.csv")
df.assign(short=df[["orig_w", "orig_h"]].min(axis=1)).groupby("source").short.describe() \
  .round(0).to_csv(OUT / "pos_resolution.csv")
if "captured_at" in df:
    mon = pd.to_datetime(df.captured_at, errors="coerce", utc=True).dt.month
    pd.crosstab(df.source, mon).to_csv(OUT / "pos_months.csv")       # sezonowość
pd.read_csv(INTERIM / "funnel_positive.csv", index_col=0).to_csv(OUT / "pos_funnel.csv")

print(pd.crosstab(df.source, df.split, margins=True))
print((df[tags] > 0.5).mean().round(3).sort_values())
