"""Przegląd ręczny w FiftyOne: zaznacz złe zdjęcia tagiem "reject", zamknij aplikację.

Widoki do przejrzenia (nie wszystkie 20k):
  sortuj malejąco po tag_text (watermarki), malejąco po other, rosnąco po trail,
  losowe 200 z każdego źródła (-> szacowany szum etykiet per źródło do rozdz. 3).
"""
import fiftyone as fo, pandas as pd
from common import ROOT, INTERIM

df = pd.read_parquet(INTERIM / "positive_normalized.parquet")
ds = fo.Dataset("positives", overwrite=True)
ds.add_samples([fo.Sample(filepath=str(ROOT / r.path), uid=r.id, source=r.source,
                          trail=float(r.trail), other=float(r.other), tag_text=float(r.tag_text),
                          top_prompt=r.top_prompt) for r in df.itertuples()])
ds.persistent = True
session = fo.launch_app(ds)
session.wait()
rej = ds.match_tags("reject").values("uid")
(INTERIM / "rejected.txt").write_text("\n".join(rej))
print("odrzucono:", len(rej))
