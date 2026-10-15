"""Przegląd ręczny obu klas w FiftyOne: złym zdjęciom nadaj tag "reject", zamknij aplikację.

Pola do filtrowania/sortowania: label (1 = pozytyw), subclass, source, trail, other, tag_text.
Widoki do przejrzenia (nie wszystkie zdjęcia):
  pozytywy: malejąco tag_text (watermarki), malejąco other, rosnąco trail, losowe 200 z każdego źródła
  negatywy urban: malejąco trail (czy to na pewno nie szlak?), losowe 200 z każdego źródła
  meme / manipulated / ai: losowe 100 z każdego źródła (czy przeróbka się udała i jest widoczna)
Losowe próbki dają szacowany szum etykiet per źródło (do rozdziału 3).
Potem: python split.py && python stats.py
"""
import fiftyone as fo, pandas as pd
from common import ROOT, INTERIM

df = pd.read_parquet(INTERIM / "normalized.parquet")
ds = fo.Dataset("dataset_review", overwrite=True)
ds.add_samples([fo.Sample(filepath=str(ROOT / r.path), uid=r.id, label=int(r.label), subclass=r.subclass,
                          source=r.source, trail=float(r.trail), other=float(r.other),
                          tag_text=float(r.tag_text), top_prompt=r.top_prompt) for r in df.itertuples()])
ds.persistent = True
session = fo.launch_app(ds)
session.wait()
rej = ds.match_tags("reject").values("uid")
(INTERIM / "rejected.txt").write_text("\n".join(rej))
print("odrzucono:", len(rej))
