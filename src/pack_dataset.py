"""Paczka zbioru do treningu (Kaggle): data/pack/trail-dataset/{img/<id>.jpg, manifest.parquet} + zip.

Uruchamiać PO przeglądzie ręcznym i split.py (bierze data/manifests/dataset.parquet).
Obrazy zmniejszane do krótszego boku 384 px (JPEG q=95), identycznie dla wszystkich klas:
trening i tak używa 160–288 px, a mniejsze pliki = szybsze dekodowanie na Kaggle (4 rdzenie CPU)
i ~3x mniejszy upload. 384 px wystarcza też na ablację rozdzielczości do 320 px.
    python pack_dataset.py [--short 384] [--limit N] [--out data/pack/trail-dataset]
"""
import argparse, shutil
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
from common import ROOT, MANIF, DATA

COLS = ["id", "split", "label", "subclass", "source", "group", "gen", "manip", "meme_style", "region", "holdout",
        "places_class", "base_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", type=int, default=384)
    ap.add_argument("--limit", type=int, default=None, help="test techniczny: N zdjęć na split")
    ap.add_argument("--out", default=str(DATA / "pack" / "trail-dataset"))
    a = ap.parse_args()
    df = pd.read_parquet(MANIF / "dataset.parquet")
    if a.limit:
        df = pd.concat([g.sample(min(len(g), a.limit), random_state=0) for _, g in df.groupby("split")])
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    (out / "img").mkdir(parents=True, exist_ok=True)
    for r in tqdm(df.itertuples(), total=len(df)):
        dst = out / "img" / f"{r.id}.jpg"
        if dst.exists():
            continue
        with Image.open(ROOT / r.path) as im:
            im = im.convert("RGB")
            s = a.short / min(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            im.save(dst, "JPEG", quality=95)
    m = df[[c for c in COLS if c in df]].copy()
    m["path"] = "img/" + m.id + ".jpg"
    m.to_parquet(out / "manifest.parquet")
    print(pd.crosstab(m.subclass, m.split, margins=True))
    if not a.limit:
        z = shutil.make_archive(str(out), "zip", out)
        print("zip do wgrania na Kaggle:", z)


if __name__ == "__main__":
    main()
