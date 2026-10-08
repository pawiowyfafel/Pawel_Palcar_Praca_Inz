"""Normalizacja (identyczna dla obu klas): bez EXIF, tylko zmniejszanie, JPEG z losową jakością.

Dobiera też zdjęcia według kwot na źródło i limitów na grupę. Zapisuje lejek selekcji.
"""
import random, numpy as np, pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm
from common import ROOT, INTERIM, PROC, load_all_meta

register_heif_opener()
OUT = PROC / "positive"
BIG = 10 ** 9
QUOTA = {"places365": 8000, "mapillary": 5000, "unsplash": 4000, "openimages": 2000,
         "wikimedia": 1500, "own": BIG}
PER_GROUP = {"unsplash": 15, "wikimedia": 20, "mapillary": 6, "places365": BIG,
             "openimages": BIG, "own": BIG}
CLIP_OTHER_MAX, CLIP_TEXT_MAX = 0.30, 0.5


def grayscale(im):
    a = np.asarray(im.resize((64, 64)), dtype=np.int16)
    return (np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < 4


def normalize(src, dst, seed):
    rng = random.Random(seed)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if min(w, h) < 256 or max(w, h) / min(w, h) > 2.2 or grayscale(im):
        return None
    target = rng.randint(384, 768)
    if min(w, h) > target:                                   # tylko zmniejszanie
        s = target / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    q = rng.randint(70, 95)
    im.save(dst, "JPEG", quality=q)                          # bez exif= -> metadane usunięte
    return {"orig_w": w, "orig_h": h, "w": im.width, "h": im.height, "jpeg_q": q}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = load_all_meta()
    m = meta.merge(pd.read_parquet(INTERIM / "clip.parquet"), on="id", how="inner")
    funnel = {"raw": meta.groupby("source").size()}
    own = m.source == "own"                                  # T3 bez filtra CLIP
    m = m[own | ((m.other < CLIP_OTHER_MAX) & (m.tag_text < CLIP_TEXT_MAX))]
    funnel["clip_ok"] = m.groupby("source").size()
    m = m.sample(frac=1, random_state=0)
    m = m[m.groupby(["source", "group"]).cumcount() < m.source.map(PER_GROUP)]
    m = m[m.groupby("source").cumcount() < m.source.map(QUOTA)]
    funnel["quota"] = m.groupby("source").size()

    recs = []
    for r in tqdm(m.itertuples(), total=len(m)):
        dst = OUT / f"{r.id}.jpg"
        try:
            info = normalize(ROOT / r.raw_path, dst, seed=r.id)
        except Exception:
            info = None
        if info:
            recs.append({"id": r.id, "path": dst.relative_to(ROOT).as_posix(), **info})
    out = m.merge(pd.DataFrame(recs), on="id")
    out["label"], out["subclass"] = 1, "nature"
    funnel["normalized"] = out.groupby("source").size()
    out.to_parquet(INTERIM / "positive_normalized.parquet")
    f = pd.DataFrame(funnel).fillna(0).astype(int)
    f.to_csv(INTERIM / "funnel_positive.csv")
    print(f)


if __name__ == "__main__":
    main()
