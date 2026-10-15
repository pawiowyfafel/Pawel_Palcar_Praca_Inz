"""Normalizacja (identyczna dla obu klas): bez EXIF, tylko zmniejszanie, JPEG z losową jakością.

Dobiera też zdjęcia według kwot na źródło i limitów na grupę – w stabilnej kolejności (hash id),
więc dodanie nowych danych nie zmienia wyboru w pozostałych źródłach. Zapisuje lejek selekcji.
Filtr CLIP: pozytywy – odrzuć "other"/tekst; negatywy "urban" – odrzuć te, które wyglądają jak szlak
(sprzeczna etykieta); negatywy syntetyczne (meme/ai/manipulated) – bez filtra.
"""
import hashlib, random, numpy as np, pandas as pd
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm
from common import ROOT, INTERIM, PROC, load_all_meta

register_heif_opener()
BIG = 10 ** 9
QUOTA = {  # pozytywy
         "places365": 8000, "mapillary": 5000, "unsplash": 4000, "openimages": 2000, "wikimedia": 1500,
         "own": BIG,
         # negatywy
         "places365_urban": 2500, "mapillary_city": 1500, "coco": 1000, "memexgen": 2000,
         "synth_meme": BIG, "synth_screen": BIG, "synth_manip": BIG, "kaggle_ai": BIG, "kaggle_inpaint": BIG}
PER_GROUP = {"unsplash": 15, "wikimedia": 20, "mapillary": 6, "mapillary_city": 3}
CLIP_OTHER_MAX, CLIP_TEXT_MAX, URBAN_TRAIL_MAX, MANIP_MIN_FRAC = 0.30, 0.5, 0.5, 0.005
NON_PHOTO_GENS = {"illustrious"}


def grayscale(im):
    a = np.asarray(im.resize((64, 64)), dtype=np.int16)
    return (np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean()) < 4


def normalize(src, dst, seed, check_gray=True):
    rng = random.Random(seed)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    if min(w, h) < 256 or max(w, h) / min(w, h) > 2.2 or (check_gray and grayscale(im)):
        return None
    target = rng.randint(384, 768)
    if min(w, h) > target:                                   # tylko zmniejszanie
        s = target / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    q = rng.randint(70, 95)
    im.save(dst, "JPEG", quality=q)                          # bez exif= -> metadane usunięte
    return {"orig_w": w, "orig_h": h, "w": im.width, "h": im.height, "jpeg_q": q}


def main():
    meta = load_all_meta()
    meta["label"] = meta.get("label", pd.Series(1, index=meta.index)).fillna(1).astype(int)
    meta["subclass"] = meta.get("subclass", pd.Series("nature", index=meta.index)).fillna("nature")
    m = meta.merge(pd.read_parquet(INTERIM / "clip.parquet"), on="id", how="inner")
    funnel = {"raw": meta.groupby("source").size()}

    pos_ok = (m.label == 1) & ((m.source == "own") | ((m.other < CLIP_OTHER_MAX) & (m.tag_text < CLIP_TEXT_MAX)))
    urban_ok = (m.subclass == "urban") & (m.trail < URBAN_TRAIL_MAX)
    synth_ok = (m.label == 0) & (m.subclass != "urban")
    if "mask_frac" in m:   # prawie niewidoczna manipulacja (<0,5% kadru) = szum etykiet
        synth_ok &= ~(m.mask_frac < MANIP_MIN_FRAC)
    if "gen" in m:         # generatory stylizowane (anime) – nie są fotorealistyczną podróbką
        synth_ok &= ~m.gen.isin(NON_PHOTO_GENS)
    m = m[pos_ok | urban_ok | synth_ok]
    funnel["clip_ok"] = m.groupby("source").size()

    m["_k"] = m.id.map(lambda s: hashlib.md5(s.encode()).hexdigest())    # stabilna kolejność
    m = m.sort_values("_k")
    m = m[m.groupby(["source", "group"]).cumcount() < m.source.map(PER_GROUP).fillna(BIG)]
    m = m[m.groupby("source").cumcount() < m.source.map(QUOTA).fillna(BIG)]
    funnel["quota"] = m.groupby("source").size()

    cache_f = INTERIM / "normalize_cache.parquet"
    cache = pd.read_parquet(cache_f).set_index("id") if cache_f.exists() else pd.DataFrame()
    recs = []
    for r in tqdm(m.itertuples(), total=len(m)):
        out_dir = PROC / ("positive" if r.label == 1 else "negative")
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{r.id}.jpg"
        if dst.exists() and r.id in cache.index:
            info = cache.loc[r.id].to_dict()
        else:
            try:   # syntetyczne zrzuty/memy bywają szare – filtr czarno-białych tylko dla zdjęć
                info = normalize(ROOT / r.raw_path, dst, seed=r.id, check_gray=r.label == 1)
            except Exception:
                info = None
        if info:
            recs.append({"id": r.id, "path": dst.relative_to(ROOT).as_posix(),
                         **{k: info[k] for k in ("orig_w", "orig_h", "w", "h", "jpeg_q")}})
    rec = pd.DataFrame(recs)
    rec.drop(columns="path").to_parquet(cache_f)
    out = m.drop(columns="_k").merge(rec, on="id")
    funnel["normalized"] = out.groupby("source").size()
    out.to_parquet(INTERIM / "normalized.parquet")
    f = pd.DataFrame(funnel).fillna(0).astype(int)
    f.to_csv(INTERIM / "funnel.csv")
    print(f)
    print(pd.crosstab(out.subclass, out.label, margins=True))


if __name__ == "__main__":
    main()
