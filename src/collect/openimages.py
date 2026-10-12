"""Open Images V7 (Google): zdjęcia z Flickra na licencji CC BY 2.0.

Zastępuje Flickr API (od 2025 r. klucze API tylko dla kont Pro). FiftyOne nie nadaje się tutaj,
bo obsługuje tylko 600 klas obiektów ("boxable"), a nie klasy scen jak Mountain czy Trail.

1. Strumieniowo czytamy etykiety zweryfikowane przez ludzi (2.7 GB) -> zdjęcia z naszymi klasami.
2. Strumieniowo czytamy metadane (2.7 GB) -> autor, licencja, link (atrybucja CC BY), obrót.
3. Obrazy pobieramy z publicznego bucketu S3 (tylko ~20% jest tam dostępne), a w razie 404
   z Flickra (miniatura 640 px, potem oryginał). Zmniejszamy do krótszego boku 1024 px.
Wyniki kroków 1–2 są cache'owane w data/raw/openimages/.
"""
import io
import numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
from common import Meta, uid, get, S, RAW, cli_limit

BASE = "https://storage.googleapis.com/openimages"
CLASSES_URL = f"{BASE}/v7/oidv7-class-descriptions.csv"
LABELS_URL = f"{BASE}/v7/oidv7-train-annotations-human-imagelabels.csv"
META_URL = f"{BASE}/v6/oidv6-train-images-with-labels-with-rotation.csv"
IMG_URL = "https://open-images-dataset.s3.amazonaws.com/train/{}.jpg"
D = RAW / "openimages"

# nazwy sprawdzone w oidv7-class-descriptions.csv
CLASSES = ["Hiking", "Trail", "Footpath", "Path", "Mountain", "Mountain range", "Mountain pass",
           "Ridge", "Summit", "Highland", "Hill", "Forest", "Old-growth forest", "Wilderness",
           "Valley", "Waterfall", "Glacier", "Nature reserve"]
PER_AUTHOR, MAX_RAW, SHORT = 15, 6000, 1024
META_COLS = ["ImageID", "OriginalURL", "Thumbnail300KURL", "OriginalLandingURL", "License",
             "Author", "Title", "Rotation"]


def stream_csv(url, chunk_filter, cache, stop=None):
    """Czyta duży CSV przez HTTP kawałkami, zostawia przefiltrowane wiersze, zapisuje cache."""
    if cache.exists():
        return pd.read_csv(cache)
    parts = []
    with S.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        for ch in tqdm(pd.read_csv(r.raw, chunksize=1_000_000), desc=url.rsplit("/", 1)[-1]):
            parts.append(chunk_filter(ch))
            if stop and stop(parts):
                break
    df = pd.concat(parts, ignore_index=True)
    if not stop:                       # niepełny wynik (--limit) nie trafia do cache
        df.to_csv(cache, index=False)
    return df


def main():
    limit = cli_limit()
    D.mkdir(parents=True, exist_ok=True)
    cls = pd.read_csv(io.StringIO(get(CLASSES_URL).text), header=None, names=["mid", "name"])
    mids = dict(cls[cls.name.isin(CLASSES)][["mid", "name"]].values)
    assert len(mids) == len(CLASSES), f"brak klas: {set(CLASSES) - set(mids.values())}"

    # 1. zdjęcia z co najmniej jedną z naszych klas (Confidence=1 -> potwierdzone przez człowieka)
    stop = (lambda ps: sum(map(len, ps)) >= limit * 50) if limit else None
    lab = stream_csv(LABELS_URL, lambda c: c[c.LabelName.isin(mids) & (c.Confidence == 1)],
                     D / "labels_filtered.csv", stop)
    lab["name"] = lab.LabelName.map(mids)
    labels = lab.groupby("ImageID").name.agg(lambda s: sorted(set(s)))
    print(f"zdjęć z naszymi klasami: {len(labels)}")
    print(lab.name.value_counts().to_string())

    # 2. metadane tylko dla tych zdjęć
    ids = set(labels.index)
    found = lambda ps: sum(map(len, ps)) >= len(ids)
    md = stream_csv(META_URL, lambda c: c[c.ImageID.isin(ids)][META_COLS],
                    D / "meta_filtered.csv", found if limit else None)
    # losowanie ważone 1/liczność najrzadszej klasy: inaczej ~80% to "Mountain" (panoramy),
    # a klasy szlakowe (Trail, Footpath, Path) prawie by nie wystąpiły
    freq = lab.name.value_counts()
    w = labels.map(lambda ls: max(1 / freq[l] for l in ls))
    md = md[md.ImageID.isin(ids)]
    u = np.random.default_rng(0).random(len(md))
    md = md.iloc[np.argsort(-np.log(u) / md.ImageID.map(w).values)]   # losowa kolejność ważona
    md = md[md.groupby(md.Author.fillna(md.ImageID)).cumcount() < PER_AUTHOR].head(MAX_RAW)
    print(f"kandydatów (max {PER_AUTHOR} na autora): {len(md)}")

    # 3. pobranie obrazów
    meta = Meta("openimages")
    for r in tqdm(md.itertuples(), total=len(md)):
        if limit and meta.added >= limit:
            break
        id_ = uid("openimages", r.ImageID)
        if meta.has(id_):
            continue
        # S3 ma tylko podzbiór "boxable" (~1.9 mln); reszta jest nadal na Flickrze
        resp = None
        for url in (IMG_URL.format(r.ImageID), r.Thumbnail300KURL, r.OriginalURL):
            if isinstance(url, str) and (resp := get(url, timeout=60, retries=2)) is not None:
                break
        if resp is None:
            continue
        try:
            im = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            continue
        if pd.notna(r.Rotation) and r.Rotation:
            im = im.rotate(-r.Rotation, expand=True)       # Rotation = obrót zgodnie z zegarem
        s = SHORT / min(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=95)
        author = r.Author if isinstance(r.Author, str) else None
        meta.add(id_, buf.getvalue(), source_id=r.ImageID, url=r.OriginalLandingURL,
                 author=author, group=f"oi:{author or r.ImageID}", oi_labels=labels[r.ImageID],
                 license=f"CC BY 2.0 ({r.License})")
    meta.close()


if __name__ == "__main__":
    main()
