"""Unsplash Lite (25k zdjęć, licencja pozwala na użytek komercyjny i niekomercyjny).

Archiwum (~320 MB, same metadane TSV) pobierane jest automatycznie,
a obrazy ściągane z CDN Unsplash w szerokości 1024 px.
"""
import glob, zipfile
import pandas as pd
from tqdm import tqdm
from common import Meta, uid, get, S, RAW, cli_limit

ZIP_URL = "https://unsplash.com/data/lite/latest"
D = RAW / "unsplash" / "lite"
KEEP = {"mountain", "mountains", "hiking", "hike", "trail", "path", "footpath", "forest",
        "woods", "woodland", "valley", "alps", "peak", "ridge", "trekking", "glacier",
        "canyon", "waterfall", "creek", "stream", "meadow", "snow", "hill", "wilderness"}
DROP = {"city", "building", "architecture", "street", "urban", "town", "village", "house",
        "interior", "room", "beach", "ocean", "sea", "coast", "car", "road", "food",
        "portrait", "fashion", "drone", "aerial"}
PER_PHOTOGRAPHER, MAX_RAW = 15, 6000


def ensure_lite():
    if glob.glob(str(D / "photos.*")):
        return
    D.mkdir(parents=True, exist_ok=True)
    z = D.parent / "lite.zip"
    print("pobieram Unsplash Lite (~320 MB)...")
    with S.get(ZIP_URL, stream=True, timeout=120) as r, z.open("wb") as f:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(D)
    z.unlink()


def read(name):
    files = sorted(glob.glob(str(D / f"{name}.*")))
    return pd.concat([pd.read_csv(f, sep="\t" if ".tsv" in f else ",", low_memory=False)
                      for f in files], ignore_index=True)


def main():
    limit = cli_limit()
    ensure_lite()
    photos, kw = read("photos"), read("keywords")
    kw["keyword"] = kw["keyword"].astype(str).str.lower().str.strip()
    kw["s"] = kw.keyword.isin(KEEP).astype(int) - 2 * kw.keyword.isin(DROP).astype(int)
    score = kw.groupby("photo_id")["s"].sum()
    sel = photos[photos.photo_id.isin(score[score > 0].index)].sample(frac=1, random_state=0)
    sel = sel[sel.groupby("photographer_username").cumcount() < PER_PHOTOGRAPHER].head(MAX_RAW)
    print("kandydatów:", len(sel))

    meta = Meta("unsplash")
    for r in tqdm(sel.itertuples(), total=len(sel)):
        if limit and meta.added >= limit:
            break
        id_ = uid("unsplash", r.photo_id)
        if meta.has(id_):
            continue
        resp = get(f"{r.photo_image_url}?w=1024&fm=jpg&q=90")
        if resp is None:
            continue
        meta.add(id_, resp.content, source_id=r.photo_id, url=r.photo_url,
                 author=r.photographer_username, group=f"unsplash:{r.photographer_username}",
                 camera=f"{getattr(r, 'exif_camera_make', '')} {getattr(r, 'exif_camera_model', '')}".strip(),
                 lat=getattr(r, "photo_location_latitude", None),
                 lon=getattr(r, "photo_location_longitude", None),
                 license="Unsplash License (Lite dataset)")
    meta.close()


if __name__ == "__main__":
    main()
