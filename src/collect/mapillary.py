"""Mapillary: crowdsourcingowe zdjęcia uliczne/szlakowe (CC BY-SA 4.0).

Zostają zdjęcia <=20 m od ścieżki z OSM i >=60 m od drogi, max 6 klatek z sekwencji.
Wymaga MAPILLARY_TOKEN w .env.
"""
import os, random
from collections import defaultdict
from datetime import datetime, timezone
import mercantile, numpy as np, shapely
from shapely.geometry import LineString
from pyproj import Transformer
from tqdm import tqdm
from common import Meta, uid, get, S, cli_limit

H = {"Authorization": f"OAuth {os.environ['MAPILLARY_TOKEN']}"}
API = "https://graph.mapillary.com/images"
OVERPASS = "https://overpass-api.de/api/interpreter"
FIELDS = "id,thumb_1024_url,captured_at,sequence,computed_geometry,geometry,is_pano,creator,camera_type"

# (west, south, east, north) — przybliżone, popraw na bboxfinder.com
REGIONS = {
    "tatry":      (19.70, 49.15, 20.30, 49.32),
    "beskid_zyw": (19.10, 49.45, 19.70, 49.65),
    "gorce":      (19.95, 49.48, 20.30, 49.62),
    "pieniny":    (20.30, 49.36, 20.55, 49.45),
    "karkonosze": (15.45, 50.70, 15.85, 50.85),
    "bieszczady": (22.30, 49.00, 22.80, 49.25),   # holdout geograficzny -> test_ood
}
HOLDOUT = {"bieszczady"}
PATHS = "path|footway|track|bridleway|steps"
ROADS = ("motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|"
         "living_street|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")
PATH_MAX_M, ROAD_MIN_M, PER_SEQ, PER_REGION = 20, 60, 6, 2500
BAD_CAM = {"fisheye", "spherical", "equirectangular"}
TR = Transformer.from_crs(4326, 3035, always_xy=True)   # metry (ETRS89-LAEA)


def osm_lines(bbox, regex):
    w, s, e, n = bbox
    q = f'[out:json][timeout:300];way["highway"~"^({regex})$"]({s},{w},{n},{e});out geom;'
    r = S.post(OVERPASS, data={"data": q}, timeout=360)
    r.raise_for_status()
    lines = []
    for el in r.json()["elements"]:
        g = el.get("geometry")
        if g and len(g) > 1:
            xs, ys = TR.transform([p["lon"] for p in g], [p["lat"] for p in g])
            lines.append(LineString(np.c_[xs, ys]))
    return lines


def nearest(lines, pts, max_d):
    d = np.full(len(pts), np.inf)
    if lines:
        tree = shapely.STRtree(lines)
        idx, dist = tree.query_nearest(pts, max_distance=max_d, return_distance=True,
                                       all_matches=False)
        d[idx[0]] = dist
    return d


def list_images(bbox):
    out, stack = {}, list(mercantile.tiles(*bbox, zooms=14))
    while stack:
        t = stack.pop()
        b = mercantile.bounds(t)
        r = get(API, headers=H, params={"fields": FIELDS, "limit": 2000,
                                        "bbox": f"{b.west},{b.south},{b.east},{b.north}"})
        if r is None:
            continue
        data = r.json().get("data", [])
        if len(data) >= 2000 and t.z < 18:      # limit 2000 -> dziel kafelek
            stack.extend(mercantile.children(t))
            continue
        out.update({im["id"]: im for im in data})
    return list(out.values())


def coords(im):
    return (im.get("computed_geometry") or im.get("geometry") or {}).get("coordinates")


def main():
    limit = cli_limit()
    meta, rng = Meta("mapillary"), random.Random(0)
    for region, bbox in REGIONS.items():
        imgs = [im for im in list_images(bbox)
                if not im.get("is_pano") and im.get("camera_type") not in BAD_CAM and coords(im)]
        print(f"{region}: {len(imgs)} zdjęć w bboxie")
        if not imgs:
            continue
        xy = np.array([coords(im)[:2] for im in imgs])
        xs, ys = TR.transform(xy[:, 0], xy[:, 1])
        pts = shapely.points(xs, ys)
        ok = (nearest(osm_lines(bbox, PATHS), pts, PATH_MAX_M) <= PATH_MAX_M) & \
             (nearest(osm_lines(bbox, ROADS), pts, ROAD_MIN_M) >= ROAD_MIN_M)
        by_seq = defaultdict(list)
        for im, k in zip(imgs, ok):
            if k:
                by_seq[im.get("sequence")].append(im)
        chosen = []
        for lst in by_seq.values():
            lst.sort(key=lambda x: x.get("captured_at") or 0)
            chosen += lst[::max(1, len(lst) // PER_SEQ)][:PER_SEQ]
        rng.shuffle(chosen)
        chosen = chosen[:PER_REGION]
        print(f"{region}: {int(ok.sum())} na szlaku, {len(by_seq)} sekwencji, biorę {len(chosen)}")

        for im in tqdm(chosen, desc=region):
            if limit and meta.added >= limit:
                break
            id_ = uid("mapillary", im["id"])
            if meta.has(id_):
                continue
            r = get(im["thumb_1024_url"]) if im.get("thumb_1024_url") else None
            if r is None:   # URL miniatur wygasają -> odśwież
                rr = get(f"https://graph.mapillary.com/{im['id']}", headers=H,
                         params={"fields": "thumb_1024_url"})
                url = rr.json().get("thumb_1024_url") if rr else None
                r = get(url) if url else None
            if r is None:
                continue
            lon, lat = coords(im)[:2]
            ts = im.get("captured_at")
            meta.add(id_, r.content, source_id=im["id"], group=f"mly_seq:{im.get('sequence')}",
                     region=region, holdout="geo" if region in HOLDOUT else None,
                     lon=lon, lat=lat, camera_type=im.get("camera_type"),
                     captured_at=datetime.fromtimestamp(ts / 1000, timezone.utc).isoformat() if ts else None,
                     author=(im.get("creator") or {}).get("username"),
                     license="CC BY-SA 4.0")
        if limit and meta.added >= limit:
            break
    meta.close()


if __name__ == "__main__":
    main()
