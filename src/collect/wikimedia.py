"""Wikimedia Commons: zdjęcia szlaków z kategorii (licencje CC, per plik). W całości -> test_ood."""
import random, re, time
from collections import deque
from tqdm import tqdm
from common import Meta, uid, get, cli_limit

API = "https://commons.wikimedia.org/w/api.php"
# Kategorie sprawdzone 2026-10-08; nieistniejące są pomijane z ostrzeżeniem
SEEDS = ["Category:Hiking trails in Poland", "Category:Hiking trails in Slovakia",
         "Category:Hiking in Poland", "Category:Trails in Poland",
         "Category:Tatra Mountains", "Category:Beskid", "Category:Bieszczady"]
BAD = re.compile(r"map|logo|diagram|drawing|painting|postcard|histor|aerial|panoram|video|"
                 r"interior|church|hotel|ski|cable car|stamp|coat of arms|people", re.I)
MAX_DEPTH, MAX_FILES, WIDTH = 2, 2500, 960


def members(cat, cmtype):
    params = dict(action="query", format="json", formatversion=2, list="categorymembers",
                  cmtitle=cat, cmtype=cmtype, cmlimit=500)
    while True:
        r = get(API, params=params)
        time.sleep(0.3)
        if r is None:
            return
        d = r.json()
        yield from d.get("query", {}).get("categorymembers", [])
        if "continue" not in d:
            return
        params.update(d["continue"])


def crawl():
    seen, files, q = set(), {}, deque((c, 0) for c in SEEDS)
    while q and len(files) < MAX_FILES * 4:
        cat, depth = q.popleft()
        if cat in seen:
            continue
        seen.add(cat)
        n0 = len(files)
        for m in members(cat, "file"):
            t = m["title"]
            if t.lower().endswith((".jpg", ".jpeg")) and not BAD.search(t):
                files.setdefault(t, cat)
        if depth < MAX_DEPTH:
            for sub in members(cat, "subcat"):
                if not BAD.search(sub["title"]):
                    q.append((sub["title"], depth + 1))
        if depth == 0:
            print(f"{cat}: +{len(files) - n0} plików" + ("" if len(files) > n0 else "  (pusta/nie istnieje?)"))
    return files


def strip_html(s):
    return re.sub(r"<[^>]+>", "", s or "").strip()


def main():
    limit = cli_limit()
    cap = min(MAX_FILES, limit) if limit else MAX_FILES
    files = crawl()
    titles = list(files)
    random.Random(0).shuffle(titles)
    meta = Meta("wikimedia")
    for i in tqdm(range(0, len(titles), 50)):
        r = get(API, params=dict(action="query", format="json", formatversion=2,
                                 titles="|".join(titles[i:i + 50]), prop="imageinfo",
                                 iiprop="url|size|mime|extmetadata", iiurlwidth=WIDTH,
                                 iiextmetadatafilter="LicenseShortName|Artist|DateTimeOriginal"))
        for p in (r.json().get("query", {}).get("pages", []) if r else []):
            if meta.added >= cap:
                break
            ii = (p.get("imageinfo") or [{}])[0]
            if ii.get("mime") != "image/jpeg" or min(ii.get("width", 0), ii.get("height", 0)) < 600:
                continue
            em = ii.get("extmetadata", {})
            lic = em.get("LicenseShortName", {}).get("value", "")
            if re.search(r"\bnd\b", lic.lower()):          # pomijamy ND
                continue
            id_ = uid("wikimedia", p["title"])
            if meta.has(id_):
                continue
            img = get(ii.get("thumburl"))
            time.sleep(0.2)
            if img is None:
                continue
            author = strip_html(em.get("Artist", {}).get("value"))
            meta.add(id_, img.content, source_id=p["title"], url=ii.get("descriptionurl"),
                     license=lic, author=author, group=f"wm:{author or id_}",
                     category=files[p["title"]], holdout="source",
                     captured_at=strip_html(em.get("DateTimeOriginal", {}).get("value")) or None)
        if meta.added >= cap:
            break
    meta.close()


if __name__ == "__main__":
    main()
