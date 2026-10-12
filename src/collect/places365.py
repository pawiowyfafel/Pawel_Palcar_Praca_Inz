"""Places365-Standard (MIT CSAIL): czytanie wybranych klas z archiwum .tar (26 GB) zapytaniami HTTP Range.

Archiwum jest posortowane po klasach (każda klasa to ciągły fragment pliku), więc dla każdej
klasy wyszukiwaniem binarnym znajdujemy jej początek (próbki po 128 KB) i strumieniujemy tylko ją.
Zamiast 26 GB pobieramy ~0.5 GB.
"""
import random, tarfile
from common import Meta, uid, S, cli_limit

URL = "https://data.csail.mit.edu/places/places365/train_256_places365standard.tar"
LICENSE = "Places365 (research, non-commercial)"

# klasa -> limit surowych zdjęć (przed filtrem CLIP); w klasie jest 3068–5000 zdjęć
CLASSES = {
    "m/mountain_path": 3000, "f/forest_path": 3000, "m/mountain": 2000,
    "m/mountain_snowy": 1500, "f/forest/broadleaf": 1500, "f/forest_road": 1000,
    "v/valley": 1200, "f/field/wild": 1000, "c/creek": 800, "c/cliff": 600,
    "s/snowfield": 600, "r/river": 500, "w/waterfall": 500, "p/pasture": 400,
    "g/glacier": 300, "c/canyon": 300, "m/marsh": 300, "s/swamp": 200,
    "r/rock_arch": 200, "c/crevasse": 150,
    # trudne pozytywy: infrastruktura szlakowa
    "c/cabin/outdoor": 400, "c/chalet": 300, "h/hunting_lodge/outdoor": 200,
    "r/rope_bridge": 200, "c/campsite": 200,
}
KEEP_P = 0.7          # losowe próbkowanie, by nie brać tylko pierwszych plików klasy
PROBE, PREC = 128 * 1024, 2 * 1024 * 1024


def key(cls):
    """'f/forest/broadleaf' -> ('f', 'forest', 'broadleaf'); kolejność jak w archiwum."""
    return tuple(cls.split("/"))


def cls_of(name):
    """'data_256/f/forest/broadleaf/00000001.jpg' -> 'f/forest/broadleaf'"""
    return name.split("/", 1)[1].rsplit("/", 1)[0]


def valid_header(h):
    if h[257:262] != b"ustar":
        return False
    try:
        chk = int(h[148:156].rstrip(b"\0 ").decode() or "0", 8)
    except ValueError:
        return False
    return chk == sum(h[:148]) + 32 * 8 + sum(h[156:512])


def header_at(off):
    """Pierwszy nagłówek tar na pozycji >= off: (offset, nazwa pliku)."""
    off -= off % 512
    b = S.get(URL, headers={"Range": f"bytes={off}-{off + PROBE - 1}"}, timeout=60).content
    for i in range(0, len(b) - 511, 512):
        if valid_header(b[i:i + 512]):
            return off + i, b[i:i + 100].rstrip(b"\0").decode()
    raise RuntimeError(f"brak nagłówka tar w okolicy {off}")


def find_start(cls, size):
    """Offset nagłówka nieco przed początkiem klasy (wyszukiwanie binarne)."""
    lo, hi, k = 0, size, key(cls)
    while hi - lo > PREC:
        mid = (lo + hi) // 2
        if key(cls_of(header_at(mid)[1])) >= k:
            hi = mid
        else:
            lo = mid
    return header_at(lo)[0]


def main():
    limit = cli_limit()
    size = int(S.head(URL, timeout=60).headers["Content-Length"])
    meta, rng = Meta("places365"), random.Random(0)
    for c, want in CLASSES.items():
        start, n, seen = find_start(c, size), 0, 0
        with S.get(URL, headers={"Range": f"bytes={start}-"}, stream=True, timeout=120) as r:
            r.raise_for_status()
            with tarfile.open(fileobj=r.raw, mode="r|") as tar:
                for m in tar:
                    if not m.isfile():
                        continue
                    k = key(cls_of(m.name))
                    if k < key(c):
                        continue
                    if k > key(c) or n >= want or (limit and meta.added >= limit):
                        break
                    seen += 1
                    if rng.random() > KEEP_P:
                        continue
                    id_ = uid("places365", m.name)
                    if not meta.has(id_):
                        meta.add(id_, tar.extractfile(m).read(), source_id=m.name,
                                 places_class=c, group=f"places:{id_}", license=LICENSE)
                    n += 1
        print(f"{c}: {n} zdjęć (przejrzano {seen})" + ("  <-- BRAK, sprawdź nazwę klasy" if not n else ""))
        if limit and meta.added >= limit:
            break
    meta.close()


if __name__ == "__main__":
    main()
