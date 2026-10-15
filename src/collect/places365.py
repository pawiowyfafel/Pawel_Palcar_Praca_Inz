"""Places365-Standard (MIT CSAIL): czytanie wybranych klas z archiwum .tar (26 GB) zapytaniami HTTP Range.

Archiwum jest posortowane po klasach (każda klasa to ciągły fragment pliku), więc dla każdej
klasy wyszukiwaniem binarnym znajdujemy jej początek (próbki po 128 KB) i strumieniujemy tylko ją.
Zamiast 26 GB pobieramy ~0.5 GB.
"""
import random, tarfile
from common import Meta, uid, S, cli_limit, cli_choice

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

# --set urban: negatywy (miasto, wnętrza, inne sceny niż szlak), ~4 000 surowych
URBAN = {
    # ulica i miasto
    "s/street": 200, "d/downtown": 150, "p/parking_lot": 150, "h/highway": 120, "c/crosswalk": 100,
    "a/alley": 100, "p/plaza": 100, "r/residential_neighborhood": 120, "s/shopfront": 100,
    "g/gas_station": 80, "c/construction_site": 80, "s/skyscraper": 60,
    "a/apartment_building/outdoor": 80, "b/bridge": 60, "t/train_station/platform": 60,
    "b/bus_station/indoor": 50, "s/subway_station/platform": 60, "a/airport_terminal": 50,
    "p/parking_garage/indoor": 50,
    # wnętrza
    "k/kitchen": 120, "b/bedroom": 120, "l/living_room": 120, "o/office": 100, "h/home_office": 80,
    "d/dining_room": 80, "b/bathroom": 80, "c/corridor": 60, "l/lobby": 50, "c/classroom": 60,
    "r/restaurant": 80, "f/fastfood_restaurant": 60, "c/coffee_shop": 60, "b/bar": 60,
    "s/supermarket": 60, "s/shopping_mall/indoor": 60, "f/food_court": 40, "m/market/indoor": 40,
    "h/hospital_room": 40, "p/pharmacy": 30, "g/gymnasium/indoor": 40, "c/car_interior": 80,
    "b/bus_interior": 40, "c/church/indoor": 40,
    # inne sceny niż szlak
    "s/stadium/soccer": 50, "p/playground": 60, "a/amusement_park": 40,
    "s/swimming_pool/outdoor": 40, "g/golf_course": 40, "b/beach": 100, "c/coast": 60, "o/ocean": 40,
    # trudne negatywy: zieleń w mieście, stoki narciarskie (decyzja 2026-10-14)
    "p/park": 250, "f/formal_garden": 120, "b/botanical_garden": 120, "l/lawn": 60, "y/yard": 60,
    "c/campus": 60, "s/ski_resort": 100, "s/ski_slope": 100,
}
HARD = {"p/park", "f/formal_garden", "b/botanical_garden", "l/lawn", "y/yard", "c/campus",
        "s/ski_resort", "s/ski_slope"}
SETS = {"nature": ("places365", CLASSES, {}),
        "urban": ("places365_urban", URBAN, {"label": 0, "subclass": "urban"})}
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
    name, classes, extra = SETS[cli_choice("set", list(SETS), "nature")]
    size = int(S.head(URL, timeout=60).headers["Content-Length"])
    meta, rng = Meta(name), random.Random(0)
    for c, want in classes.items():
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
                                 places_class=c, group=f"places:{id_}", license=LICENSE,
                                 **extra, **({"hard": True} if c in HARD else {}))
                    n += 1
        print(f"{c}: {n} zdjęć (przejrzano {seen})" + ("  <-- BRAK, sprawdź nazwę klasy" if not n else ""))
        if limit and meta.added >= limit:
            break
    meta.close()


if __name__ == "__main__":
    main()
