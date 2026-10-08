"""Places365-Standard (MIT CSAIL): strumieniowe filtrowanie archiwum .tar (~26 GB).

Archiwum czytane jest w locie, na dysk trafiają tylko wybrane klasy (~0.5 GB).
Klasy w archiwum są ułożone alfabetycznie, więc przejście całości trwa 30–90 min.
"""
import random, tarfile
from collections import Counter
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
KEEP_P = 0.7  # losowe próbkowanie, by nie brać tylko pierwszych plików klasy


def cls_of(name):
    n = "/" + name
    for c in CLASSES:
        if f"/{c}/" in n:          # '/m/mountain/' nie pasuje do '/m/mountain_path/'
            return c
    return None


def main():
    limit = cli_limit()
    meta, cnt, rng = Meta("places365"), Counter(), random.Random(0)
    with S.get(URL, stream=True, timeout=120) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|") as tar:     # tryb strumieniowy
            for m in tar:
                if not m.isfile() or not m.name.endswith(".jpg"):
                    continue
                c = cls_of(m.name)
                if c is None or cnt[c] >= CLASSES[c] or rng.random() > KEEP_P:
                    continue
                id_ = uid("places365", m.name)
                if not meta.has(id_):
                    meta.add(id_, tar.extractfile(m).read(), source_id=m.name,
                             places_class=c, group=f"places:{id_}", license=LICENSE)
                cnt[c] += 1
                if all(cnt[k] >= v for k, v in CLASSES.items()) or (limit and meta.added >= limit):
                    break
    meta.close()
    print(dict(cnt))


if __name__ == "__main__":
    main()
