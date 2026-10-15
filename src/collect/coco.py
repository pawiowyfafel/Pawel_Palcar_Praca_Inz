"""COCO 2017 (val, 5k zdjęć): negatywy "urban" – jedzenie, wnętrza, zwierzęta domowe, ludzie z bliska, sport.

Z archiwum adnotacji (250 MB) czytamy zapytaniami Range tylko instances_val2017.json (~25 MB);
ten sam plik (maski obiektów) wykorzysta później synth/manipulate_cpu.py do splicingu.
Obrazy pobieramy pojedynczo z images.cocodataset.org.
"""
import io, json, random, zipfile
from collections import defaultdict
from tqdm import tqdm
from common import Meta, uid, get, S, RAW, cli_limit

ANN_ZIP = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
ANN_MEMBER = "annotations/instances_val2017.json"
IMG_URL = "http://images.cocodataset.org/val2017/{}"
D = RAW / "coco"
ANN = D / "instances_val2017.json"

# grupa -> (kategorie COCO, ile zdjęć)
GROUPS = {
    "food":    ({"pizza", "sandwich", "cake", "donut", "hot dog", "banana", "broccoli", "bowl", "cup",
                 "wine glass", "fork"}, 200),
    "indoor":  ({"couch", "bed", "dining table", "tv", "laptop", "toilet", "sink", "refrigerator",
                 "oven", "microwave", "book", "clock", "keyboard"}, 250),
    "pets":    ({"cat", "dog"}, 150),
    "people":  ({"person"}, 150),            # człowiek zajmuje >=35% kadru (selfie-podobne)
    "sport":   ({"tennis racket", "skateboard", "surfboard", "sports ball", "baseball bat",
                 "skis", "snowboard", "frisbee"}, 120),
    "traffic": ({"car", "bus", "train", "truck", "traffic light", "motorcycle", "airplane"}, 130),
}
PERSON_MIN_AREA = 0.35


class HttpFile(io.RawIOBase):
    """Plik po HTTP z obsługą seek() przez nagłówek Range (zipfile czyta tylko potrzebne bajty)."""

    def __init__(self, url):
        h = S.head(url, timeout=60, allow_redirects=True)      # HF przekierowuje na CDN
        self.url, self.pos = h.url, 0
        self.size = int(h.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        b = S.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120).content
        self.pos += len(b)
        return b

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def ensure_ann():
    if not ANN.exists():
        D.mkdir(parents=True, exist_ok=True)
        print("pobieram instances_val2017.json z archiwum (Range)...")
        with zipfile.ZipFile(io.BufferedReader(HttpFile(ANN_ZIP), 1 << 20)) as zf:
            ANN.write_bytes(zf.read(ANN_MEMBER))
    return json.loads(ANN.read_text(encoding="utf-8"))


def main():
    limit = cli_limit()
    ann = ensure_ann()
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    lic = {l["id"]: l["name"] for l in ann["licenses"]}
    imgs = {im["id"]: im for im in ann["images"]}
    names, person_area = defaultdict(set), defaultdict(float)
    for a in ann["annotations"]:
        n = cats[a["category_id"]]
        names[a["image_id"]].add(n)
        if n == "person":
            im = imgs[a["image_id"]]
            person_area[a["image_id"]] = max(person_area[a["image_id"]],
                                             a["area"] / (im["width"] * im["height"]))

    rng, used, pick = random.Random(0), set(), []
    for g, (want, n) in GROUPS.items():
        cand = [i for i in imgs if i not in used and names[i] & want
                and (g != "people" or person_area[i] >= PERSON_MIN_AREA)]
        rng.shuffle(cand)
        for i in cand[:n]:
            used.add(i)
            pick.append((g, i))
    print(f"wybrano {len(pick)} zdjęć: " + ", ".join(f"{g} {sum(1 for x, _ in pick if x == g)}" for g in GROUPS))

    meta = Meta("coco")
    for g, i in tqdm(pick):
        if limit and meta.added >= limit:
            break
        im = imgs[i]
        id_ = uid("coco", im["file_name"])
        if meta.has(id_):
            continue
        r = get(IMG_URL.format(im["file_name"]))
        if r is None:
            continue
        meta.add(id_, r.content, source_id=im["file_name"], url=im.get("flickr_url"),
                 group=f"coco:{im['file_name']}", coco_group=g, coco_labels=sorted(names[i]),
                 license=lic.get(im["license"]), label=0, subclass="urban")
    meta.close()


if __name__ == "__main__":
    main()
