"""Places365 – plan awaryjny: mirror na Hugging Face (parquet, ~19 GB strumienia).

Uwaga: nazwy klas w mirrorze mają spacje zamiast podkreślników ('mountain path').
"""
import io
from datasets import load_dataset
from common import Meta, uid, cli_limit
from collect.places365 import CLASSES, LICENSE

LABEL = "label"


def main():
    limit = cli_limit()
    ds = load_dataset("ljnlonoljpiljm/places365-256px", split="train", streaming=True)
    names = ds.features[LABEL].names
    want_short = {c.split("/", 1)[1]: c for c in CLASSES}       # 'mountain_path' -> 'm/mountain_path'
    want = {i: want_short[n.replace(" ", "_")] for i, n in enumerate(names)
            if n.replace(" ", "_") in want_short}
    assert len(want) == len(CLASSES), f"nie znaleziono klas: {set(CLASSES) - set(want.values())}"

    meta, cnt = Meta("places365"), {c: 0 for c in CLASSES}
    for k, ex in enumerate(ds):
        c = want.get(ex[LABEL])
        if c is None or cnt[c] >= CLASSES[c]:
            continue
        id_ = uid("places365", f"hf:{k}")
        if not meta.has(id_):
            buf = io.BytesIO()
            ex["image"].convert("RGB").save(buf, "JPEG", quality=95)
            meta.add(id_, buf.getvalue(), source_id=f"hf:{k}", places_class=c,
                     group=f"places:{id_}", license=LICENSE)
        cnt[c] += 1
        if all(cnt[x] >= v for x, v in CLASSES.items()) or (limit and meta.added >= limit):
            break
    meta.close()
    print(cnt)


if __name__ == "__main__":
    main()
