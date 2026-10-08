"""Open Images V7 (Google): zdjęcia z Flickra na licencji CC BY 2.0, pobierane przez FiftyOne.

Zastępuje Flickr API (od 2025 r. klucze API tylko dla kont Pro).
Nazwy klas sprawdzone w oidv7-class-descriptions.csv.
"""
from pathlib import Path
import fiftyone.zoo as foz
from common import Meta, uid, cli_limit

CLASSES = ["Hiking", "Trail", "Footpath", "Path", "Mountain", "Mountain range", "Mountain pass",
           "Ridge", "Summit", "Highland", "Hill", "Forest", "Old-growth forest", "Wilderness",
           "Valley", "Waterfall", "Glacier", "Nature reserve"]
MAX_RAW = 6000


def main():
    limit = cli_limit()
    n = min(MAX_RAW, limit) if limit else MAX_RAW
    ds = foz.load_zoo_dataset("open-images-v7", split="train",
                              label_types=["classifications"], classes=CLASSES,
                              max_samples=n, shuffle=True, seed=0,
                              dataset_name=f"oi_trail_pool_{n}")
    meta = Meta("openimages")
    for s in ds.iter_samples(progress=True):
        oid = Path(s.filepath).stem
        id_ = uid("openimages", oid)
        if meta.has(id_):
            continue
        pl = s["positive_labels"]
        labels = [c.label for c in pl.classifications] if pl else []
        meta.add(id_, path=s.filepath, source_id=oid, oi_labels=labels,
                 group=f"oi:{oid}", license="CC BY 2.0 (Open Images, Flickr)")
    meta.close()


if __name__ == "__main__":
    main()
