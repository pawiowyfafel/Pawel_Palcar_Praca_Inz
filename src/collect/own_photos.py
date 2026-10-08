"""Własne zdjęcia (tylko test_real / T3). Wrzucaj do data/raw/own/<osoba>/ (JPG, HEIC, PNG)."""
from pillow_heif import register_heif_opener
register_heif_opener()
from PIL import Image
from common import Meta, uid, RAW

EXT = {".jpg", ".jpeg", ".heic", ".heif", ".png"}


def main():
    meta = Meta("own")
    for person in sorted(p for p in (RAW / "own").iterdir() if p.is_dir() and p.name != "img"):
        for f in sorted(person.rglob("*")):
            if f.suffix.lower() not in EXT:
                continue
            id_ = uid("own", f"{person.name}/{f.name}")
            if meta.has(id_):
                continue
            try:
                with Image.open(f) as im:
                    ex = im.getexif()
                    dt = ex.get_ifd(0x8769).get(36867) or ex.get(306) or ""
                    cam = f"{ex.get(271, '')} {ex.get(272, '')}".strip()
            except Exception:
                continue
            # GPS celowo NIE zapisujemy (RODO)
            meta.add(id_, path=f, source_id=f.name, author=person.name, camera=cam,
                     captured_at=str(dt) or None, group=f"own:{person.name}:{str(dt)[:10]}",
                     holdout="real", license="własne / za zgodą autora")
    meta.close()


if __name__ == "__main__":
    main()
