"""Przegląd ręczny obu klas w FiftyOne: złym zdjęciom nadaj tag "reject".

Uruchomienie (PowerShell, z katalogu Pawel_Palcar_Praca_Inz):
    & .\\.venv\\Scripts\\python.exe .\\src\\review.py            -> otwiera http://localhost:5151
    & .\\.venv\\Scripts\\python.exe .\\src\\review.py --export   -> tylko zapis listy odrzuconych
Tagi zapisują się w bazie FiftyOne na bieżąco; ponowne uruchomienie wczytuje istniejący zbiór (z tagami)
i dokłada tylko nowe zdjęcia. Lista odrzuconych (data/interim/rejected.txt) zapisuje się przy wyjściu,
także po Ctrl+C.

Pola do filtrowania/sortowania: label (1 = pozytyw), subclass, source, trail, other, tag_text.
Widoki do przejrzenia (nie wszystkie zdjęcia):
  pozytywy: malejąco tag_text (watermarki), malejąco other, rosnąco trail, losowe 200 z każdego źródła
  negatywy urban: malejąco trail (czy to na pewno nie szlak?), losowe 200 z każdego źródła
  meme / manipulated / ai: losowe 100 z każdego źródła (czy przeróbka się udała i jest widoczna)
Losowe próbki dają szacowany szum etykiet per źródło (do rozdziału 3).
Potem: python split.py ; python stats.py
"""
import sys
import fiftyone as fo, pandas as pd
from fiftyone import ViewField as F
from common import ROOT, INTERIM

NAME = "dataset_review"
POS_SAMPLE, NEG_SAMPLE = 200, 100      # losowe próbki na źródło (szacowanie szumu etykiet)


def setup_views(ds):
    """Gotowe widoki (lista "Unsaved view" w lewym górnym rogu aplikacji)."""
    pos, neg = ds.match(F("label") == 1), ds.match(F("label") == 0)
    views = {
        "A1 pozytywy - moze nie szlak (duze other)": pos.sort_by("other", reverse=True).limit(400),
        "A2 pozytywy - tekst lub watermark": pos.sort_by("tag_text", reverse=True).limit(200),
        "A3 pozytywy - malo szlaku (male trail)": pos.sort_by("trail").limit(300),
        "B1 negatywy urban - wygladaja jak szlak": neg.match(F("subclass") == "urban")
                                                      .sort_by("trail", reverse=True).limit(300),
    }
    for s in sorted(ds.distinct("source")):
        sub = ds.match(F("source") == s)
        n = POS_SAMPLE if sub.first().label == 1 or s in ("places365_urban", "mapillary_city", "coco") else NEG_SAMPLE
        views[f"C losowe - {s}"] = sub.take(n, seed=0)
    views["Z odrzucone (tag reject)"] = ds.match_tags("reject")
    for name, v in views.items():
        ds.save_view(name, v, overwrite=True)
    print(f"widoki: {len(views)}")


FIELDS = ["label", "subclass", "source", "trail", "other", "tag_text", "top_prompt"]


def sync(ds):
    """Zbiór przeglądu = aktualny normalized.parquet: nowe zdjęcia dodaj, usunięte skasuj,
    pola (np. etykietę po zmianie definicji) zaktualizuj. Tagi (reject) zostają."""
    df = pd.read_parquet(INTERIM / "normalized.parquet").set_index("id")
    df["label"] = df.label.astype(int)
    have = set(ds.values("uid")) if len(ds) else set()
    gone = have - set(df.index)
    if gone:
        ds.delete_samples(ds.match(F("uid").is_in(list(gone))))
    new = df[~df.index.isin(have)]
    if len(new):
        ds.add_samples([fo.Sample(filepath=str(ROOT / r.path), uid=i, **{f: getattr(r, f) for f in FIELDS})
                        for i, r in new.iterrows()])
    old = df[df.index.isin(have)]
    for f in FIELDS:
        ds.set_values(f, {i: (int(v) if f == "label" else v) for i, v in old[f].items()}, key_field="uid")
    print(f"zdjęć w przeglądzie: {len(ds)} (nowych: {len(new)}, usuniętych: {len(gone)}), "
          f"odrzuconych dotąd: {len(ds.match_tags('reject'))}")


def export(ds):
    rej = ds.match_tags("reject").values("uid")
    (INTERIM / "rejected.txt").write_text("\n".join(rej))
    print("odrzucono:", len(rej), "->", INTERIM / "rejected.txt")


def main():
    if fo.dataset_exists(NAME):
        ds = fo.load_dataset(NAME)
    else:
        ds = fo.Dataset(NAME)
        ds.persistent = True
    if "--export" in sys.argv:
        return export(ds)
    if "--views" in sys.argv:            # synchronizacja z normalized.parquet i gotowe widoki, bez aplikacji
        sync(ds)
        return setup_views(ds)

    sync(ds)
    setup_views(ds)
    try:
        session = fo.launch_app(ds)
        session.wait()
    finally:
        export(ds)


if __name__ == "__main__":
    main()
