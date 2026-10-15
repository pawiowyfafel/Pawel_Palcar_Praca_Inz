"""Przygotowanie paczki wejściowej dla Kaggle: jobs.csv + zdjęcia źródłowe + maski + run_gen.py.

AI (scenariusz ataku, decyzja 2026-10-14):
  sd15, sdxl : po 500 (train/val) + 125 (test), ~1/3 img2img
  flux       : 250 (test_ood – generator niewidziany w treningu)
Inpainting dyfuzyjny ("fałszywy raport warunków"):
  sd15inp    : 300 (train/val) + 150 (test)
  sdxlinp    : 150 (test_ood)
Źródła img2img/inpaintingu = pozytywy z tego samego splitu co rola zadania (dziedziczą group),
tylko z Mapillary/Unsplash/Open Images/Wikimedia (Places365 ma 256 px -> rozmyte wyniki = skrót).
Wynik: data/kaggle/trail-gen-input.zip
"""
import csv, random, shutil
import pandas as pd
from PIL import Image
from common import DATA, ROOT, MANIF, uid, load_all_meta
from synth.manipulate_cpu import freeform_mask

OUT = DATA / "kaggle" / "trail-gen-input"
SRC_OK = {"mapillary", "unsplash", "openimages", "wikimedia"}
PLAN = [  # (gen, task, rola, ile, odsetek i2i)
    ("sd15", "ai", "train", 425, 1 / 3), ("sd15", "ai", "val", 75, 1 / 3), ("sd15", "ai", "test", 125, 1 / 3),
    ("sdxl", "ai", "train", 425, 1 / 3), ("sdxl", "ai", "val", 75, 1 / 3), ("sdxl", "ai", "test", 125, 1 / 3),
    ("flux", "ai", "test_ood", 250, 1 / 3),
    ("sd15inp", "inpaint", "train", 255, 1), ("sd15inp", "inpaint", "val", 45, 1),
    ("sd15inp", "inpaint", "test", 150, 1), ("sdxlinp", "inpaint", "test_ood", 150, 1),
]
CONDITION = ["muddy", "snowy", "icy", "with a fallen tree across it", "foggy", "flooded with puddles",
             "dry and dusty", "covered in autumn leaves", "after heavy rain", "with melting snow"]
TERRAIN = ["mountain hiking trail", "forest path in a beech forest", "rocky ridge trail", "trail through a meadow",
           "stone steps on a mountain trail", "trail through dwarf pine", "narrow path along a stream",
           "trail near a mountain hut", "trail with a wooden signpost", "path in a spruce forest"]
STYLE = ["phone photo", "amateur snapshot", "photo taken with a smartphone", "overcast day photo",
         "slightly blurry phone photo", "realistic photo", "photo, natural light"]
REGION = ["", "in the Tatra mountains", "in the Carpathians", "in the Alps", "in Poland", ""]
INPAINT = ["deep fresh snow covering the path", "a large fallen tree blocking the path",
           "a big muddy puddle flooding the path", "dry dirt path", "thick ice on the path",
           "a landslide of rocks on the path", "a deep stream crossing the path", "a broken wooden bridge"]
T2I_SIZES = {"sd15": [(768, 512), (512, 768)], "sdxl": [(1152, 896), (896, 1152)],
             "flux": [(1024, 768), (768, 1024)]}


def prompt(rng):
    p = f"{rng.choice(TERRAIN)} {rng.choice(REGION)}, {rng.choice(CONDITION)}, {rng.choice(STYLE)}"
    return " ".join(p.split())


def main():
    rng = random.Random(0)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "src").mkdir(parents=True)
    (OUT / "masks").mkdir()
    pos = pd.read_parquet(MANIF / "positive.parquet")
    pos = pos[pos.source.isin(SRC_OK) & (pos[["orig_w", "orig_h"]].max(axis=1) >= 768)]
    pos = pos.merge(load_all_meta()[["id", "raw_path"]], on="id")
    by_split = {s: g.sample(frac=1, random_state=0) for s, g in pos.groupby("split")}
    used, rows = set(), []
    for gen, task, role, n, p_i2i in PLAN:
        pool = by_split[role]
        for k in range(n):
            mode = ("inpaint" if task == "inpaint" else "i2i") if rng.random() < p_i2i else "t2i"
            id_ = uid("kaggle", f"{gen}:{role}:{k}")
            r = dict(id=id_, task=task, gen=gen, mode=mode, role=role, seed=rng.randint(0, 2 ** 31 - 1),
                     prompt="", strength="", src="", mask="", w="", h="", base_id="", group=f"kaggle:{id_}")
            if mode == "t2i":
                r["prompt"] = prompt(rng)
                r["w"], r["h"] = rng.choice(T2I_SIZES[gen])
            else:
                b = next(x for x in pool.itertuples() if x.id not in used)
                used.add(b.id)
                im = Image.open(ROOT / b.raw_path).convert("RGB")
                im.thumbnail((1024, 1024))
                im.save(OUT / "src" / f"{b.id}.jpg", quality=90)
                r.update(src=f"{b.id}.jpg", base_id=b.id, group=b.group)
                if mode == "i2i":
                    r["prompt"] = prompt(rng)
                    r["strength"] = round(rng.uniform(0.3, 0.6), 2)
                else:
                    r["prompt"] = f"{rng.choice(INPAINT)}, realistic phone photo"
                    freeform_mask(im.size, rng, 0.08, 0.25).save(OUT / "masks" / f"{id_}.png")
                    r["mask"] = f"{id_}.png"
            rows.append(r)
    with open(OUT / "jobs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    shutil.copy(ROOT / "kaggle" / "run_gen.py", OUT / "run_gen.py")
    shutil.make_archive(str(OUT), "zip", OUT)
    df = pd.DataFrame(rows)
    print(pd.crosstab([df["gen"], df["mode"]], df["role"], margins=True))
    print("paczka:", str(OUT) + ".zip")


if __name__ == "__main__":
    main()
