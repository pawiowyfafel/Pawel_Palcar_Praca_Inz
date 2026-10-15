"""Publiczne zbiory manipulacji (negatywy "manipulated"). Duże archiwa czytamy fragmentami (HTTP Range).

  defacto   – DEFACTO inpainting (HF bahakirbasoglu/DefactoInpaintingDataset, zip 14.6 GB): 1 000 -> trening
  cocoglide – CocoGlide (TruFor, inpainting dyfuzyjny GLIDE, zip 120 MB): 512 -> test_ood
  imd2020   – IMD2020 (HF KhangTruong/IMD2020, split test, parquet): 500 -> test_ood
Uruchomienie: python -m collect.forgery [--only defacto|cocoglide|imd2020]
"""
import io, random, zipfile
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tqdm import tqdm
from common import Meta, uid, cli_limit, cli_choice
from collect.coco import HttpFile

ZIPS = {
    "defacto": ("https://huggingface.co/datasets/bahakirbasoglu/DefactoInpaintingDataset/resolve/main/dataset.zip",
                "dataset/inpainting_img/img/", 1000, None, "DEFACTO (research)"),
    "cocoglide": ("https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip",
                  "fake/", 512, "source", "CocoGlide (research, COCO-based)"),
}
IMD, IMD_N = "datasets/KhangTruong/IMD2020/data/test-*.parquet", 500


def from_zip(meta, name, limit):
    url, prefix, n, holdout, lic = ZIPS[name]
    zf = zipfile.ZipFile(io.BufferedReader(HttpFile(url), 1 << 16))
    names = sorted(x for x in zf.namelist() if x.startswith(prefix) and not x.endswith("/"))
    random.Random(0).shuffle(names)
    print(f"{name}: {len(names)} plików, biorę {n}")
    for x in tqdm(names[:n], desc=name):
        if limit and meta.added >= limit:
            return
        id_ = uid(name, x)
        if not meta.has(id_):
            meta.add(id_, zf.read(x), source_id=x, group=f"{name}:{x}", label=0, subclass="manipulated",
                     manip=f"public_{name}", holdout=holdout, license=lic)


def imd2020(meta, limit):
    fs = HfFileSystem()
    files = sorted(fs.glob(IMD))
    got = 0
    for f in files:
        pf = pq.ParquetFile(fs.open(f, "rb", cache_type="none"), pre_buffer=False)
        for rg in range(pf.metadata.num_row_groups):
            if got >= IMD_N or (limit and meta.added >= limit):
                return
            imgs = pf.read_row_group(rg, columns=["image"]).column("image").to_pylist()
            for k, im in enumerate(imgs):
                id_ = uid("imd2020", f"{f}:{rg}:{k}")
                if got >= IMD_N:
                    break
                got += 1
                if not meta.has(id_) and im and im.get("bytes"):
                    meta.add(id_, im["bytes"], source_id=f"{f.rsplit('/', 1)[-1]}:{rg}:{k}",
                             group=f"imd2020:{id_}", label=0, subclass="manipulated",
                             manip="public_imd2020", holdout="source", license="IMD2020 (research)")
            print(f"imd2020: {got}/{IMD_N}")


def main():
    limit = cli_limit()
    only = cli_choice("only", ["defacto", "cocoglide", "imd2020", "all"], "all")
    meta = Meta("forgery")
    for name in ZIPS:
        if only in ("all", name):
            from_zip(meta, name, limit)
    if only in ("all", "imd2020"):
        imd2020(meta, limit)
    meta.close()


if __name__ == "__main__":
    main()
