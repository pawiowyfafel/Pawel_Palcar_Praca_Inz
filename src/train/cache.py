"""Jednorazowy cache zdjęć: dekodowanie JPEG raz na sesję zamiast w każdej epoce.

Każde zdjęcie z manifestu -> krótszy bok 256 px -> środek 256×256 -> uint8 w pliku memmap (N×256×256×3).
Na Kaggle ~7,5 GB na lokalnym dysku; system trzyma plik w pamięci podręcznej, więc dwa treningi równolegle
czytają te same dane bez kopiowania. Ewaluacja "Resize(256) + CenterCrop(224)" = środek 224 z cache'u.
    python -m train.cache --data-root /kaggle/input/trail-dataset --out /tmp/cache [--workers 4]
"""
import argparse, json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image

SIZE = 256


def load_square(path, size=SIZE):
    with Image.open(path) as im:
        im.draft("RGB", (size, size))                    # szybsze dekodowanie JPEG ze skalowaniem DCT
        im = im.convert("RGB")
        s = size / min(im.size)
        im = im.resize((max(size, round(im.width * s)), max(size, round(im.height * s))), Image.BILINEAR)
        l, t = (im.width - size) // 2, (im.height - size) // 2
        return np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.uint8)


def _work(args):
    i, path, out, n = args
    mm = np.lib.format.open_memmap(out, mode="r+")
    mm[i] = load_square(path)
    return i


def build(data_root, out, workers=4):
    data_root, out = Path(data_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(data_root / "manifest.parquet").reset_index(drop=True)
    arr = out / "images.npy"
    done = out / "done.json"
    if done.exists() and json.loads(done.read_text())["n"] == len(df):
        print("cache gotowy:", arr)
        return
    np.lib.format.open_memmap(arr, mode="w+", dtype=np.uint8, shape=(len(df), SIZE, SIZE, 3))
    t = time.time()
    jobs = [(i, str(data_root / p), str(arr), len(df)) for i, p in enumerate(df.path)]
    with Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(_work, jobs, chunksize=64)):
            if k % 5000 == 0:
                print(f"cache {k}/{len(df)}  {time.time() - t:.0f} s", flush=True)
    df.assign(cache_idx=np.arange(len(df))).to_parquet(out / "manifest.parquet")
    done.write_text(json.dumps({"n": len(df), "size": SIZE, "s": round(time.time() - t)}))
    print(f"cache gotowy: {len(df)} zdjęć w {time.time() - t:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    build(a.data_root, a.out, a.workers)
