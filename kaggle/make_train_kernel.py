"""Buduje notebook Kaggle do treningu: wkleja aktualny kod z repo (src/train/*.py, configs/*.yaml)
i uruchamia kolejkę runów na 2× T4. Dane: dataset Kaggle z paczką pack_dataset.py (manifest.parquet + img/).

    python kaggle/make_train_kernel.py --session 1 --groups E1_modele [--priority P1] [--hours 11]
    kaggle kernels push -p data/kaggle/train_s1
Wynik sesji: /kaggle/working/runs.zip (runy: config, epochs.csv, best.pt, preds.parquet, eval.json, logi).
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER, DATASET = "ameliaschabowy", "ameliaschabowy/trail-dataset"
FILES = sorted((ROOT / "src" / "train").glob("*.py")) + sorted((ROOT / "configs").glob("*.yaml"))


def notebook(a):
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    cell = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    write = [cell(f"%%writefile /kaggle/working/code/{f.relative_to(ROOT).as_posix()}\n" + f.read_text(encoding="utf-8"))
             for f in FILES]
    q = f"--hours {a.hours}" + (f" --groups {a.groups}" if a.groups else "") + (f" --priority {a.priority}" if a.priority else "")
    return {"cells": [
        md(f"# Trening modeli – sesja {a.session}\n\nAccelerator **GPU T4 x2**, Internet **On**, Input: `{DATASET}`.\n"
           f"Kolejka: `{q}`. Wynik: `runs.zip` w Output."),
        cell("!mkdir -p /kaggle/working/code/src/train /kaggle/working/code/configs\n"
             "!pip install -q timm"),
        *write,
        cell("import glob, os, torch\n"
             "DATA = os.path.dirname(glob.glob('/kaggle/input/**/manifest.parquet', recursive=True)[0])\n"
             "print('dane:', DATA, '| GPU:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n"
             "assert torch.cuda.device_count() >= 1"),
        cell(f"!cd /kaggle/working/code/src && python -m train.runqueue --data-root $DATA --out /kaggle/working/runs "
             f"--gpus 2 --workers 2 {q}"),
        cell("import shutil\nshutil.make_archive('/kaggle/working/runs', 'zip', '/kaggle/working/runs')\n"
             "shutil.rmtree('/kaggle/working/runs'); shutil.rmtree('/kaggle/working/code')\n!ls -la /kaggle/working"),
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--priority", default=None)
    ap.add_argument("--hours", type=float, default=11.0)
    a = ap.parse_args()
    out = ROOT / "data" / "kaggle" / f"train_s{a.session}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.ipynb").write_text(json.dumps(notebook(a), ensure_ascii=False, indent=1), encoding="utf-8")
    slug = f"trail-train-s{a.session}"
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}", "title": slug, "code_file": "train.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [],
        "kernel_sources": [], "model_sources": []}, indent=1))
    print("gotowe:", out)


if __name__ == "__main__":
    main()
