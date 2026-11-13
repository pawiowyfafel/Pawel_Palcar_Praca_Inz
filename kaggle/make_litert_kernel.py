"""Notebook Kaggle (CPU, bez limitu GPU): eksport LiteRT/TFLite runów z wyniku wskazanej sesji (post.zip/runs.zip).

    python kaggle/make_litert_kernel.py --source trail-train-s2 --runs E1_cnn_multi_s0,E1_mnv3s_multi_s0,...
    kaggle kernels push -p data/kaggle/litert
Wynik: litert.zip (runy z export/*.tflite i export/litert.json).
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = "ameliaschabowy"
FILES = [ROOT / "src" / "train" / f for f in ("__init__.py", "data.py", "models.py", "export.py", "litert_export.py")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--runs", required=True)
    a = ap.parse_args()
    cell = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    runs = a.runs.split(",")
    cells = [cell("!mkdir -p /kaggle/working/code/src/train\n!pip install -q litert-torch ai-edge-litert ai-edge-quantizer timm torchao scikit-learn pandas")]
    cells += [cell(f"%%writefile /kaggle/working/code/src/train/{f.name}\n" + (f.read_text(encoding="utf-8") or "# pakiet\n"))
              for f in FILES]
    cells += [
        cell(f"RUNS = {runs!r}\n"
             "import glob, os, zipfile, subprocess, shutil\n"
             "src = [z for z in glob.glob('/kaggle/input/**/*.zip', recursive=True) if os.path.basename(z) in ('post.zip', 'runs.zip')]\n"
             "for z in src:\n"
             "    zf = zipfile.ZipFile(z)\n"
             "    zf.extractall('/kaggle/working/lrt', [m for m in zf.namelist() if m.split('/')[0] in RUNS])\n"
             "print(sorted(os.listdir('/kaggle/working/lrt')))\n"
             "DATA = os.path.dirname(glob.glob('/kaggle/input/**/manifest.parquet', recursive=True)[0])\n"
             "subprocess.run('cd /kaggle/working/code/src && python -m train.litert_export ' + "
             "' '.join(f'/kaggle/working/lrt/{r}' for r in RUNS) + f' --data-root {DATA}', shell=True)\n"
             "for r in RUNS:   # tylko to, co potrzebne: tflite + raport\n"
             "    for f in glob.glob(f'/kaggle/working/lrt/{r}/*') + glob.glob(f'/kaggle/working/lrt/{r}/export/*'):\n"
             "        if os.path.isfile(f) and not f.endswith(('.tflite', 'litert.json', 'config.json')):\n"
             "            os.remove(f)\n"
             "    shutil.rmtree(f'/kaggle/working/lrt/{r}/wandb', ignore_errors=True)\n"
             "shutil.make_archive('/kaggle/working/litert', 'zip', '/kaggle/working/lrt')\n"
             "shutil.rmtree('/kaggle/working/lrt'); shutil.rmtree('/kaggle/working/code')\n"
             "!ls -la /kaggle/working"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "data" / "kaggle" / "litert"
    out.mkdir(parents=True, exist_ok=True)
    (out / "litert.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/trail-litert-export", "title": "trail-litert-export", "code_file": "litert.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
        "enable_internet": True, "dataset_sources": [f"{USER}/trail-dataset"], "competition_sources": [],
        "kernel_sources": [f"{USER}/{a.source}"], "model_sources": []}, indent=1))
    print("gotowe:", out)


if __name__ == "__main__":
    main()
