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


def wandb_key():
    """WANDB_API_KEY z .env (plik poza gitem). Notebook z kluczem powstaje w data/ (też poza gitem)."""
    env = ROOT / ".env"
    for line in (env.read_text(encoding="utf-8").splitlines() if env.exists() else []):
        if line.startswith("WANDB_API_KEY="):
            return line.split("=", 1)[1].strip()
    return ""


def post_cells(a, cell):
    """Po treningu: runy z poprzedniej sesji (wynik notebooka podpięty jako wejście) -> test odporności,
    eksport ONNX (FP32/FP16/INT8) i próba LiteRT w osobnym środowisku (ai-edge-torch wymaga własnej wersji torch)."""
    if not a.post_runs:
        return []
    runs = a.post_runs.split(",")
    return [
        cell(f"POST = {runs!r}\n"
             "import zipfile, glob\n"
             "src = glob.glob('/kaggle/input/**/runs.zip', recursive=True)[0]\n"
             "zf = zipfile.ZipFile(src)\n"
             "zf.extractall('/kaggle/working/post', [m for m in zf.namelist() if m.split('/')[0] in POST])\n"
             "print('runy do eksportu:', sorted(os.listdir('/kaggle/working/post')))"),
        cell("!pip install -q onnx onnxruntime onnxconverter-common\n"
             "for r in POST:\n"
             "    !cd /kaggle/working/code/src && python -m train.robustness /kaggle/working/post/{r} --data-root $DATA --n 3000\n"
             "    !cd /kaggle/working/code/src && python -m train.export /kaggle/working/post/{r} --data-root $DATA --check 1000"),
        cell("# LiteRT: osobne środowisko; gdy venv się nie da utworzyć – instalacja w głównym (trening już skończony)\n"
             "import subprocess\n"
             "PY = '/tmp/lrt/bin/python'\n"
             "if subprocess.run('python -m venv /tmp/lrt', shell=True).returncode != 0:\n"
             "    PY = 'python'\n"
             "r = subprocess.run(f'{PY} -m pip install -q ai-edge-torch ai-edge-litert timm pandas', shell=True)\n"
             "if r.returncode != 0:\n"
             "    subprocess.run(f'{PY} -m pip install -q litert-torch ai-edge-litert timm pandas', shell=True)\n"
             "runs = ' '.join(f'/kaggle/working/post/{r}' for r in POST)\n"
             "subprocess.run(f'cd /kaggle/working/code/src && {PY} -m train.litert_export {runs}', shell=True)"),
        cell("import shutil\n"
             "shutil.make_archive('/kaggle/working/post', 'zip', '/kaggle/working/post')\n"
             "shutil.rmtree('/kaggle/working/post')"),
    ]


def notebook(a):
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    cell = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    # pusty plik (__init__.py) -> komentarz: %%writefile z pustą treścią jest błędem w Jupyterze
    write = [cell(f"%%writefile /kaggle/working/code/{f.relative_to(ROOT).as_posix()}\n"
                  + (f.read_text(encoding="utf-8") or "# pakiet\n")) for f in FILES]
    q = f"--hours {a.hours}" + (f" --groups {a.groups}" if a.groups else "") + (f" --priority {a.priority}" if a.priority else "")
    return {"cells": [
        md(f"# Trening modeli – sesja {a.session}\n\nAccelerator **GPU T4 x2**, Internet **On**, Input: `{DATASET}`.\n"
           f"Kolejka: `{q}`. Wynik: `runs.zip` w Output."),
        cell("!mkdir -p /kaggle/working/code/src/train /kaggle/working/code/configs\n"
             "!pip install -q timm wandb"),
        # klucz W&B: sekret Kaggle WANDB_API_KEY (Add-ons -> Secrets), a gdy go nie ma – wartość z lokalnego .env
        cell("import os\n"
             "try:\n"
             "    from kaggle_secrets import UserSecretsClient\n"
             "    os.environ['WANDB_API_KEY'] = UserSecretsClient().get_secret('WANDB_API_KEY')\n"
             "except Exception:\n"
             f"    os.environ['WANDB_API_KEY'] = {wandb_key()!r}\n"
             "print('W&B:', 'klucz ustawiony' if os.environ['WANDB_API_KEY'] else 'BRAK klucza – trening bez W&B')"),
        *write,
        cell("import glob, os, torch, zipfile\n"
             "found = glob.glob('/kaggle/input/**/manifest.parquet', recursive=True)\n"
             "if not found:   # Kaggle nie rozpakował archiwum -> rozpakuj na dysk roboczy\n"
             "    z = glob.glob('/kaggle/input/**/*.zip', recursive=True)[0]\n"
             "    zipfile.ZipFile(z).extractall('/kaggle/working/data')\n"
             "    found = glob.glob('/kaggle/working/data/**/manifest.parquet', recursive=True)\n"
             "DATA = os.path.dirname(found[0])\n"
             "print('dane:', DATA, '| GPU:', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])\n"
             "assert torch.cuda.device_count() >= 1"),
        # jednorazowy cache uint8 na lokalnym dysku: dekodowanie raz zamiast co epokę, wolny /kaggle/input tylko raz
        cell("!cd /kaggle/working/code/src && python -m train.cache --data-root $DATA --out /tmp/cache --workers 4"),
        cell(f"!cd /kaggle/working/code/src && python -m train.runqueue --data-root $DATA --out /kaggle/working/runs "
             f"--cache-dir /tmp/cache --gpus 2 --workers 2 {q}") if (a.groups or a.priority) else md("(bez treningów)"),
        *post_cells(a, cell),
        cell("import shutil, os\nos.makedirs('/kaggle/working/runs', exist_ok=True)\n"
             "shutil.make_archive('/kaggle/working/runs', 'zip', '/kaggle/working/runs')\n"
             "shutil.rmtree('/kaggle/working/runs'); shutil.rmtree('/kaggle/working/code')\n"
             "shutil.rmtree('/kaggle/working/data', ignore_errors=True)   # rozpakowane dane nie trafiają do Output\n"
             "!ls -la /kaggle/working"),
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--priority", default=None)
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--post-runs", default=None, help="runy z poprzedniej sesji do testu odporności i eksportu")
    ap.add_argument("--post-kernel", default=None, help="notebook, którego wynik (runs.zip) podpinamy, np. trail-train-s1")
    a = ap.parse_args()
    out = ROOT / "data" / "kaggle" / f"train_s{a.session}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.ipynb").write_text(json.dumps(notebook(a), ensure_ascii=False, indent=1), encoding="utf-8")
    slug = f"trail-train-s{a.session}"
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{slug}", "title": slug, "code_file": "train.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [DATASET], "competition_sources": [],
        "kernel_sources": [f"{USER}/{a.post_kernel}"] if a.post_kernel else [], "model_sources": []}, indent=1))
    print("gotowe:", out)


if __name__ == "__main__":
    main()
