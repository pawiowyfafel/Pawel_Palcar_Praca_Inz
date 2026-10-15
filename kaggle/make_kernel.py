"""Buduje notebook Kaggle z wklejonym aktualnym run_gen.py (repo = jedyne źródło kodu).

    python kaggle/make_kernel.py 2        -> data/kaggle/kernel_s2/ (notebook + kernel-metadata.json)
    kaggle kernels push -p data/kaggle/kernel_s2
Tworzy też kaggle/generate.ipynb (sesja 1) do ręcznego importu.
Dataset wejściowy (jobs.csv, src/, masks/) pozostaje bez zmian.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER, DATASET = "ameliaschabowy", "ameliaschabowy/trail-gen-input"
PLANS = {0: {0: ["sd15", "sd15inp", "pixart"], 1: ["sdxl", "sdxlinp"]},   # sesja testowa
         1: {0: ["sd15", "sd15inp"], 1: ["sdxl"]},
         2: {0: ["pixart"], 1: ["sdxlinp"]},
         3: {0: ["sd15inp"]}}                    # dokończenie sd15inp (sesja 1 przerwana przez filtr NSFW)
EXTRA = {0: "--limit 3", 3: "--exclude /kaggle/working/done.txt"}       # 0: po 3 obrazy na generator
# sesje dokańczające: id zadań zrobionych wcześniej (z częściowego wyniku pobranego lokalnie)
DONE = {3: "data/kaggle/run_s1/sd15inp/meta.jsonl"}


def notebook(session):
    code = (ROOT / "kaggle" / "run_gen.py").read_text(encoding="utf-8")
    md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s}
    cell = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
    return {"cells": [
        md(f"# Generacja obrazów AI i inpaintingu – sesja {session}\n\n"
           "Accelerator: **GPU T4 x2**, Internet: **On**, Input: dataset `trail-gen-input`.\n"
           "Wynik: pliki `*.zip` w Output (`/kaggle/working`)."),
        cell(f"PLAN = {PLANS[session]!r}\nEXTRA = {EXTRA.get(session, '')!r}"),
        # torchao z obrazu Kaggle jest za stary dla najnowszych diffusers/peft (ImportError: FqnToConfig)
        cell("!pip uninstall -y -q torchao\n"
             "!pip install -q -U diffusers transformers accelerate bitsandbytes sentencepiece protobuf"),
        # kontrola przed długimi zadaniami: błąd importu zatrzyma notebook od razu
        cell("import torch, diffusers, transformers\n"
             "from diffusers import (AutoPipelineForText2Image, AutoPipelineForImage2Image,\n"
             "                       AutoPipelineForInpainting, PixArtSigmaPipeline)\n"
             "print('torch', torch.__version__, '| GPU:', [torch.cuda.get_device_name(i)\n"
             "      for i in range(torch.cuda.device_count())])\n"
             "print('diffusers', diffusers.__version__, '| transformers', transformers.__version__)\n"
             "assert torch.cuda.device_count() >= 1, 'brak GPU'"),
        cell("%%writefile /kaggle/working/run_gen.py\n" + code),
        *([cell("%%writefile /kaggle/working/done.txt\n" + "\n".join(
            json.loads(l)["id"] for l in (ROOT / DONE[session]).read_text(encoding="utf-8").splitlines() if l.strip()))]
          if session in DONE else []),
        cell("import glob, os, subprocess\n"
             "INPUT = os.path.dirname(glob.glob('/kaggle/input/**/jobs.csv', recursive=True)[0])\n"
             "print('dane wejściowe:', INPUT)\n"
             "procs = []\n"
             "for gpu, gens in PLAN.items():\n"
             "    cmd = ' && '.join(f'python /kaggle/working/run_gen.py --gen {g} --input {INPUT} --hours 11 {EXTRA} '\n"
             "                     f'> /kaggle/working/{g}.log 2>&1' for g in gens)\n"
             "    procs.append(subprocess.Popen(cmd, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))))\n"
             "print('uruchomiono:', PLAN)"),
        cell("import time\n"
             "from IPython.display import clear_output\n"
             "while any(p.poll() is None for p in procs):\n"
             "    clear_output(wait=True)\n"
             "    for g in [g for gens in PLAN.values() for g in gens]:\n"
             "        f = f'/kaggle/working/{g}.log'\n"
             "        if os.path.exists(f):\n"
             "            print(f'== {g}:', ''.join(open(f).readlines()[-2:]))\n"
             "    time.sleep(60)\n"
             "print('KONIEC, kody wyjścia:', [p.returncode for p in procs])"),
        cell("# sprzątanie: zostaw tylko zipy i logi (inaczej Output zawiera każdy obraz dwa razy)\n"
             "import shutil\n"
             "for gens in PLAN.values():\n"
             "    for g in gens:\n"
             "        d = f'/kaggle/working/{g}'\n"
             "        if os.path.exists(d + '.zip') and os.path.isdir(d):\n"
             "            shutil.rmtree(d)\n"
             "!ls -la /kaggle/working"),
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5}


def main():
    session = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    (ROOT / "kaggle" / "generate.ipynb").write_text(json.dumps(notebook(1), ensure_ascii=False, indent=1),
                                                     encoding="utf-8")
    out = ROOT / "data" / "kaggle" / f"kernel_s{session}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "generate.ipynb").write_text(json.dumps(notebook(session), ensure_ascii=False, indent=1), encoding="utf-8")
    slug = "trail-gen-test" if session == 0 else f"trail-gen-session-{session}"
    meta = {"id": f"{USER}/{slug}", "title": slug,
            "code_file": "generate.ipynb", "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("gotowe:", out, PLANS[session])


if __name__ == "__main__":
    main()
