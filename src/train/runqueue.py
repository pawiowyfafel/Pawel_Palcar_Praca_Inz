"""Kolejka runów na 1–2 GPU (Kaggle T4 x2): każdy GPU wykonuje swoją listę po kolei (trening + ocena).

    python -m train.runqueue --data-root /kaggle/input/trail-dataset --out /kaggle/working/runs \
        [--groups E1_modele] [--priority P1] [--hours 11] [--gpus 2]
Pomija runy, które mają już summary.json (wznawianie). Nie zaczyna runu, jeśli do limitu czasu zostało mniej
niż jego szacowany czas.
"""
import argparse, os, subprocess, sys, threading, time
from pathlib import Path
from train.experiments import ROOT, expand

EST_MIN = {"cnn": 22, "mnv3s": 14, "mnv3l": 17, "effl0": 18, "effb3": 45}     # szacunek na T4


def worker(gpu, runs, args, t_end, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
    for name, o in runs:
        if time.time() + EST_MIN[o["model"]] * 60 > t_end:
            log(f"[gpu{gpu}] koniec czasu – pomijam {name} i kolejne")
            return
        sets = [f"{k}={v}" for k, v in o.items()] + [f"data_root={args.data_root}", f"out_dir={args.out}",
                                                     f"workers={args.workers}"]
        t = time.time()
        with open(Path(args.out) / f"{name}.log", "w") as f:
            r = subprocess.run([sys.executable, "-m", "train.train", "--config", str(ROOT / "configs" / "base.yaml"),
                                "--set", *sets], env=env, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT / "src")
            if r.returncode == 0:
                subprocess.run([sys.executable, "-m", "train.evaluate", str(Path(args.out) / name)], env=env,
                               stdout=f, stderr=subprocess.STDOUT, cwd=ROOT / "src")
        log(f"[gpu{gpu}] {name}: {'OK' if r.returncode == 0 else 'BŁĄD ' + str(r.returncode)} "
            f"({(time.time() - t) / 60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", default=None)
    ap.add_argument("--priority", default=None)
    ap.add_argument("--hours", type=float, default=11.0)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2, help="wątki DataLoadera na run (Kaggle: 4 CPU / 2 GPU)")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    runs = [r for r in expand(args.groups.split(",") if args.groups else None,
                              args.priority.split(",") if args.priority else None)
            if not (Path(args.out) / r[0] / "summary.json").exists()]
    runs.sort(key=lambda r: -EST_MIN[r[1]["model"]])                  # najdłuższe najpierw, potem wypełnianie
    lanes, load = [[] for _ in range(args.gpus)], [0.0] * args.gpus
    for r in runs:                                                     # przydział zachłanny do najmniej obciążonego
        g = load.index(min(load))
        lanes[g].append(r)
        load[g] += EST_MIN[r[1]["model"]]
    lock = threading.Lock()

    def log(msg):
        with lock:
            print(time.strftime("%H:%M"), msg, flush=True)

    log(f"runów: {len(runs)}, szacowany czas na GPU: {[round(x / 60, 1) for x in load]} h")
    t_end = time.time() + args.hours * 3600
    th = [threading.Thread(target=worker, args=(g, lanes[g], args, t_end, log)) for g in range(args.gpus)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    log("kolejka zakończona")


if __name__ == "__main__":
    main()
