"""Rozwija configs/experiments.yaml w listę runów (nazwa + nadpisania).

    python -m train.experiments [--groups E1_modele,E3_krzywa_ai] [--priority P1]   -> lista runów i liczba
"""
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def run_name(group, o, seed):
    parts = [group.split("_")[0], o["model"], o.get("head", "multi")]
    if o.get("ai_train_n") is not None:
        parts.append(f"ai{o['ai_train_n']}")
    if o.get("img_size"):
        parts.append(f"r{o['img_size']}")
    return "_".join(parts + [f"s{seed}"])


def expand(groups=None, priority=None, path=ROOT / "configs" / "experiments.yaml"):
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = []
    for g, gs in spec["groups"].items():
        if groups and g not in groups:
            continue
        if priority and gs["priority"] not in priority:
            continue
        for o in gs["runs"]:
            for s in gs.get("seeds", spec["seeds"]):
                name = run_name(g, o, s)
                out.append((name, {**o, "seed": s, "name": name, "group": g}))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default=None)
    ap.add_argument("--priority", default=None, help="np. P1 albo P1,P2")
    a = ap.parse_args()
    runs = expand(a.groups.split(",") if a.groups else None, a.priority.split(",") if a.priority else None)
    for n, o in runs:
        print(n, o)
    print("runów:", len(runs))
