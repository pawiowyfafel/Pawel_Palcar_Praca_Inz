"""Pomiar czasu inferencji modeli .tflite na telefonach z Androidem (adb + benchmark_model z LiteRT/TFLite).

    python -m train.phone_bench                       # wszystkie podłączone telefony, 4 modele E1 × 3 formaty × 3 konfiguracje
    python -m train.phone_bench --serial <id> --configs cpu1,cpu4 --num-runs 50

Wymagania: telefon z włączonym debugowaniem USB (adb devices pokazuje "device"), binarki benchmark_model
w data/tools/ (android_aarch64_benchmark_model, android_arm_benchmark_model – nightly TFLite z Google Storage).

Konfiguracje:
  cpu1 – CPU, 1 wątek (XNNPACK)          – najgorszy realistyczny przypadek (telefon zajęty innymi zadaniami)
  cpu4 – CPU, 4 wątki (XNNPACK)          – typowe ustawienie w aplikacji
  gpu  – delegat GPU (OpenCL/OpenGL)     – zapisujemy też, ile węzłów grafu przejął delegat

Wynik: results/models/telefon_benchmark.csv (dopisywany; pomiary już wykonane są pomijane – można przerwać i wznowić)
       results/models/telefon_logi/<telefon>/<model>_<format>_<konfig>.txt (surowe wyjście benchmark_model)
Czasy w ms: init (wczytanie + delegat), first (pierwsza inferencja), avg/std/min/max z --num-runs przebiegów
po --warmup-runs rozgrzewki. Między pomiarami przerwa --cooldown s (ogranicza wpływ nagrzewania – temperaturę
baterii zapisujemy przed każdym pomiarem).
"""
import argparse, os, re, shutil, subprocess, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "data" / "tools"
OUT = ROOT / "results" / "models"
REMOTE = "/data/local/tmp/trail"
RUNS = ["E1_cnn_multi_s0", "E1_mnv3s_multi_s0", "E1_mnv3l_multi_s0", "E1_effl0_multi_s0"]
CONFIGS = {"cpu1": ["--num_threads=1", "--use_xnnpack=true"],
           "cpu2": ["--num_threads=2", "--use_xnnpack=true"],
           "cpu4": ["--num_threads=4", "--use_xnnpack=true"],
           "gpu": ["--num_threads=4", "--use_gpu=true"]}
BIN = {"arm64-v8a": "android_aarch64_benchmark_model", "armeabi-v7a": "android_arm_benchmark_model",
       "armeabi": "android_arm_benchmark_model"}


def find_adb():
    a = shutil.which("adb")
    if a:
        return a
    p = Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "platform-tools" / "adb.exe"
    if p.exists():
        return str(p)
    raise SystemExit("Nie znaleziono adb (PATH ani %LOCALAPPDATA%\\Android\\Sdk\\platform-tools).")


class Phone:
    def __init__(self, adb, serial):
        self.adb, self.serial = adb, serial

    def run(self, *args, timeout=900):
        r = subprocess.run([self.adb, "-s", self.serial, *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return r.stdout + r.stderr

    def sh(self, cmd, timeout=900):
        return self.run("shell", cmd, timeout=timeout)

    def prop(self, k):
        return self.sh(f"getprop {k}").strip()

    def info(self):
        mem = re.search(r"MemTotal:\s+(\d+)", self.sh("cat /proc/meminfo"))
        soc = self.prop("ro.soc.model") or self.prop("ro.board.platform") or self.prop("ro.hardware")
        return {"telefon": f"{self.prop('ro.product.manufacturer')} {self.prop('ro.product.model')}".strip(),
                "soc": soc, "soc_producent": self.prop("ro.soc.manufacturer"),
                "android": self.prop("ro.build.version.release"), "sdk": self.prop("ro.build.version.sdk"),
                "abi": self.prop("ro.product.cpu.abi"), "rdzenie": self.sh("nproc").strip(),
                "ram_GB": round(int(mem.group(1)) / 2 ** 20, 1) if mem else None}

    def battery_temp(self):
        m = re.search(r"temperature:\s*(\d+)", self.sh("dumpsys battery"))
        return int(m.group(1)) / 10 if m else None


def parse(log):
    r = {}
    m = re.search(r"Inference timings in us: Init: ([\d.]+), First inference: ([\d.]+)", log)
    if m:
        r["init_ms"], r["first_ms"] = float(m.group(1)) / 1e3, float(m.group(2)) / 1e3
    stats = re.findall(r"count=(\d+) first=\S+ curr=\S+ min=([\d.]+) max=([\d.]+) avg=([\d.]+) std=([\d.]+)", log)
    if stats:                                        # ostatnia linia = właściwy pomiar (pierwsza to rozgrzewka)
        n, mn, mx, avg, std = stats[-1]
        r.update(n=int(n), avg_ms=float(avg) / 1e3, std_ms=float(std) / 1e3, min_ms=float(mn) / 1e3,
                 max_ms=float(mx) / 1e3)
    m = re.search(r"Memory footprint delta from the start of the tool \(MB\): init=([\d.]+) overall=([\d.]+)", log)
    if m:
        r["pamiec_init_MB"], r["pamiec_przyrost_MB"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"[Pp]eak memory footprint \(MB\)[^:]*: ([\d.]+)", log)
    if m:
        r["pamiec_szczyt_MB"] = float(m.group(1))
    m = re.findall(r"Replacing (\d+) out of (\d+) node\(s\) with delegate \((\S+)\)", log)
    if m:
        r["delegat"] = ";".join(f"{d}:{a}/{b}" for a, b, d in m)
    if "avg_ms" not in r:
        err = [l for l in log.splitlines() if re.search(r"(?i)error|fail|abort", l)]
        r["blad"] = " | ".join(err[-3:])[:400] or log.strip()[-400:]
    return r


def summarize():
    """Mediana (oraz min–max) średnich czasów ze wszystkich pomiarów i powtórzeń -> telefon_podsumowanie.csv."""
    parts = [pd.read_csv(f) for f in (OUT / "telefon_benchmark.csv", OUT / "telefon_powtorzenia.csv") if f.exists()]
    d = pd.concat(parts, ignore_index=True)
    d = d[d.avg_ms.notna()]
    key = ["telefon", "soc", "model", "format", "konfig"]
    s = d.groupby(key).agg(mediana_ms=("avg_ms", "median"), min_ms=("avg_ms", "min"), max_ms=("avg_ms", "max"),
                           n_pomiarow=("avg_ms", "size"), init_ms=("init_ms", "median"),
                           pamiec_szczyt_MB=("pamiec_szczyt_MB", "median"), MB=("MB", "first"),
                           delegat=("delegat", "first")).round(2).reset_index()
    s.to_csv(OUT / "telefon_podsumowanie.csv", index=False)
    print(s.pivot_table(index=["model", "format"], columns=["telefon", "konfig"], values="mediana_ms").round(1).to_string())
    print("zapisano:", OUT / "telefon_podsumowanie.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", default=None, help="tylko ten telefon (z 'adb devices'); domyślnie wszystkie")
    ap.add_argument("--runs", default=",".join(RUNS))
    ap.add_argument("--formats", default="fp32,int8dyn,int8")
    ap.add_argument("--configs", default="cpu1,cpu2,cpu4,gpu")
    ap.add_argument("--summary", action="store_true", help="tylko zbuduj telefon_podsumowanie.csv z zapisanych pomiarów")
    ap.add_argument("--num-runs", type=int, default=100)
    ap.add_argument("--warmup-runs", type=int, default=20)
    ap.add_argument("--cooldown", type=float, default=10)
    ap.add_argument("--csv", default="telefon_benchmark.csv")
    ap.add_argument("--repeat", action="store_true", help="mierz ponownie także to, co już jest w CSV")
    a = ap.parse_args()
    if a.summary:
        return summarize()
    adb = find_adb()
    devs = [l.split()[0] for l in subprocess.run([adb, "devices"], capture_output=True, text=True).stdout.splitlines()[1:]
            if l.strip().endswith("device")]
    if a.serial:
        devs = [d for d in devs if d == a.serial]
    if not devs:
        raise SystemExit("Brak telefonu w stanie 'device' – sprawdź kabel, debugowanie USB i zgodę na komputer na ekranie telefonu.")

    out_csv = OUT / a.csv
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    rows = []
    for serial in devs:
        ph = Phone(adb, serial)
        info = ph.info()
        print(f"\n=== {info['telefon']} ({info['soc']}, Android {info['android']}, {info['abi']}, {info['ram_GB']} GB)", flush=True)
        binary = TOOLS / BIN.get(info["abi"], "android_aarch64_benchmark_model")
        if not binary.exists():
            raise SystemExit(f"Brak {binary} – pobierz binarkę benchmark_model do data/tools/.")
        ph.sh(f"mkdir -p {REMOTE}")
        ph.run("push", str(binary), f"{REMOTE}/benchmark_model")
        ph.sh(f"chmod 755 {REMOTE}/benchmark_model")
        tag = re.sub(r"[^\w.-]+", "_", info["telefon"])
        logdir = OUT / "telefon_logi" / tag
        logdir.mkdir(parents=True, exist_ok=True)
        for run in a.runs.split(","):
            model = run.split("_")[1]
            for fmt in a.formats.split(","):
                src = ROOT / "runs" / run / "export" / f"model_{fmt}.tflite"
                if not src.exists():
                    print("brak", src)
                    continue
                remote = f"{REMOTE}/{model}_{fmt}.tflite"
                pushed = False
                for cfg in a.configs.split(","):
                    key = dict(telefon=info["telefon"], model=model, format=fmt, konfig=cfg)
                    if not a.repeat and len(done) and ((done[list(key)] == pd.Series(key)).all(axis=1)
                                                       & done.get("avg_ms", pd.Series(dtype=float)).notna()).any():
                        continue
                    if not pushed:
                        ph.run("push", str(src), remote)
                        pushed = True
                    time.sleep(a.cooldown)
                    temp = ph.battery_temp()
                    cmd = (f"{REMOTE}/benchmark_model --graph={remote} --warmup_runs={a.warmup_runs} "
                           f"--num_runs={a.num_runs} --report_peak_memory_footprint=true " + " ".join(CONFIGS[cfg]))
                    t0 = time.time()
                    log = ph.sh(cmd)
                    (logdir / f"{model}_{fmt}_{cfg}.txt").write_text(cmd + "\n\n" + log, encoding="utf-8")
                    r = {**info, **key, "MB": round(src.stat().st_size / 2 ** 20, 2), "temp_baterii_C": temp,
                         **parse(log), "czas_s": round(time.time() - t0, 1)}
                    rows.append(r)
                    print(f"{model:6s} {fmt:8s} {cfg:5s} " + (f"avg {r['avg_ms']:7.2f} ms ± {r['std_ms']:.2f}"
                          + (f"  [{r['delegat']}]" if r.get("delegat") else "") if "avg_ms" in r else f"BŁĄD: {r['blad']}"),
                          flush=True)
                    df = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)   # zapis po każdym pomiarze
                    df.to_csv(out_csv, index=False)
                ph.sh(f"rm -f {remote}")
        ph.sh(f"rm -rf {REMOTE}")
    print("\nzapisano:", out_csv)


if __name__ == "__main__":
    main()
