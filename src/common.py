import argparse, hashlib, json, math, os, time
from pathlib import Path
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA = ROOT / "data"
RAW, INTERIM, PROC, MANIF = DATA / "raw", DATA / "interim", DATA / "processed", DATA / "manifests"
for p in (RAW, INTERIM, PROC, MANIF):
    p.mkdir(parents=True, exist_ok=True)

# Wikimedia i Overpass wymagają sensownego User-Agenta z kontaktem (CONTACT_EMAIL w .env)
UA = f"AGH-thesis-dataset/0.1 (engineering thesis; mailto:{os.environ.get('CONTACT_EMAIL', 'unknown')})"
S = requests.Session()
S.headers["User-Agent"] = UA


def get(url, params=None, headers=None, timeout=30, retries=4):
    for i in range(retries):
        try:
            r = S.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(5 * 2 ** i)
                continue
            return None
        except requests.RequestException:
            time.sleep(2 * 2 ** i)
    return None


def cli_limit():
    """--limit N: maks. liczba NOWYCH zdjęć w tym uruchomieniu (do przebiegów testowych)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args().limit


def uid(source, source_id):
    return hashlib.sha1(f"{source}:{source_id}".encode()).hexdigest()[:16]


def _clean(v):
    return None if isinstance(v, float) and math.isnan(v) else v


class Meta:
    """data/raw/<source>/meta.jsonl + img/. Append-only, wznawia się po przerwaniu."""

    def __init__(self, source):
        self.source = source
        self.dir = RAW / source
        (self.dir / "img").mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "meta.jsonl"
        self.ids = set()
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                self.ids = {json.loads(l)["id"] for l in f if l.strip()}
        self.f = self.path.open("a", encoding="utf-8")
        self.added = 0

    def has(self, id_):
        return id_ in self.ids

    def add(self, id_, data=None, path=None, **rec):
        if data is not None:
            p = self.dir / "img" / f"{id_}.jpg"
            p.write_bytes(data)
        else:
            p = Path(path).resolve()
        rp = p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)
        rec = {"id": id_, "source": self.source, "raw_path": rp,
               **{k: _clean(v) for k, v in rec.items()}}
        self.f.write(json.dumps(rec, ensure_ascii=False,
                                default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        self.f.flush()
        self.ids.add(id_)
        self.added += 1

    def close(self):
        self.f.close()
        print(f"[{self.source}] dodano {self.added}, razem {len(self.ids)}")


def load_all_meta():
    import pandas as pd
    frames = [pd.read_json(p, lines=True, dtype=False, convert_dates=False)
              for p in RAW.glob("*/meta.jsonl") if p.stat().st_size]
    return pd.concat(frames, ignore_index=True)


class UF:
    """Union-find do klastrów duplikatów i grup."""

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)
