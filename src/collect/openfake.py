"""OpenFake (HF ComplexDataLab/OpenFake, split test, CC BY-NC 4.0): obrazy z najnowszych generatorów -> test_ood.

Test to 13 plików parquet (~68 GB). Obrazy leżą w grupach wierszy po ~146 (~100 MB), prompty zajmują ~50 KB.
1. indeks: czytamy tylko małe kolumny (prompt, label, model) ze wszystkich grup wierszy (~30 MB),
2. wybór: fałszywe (label != real) sceny plenerowe/przyrodnicze, różne generatory,
3. pobieramy kolumnę obrazów tylko z grup wierszy z największą liczbą trafień.
"""
import re
import pandas as pd, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tqdm import tqdm
from common import Meta, uid, INTERIM, cli_limit

REPO = "datasets/ComplexDataLab/OpenFake"
WANT, MAX_PER_MODEL = 400, 60
NATURE = re.compile(r"\b(mountain|forest|trail|hik|landscape|valley|river|lake|flood|landslide|snow|"
                    r"storm|wildfire|cliff|waterfall|canyon|meadow|woods|hill|glacier|avalanche|mud|"
                    r"rainforest|national park|wilderness|path)\w*", re.I)
NOT = re.compile(r"\b(city|street|building|portrait|crowd|protest|politic|president|celebrity|logo|"
                 r"poster|cartoon|anime|text|meme|interior|room|office|stadium)\w*", re.I)
fs = HfFileSystem()


def pfile(path):
    return pq.ParquetFile(fs.open(path, "rb", cache_type="none"), pre_buffer=False)


def index(files):
    f_idx = INTERIM / "openfake_test_index.parquet"
    if f_idx.exists():
        return pd.read_parquet(f_idx)
    parts = []
    for fi, f in enumerate(tqdm(files, desc="indeks")):
        pf = pfile(f)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg, columns=["prompt", "label", "model"]).to_pandas()
            t["file"], t["rg"], t["row"] = fi, rg, range(len(t))
            parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(f_idx)
    return df


def main():
    limit = cli_limit()
    files = sorted(fs.glob(f"{REPO}/core/test-*.parquet"))
    df = index(files)
    print(f"wierszy: {len(df)}, etykiety: {df.label.value_counts().to_dict()}")
    hit = df[(df.label.astype(str).str.lower() != "real") & df.prompt.str.contains(NATURE, na=False)
             & ~df.prompt.str.contains(NOT, na=False)]
    print(f"fałszywe sceny przyrodnicze: {len(hit)}; generatory: {hit.model.value_counts().head(15).to_dict()}")
    hit = hit.sample(frac=1, random_state=0)
    hit = hit[hit.groupby("model").cumcount() < MAX_PER_MODEL]
    # grupy wierszy z największą liczbą trafień, aż uzbieramy WANT
    rgs = hit.groupby(["file", "rg"]).size().sort_values(ascending=False)
    chosen, n = [], 0
    for (fi, rg), c in rgs.items():
        chosen.append((fi, rg))
        n += c
        if n >= WANT:
            break
    print(f"pobieram {len(chosen)} grup wierszy (~{len(chosen) * 100} MB) z {n} trafieniami")

    meta = Meta("openfake")
    for fi, rg in tqdm(chosen, desc="obrazy"):
        if limit and meta.added >= limit:
            break
        rows = hit[(hit.file == fi) & (hit.rg == rg)]
        if all(meta.has(uid("openfake", f"{fi}:{rg}:{r}")) for r in rows.row):
            continue
        img = pfile(files[fi]).read_row_group(rg, columns=["image"]).column("image").to_pylist()
        for r in rows.itertuples():
            id_ = uid("openfake", f"{fi}:{rg}:{r.row}")
            if not meta.has(id_) and img[r.row] and img[r.row].get("bytes"):
                meta.add(id_, img[r.row]["bytes"], source_id=f"test:{fi}:{rg}:{r.row}", gen=r.model,
                         prompt=r.prompt, group=f"openfake:{id_}", label=0, subclass="ai",
                         holdout="generator", license="CC BY-NC 4.0 (OpenFake)")
    meta.close()


if __name__ == "__main__":
    main()
