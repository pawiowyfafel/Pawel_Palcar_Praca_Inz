"""Deduplikacja: pHash <=4 -> duplikat (usuwany), pHash <=10 lub CLIP >=0.95 -> wspólny klaster (ten sam split)."""
import numpy as np, pandas as pd, imagehash
from PIL import Image
from tqdm import tqdm
from common import ROOT, INTERIM, UF

PH_DUP, PH_NEAR, CLIP_NEAR, CH = 4, 10, 0.95, 256


def main():
    df = pd.read_parquet(INTERIM / "positive_normalized.parquet").reset_index(drop=True)
    n = len(df)
    hs = np.array([int(str(imagehash.phash(Image.open(ROOT / p))), 16) for p in tqdm(df.path)],
                  dtype=np.uint64)
    area = (df.orig_w * df.orig_h).values
    uf, drop = UF(n), set()

    for i0 in tqdm(range(0, n, CH), desc="phash"):
        d = np.bitwise_count(hs[i0:i0 + CH, None] ^ hs[None, :])
        for a, b in zip(*np.nonzero(d <= PH_NEAR)):
            a = a + i0
            if a < b:
                uf.union(a, b)
                if d[a - i0, b] <= PH_DUP:
                    drop.add(a if area[a] < area[b] else b)

    emb = np.load(INTERIM / "clip_emb.npy")
    ids = pd.read_parquet(INTERIM / "clip_emb_ids.parquet").id
    pos = pd.Series(np.arange(len(ids)), index=ids.values)
    E = emb[pos[df.id].values].astype(np.float32)
    for i0 in tqdm(range(0, n, 1000), desc="clip"):
        for a, b in zip(*np.nonzero(E[i0:i0 + 1000] @ E.T >= CLIP_NEAR)):
            a = a + i0
            if a < b:
                uf.union(a, b)

    df["phash"] = [f"{h:016x}" for h in hs]
    df["dup_cluster"] = [uf.find(i) for i in range(n)]
    df["is_dup"] = df.index.isin(drop)
    df.to_parquet(INTERIM / "positive_dedup.parquet")
    sizes = df.dup_cluster.value_counts()
    print(f"duplikaty: {len(drop)}, klastry wielo-elementowe: {(sizes > 1).sum()}, "
          f"największy klaster: {sizes.max()}")   # bardzo duży klaster -> obniż PH_NEAR


if __name__ == "__main__":
    main()
