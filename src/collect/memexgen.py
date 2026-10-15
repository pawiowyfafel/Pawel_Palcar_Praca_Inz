"""MemeXGen (HF YZhao09/MemeXGen, CC BY 4.0): prawdziwe memy, negatywy "meme".

Każda para ma oryginał (1_orig.png) i wersję wygenerowaną przez FLUX (2_flux.png).
Bierzemy WYŁĄCZNIE oryginały (inaczej do "meme" trafiłyby obrazy AI):
  US_to_CH: oryginalne memy angielskie -> pula treningowa
  Ch_to_US: oryginalne memy chińskie  -> test_ood (inna kultura i język napisów)
"""
import random
from huggingface_hub import HfApi
from tqdm import tqdm
from common import Meta, uid, get, cli_limit

REPO = "YZhao09/MemeXGen"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
PARTS = {"US_to_CH_image_pairs": (1500, None), "Ch_to_US_image_pairs": (500, "source")}


def orig_url(folder):
    """US_to_CH: <f>/1_orig.png;  Ch_to_US: <f>/original_<uuid>.jpg (f = .../pair_<uuid>)."""
    if folder.startswith("US_to_CH"):
        return BASE + folder + "/1_orig.png"
    return BASE + folder + f"/original_{folder.rsplit('pair_', 1)[-1]}.jpg"


def main():
    limit = cli_limit()
    api, meta, rng = HfApi(), Meta("memexgen"), random.Random(0)
    for part, (n, holdout) in PARTS.items():
        folders = sorted(x.path for x in api.list_repo_tree(REPO, path_in_repo=part, repo_type="dataset"))
        rng.shuffle(folders)
        print(f"{part}: {len(folders)} par, biorę {n}")
        added = 0
        for f in tqdm(folders, desc=part):
            if added >= n or (limit and meta.added >= limit):
                break
            id_ = uid("memexgen", f)
            if meta.has(id_):
                added += 1
                continue
            r = get(orig_url(f), timeout=60)
            if r is None:
                continue
            meta.add(id_, r.content, source_id=f, url=f"https://huggingface.co/datasets/{REPO}",
                     group=f"memexgen:{f}", license="CC BY 4.0", label=0, subclass="meme",
                     meme_lang="en" if part.startswith("US") else "zh", holdout=holdout)
            added += 1
    meta.close()


if __name__ == "__main__":
    main()
