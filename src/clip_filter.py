"""Zero-shot CLIP: ocena trail/nature/other + tagi trudnych pozytywów + wykrycie tekstu/watermarku.

Dwa etapy:
1. embeddingi obrazów (wolne, przyrostowo: tylko nowe zdjęcia) -> clip_emb.npy + clip_emb_ids.parquet
2. ocena promptami z zapisanych embeddingów (sekundy) -> clip.parquet
Zmiana promptów wymaga tylko etapu 2:  python clip_filter.py --rescore
"""
import sys
import numpy as np, pandas as pd, torch, open_clip
from PIL import Image
from pillow_heif import register_heif_opener
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from common import ROOT, INTERIM, load_all_meta

register_heif_opener()
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
GROUPS = {
    "trail": ["a photo of a hiking trail", "a photo of a mountain path", "a photo of a forest path",
              "a photo taken on a hiking trail in the mountains", "a photo of a muddy trail",
              "a photo of a snowy mountain trail", "a photo of a rocky trail",
              # trudne pozytywy (bez nich lądowały w "a screenshot")
              "a photo of a trail signpost", "a photo of hikers walking on a mountain trail",
              "a photo of a wooden footbridge on a trail", "a photo of stone steps on a mountain trail"],
    "nature": ["a photo of mountains", "a photo of a forest", "a photo of a mountain valley",
               "a photo of a waterfall", "a photo of a mountain meadow", "a photo of a mountain hut",
               "a photo of a mountain stream", "a photo of a rocky cliff", "a photo of a mountain lake",
               "a photo of a foggy forest", "a photo of a night sky over mountains"],
    "other": ["a photo of a city street", "a photo of a building", "a photo of an indoor room",
              "a photo of a beach", "a photo of a car", "a screenshot", "a meme with text",
              "a map", "a painting", "a drawing", "an aerial drone photo",
              "a black and white historical photo", "a close-up portrait of a person",
              "a photo of food", "a close-up photo of an animal", "a ski slope with skiers",
              "a photo of an asphalt road"],
}
# CLIP słabo rozumie przeczenia ("a photo with no signs"), więc tag porównujemy z neutralnym opisem
NEUTRAL = "a photo of a hiking trail in nature"
TAGS = {
    "people": "a photo of people on a trail",
    "fog":    "a foggy photo with low visibility",
    "night":  "a photo taken at night or at dusk",
    "snow":   "a photo of a trail covered with snow",
    "mud":    "a photo of mud and puddles on a path",
    "sign":   "a photo of a trail signpost or trail marker",
    "hut":    "a photo of a mountain hut or shelter",
    "text":   "an image with overlaid text or a watermark",
}


class DS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            with Image.open(ROOT / self.paths[i]) as im:
                return self.tf(im.convert("RGB")), i
        except Exception:
            return None


def collate(b):
    b = [x for x in b if x is not None]
    return (torch.stack([x[0] for x in b]), torch.tensor([x[1] for x in b])) if b else None


@torch.no_grad()
def embed(model, tf, dev):
    """Etap 1: embeddingi tylko dla zdjęć, których jeszcze nie ma w clip_emb.npy."""
    meta = load_all_meta()
    ef, idf = INTERIM / "clip_emb.npy", INTERIM / "clip_emb_ids.parquet"
    old_emb, old_ids = (np.load(ef), pd.read_parquet(idf).id) if ef.exists() else (None, pd.Series(dtype=str))
    todo = meta[~meta.id.isin(old_ids)].reset_index(drop=True)
    print(f"embeddingi do policzenia: {len(todo)} z {len(meta)}")
    if todo.empty:
        return
    dl = DataLoader(DS(todo.raw_path.tolist(), tf), batch_size=64, num_workers=4, collate_fn=collate)
    ids, embs = [], []
    for batch in tqdm(dl):
        if batch is None:
            continue
        x, idx = batch
        f = model.encode_image(x.to(dev))
        embs.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float16))
        ids.append(todo.id.values[idx.numpy()])
    if old_emb is not None:
        embs, ids = [old_emb] + embs, [old_ids.values] + ids
    np.save(ef, np.concatenate(embs))                       # wiersz i <-> id w wierszu i
    pd.DataFrame({"id": np.concatenate(ids)}).to_parquet(idf)


@torch.no_grad()
def score(model, dev):
    """Etap 2: ocena wszystkich zapisanych embeddingów promptami."""
    tok = open_clip.get_tokenizer(MODEL)

    def enc(txt):
        t = model.encode_text(tok(txt).to(dev))
        return t / t.norm(dim=-1, keepdim=True)

    E = torch.from_numpy(np.load(INTERIM / "clip_emb.npy").astype(np.float32)).to(dev)
    ids = pd.read_parquet(INTERIM / "clip_emb_ids.parquet").id.values
    prompts = [p for g in GROUPS.values() for p in g]
    owner = np.array([g for g, ps in GROUPS.items() for _ in ps])
    p = (100 * E @ enc(prompts).T).softmax(-1).cpu().numpy()
    out = {g: p[:, owner == g].sum(1) for g in GROUPS}
    out["top_prompt"] = np.array(prompts)[p.argmax(1)]
    for k, t in TAGS.items():
        out[f"tag_{k}"] = (100 * E @ enc([t, NEUTRAL]).T).softmax(-1)[:, 0].cpu().numpy()
    res = pd.DataFrame(out)
    res.insert(0, "id", ids)
    res.to_parquet(INTERIM / "clip.parquet")
    src = load_all_meta()[["id", "source"]]
    print(res.merge(src).groupby("source")[["trail", "nature", "other"]].mean().round(2))


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, tf = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=dev)
    model.eval()
    if "--rescore" not in sys.argv:
        embed(model, tf, dev)
    score(model, dev)


if __name__ == "__main__":
    main()
