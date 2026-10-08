"""Zero-shot CLIP: ocena trail/nature/other + tagi trudnych pozytywów + wykrycie tekstu/watermarku."""
import numpy as np, pandas as pd, torch, open_clip
from PIL import Image
from pillow_heif import register_heif_opener
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from common import ROOT, INTERIM, load_all_meta

register_heif_opener()
GROUPS = {
    "trail": ["a photo of a hiking trail", "a photo of a mountain path", "a photo of a forest path",
              "a photo taken on a hiking trail in the mountains", "a photo of a muddy trail",
              "a photo of a snowy mountain trail", "a photo of a rocky trail"],
    "nature": ["a photo of mountains", "a photo of a forest", "a photo of a mountain valley",
               "a photo of a waterfall", "a photo of a mountain meadow", "a photo of a mountain hut",
               "a photo of a mountain stream"],
    "other": ["a photo of a city street", "a photo of a building", "a photo of an indoor room",
              "a photo of a beach", "a photo of a car", "a screenshot", "a meme with text",
              "a map", "a painting", "a drawing", "an aerial drone photo",
              "a black and white historical photo", "a close-up portrait of a person",
              "a photo of food", "a close-up photo of an animal", "a ski slope with skiers"],
}
TAGS = {
    "people": ("a photo with people in it", "a photo with no people"),
    "fog":    ("a foggy photo", "a clear photo"),
    "night":  ("a photo taken at night", "a photo taken during the day"),
    "snow":   ("a photo with snow", "a photo without snow"),
    "mud":    ("a photo of mud or puddles on a path", "a photo of a dry path"),
    "sign":   ("a photo with a trail signpost", "a photo with no signs"),
    "hut":    ("a photo of a mountain hut or shelter", "a photo with no buildings"),
    "text":   ("an image with overlaid text or a watermark", "a photo with no text"),
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
def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, tf = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k",
                                                         device=dev)
    tok = open_clip.get_tokenizer("ViT-B-32")
    model.eval()

    def enc(txt):
        t = model.encode_text(tok(txt).to(dev))
        return t / t.norm(dim=-1, keepdim=True)

    prompts = [p for g in GROUPS.values() for p in g]
    owner = np.array([g for g, ps in GROUPS.items() for _ in ps])
    T = enc(prompts)
    TT = {k: enc(list(v)) for k, v in TAGS.items()}

    meta = load_all_meta()
    dl = DataLoader(DS(meta.raw_path.tolist(), tf), batch_size=64, num_workers=4, collate_fn=collate)
    emb = np.zeros((len(meta), 512), np.float16)
    rows = []
    for batch in tqdm(dl):
        if batch is None:
            continue
        x, idx = batch
        f = model.encode_image(x.to(dev))
        f = f / f.norm(dim=-1, keepdim=True)
        p = (100 * f @ T.T).softmax(-1).cpu().numpy()
        out = {g: p[:, owner == g].sum(1) for g in GROUPS}
        out["top_prompt"] = np.array(prompts)[p.argmax(1)]
        for k, t in TT.items():
            out[f"tag_{k}"] = (100 * f @ t.T).softmax(-1)[:, 0].cpu().numpy()
        df = pd.DataFrame(out)
        df["id"] = meta.id.values[idx.numpy()]
        rows.append(df)
        emb[idx.numpy()] = f.cpu().numpy().astype(np.float16)
    res = pd.concat(rows, ignore_index=True)
    res.to_parquet(INTERIM / "clip.parquet")
    np.save(INTERIM / "clip_emb.npy", emb)
    meta[["id"]].to_parquet(INTERIM / "clip_emb_ids.parquet")
    print(res.merge(meta[["id", "source"]]).groupby("source")[["trail", "nature", "other"]].mean().round(2))


if __name__ == "__main__":
    main()
