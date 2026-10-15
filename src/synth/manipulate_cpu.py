"""Manipulacje zdjęć szlaków liczone na CPU (negatywy "manipulated"). Każda dziedziczy group po oryginale.

  removal  – LaMa (big-lama, TorchScript) wypełnia nieregularną maskę: "magiczna gumka" w telefonie
  splice   – obiekt z COCO (maska z adnotacji) wklejony na szlak; połowa na twardo,
             połowa z wygładzeniem krawędzi i dopasowaniem kolorów (LAB)
  copymove – fragment zdjęcia powielony w innym miejscu tego samego zdjęcia (OpenCV)

Uruchomienie: python -m synth.manipulate_cpu [--only removal|splice|copymove] [--limit N]
"""
import io, json, math, random
import cv2, numpy as np, pandas as pd, torch
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
from common import Meta, uid, RAW, MANIF, ROOT, load_all_meta, cli_limit, cli_choice
from synth.textlib import load_rgb

N = {"removal": 1000, "splice": 1000, "copymove": 500}
LAMA = RAW / "models" / "big-lama.pt"
COCO_ANN = RAW / "coco" / "instances_val2017.json"
DONORS = {"person", "dog", "bear", "cow", "horse", "sheep", "backpack", "car", "bicycle", "motorcycle",
          "umbrella", "bench", "suitcase", "bird", "cat", "elephant", "giraffe", "fire hydrant",
          "stop sign", "chair", "potted plant", "skis", "snowboard"}


# ---------- removal (LaMa) ----------
class Lama:
    def __init__(self):
        torch.set_num_threads(max(1, torch.get_num_threads()))
        self.model = torch.jit.load(str(LAMA), map_location="cpu").eval()

    @torch.inference_mode()
    def __call__(self, img: Image.Image, mask: Image.Image) -> Image.Image:
        a = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None]
        m = torch.from_numpy((np.asarray(mask) > 127).astype(np.float32))[None, None]
        h, w = a.shape[2:]
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        a = torch.nn.functional.pad(a, (0, pw, 0, ph), mode="reflect")
        m = torch.nn.functional.pad(m, (0, pw, 0, ph), mode="reflect")
        out = self.model(a, m)[0, :, :h, :w].permute(1, 2, 0).numpy()
        return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8))


def freeform_mask(size, rng, frac_lo=0.03, frac_hi=0.15):
    """Nieregularna maska (pociągnięcia pędzla), głównie w dolnych 2/3 kadru – tam leży ścieżka."""
    W, H = size
    target = rng.uniform(frac_lo, frac_hi)
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    cx, cy = rng.uniform(0.15, 0.85) * W, rng.uniform(0.35, 0.85) * H
    r = math.sqrt(target * W * H) / 2
    for _ in range(rng.randint(3, 7)):
        pts, x, y = [], cx + rng.uniform(-r, r), cy + rng.uniform(-r, r)
        for _ in range(rng.randint(3, 6)):
            a = rng.uniform(0, 2 * math.pi)
            x = min(max(x + math.cos(a) * r * rng.uniform(0.3, 0.9), 0), W)
            y = min(max(y + math.sin(a) * r * rng.uniform(0.3, 0.9), 0), H)
            pts.append((x, y))
        d.line(pts, fill=255, width=int(r * rng.uniform(0.4, 0.8)), joint="curve")
    return m.filter(ImageFilter.MaxFilter(9))


# ---------- splice (COCO) ----------
def load_donors():
    ann = json.loads(COCO_ANN.read_text(encoding="utf-8"))
    have = {r.source_id: r.raw_path for r in load_all_meta().query("source == 'coco'").itertuples()}
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {im["id"]: im for im in ann["images"] if im["file_name"] in have}
    out = []
    for a in ann["annotations"]:
        im = imgs.get(a["image_id"])
        if im is None or a["iscrowd"] or cats[a["category_id"]] not in DONORS \
                or not isinstance(a["segmentation"], list):
            continue
        x, y, w, h = a["bbox"]
        frac = a["area"] / (im["width"] * im["height"])
        if 0.02 <= frac <= 0.35 and x > 2 and y > 2 and x + w < im["width"] - 2 and y + h < im["height"] - 2:
            out.append({"path": have[im["file_name"]], "file": im["file_name"], "cat": cats[a["category_id"]],
                        "bbox": (x, y, w, h), "poly": a["segmentation"], "size": (im["width"], im["height"])})
    return out


def cut_object(don):
    img = Image.open(ROOT / don["path"]).convert("RGB")
    if img.size != tuple(don["size"]):
        return None
    m = Image.new("L", img.size, 0)
    for p in don["poly"]:
        ImageDraw.Draw(m).polygon(list(zip(p[::2], p[1::2])), fill=255)
    x, y, w, h = map(int, don["bbox"])
    return img.crop((x, y, x + w, y + h)), m.crop((x, y, x + w, y + h))


def color_match(obj, ref):
    """Przeniesienie średniej i odchylenia w LAB z otoczenia na obiekt (w połowie, by nie przesadzić)."""
    o = cv2.cvtColor(np.asarray(obj), cv2.COLOR_RGB2LAB).astype(np.float32)
    r = cv2.cvtColor(np.asarray(ref), cv2.COLOR_RGB2LAB).astype(np.float32)
    om, os_ = o.reshape(-1, 3).mean(0), o.reshape(-1, 3).std(0) + 1e-3
    rm, rs = r.reshape(-1, 3).mean(0), r.reshape(-1, 3).std(0) + 1e-3
    t = (o - om) / os_ * rs + rm
    t = 0.5 * o + 0.5 * t
    return Image.fromarray(cv2.cvtColor(np.clip(t, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


def splice(img, don, rng):
    cut = cut_object(don)
    if cut is None:
        return None, None
    obj, m = cut
    W, H = img.size
    s = min(rng.uniform(0.15, 0.45) * H / obj.height, 0.6 * W / obj.width)
    obj = obj.resize((max(4, int(obj.width * s)), max(4, int(obj.height * s))), Image.LANCZOS)
    m = m.resize(obj.size, Image.LANCZOS)
    x = rng.randint(0, max(0, W - obj.width))
    y = min(H - obj.height, int(rng.uniform(0.55, 0.97) * H) - obj.height)   # "stoi" na ziemi
    y = max(0, y)
    mode = rng.choice(["hard", "blend"])
    if mode == "blend":
        ref = img.crop((max(0, x - obj.width // 2), max(0, y - obj.height // 2),
                        min(W, x + obj.width * 3 // 2), min(H, y + obj.height * 3 // 2)))
        obj = color_match(obj, ref)
        m = m.filter(ImageFilter.GaussianBlur(rng.uniform(1.5, 3.5)))
    else:
        m = m.point(lambda v: 255 if v > 127 else 0)
    img = img.copy()
    img.paste(obj, (x, y), m)
    frac = (np.asarray(m) > 127).sum() / (W * H)
    return img, {"splice_mode": mode, "donor": don["file"], "donor_cat": don["cat"], "mask_frac": round(frac, 4)}


# ---------- copy-move ----------
def copymove(img, rng):
    a = np.asarray(img).copy()
    H, W = a.shape[:2]
    pw, ph = int(W * rng.uniform(0.1, 0.3)), int(H * rng.uniform(0.1, 0.3))
    sx, sy = rng.randint(0, W - pw), rng.randint(int(H * 0.3), H - ph)
    patch = a[sy:sy + ph, sx:sx + pw]
    s = rng.uniform(0.8, 1.2)
    patch = cv2.resize(patch, (max(4, int(pw * s)), max(4, int(ph * s))))
    if rng.random() < 0.3:
        patch = patch[:, ::-1]
    ph2, pw2 = patch.shape[:2]
    for _ in range(20):                                   # cel nienakładający się na źródło
        dx, dy = rng.randint(0, W - pw2), rng.randint(int(H * 0.3), H - ph2)
        if abs(dx - sx) > pw or abs(dy - sy) > ph:
            break
    mode = rng.choice(["hard", "blend"])
    if mode == "hard":
        a[dy:dy + ph2, dx:dx + pw2] = patch
    else:
        mk = np.zeros((ph2, pw2), np.float32)
        cv2.ellipse(mk, (pw2 // 2, ph2 // 2), (pw2 // 2 - 2, ph2 // 2 - 2), 0, 0, 360, 1, -1)
        mk = cv2.GaussianBlur(mk, (0, 0), max(1, min(pw2, ph2) / 12))[..., None]
        reg = a[dy:dy + ph2, dx:dx + pw2].astype(np.float32)
        a[dy:dy + ph2, dx:dx + pw2] = (mk * patch + (1 - mk) * reg).astype(np.uint8)
    return Image.fromarray(a), {"cm_mode": mode, "mask_frac": round(pw2 * ph2 / (W * H), 4)}


def main():
    limit = cli_limit()
    only = cli_choice("only", list(N) + ["all"], "all")
    meta = Meta("synth_manip")
    pos = pd.read_parquet(MANIF / "positive.parquet", columns=["id", "group", "source"])
    pos = pos[pos.source != "own"].merge(load_all_meta()[["id", "raw_path"]], on="id")
    lama = Lama() if only in ("all", "removal") else None
    donors = load_donors() if only in ("all", "splice") else None
    if donors is not None:
        print(f"obiektów-dawców z COCO: {len(donors)}")
    for k, (tech, n) in enumerate(N.items()):
        if only not in ("all", tech):
            continue
        for b in tqdm(list(pos.sample(n, random_state=10 + k).itertuples()), desc=tech):
            if limit and meta.added >= limit:
                break
            id_ = uid("synth_manip", f"{b.id}:{tech}")
            if meta.has(id_):
                continue
            rng = random.Random(id_)
            img = load_rgb(ROOT / b.raw_path)
            if tech == "removal":
                m = freeform_mask(img.size, rng)
                out, info = lama(img, m), {"mask_frac": round((np.asarray(m) > 127).mean(), 4)}
            elif tech == "splice":
                out, info = splice(img, rng.choice(donors), rng)
                if out is None:
                    continue
            else:
                out, info = copymove(img, rng)
            buf = io.BytesIO()
            out.save(buf, "JPEG", quality=95)
            meta.add(id_, buf.getvalue(), label=0, subclass="manipulated", manip=tech, base_id=b.id,
                     base_source=b.source, group=b.group, source_id=f"pos:{b.id}",
                     license="syntetyczne (własne, na zdjęciu źródłowym)", **info)
    meta.close()


if __name__ == "__main__":
    main()
