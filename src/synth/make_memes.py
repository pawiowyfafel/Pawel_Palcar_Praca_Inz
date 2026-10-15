"""Syntetyczne memy (negatywy "meme").

Bazy:
  - nasze pozytywy (2 000)  -> mem dziedziczy group po zdjęciu bazowym (ten sam split, brak przecieku)
  - szablony Imgflip (1 000) -> group = szablon (memy z jednego szablonu w jednym splicie)
Style: impact (góra/dół), bar (biały pasek z podpisem), labels (etykiety na obiektach),
       emoji (naklejki), vulgar (wulgarne napisy – przypadek "ktoś dopisał wulgaryzm do zdjęcia").
"""
import io, random
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from common import Meta, uid, get, RAW, MANIF, ROOT, load_all_meta, cli_limit
from synth.textlib import (IMPACT, ARIAL_B, SEGOE, CAPTIONS, BAR_CAPTIONS, LABEL_WORDS, VULGAR,
                           fit, draw_lines, paste_emoji, load_rgb)

ON_POSITIVES = {"impact": 600, "bar": 500, "labels": 250, "emoji": 250, "vulgar": 400}
ON_TEMPLATES = {"impact": 600, "bar": 250, "vulgar": 150}
TPL_DIR = RAW / "imgflip" / "templates"


def impact(im, rng, pair=None):
    top, bottom = pair or rng.choice(CAPTIONS)
    d = ImageDraw.Draw(im)
    W, H = im.size
    for text, at_top in ((top, True), (bottom, False)):
        if not text:
            continue
        f, lines = fit(d, text.upper(), IMPACT, W * 0.94, H * 0.22, H * 0.13)
        h = len(lines) * f.size * 1.15
        y = H * 0.02 if at_top else H - h - H * 0.03
        draw_lines(d, lines, f, W / 2, y, "white", stroke=max(2, f.size // 14))
    return im


def bar(im, rng, text=None):
    text = text or rng.choice(BAR_CAPTIONS)
    W, H = im.size
    d0 = ImageDraw.Draw(im)
    f, lines = fit(d0, text, rng.choice([SEGOE, ARIAL_B]), W * 0.92, H * 0.35, W * 0.06)
    bh = int(len(lines) * f.size * 1.15 + f.size * 1.2)
    out = Image.new("RGB", (W, H + bh), "white")
    out.paste(im, (0, bh))
    draw_lines(ImageDraw.Draw(out), lines, f, W * 0.04, f.size * 0.6, "black", align="left")
    return out


def labels(im, rng):
    d = ImageDraw.Draw(im)
    W, H = im.size
    for w in rng.sample(LABEL_WORDS, rng.randint(2, 4)):
        f, lines = fit(d, w, IMPACT, W * 0.4, H * 0.12, H * 0.08)
        x, y = rng.uniform(0.25, 0.75) * W, rng.uniform(0.1, 0.8) * H
        draw_lines(d, lines, f, x, y, "white", stroke=max(2, f.size // 12))
    return im


def emoji(im, rng):
    im = paste_emoji(im, rng)
    return impact(im, rng, ("", rng.choice(CAPTIONS)[1])) if rng.random() < 0.5 else im


def vulgar(im, rng):
    pair = rng.choice(VULGAR)
    if rng.random() < 0.7:
        return impact(im, rng, pair)
    return bar(im, rng, " ".join(p for p in pair if p).lower())


STYLES = {"impact": impact, "bar": bar, "labels": labels, "emoji": emoji, "vulgar": vulgar}


def templates():
    """Pobiera ~100 popularnych szablonów z Imgflip (API bez klucza)."""
    TPL_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for m in get("https://api.imgflip.com/get_memes").json()["data"]["memes"]:
        p = TPL_DIR / f"{m['id']}.jpg"
        if not p.exists():
            r = get(m["url"])
            if r is None:
                continue
            p.write_bytes(r.content)
        out.append((m["id"], m["name"], p))
    return out


def save(meta, rng, base_img, style, id_, **rec):
    im = STYLES[style](base_img, rng)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    meta.add(id_, buf.getvalue(), label=0, subclass="meme", meme_style=style,
             license="syntetyczne (własne)", **rec)


def main():
    limit = cli_limit()
    rng, meta = random.Random(0), Meta("synth_meme")

    # 1) memy na pozytywach
    pos = pd.read_parquet(MANIF / "positive.parquet", columns=["id", "group", "source"])
    pos = pos[pos.source != "own"].merge(load_all_meta()[["id", "raw_path"]], on="id")
    jobs = [s for s, n in ON_POSITIVES.items() for _ in range(n)]
    bases = pos.sample(len(jobs), random_state=0)
    for style, b in tqdm(list(zip(jobs, bases.itertuples())), desc="pozytywy"):
        if limit and meta.added >= limit:
            break
        id_ = uid("synth_meme", f"pos:{b.id}:{style}")
        if not meta.has(id_):
            save(meta, random.Random(id_), load_rgb(ROOT / b.raw_path), style, id_,
                 base_id=b.id, base_source=b.source, group=b.group, source_id=f"pos:{b.id}")

    # 2) memy na szablonach Imgflip
    tpl = templates()
    jobs = [s for s, n in ON_TEMPLATES.items() for _ in range(n)]
    rng.shuffle(jobs)
    for k, style in enumerate(tqdm(jobs, desc="szablony")):
        if limit and meta.added >= limit:
            break
        tid, name, p = tpl[k % len(tpl)]
        id_ = uid("synth_meme", f"tpl:{tid}:{k}")
        if not meta.has(id_):
            save(meta, random.Random(id_), load_rgb(p), style, id_, template=name,
                 group=f"imgflip:{tid}", source_id=f"imgflip:{tid}")
    meta.close()


if __name__ == "__main__":
    main()
