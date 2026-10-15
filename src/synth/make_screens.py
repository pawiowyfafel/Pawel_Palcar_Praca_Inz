"""Syntetyczne zrzuty ekranu telefonu z naszymi pozytywami (negatywy "meme", decyzja 2026-10-14).

Warianty (ogólny interfejs, bez prawdziwych logotypów/nazw marek):
  gallery – przeglądarka zdjęć: czarne tło, pasek u góry i ikony u dołu
  feed    – kanał społecznościowy: nagłówek, avatar, zdjęcie, polubienia, podpis
  chat    – komunikator: dymki wiadomości ze zdjęciem
Zrzut dziedziczy group po zdjęciu bazowym.
"""
import io, random
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm
from common import Meta, uid, MANIF, ROOT, load_all_meta, cli_limit
from synth.textlib import SEGOE, SEGOE_B, SYMBOL, BAR_CAPTIONS, font, load_rgb, paste_emoji

N = {"gallery": 170, "feed": 200, "chat": 130}
W, H = 1080, 2280                      # proporcje telefonu (~19:9), po normalizacji < 2.2:1
NAMES = ["ania_w_gorach", "tomek.hikes", "kasia__", "mountain.mike", "p.nowak", "trail_runner_88",
         "Mama", "Grupa: Tatry 2025", "Kuba", "ola.travels"]


def status_bar(d, rng, dark):
    fg = "white" if dark else "black"
    d.text((50, 22), f"{rng.randint(6, 23)}:{rng.randint(0, 59):02d}", font=font(SEGOE_B, 40), fill=fg)
    x = W - 60
    d.rectangle([x - 70, 30, x, 62], outline=fg, width=3)                     # bateria
    d.rectangle([x - 66, 34, x - 66 + int(62 * rng.uniform(0.15, 1)), 58], fill=fg)
    for i in range(4):                                                        # zasięg
        d.rectangle([x - 190 + i * 18, 62 - (i + 1) * 8, x - 178 + i * 18, 62], fill=fg)


def fit_photo(photo, max_w, max_h):
    s = min(max_w / photo.width, max_h / photo.height)
    return photo.resize((int(photo.width * s), int(photo.height * s)), Image.LANCZOS)


def gallery(photo, rng):
    im = Image.new("RGB", (W, H), "black")
    d = ImageDraw.Draw(im)
    status_bar(d, rng, True)
    d.text((50, 110), "‹", font=font(SEGOE, 80), fill="white")
    d.text((W / 2 - 160, 125), f"{rng.randint(1, 28)} {rng.choice(['lip', 'sie', 'paź', 'Jul', 'Aug'])} "
           f"{rng.randint(2019, 2025)}", font=font(SEGOE_B, 44), fill="white")
    p = fit_photo(photo, W, H - 520)
    im.paste(p, ((W - p.width) // 2, (H - p.height) // 2))
    for i, s in enumerate(["⤴", "♡", "✎", "🗑"]):
        d.text((110 + i * 250, H - 170), s, font=font(SYMBOL, 64), fill="white")
    return im


def feed(photo, rng):
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    status_bar(d, rng, False)
    d.line([0, 230, W, 230], fill=(220, 220, 220), width=2)
    d.text((50, 130), rng.choice(["Odkrywaj", "Główna", "Home", "Feed"]), font=font(SEGOE_B, 56), fill="black")
    y = 270
    d.ellipse([40, y, 140, y + 100], fill=tuple(rng.randint(80, 220) for _ in range(3)))
    d.text((165, y + 22), rng.choice(NAMES), font=font(SEGOE_B, 42), fill="black")
    p = fit_photo(photo, W, 1300)
    p = p.resize((W, int(p.height * W / p.width)))
    im.paste(p, (0, y + 130))
    y2 = y + 150 + p.height
    d.text((50, y2), "♡   💬   ➤", font=font(SYMBOL, 60), fill="black")
    d.text((50, y2 + 100), f"{rng.randint(3, 9999)} polubień", font=font(SEGOE_B, 40), fill="black")
    d.text((50, y2 + 160), rng.choice(BAR_CAPTIONS).split("\n")[0][:60], font=font(SEGOE, 38), fill="black")
    d.text((50, y2 + 230), f"{rng.randint(1, 23)} godz. temu", font=font(SEGOE, 32), fill=(140, 140, 140))
    if rng.random() < 0.3:
        paste_emoji(im, rng, n=2)
    return im


def chat(photo, rng):
    bg = rng.choice([(255, 255, 255), (236, 229, 221), (18, 18, 18)])
    dark = bg[0] < 100
    im = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(im)
    status_bar(d, rng, dark)
    fg = "white" if dark else "black"
    d.text((140, 130), rng.choice(NAMES), font=font(SEGOE_B, 48), fill=fg)
    d.ellipse([40, 125, 115, 200], fill=(150, 170, 200))
    y = 300
    for _ in range(rng.randint(1, 3)):                                        # wiadomości przed zdjęciem
        msg = rng.choice(["patrz jaki widok", "byliśmy dziś tutaj", "zobacz jak tam jest",
                          "warunki na szlaku 👇", "look at this", "ale błoto xd"])
        mine = rng.random() < 0.5
        tw = d.textlength(msg, font=font(SEGOE, 42)) + 60
        x = W - tw - 40 if mine else 40
        d.rounded_rectangle([x, y, x + tw, y + 90], 40, fill=(0, 132, 255) if mine else (228, 230, 235))
        d.text((x + 30, y + 18), msg, font=font(SEGOE, 42), fill="white" if mine else "black",
               embedded_color=True)
        y += 120
    p = fit_photo(photo, int(W * 0.72), H - y - 400)
    x = W - p.width - 40 if rng.random() < 0.5 else 40
    mask = Image.new("L", p.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, p.width, p.height], 40, fill=255)
    im.paste(p, (x, y), mask)
    d.rounded_rectangle([30, H - 150, W - 30, H - 50], 50, outline=(180, 180, 180), width=3)
    d.text((80, H - 130), "Wiadomość…", font=font(SEGOE, 40), fill=(150, 150, 150))
    return im


VARIANTS = {"gallery": gallery, "feed": feed, "chat": chat}


def main():
    limit = cli_limit()
    meta = Meta("synth_screen")
    pos = pd.read_parquet(MANIF / "positive.parquet", columns=["id", "group", "source"])
    pos = pos[pos.source != "own"].merge(load_all_meta()[["id", "raw_path"]], on="id")
    jobs = [v for v, n in N.items() for _ in range(n)]
    bases = pos.sample(len(jobs), random_state=1)
    for v, b in tqdm(list(zip(jobs, bases.itertuples()))):
        if limit and meta.added >= limit:
            break
        id_ = uid("synth_screen", f"{b.id}:{v}")
        if meta.has(id_):
            continue
        rng = random.Random(id_)
        im = VARIANTS[v](load_rgb(ROOT / b.raw_path), rng).resize((W // 2, H // 2), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=rng.randint(85, 95))
        meta.add(id_, buf.getvalue(), label=0, subclass="meme", meme_style=f"screen_{v}",
                 base_id=b.id, base_source=b.source, group=b.group, source_id=f"pos:{b.id}",
                 license="syntetyczne (własne)")
    meta.close()


if __name__ == "__main__":
    main()
