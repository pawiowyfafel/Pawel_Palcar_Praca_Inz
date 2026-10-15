"""Wspólne narzędzia do rysowania tekstu (memy, zrzuty ekranu). Czcionki systemowe Windows."""
import random
from PIL import Image, ImageDraw, ImageFont

FONTS = "C:/Windows/Fonts/"
IMPACT, ARIAL, ARIAL_B = FONTS + "impact.ttf", FONTS + "arial.ttf", FONTS + "arialbd.ttf"
SEGOE, SEGOE_B, EMOJI = FONTS + "segoeui.ttf", FONTS + "segoeuib.ttf", FONTS + "seguiemj.ttf"
SYMBOL = FONTS + "seguisym.ttf"        # ikony interfejsu (♡ ✎ 🗑 💬 ➤) – Segoe UI ich nie ma
EMOJIS = ["😂", "🤣", "😭", "🔥", "💀", "👀", "😎", "🙃", "😅", "🤡", "💯", "🥶", "🥵", "🤔", "😱",
          "👍", "🙏", "⛰️", "🌲", "🐻", "🥾"]   # lista, nie list(str): emoji bywają wieloznakowe

# Podpisy (góra, dół). Mieszanka PL/EN, część tematyczna (szlak), część ogólna.
CAPTIONS = [
    ("KIEDY MÓWIĄ ŻE TO TYLKO", "GODZINKA SPACERU"), ("JA NA SZLAKU", "PO 5 MINUTACH"),
    ("NIKT:", "ABSOLUTNIE NIKT:"), ("PLAN NA WEEKEND", "VS RZECZYWISTOŚĆ"),
    ("KIEDY MAPA MÓWI 2 KM", "A JEST 9 KM POD GÓRĘ"), ("MOJA KONDYCJA", "W STYCZNIU"),
    ("POV:", "ZAPOMNIAŁEŚ WODY"), ("MÓWIŁEM ŻE BĘDZIE BŁOTO", "ALE NIKT NIE SŁUCHAŁ"),
    ("TO JEST FINE", "WSZYSTKO JEST FINE"), ("ZNOWU PONIEDZIAŁEK", "ZNOWU PRACA"),
    ("KIEDY WIFI NIE DZIAŁA", ""), ("", "TYLKO PRAWDZIWI WIEDZĄ"), ("JA:", "TAKŻE JA:"),
    ("WHEN THE TRAIL SAYS EASY", "AND IT'S STRAIGHT UP"), ("ONE DOES NOT SIMPLY", "WALK UP THIS HILL"),
    ("NOBODY:", "ME AT 6AM ON A MOUNTAIN"), ("I SHOULD BUY", "BETTER SHOES"),
    ("THIS IS FINE", ""), ("", "AND I TOOK THAT PERSONALLY"), ("ME EXPLAINING", "WHY WE ARE LOST"),
    ("EXPECTATION", "REALITY"), ("WHEN YOU SEE IT", ""), ("Y U NO", "HAVE SIGNAL"),
    ("SUCH VIEW", "MUCH WOW"), ("MONDAY", "AGAIN"), ("", "SAID NO ONE EVER"),
]
BAR_CAPTIONS = [
    "kiedy w końcu dochodzisz na szczyt a tam mgła", "ja po tym jak powiedziałem 'to tylko spacer'",
    "nikt:\nja w górach o 5 rano:", "mój weekend vs weekend influencerów", "pov: zgubiłeś szlak",
    "me when the trail is 'moderate'", "nobody:\nmy legs after day one:", "this is where the fun begins",
    "when the forecast said sunny", "tell me you're from the city without telling me",
    "jak wygląda 'krótki skrót' według mojego taty", "ten moment kiedy schronisko jest zamknięte",
    "my last brain cell trying to read the map", "kiedy kumpel mówi że zna drogę",
]
LABEL_WORDS = ["ja", "moje plany", "kondycja", "szef", "weekend", "me", "my motivation", "the trail",
               "rent", "sleep", "kawa", "deadline", "my friends", "wifi", "egzamin", "the view"]
# wulgarne/obraźliwe napisy (bez treści nienawistnych wobec grup) – przypadek "ktoś dopisał wulgaryzm"
VULGAR = [
    ("KURWA", "ZNOWU BŁOTO"), ("CO TO KURWA JEST", ""), ("JEBAĆ TEN SZLAK", ""), ("CHUJ", "Z TYM"),
    ("PIERDOLĘ", "NIE IDĘ"), ("SPIERDALAJ", ""), ("ZAJEBISTY WIDOK", "XD"), ("JA PIERDOLĘ", "ILE SCHODÓW"),
    ("FUCK THIS", "TRAIL"), ("WTF", "IS THIS SHIT"), ("FUCK", "OFF"), ("SHIT", "HAPPENS"),
    ("HOLY SHIT", ""), ("", "FUCKING FINALLY"), ("BULLSHIT", "WEATHER FORECAST"), ("DUPA", "BOLI"),
]


def font(path, size):
    return ImageFont.truetype(path, max(8, int(size)))


def wrap(draw, text, fnt, max_w):
    """Łamie tekst na linie mieszczące się w max_w."""
    lines = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split():
            t = (cur + " " + w).strip()
            if draw.textlength(t, font=fnt) <= max_w or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def fit(draw, text, path, max_w, max_h, start):
    """Największa czcionka, przy której tekst (z łamaniem) mieści się w prostokącie."""
    size = start
    while size > 10:
        f = font(path, size)
        lines = wrap(draw, text, f, max_w)
        h = len(lines) * size * 1.15
        if h <= max_h and all(draw.textlength(l, font=f) <= max_w for l in lines):
            return f, lines
        size *= 0.9
    f = font(path, 10)
    return f, wrap(draw, text, f, max_w)


def draw_lines(draw, lines, fnt, cx, y, fill, stroke=0, stroke_fill="black", align="center"):
    size = fnt.size
    for i, l in enumerate(lines):
        w = draw.textlength(l, font=fnt)
        x = cx - w / 2 if align == "center" else cx
        draw.text((x, y + i * size * 1.15), l, font=fnt, fill=fill,
                  stroke_width=stroke, stroke_fill=stroke_fill)


def paste_emoji(im, rng: random.Random, n=None):
    d = ImageDraw.Draw(im)
    for _ in range(n or rng.randint(1, 4)):
        s = int(min(im.size) * rng.uniform(0.08, 0.2))
        x, y = rng.randint(0, max(0, im.width - s)), rng.randint(0, max(0, im.height - s))
        d.text((x, y), rng.choice(EMOJIS), font=font(EMOJI, s), embedded_color=True)
    return im


def load_rgb(path, max_side=1024):
    im = Image.open(path).convert("RGB")
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    return im
