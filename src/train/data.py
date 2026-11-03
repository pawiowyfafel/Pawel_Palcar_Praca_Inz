"""Dane do treningu: manifest -> Dataset, etykiety (binarne / 5 klas), augmentacje, podzbiór AI (krzywa uczenia).

Etykieta binarna: 1 = NARUSZENIE (negatyw), 0 = autentyczne zdjęcie z wycieczki. Model zwraca więc
p_violation, a reguła trójstopniowa (evaluate.py) porównuje je z progami t_przepuść / t_blokuj.
"""
import hashlib, io, random
import pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

CLASSES = ["nature", "urban", "meme", "manipulated", "ai"]          # głowica 5-klasowa; 0 = autentyczne
OBVIOUS = ["urban", "meme"]                                         # naruszenia "oczywiste" (mogą być blokowane)


def stable_order(ids):
    return ids.map(lambda s: hashlib.md5(s.encode()).hexdigest())


def load_manifest(path, ai_train_n=None):
    """Wczytuje manifest. ai_train_n: zostaw w train tylko N obrazów AI (stała kolejność wg hasha id),
    walidacja i testy pozostają nietknięte – krzywa uczenia porównuje modele na tych samych danych."""
    df = pd.read_parquet(path)
    if ai_train_n is not None:
        ai_tr = df[(df.split == "train") & (df.subclass == "ai")]
        keep = ai_tr.assign(_k=stable_order(ai_tr.id)).sort_values("_k").head(ai_train_n).index
        df = df.drop(ai_tr.index.difference(keep))
    return df.reset_index(drop=True)


class RandomJPEG:
    """Ponowna kompresja JPEG – odporność na komunikatory i różne telefony."""

    def __init__(self, p=0.3, q=(50, 95)):
        self.p, self.q = p, q

    def __call__(self, im):
        if random.random() > self.p:
            return im
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=random.randint(*self.q))
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


class RandomDownscale:
    """Losowe zmniejszenie rozdzielczości – przeciw skrótowi 'rozdzielczość -> klasa' (np. Places365 256 px)."""

    def __init__(self, p=0.2, scale=(0.4, 0.9)):
        self.p, self.scale = p, scale

    def __call__(self, im):
        if random.random() > self.p:
            return im
        s = random.uniform(*self.scale)
        return im.resize((max(32, int(im.width * s)), max(32, int(im.height * s))), Image.BILINEAR)


def transforms(size, mean, std, train):
    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        return T.Compose([T.Resize(int(round(size / 0.875))), T.CenterCrop(size), *norm])
    # Bez mocnego rozmycia i szumu: niszczą ślady generacji AI i manipulacji, które model ma wykrywać.
    return T.Compose([RandomJPEG(), RandomDownscale(),
                      T.RandomResizedCrop(size, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                      T.RandomApply([T.ColorJitter(0.15, 0.15, 0.1, 0.02)], p=0.5), *norm])


class TrailDataset(Dataset):
    def __init__(self, df, root, tf, head):
        self.paths = [str(root / p) for p in df.path]
        self.tf, self.head = tf, head
        if head == "binary":
            self.y = torch.tensor((df.label == 0).astype("float32").values)
        else:
            self.y = torch.tensor(df.subclass.map(CLASSES.index).values)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), self.y[i], i


def class_weights(df, head):
    """Wagi w funkcji straty zamiast dokładania danych (pozytywów ~21 tys., negatywów ~17 tys.)."""
    tr = df[df.split == "train"]
    if head == "binary":
        n_viol = (tr.label == 0).sum()
        return torch.tensor((len(tr) - n_viol) / max(1, n_viol))        # pos_weight dla BCE
    cnt = tr.subclass.value_counts().reindex(CLASSES).fillna(1).values
    w = torch.tensor(len(tr) / (len(CLASSES) * cnt), dtype=torch.float32)
    return w / w.mean()
