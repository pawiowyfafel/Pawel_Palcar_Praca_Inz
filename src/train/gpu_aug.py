"""Augmentacje i przygotowanie wejścia na GPU, całymi paczkami (wejście: uint8 N×256×256×3 z cache'u).

Trening: losowe przycięcie ze skalowaniem (RandomResizedCrop, skala 0,5–1), odbicie poziome, jasność /
kontrast / nasycenie (p=0,5), zmniejszenie rozdzielczości ×0,4–0,9 (p=0,2), kompresja JPEG q 50–95 (p=0,3),
normalizacja ImageNet. Bez rozmycia i szumu (niszczą ślady AI i manipulacji).
Ewaluacja: Resize(round(size/0,875)) + CenterCrop(size) – jak standardowy potok ImageNet i aplikacja na telefonie.
"""
import math, random
import torch, torch.nn.functional as F
from torchvision.io import decode_jpeg, encode_jpeg


class GPUAugment:
    def __init__(self, size, mean, std, train, p_jpeg=0.3, p_down=0.2, p_color=0.5):
        self.size, self.train = size, train
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.p_jpeg, self.p_down, self.p_color = p_jpeg, p_down, p_color

    def _crop_boxes(self, n, h, w):
        """Pudełka RandomResizedCrop (skala 0,5–1, proporcje 3/4–4/3) jako współrzędne affine_grid."""
        th = torch.empty(n, 2, 3)
        for i in range(n):
            for _ in range(10):
                area = random.uniform(0.5, 1.0)
                r = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
                cw, ch = math.sqrt(area * r), math.sqrt(area / r)
                if cw <= 1 and ch <= 1:
                    break
            else:
                cw = ch = 1.0
            cx, cy = random.uniform(-(1 - cw), 1 - cw), random.uniform(-(1 - ch), 1 - ch)
            flip = -1.0 if random.random() < 0.5 else 1.0
            th[i] = torch.tensor([[cw * flip, 0, cx], [0, ch, cy]])
        return th

    def _jpeg(self, x):
        """x: float 0–1 na GPU. Kompresja JPEG wybranych zdjęć (torchvision.io; na GPU, gdy dostępne)."""
        idx = [i for i in range(len(x)) if random.random() < self.p_jpeg]
        if not idx:
            return x
        u8 = (x[idx] * 255).round().clamp(0, 255).to(torch.uint8)
        q = random.randint(50, 95)
        try:                                                # kodowanie na GPU, bajty na CPU, dekodowanie na GPU
            enc = [e.cpu() for e in encode_jpeg(list(u8), quality=q)]
            dec = decode_jpeg(enc, device=x.device)
        except Exception:                                   # brak obsługi GPU w torchvision: na CPU
            enc = encode_jpeg([t.cpu() for t in u8], quality=q)
            dec = [d.to(x.device) for d in decode_jpeg(enc)]
        x[idx] = torch.stack(dec).float() / 255
        return x

    @torch.no_grad()
    def __call__(self, u8):
        x = u8.permute(0, 3, 1, 2).float() / 255                       # N×3×256×256, 0–1
        n, _, h, w = x.shape
        if self.train:
            grid = F.affine_grid(self._crop_boxes(n, h, w).to(x.device), (n, 3, self.size, self.size),
                                 align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
            m = torch.rand(n, device=x.device) < self.p_color         # jasność, kontrast, nasycenie
            if m.any():
                k = int(m.sum())
                b = torch.empty(k, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
                c = torch.empty(k, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
                s = torch.empty(k, 1, 1, 1, device=x.device).uniform_(0.9, 1.1)
                y = x[m] * b
                mean = y.mean(dim=(1, 2, 3), keepdim=True)
                y = (y - mean) * c + mean
                gray = (0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3])
                x[m] = ((y - gray) * s + gray).clamp(0, 1)
            m = torch.rand(n, device=x.device) < self.p_down           # niższa rozdzielczość
            if m.any():
                f = random.uniform(0.4, 0.9)
                small = F.interpolate(x[m], scale_factor=f, mode="bilinear", align_corners=False, antialias=True)
                x[m] = F.interpolate(small, size=(self.size, self.size), mode="bilinear", align_corners=False)
            x = self._jpeg(x)
        else:
            r = round(self.size / 0.875)
            if r != h:
                x = F.interpolate(x, size=(r, r), mode="bilinear", align_corners=False, antialias=True)
            o = (r - self.size) // 2
            x = x[:, :, o:o + self.size, o:o + self.size]
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
