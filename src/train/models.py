"""Architektury: własny mały CNN (dolna granica) + modele mobilne z timm (ImageNet) + model referencyjny.

python -m train.models  -> tabela złożoności (parametry, MAC, rozmiar FP32) do rozdziału 4.
"""
import torch, torch.nn as nn, timm
from torch.utils.flop_counter import FlopCounterMode

# nazwa -> (identyfikator timm z wagami ImageNet, domyślna rozdzielczość, rola)
ZOO = {
    "cnn":   (None, 160, "własny mały CNN bez pretreningu – dolna granica"),
    "mnv3s": ("mobilenetv3_small_100.lamb_in1k", 224, "MobileNetV3-Small – najmniejszy model mobilny"),
    "mnv3l": ("mobilenetv3_large_100.ra_in1k", 224, "MobileNetV3-Large – standard mobilny"),
    "effl0": ("efficientnet_lite0.ra_in1k", 224, "EfficientNet-Lite0 – kompromis, przyjazny INT8"),
    "effb3": ("efficientnet_b3.ra2_in1k", 288, "EfficientNet-B3 – referencja serwerowa (górna granica)"),
}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class SmallCNN(nn.Module):
    """Wejście conv stride 2 + 4 bloki conv-BN-ReLU (+ max-pool), global average pooling, dropout, klasyfikator.
    Zwykłe (nie depthwise-separable) sploty: celowo "klasyczna" sieć, by pokazać zysk z architektur mobilnych."""

    def __init__(self, num_classes, widths=(32, 64, 128, 256), drop=0.2):
        super().__init__()
        layers = [nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True)]
        c_in = 16
        for w in widths:
            layers += [nn.Conv2d(c_in, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, padding=1, groups=w, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c_in = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(c_in, num_classes))

    def forward(self, x):
        return self.head(self.features(x))

    def get_classifier(self):
        return self.head


def build(name, num_classes, pretrained=True):
    timm_id, size, _ = ZOO[name]
    if timm_id is None:
        return SmallCNN(num_classes), size, MEAN, STD
    m = timm.create_model(timm_id, pretrained=pretrained, num_classes=num_classes)
    cfg = m.pretrained_cfg
    return m, size, tuple(cfg.get("mean", MEAN)), tuple(cfg.get("std", STD))


def set_backbone_trainable(model, trainable):
    """Etap 1 transfer learningu: zamrożony backbone, uczy się tylko klasyfikator."""
    head = set(id(p) for p in model.get_classifier().parameters())
    for p in model.parameters():
        if id(p) not in head:
            p.requires_grad = trainable


def complexity(model, size):
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    return {"params_M": round(params / 1e6, 3), "MMAC": round(fc.get_total_flops() / 2 / 1e6, 1),
            "fp32_MB": round(params * 4 / 2 ** 20, 2)}


if __name__ == "__main__":
    for n, (tid, size, role) in ZOO.items():
        m, s, _, _ = build(n, 5, pretrained=False)
        print(f"{n:6s} {s:4d}px  {complexity(m, s)}  {role}")
