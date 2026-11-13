# Bibliografia i źródła użyte w projekcie

Stan: 2026-11-12. Lista roboczy, z której powstanie bibliografia pracy. Przy cytowaniu stron WWW podaj datę dostępu.
Oznaczenia: **[P]** publikacja do bibliografii, **[D]** zbiór danych, **[M]** model / wagi, **[N]** narzędzie / biblioteka, **[U]** usługa / platforma.
Pozycje ze znakiem ⚠ mają podany tylko opis bez pewnego linku. Przed oddaniem pracy uzupełnij je o DOI albo stronę wydawcy.

---

## 1. Publikacje naukowe

### Klasyfikacja obrazów i architektury mobilne
- [P] Howard A. i in., *Searching for MobileNetV3*, ICCV 2019. https://arxiv.org/abs/1905.02244
- [P] Tan M., Le Q., *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019. https://arxiv.org/abs/1905.11946
- [P] EfficientNet-Lite (Google, wariant pod urządzenia brzegowe: bez squeeze-excitation, ReLU6). https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
- [P] Wightman R., *PyTorch Image Models (timm)*. https://github.com/huggingface/pytorch-image-models
- [P] Selvaraju R. R. i in., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017. https://arxiv.org/abs/1610.02391
- [P] Geirhos R. i in., *Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence 2020. https://arxiv.org/abs/2004.07780

### Kwantyzacja i wdrożenie na urządzeniach mobilnych
- [P] Jacob B. i in., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018. https://arxiv.org/abs/1712.05877
- [P] Krishnamoorthi R., *Quantizing deep convolutional networks for efficient inference: A whitepaper*, 2018. https://arxiv.org/abs/1806.08342 (omawia problemy PTQ w sieciach z depthwise conv, por. spadek MobileNetV3-S po INT8)
- [P] Nagel M. i in., *A White Paper on Neural Network Quantization*, 2021. https://arxiv.org/abs/2106.08295

### Filtrowanie zero-shot (CLIP)
- [P] Radford A. i in., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021. https://arxiv.org/abs/2103.00020
- [P] Cherti M. i in., *Reproducible Scaling Laws for Contrastive Language-Image Learning* (OpenCLIP), CVPR 2023. https://arxiv.org/abs/2212.07143
- [P] Schuhmann C. i in., *LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models*, NeurIPS 2022. https://arxiv.org/abs/2210.08402

### Zbiory danych (publikacje źródłowe)
- [P] Zhou B., Lapedriza A., Khosla A., Oliva A., Torralba A., *Places: A 10 Million Image Database for Scene Recognition*, IEEE TPAMI 2018. https://doi.org/10.1109/TPAMI.2017.2723009
- [P] Kuznetsova A. i in., *The Open Images Dataset V4*, IJCV 2020. https://arxiv.org/abs/1811.00982
- [P] Lin T.-Y. i in., *Microsoft COCO: Common Objects in Context*, ECCV 2014. https://arxiv.org/abs/1405.0312
- [P] Zhao Y., Zhang P., Ignat O., *Beyond Translation: Cross-Cultural Meme Transcreation with Vision-Language Models* (MemeXGen), 2026. https://arxiv.org/abs/2602.02510
- [P] ⚠ Mahfoudi G. i in., *DEFACTO: Image and Face Manipulation Dataset*, EUSIPCO 2019.
- [P] ⚠ Novozámský A., Mahdian B., Saic S., *IMD2020: A Large-Scale Annotated Dataset Tailored for Detecting Manipulated Images*, WACV Workshops 2020.
- [P] Guillaro F. i in., *TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization* (zawiera CocoGlide), CVPR 2023. https://arxiv.org/abs/2212.10957
- [P] ⚠ OpenFake (ComplexDataLab): zbiór obrazów z najnowszych generatorów. Publikację wskazuje karta zbioru: https://huggingface.co/datasets/ComplexDataLab/OpenFake

### Generowanie i manipulacja obrazów
- [P] Rombach R. i in., *High-Resolution Image Synthesis with Latent Diffusion Models* (Stable Diffusion), CVPR 2022. https://arxiv.org/abs/2112.10752
- [P] Podell D. i in., *SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis*, 2023. https://arxiv.org/abs/2307.01952
- [P] Chen J. i in., *PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation*, 2024. https://arxiv.org/abs/2403.04692
- [P] Suvorov R. i in., *Resolution-robust Large Mask Inpainting with Fourier Convolutions* (LaMa), WACV 2022. https://arxiv.org/abs/2109.07161

### Wykrywanie obrazów generowanych (przegląd literatury)
- [P] Zhu M. i in., *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image*, NeurIPS 2023. https://arxiv.org/abs/2306.08571
- [P] Park J., Owens A., *Community Forensics: Using Thousands of Generators to Train Fake Image Detectors*, CVPR 2025. https://arxiv.org/abs/2411.04125
- [P] ⚠ VCT² (benchmark wykrywania obrazów AI; ~58% dokładności czołowych detektorów zero-shot). Pozycję trzeba odszukać w ACL Anthology.

---

## 2. Zbiory danych użyte w projekcie

| Źródło | Rola | Licencja | Link |
|---|---|---|---|
| [D] Places365-Standard | pozytywy (25 kategorii), stoki narciarskie, negatywy urban | badania, niekomercyjnie | http://places2.csail.mit.edu/ · archiwum: https://data.csail.mit.edu/places/places365/train_256_places365standard.tar · kategorie: https://github.com/CSAILVision/places365 |
| [D] Mapillary | pozytywy (12 regionów górskich), negatywy urban (5 miast) | CC BY-SA 4.0 | https://www.mapillary.com · API: https://www.mapillary.com/developer/api-documentation |
| [D] OpenStreetMap (przez Overpass API) | filtr geograficzny Mapillary (ścieżki i drogi) | ODbL, „© OpenStreetMap contributors” | https://www.openstreetmap.org/copyright · https://overpass-api.de |
| [D] Unsplash Lite | pozytywy | Unsplash License (Lite: komercyjnie i niekomercyjnie) | https://github.com/unsplash/datasets · https://unsplash.com/data |
| [D] Open Images V7 | pozytywy | zdjęcia CC BY 2.0, etykiety CC BY 4.0 | https://storage.googleapis.com/openimages/web/index.html |
| [D] Wikimedia Commons | pozytywy (test OOD) | CC BY / BY-SA / CC0, per plik | https://commons.wikimedia.org · API: https://commons.wikimedia.org/w/api.php |
| [D] COCO 2017 (val) | negatywy urban, obiekty do splicingu | CC BY (Flickr), per zdjęcie | https://cocodataset.org |
| [D] MemeXGen (tylko oryginały) | negatywy meme (EN: trening, ZH: test OOD) | CC BY 4.0 | https://huggingface.co/datasets/YZhao09/MemeXGen |
| [D] Imgflip (szablony memów) | baza syntetycznych memów | szablony: użycie badawcze | https://imgflip.com/api · https://api.imgflip.com/get_memes |
| [D] DEFACTO, inpainting | negatywy manipulated (trening) | badania | https://huggingface.co/datasets/bahakirbasoglu/DefactoInpaintingDataset |
| [D] CocoGlide | negatywy manipulated (test OOD) | badania (oparte na COCO) | https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip · https://github.com/grip-unina/TruFor |
| [D] IMD2020 | negatywy manipulated (test OOD) | badania | https://huggingface.co/datasets/KhangTruong/IMD2020 |
| [D] OpenFake (split test) | negatywy AI (test OOD, 20 generatorów) | CC BY-NC 4.0 | https://huggingface.co/datasets/ComplexDataLab/OpenFake |
| [D] Własne obrazy AI i inpaintingi | negatywy ai / manipulated | wygenerowane (licencje modeli, pkt 3) | repozytorium projektu, `kaggle/` |

---

## 3. Modele i wagi

### Klasyfikatory (wagi ImageNet z timm, Hugging Face Hub)
- [M] MobileNetV3-Small: `mobilenetv3_small_100.lamb_in1k` https://huggingface.co/timm/mobilenetv3_small_100.lamb_in1k
- [M] MobileNetV3-Large: `mobilenetv3_large_100.ra_in1k` https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k
- [M] EfficientNet-Lite0: `efficientnet_lite0.ra_in1k` https://huggingface.co/timm/efficientnet_lite0.ra_in1k
- [M] EfficientNet-B3: `efficientnet_b3.ra2_in1k` https://huggingface.co/timm/efficientnet_b3.ra2_in1k

### Filtr CLIP
- [M] OpenCLIP ViT-B/32 `laion2b_s34b_b79k` https://github.com/mlfoundations/open_clip

### Generatory (obrazy AI i inpainting, Kaggle)
- [M] Stable Diffusion 1.5 (CreativeML OpenRAIL-M) https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
- [M] Stable Diffusion 1.5 Inpainting https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting
- [M] SDXL base 1.0 (OpenRAIL++) https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- [M] SDXL VAE fp16-fix https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
- [M] SDXL Inpainting 0.1 https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1
- [M] PixArt-Σ XL 1024 MS (OpenRAIL++, generator testowy zamiast FLUX) https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS

### Manipulacje na CPU
- [M] LaMa (big-lama, TorchScript) https://github.com/advimman/lama · plik: https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt

### Rozważane, nieużyte
- FLUX.1-schnell: wymagał logowania na HF, zastąpiony PixArt-Σ. https://huggingface.co/black-forest-labs/FLUX.1-schnell
- Kandinsky 2.2 (zapas). https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder

### Filtry NSFW (kaskada w aplikacji, rozdz. 2.2)
- [N] NSFWJS https://github.com/infinitered/nsfwjs
- [N] NudeNet https://github.com/notAI-tech/NudeNet

---

## 4. Narzędzia i biblioteki

| Narzędzie | Zastosowanie | Link |
|---|---|---|
| [N] PyTorch, torchvision | trening, augmentacje na GPU, JPEG na GPU | https://pytorch.org |
| [N] timm | architektury i wagi | https://github.com/huggingface/pytorch-image-models |
| [N] diffusers | generacja AI i inpainting | https://github.com/huggingface/diffusers |
| [N] OpenCLIP | filtr zero-shot | https://github.com/mlfoundations/open_clip |
| [N] Hugging Face Hub, datasets | pobieranie modeli i zbiorów | https://huggingface.co |
| [N] FiftyOne | przegląd ręczny zbioru | https://github.com/voxel51/fiftyone · https://docs.voxel51.com |
| [N] ImageHash | deduplikacja (pHash) | https://github.com/JohannesBuchner/imagehash |
| [N] OpenCV | copy-move, dopasowanie kolorów | https://opencv.org |
| [N] Pillow, pillow-heif | obróbka obrazów, HEIC | https://python-pillow.org |
| [N] pandas, PyArrow | manifesty (parquet) | https://pandas.pydata.org · https://arrow.apache.org |
| [N] scikit-learn | ROC-AUC, PR-AUC | https://scikit-learn.org |
| [N] Shapely, pyproj, mercantile | filtr geograficzny Mapillary | https://shapely.readthedocs.io · https://pyproj4.github.io/pyproj · https://github.com/mapbox/mercantile |
| [N] ONNX, ONNX Runtime (+ kwantyzacja) | eksport FP32/FP16/INT8 | https://onnx.ai · https://onnxruntime.ai · https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html |
| [N] onnxconverter-common | konwersja do FP16 | https://github.com/microsoft/onnxconverter-common |
| [N] LiteRT (dawniej TensorFlow Lite) | format modeli na telefon | https://ai.google.dev/edge/litert |
| [N] litert-torch (dawniej ai-edge-torch) | PyTorch → LiteRT | https://github.com/google-ai-edge/ai-edge-torch |
| [N] ai-edge-quantizer | kwantyzacja LiteRT (INT8 dynamiczny) | https://github.com/google-ai-edge/ai-edge-quantizer |
| [N] benchmark_model (LiteRT/TFLite) + adb | pomiar czasu inferencji na telefonach | https://ai.google.dev/edge/litert/models/measurement · https://developer.android.com/tools/adb |
| [N] torchao | kwantyzacja PT2E (INT8 statyczny) | https://github.com/pytorch/ao |
| [N] Weights & Biases | śledzenie eksperymentów | https://wandb.ai · https://docs.wandb.ai |
| [N] Kaggle CLI / API | uruchamianie notebooków, datasety | https://github.com/Kaggle/kaggle-cli |

---

## 5. Usługi i platformy

- [U] Kaggle Notebooks (GPU 2× T4, ~30 h/tydzień): trening, generacja AI, eksport. https://www.kaggle.com
- [U] Weights & Biases (projekt `praca-inz-trail`). https://wandb.ai
- [U] Hugging Face Hub: modele i zbiory. https://huggingface.co
- [U] Mapillary Graph API v4. https://graph.mapillary.com
- [U] Overpass API (OpenStreetMap). https://overpass-api.de
- [U] Wikimedia Commons API. https://commons.wikimedia.org/w/api.php

---

## 6. Akty prawne i licencje

- RODO: Rozporządzenie (UE) 2016/679. https://eur-lex.europa.eu/eli/reg/2016/679/oj
- Licencje Creative Commons (CC BY, CC BY-SA, CC BY-NC). https://creativecommons.org/licenses/
- Open Database License (OpenStreetMap). https://opendatacommons.org/licenses/odbl/
- CreativeML OpenRAIL-M / OpenRAIL++ (licencje modeli SD/SDXL/PixArt): na kartach modeli w pkt 3.

---

## 7. Rozważane, ale nieużyte (do opisu w przeglądzie lub ograniczeniach)

- Community Forensics (zbyt duży: 1,1 TB). https://huggingface.co/datasets/OwensLab/CommunityForensics
- ELSA D3 (2,5 TB). https://huggingface.co/datasets/elsaEU/ELSA_D3
- GenImage (pobieranie przez Baidu, znany bias formatu PNG/JPEG). https://github.com/GenImage-Dataset/GenImage
- PolyMeme (lista URL, wiele martwych linków). https://zenodo.org/records/8354816
- RedditMemes (HSDSLab, dane usunięte z repozytorium)
- Hateful Memes, MemeXplain, MemeLens: treści nienawistne i propagandowe, świadomie pominięte.
- CASIA v2: wymagał konta Kaggle i nie był konieczny przy DEFACTO i własnych manipulacjach.
- Flickr API: od 2025 r. klucze tylko dla kont Pro, zastąpione przez Open Images.
