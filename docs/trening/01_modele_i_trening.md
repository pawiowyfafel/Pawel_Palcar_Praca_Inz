# Modele i trening: opis, uzasadnienie, plan i instrukcja

Stan: 2026-11-03. Przegląd ręczny zakończony (61 odrzuceń, szum etykiet 0–2%), zbiór końcowy 38 167 zdjęć, paczka `trail-dataset.zip` (2,7 GB, JPEG q=95) wgrywana na Kaggle. Kod przetestowany technicznie (CPU). Trening jeszcze nieuruchomiony.
Materiał do rozdziałów 4 i 5 pracy.

## 1. Zadanie modelu

Model jest **asystentem moderatora**. Dla zdjęcia zwraca prawdopodobieństwo naruszenia `p_violation`, a dwa progi dzielą zgłoszenia na trzy ścieżki:

| `p_violation` | Decyzja | Kiedy |
|---|---|---|
| < t_przepuść | **przepuść** | autentyczne zdjęcie z wycieczki |
| między progami | **flaguj** → administracja | wszystko, czego model nie jest pewien |
| ≥ t_blokuj | **zablokuj** | oczywiste naruszenie (miasto, mem, zrzut ekranu) |

Progi wyznaczamy na zbiorze walidacyjnym z kosztów błędów:
- błędna blokada autentycznego zdjęcia ≤ 0,5% (najgorszy błąd: frustracja użytkownika);
- przepuszczenie oczywistego naruszenia ≤ 2%.

Wartości są do uzgodnienia z promotorką (`evaluate.py --fbr --miss`).

## 2. Wybór modeli i uzasadnienie

Drabinka od trywialnego do serwerowego: pokazuje, ile daje architektura mobilna względem prostej sieci i ile tracimy względem dużego modelu.

| Model (`--set model=`) | Wagi startowe | Wejście | Parametry | MAC | FP32 | Rola |
|---|---|---|---|---|---|---|
| `cnn`: własny CNN | brak (od zera) | 160 | 0,40 mln | 124 mln | 1,5 MB | dolna granica |
| `mnv3s`: MobileNetV3-Small | ImageNet (`lamb_in1k`) | 224 | 1,52 mln | 56 mln | 5,8 MB | najtańszy model mobilny |
| `mnv3l`: MobileNetV3-Large | ImageNet (`ra_in1k`) | 224 | 4,21 mln | 215 mln | 16,1 MB | standard mobilny |
| `effl0`: EfficientNet-Lite0 | ImageNet (`ra_in1k`) | 224 | 3,38 mln | 384 mln | 12,9 MB | **główny kandydat** |
| `effb3`: EfficientNet-B3 | ImageNet (`ra2_in1k`) | 288 | 10,7 mln | 1 588 mln | 40,8 MB | referencja serwerowa |

Wartości zmierzone `python -m train.models`, dla głowicy 5-klasowej.

**Dlaczego te modele:**
- **Własny CNN:** odpowiada na pytanie, czy wystarczy coś trywialnego. Zwykłe sploty 3×3 (bez depthwise) i brak pretreningu. Ma mało parametrów, ale więcej obliczeń niż MobileNetV3-Small, co dobrze pokazuje, po co są bloki odwróconej rezydualności (*inverted residual*) i sploty depthwise.
- **MobileNetV3 (Small/Large):** architektury zaprojektowane pod telefony (NAS pod opóźnienie na CPU mobilnym), z blokami inverted residual, squeeze-excitation i h-swish. Small i Large wyznaczają przedział „ile kosztuje dokładność”.
- **EfficientNet-Lite0:** wariant EfficientNet przygotowany pod urządzenia brzegowe: **bez squeeze-excitation i z ReLU6 zamiast swish**, dzięki czemu dobrze znosi kwantyzację INT8 i działa na akceleratorach mobilnych. To najbardziej prawdopodobny model docelowy.
- **EfficientNet-B3:** model „serwerowy” (SE, swish, 288 px). Nie trafi na telefon, ale pokazuje, czy błędy wynikają z mobilności modelu, czy z samego zadania. To ważne dla wniosków o wykrywaniu AI.

**Dlaczego nie inne:**
- ViT, ConvNeXt, Swin: za duże na telefon, a operacje attention i LayerNorm gorzej się kwantyzują i są słabiej wspierane przez delegaty mobilne;
- MobileViT, EfficientFormer: nowsze, ale słabsze wsparcie narzędzi LiteRT i ONNX Runtime Mobile, a do tego mniej porównań w literaturze;
- modele detekcji (YOLO): zadanie to klasyfikacja całego zdjęcia, nie lokalizacja obiektów.

Biblioteka **timm** (PyTorch Image Models) daje wszystkie te architektury w jednym API z wagami ImageNet.

### Głowica wyjściowa

| `head` | Wyjście | `p_violation` | Zaleta |
|---|---|---|---|
| `multi` (domyślna) | 5 klas: nature, urban, meme, manipulated, ai | 1 − p(nature) | moderator dostaje **powód flagi**; możliwa reguła podklasowa: blokuj tylko gdy dominuje urban lub meme, a AI i manipulacje co najwyżej flaguj |
| `binary` | 1 logit | sigmoid | prostsza, punkt odniesienia (eksperyment E2) |

## 3. Dane i augmentacje

- **Wejście:** paczka `pack_dataset.py`, czyli zdjęcia z manifestu po przeglądzie, krótszy bok 384 px, JPEG q=95, identycznie dla wszystkich klas. Trening używa 160–288 px, a mniejsze pliki to szybsze dekodowanie na 4 rdzeniach Kaggle i ~3× mniejszy upload.
- **Ewaluacja:** skalowanie do size/0,875 i środkowe przycięcie (standard ImageNet).
- **Augmentacje treningowe** (`train/data.py`):
  - ponowna kompresja JPEG q 50–95 (p=0,3): odporność na komunikatory i różne telefony;
  - losowe zmniejszenie rozdzielczości ×0,4–0,9 (p=0,2): przeciw skrótowi „rozdzielczość → klasa” (np. Places365 ma 256 px);
  - RandomResizedCrop (skala 0,5–1), odbicie poziome, lekki ColorJitter (p=0,5);
  - **celowo bez rozmycia i szumu:** niszczą ślady generacji AI i manipulacji, które model ma wykrywać.
- **Nierównowaga klas** (pozytywy ~21 tys., negatywy ~17 tys., AI tylko ~850 w train): wagi klas w funkcji straty zamiast dokładania danych.

### Potok danych na GPU (zmiana po pierwszej sesji, 2026-11-05)

W pierwszej próbie GPU było wykorzystane w 12–17%. Wąskim gardłem był procesor: Kaggle ma 4 rdzenie, a dekodowanie JPEG i augmentacje PIL dawały ~50 zdjęć/s na wątek. Do tego pierwsze czytanie danych z dysku sieciowego `/kaggle/input` zajęło ~20 min. Przy tym tempie plan P1 wymagałby ~40 h GPU.

**Rozwiązanie:**
1. **`train/cache.py`:** na początku sesji wszystkie zdjęcia są raz dekodowane do pliku uint8 N×256×256×3 (krótszy bok 256, środek 256×256; ~7,5 GB) na lokalnym dysku. Dwa treningi równolegle czytają ten sam plik z pamięci podręcznej systemu.
2. **`train/gpu_aug.py`:** augmentacje całymi paczkami na GPU. Losowe przycięcie ze skalowaniem i odbiciem robi `affine_grid` + `grid_sample`; do tego jasność/kontrast/nasycenie, zmniejszenie rozdzielczości oraz kompresja JPEG (torchvision.io, na GPU). Ewaluacja: Resize(size/0,875) + CenterCrop(size), czyli środek 224 z cache'u, tak samo jak standardowy potok ImageNet i aplikacja na telefonie.
3. **Wynik:** nawet licząc augmentacje na CPU, potok jest 3,7× szybszy (739 vs 200 zdjęć/s na rdzeń). Na GPU ograniczeniem jest już tylko obliczenie modelu.

**Konsekwencje dla eksperymentów:** wszystkie modele trenujemy na wejściu 224 px (EfficientNet-B3 także, zamiast 288). Porównanie architektur odbywa się więc przy tej samej rozdzielczości, a w pracy warto to podkreślić jako uczciwe porównanie. Ablacja E4: 160 vs 256 px (maksimum przy cache'u 256). Zmiana `ColorJitter` (bez odcienia/hue) wynika z kosztu i nie zmienia sensu augmentacji.

## 4. Procedura treningu (`train/train.py`, `configs/base.yaml`)

| Element | Ustawienie | Uzasadnienie |
|---|---|---|
| Transfer learning | etap 1: zamrożony backbone **i jego statystyki BatchNorm**, 3 epoki, lr 1e-3 (tylko klasyfikator); etap 2: pełny fine-tuning, do 25 epok, lr 2e-4 | stabilny start nowej głowicy, potem dostrojenie cech |
| Optymalizator | AdamW, weight decay 0,05 | standard dla CNN z timm |
| Harmonogram | rozgrzewka 5% kroków + cosine | brak ręcznego dobierania spadków lr |
| Funkcja straty | własna CrossEntropy: wagi klas √(1/liczność) (szlak 0,65 … AI 2,58; klasa bez przykładów = 0) na składniku głównym + **jednolite** wygładzanie etykiet 0,05 na klasy obecne w treningu; BCE z `pos_weight` (binarna) | nierównowaga klas bez przesadnego faworyzowania rzadkiej klasy AI; patrz niżej |
| Precyzja | mixed precision (FP16) na GPU | ~2× szybciej na T4 |
| Early stopping | ROC-AUC `p_violation` na val **bez obrazów AI**, cierpliwość 5 epok | **to samo kryterium wyboru modelu we wszystkich eksperymentach**, także w krzywej uczenia z 0 obrazami AI; AI oceniane osobno jako scenariusz ataku |
| Powtórzenia | 3 ziarna (0, 1, 2) | wyniki jako średnia ± odchylenie |
| CNN od zera | lr 1e-3, bez etapu 1, do 40 epok, cierpliwość 8 | brak pretreningu wymaga dłuższego uczenia |
| Reprodukowalność | `config.json` w każdym runie: pełna konfiguracja, hash MD5 manifestu, liczba obrazów AI, złożoność modelu | odtworzenie dowolnego wyniku |

Wynik runu: `runs/<nazwa>/` z plikami `config.json`, `epochs.csv` (krzywe uczenia), `best.pt`, `preds.parquet` (predykcje dla val, test, test_ood i test_real, z podklasą, źródłem i generatorem), `summary.json`, `eval.json` i `decisions.csv`. **Cała ocena odbywa się offline z `preds.parquet`**, więc progi i reguły można zmieniać bez ponownego treningu.

**Dwa błędy wykryte w pierwszej pełnej sesji (2026-11-05), poprawione przed właściwym treningiem:**
1. **`nn.CrossEntropyLoss(weight=…, label_smoothing=…)` waży wagami klas także składnik wygładzania.** Przy rzadkiej klasie AI (duża waga) pcha to model do przypisywania prawdopodobieństwa „AI” każdemu zdjęciu, a przy treningu bez AI (krzywa uczenia, punkt 0) drastycznie: strata ~12,6 zamiast ~1,6, val AUC ~0,50 przez pierwsze epoki. Rozwiązanie: własna funkcja straty z jednolitym wygładzaniem tylko na klasy obecne w treningu i łagodniejsze wagi (pierwiastek z odwrotności liczności).
2. **NaN w EfficientNet-B3:** w etapie 1 BatchNorm zamrożonego backbone aktualizował statystyki. Prawdopodobnie jedna paczka z przepełnieniem FP16 psuła je trwale. Rozwiązanie: BatchNorm w trybie eval w etapie 1, pomijanie paczek z niesk. stratą (przerwanie przy >20%) i `nan_to_num` w predykcjach.

To dobry materiał do rozdziału 4 (pułapki implementacyjne) i argument za śledzeniem eksperymentów w W&B: oba błędy były widoczne od razu na krzywych uczenia.

## 5. Gdzie i jak trenujemy

- **Platforma:** **Kaggle Notebooks, GPU 2× Tesla T4**, darmowe (~30 h GPU tygodniowo, sesja do 12 h), z tego samego konta co generacja AI. Lokalny komputer nie ma GPU NVIDIA, więc służy tylko do testów technicznych na CPU. Zapasowo: Google Colab (ten sam kod).
- **Dane:** paczka wgrana jako prywatny dataset Kaggle `trail-dataset` (`manifest.parquet` + `img/`).
- **Kod:** `kaggle/make_train_kernel.py` wkleja aktualne `src/train/*.py` i `configs/*.yaml` do notebooka (repo = jedyne źródło kodu), a `kaggle kernels push` uruchamia go przez API.
- **Kolejka** (`train/runqueue.py`): runy rozdzielane między 2 GPU, od najdłuższych. Każdy run to trening + ocena. Kolejka pomija runy już zrobione i nie zaczyna runu, który nie zmieści się w limicie czasu sesji.
- **Szacowany czas na T4:**

  | Model | Czas runu |
  |---|---|
  | MobileNetV3 | ~15 min |
  | EfficientNet-Lite0 | ~18 min |
  | CNN | ~22 min |
  | EfficientNet-B3 | ~45 min |

  Do zweryfikowania w pierwszej sesji, bo wąskim gardłem może być dekodowanie JPEG na 4 rdzeniach.

### Śledzenie eksperymentów: Weights & Biases

Projekt W&B **`praca-inz-trail`**, a każdy run treningowy to osobny run W&B (`src/train/wb.py`):
- **konfiguracja:** pełny config + hash manifestu, złożoność modelu, liczba obrazów AI;
- **grupa** = eksperyment (E1_modele, E2_glowica, E3_krzywa_ai, E4_rozdzielczosc);
- **tagi:** model, głowica, `ai<N>`, ziarno, rozdzielczość;
- **co epokę:** `train/loss`, `val/auc_bez_ai`, `lr`, czas;
- **po ocenie** (`evaluate.py` dołącza do tego samego runu): progi `t_przepusc` i `t_blokuj`, ROC-AUC i PR-AUC (z 95% CI) dla każdego zbioru, metryki asystenta moderatora (pokrycie, obciążenie moderacji, błędne blokady, przepuszczone naruszenia) oraz tabela `decyzje_per_podklasa`;
- **po teście odporności:** tabela `odpornosc` i AUC dla każdej degradacji.

**Klucz:** zmienna `WANDB_API_KEY`. Na Kaggle brana jest z sekretu Kaggle o tej nazwie (*Add-ons → Secrets*), a jeśli go nie ma, z lokalnego `.env` (wartość wklejana do prywatnego notebooka w `data/`, poza gitem).
**Bez klucza** albo bez sieci trening działa normalnie, tylko bez logowania. Lokalne pliki (`epochs.csv`, `preds.parquet`, `eval.json`) są zawsze zapisywane i to one są źródłem prawdy dla tabel w pracy.

## 6. Plan eksperymentów (`configs/experiments.yaml`)

| Grupa | Pytanie | Runy | Priorytet | Liczba |
|---|---|---|---|---|
| **E1_modele** | PB1–PB3: która architektura? | cnn, mnv3s, mnv3l, effl0, effb3 × 3 ziarna, głowica 5-klasowa | P1 | 15 |
| **E3_krzywa_ai** | PB4: ile obrazów AI w treningu? | effl0 z 0/250/500/750 AI (+ „wszystkie” = E1) × 3 ziarna; mnv3l, mnv3s, cnn z 0 AI × 3 ziarna | P1 | 21 |
| **E2_glowica** | binarna vs 5-klasowa | effl0, mnv3l binarnie × 3 ziarna | P2 | 6 |
| **E4_rozdzielczosc** | czy wyższa rozdzielczość pomaga przy AI i manipulacjach? | effl0 160 i 288 px, 1 ziarno | P3 | 2 |

**Budżet:**
- **P1** (36 runów) to ~6 h na każdym z 2 GPU, czyli **jedna sesja Kaggle**;
- P2 i P3 to druga, krótsza sesja;
- razem ~15–20 h z limitu ~30 h tygodniowo.

## 7. Ocena (`train/evaluate.py`)

- **Metryki klasyczne:** ROC-AUC (z bootstrapowym 95% CI), PR-AUC, dla val, test, test_ood i test_real.
- **Asystent moderatora:**
  - progi t_przepuść i t_blokuj z val,
  - dla każdego zbioru: pokrycie automatyczne, obciążenie moderacji, błędne blokady autentycznych, przepuszczone naruszenia,
  - dla każdej podklasy i źródła: odsetek przepuszczonych, flagowanych i zablokowanych (`decisions.csv`);
  - dwie reguły: `score` (sam `p_violation`) i `subclass` (blokada tylko naruszeń oczywistych).
- **Agregacja ziaren:** średnia ± sd (`evaluate.py run1 run2 run3 --out results/models/<nazwa>.csv`).
- **Odporność** (`train/robustness.py`): JPEG q 90/70/50, skala ×0,5/×0,25, „komunikator”, „zrzut ekranu”, czyli AUC i decyzje po degradacji.
- **Do zrobienia po treningu:** Grad-CAM (podgląd, na co patrzy model) i galeria najczęstszych błędów.

## 8. Eksport na telefon i kwantyzacja (`train/export.py`)

- **Ścieżka podstawowa (sprawdzona):** PyTorch → ONNX FP32 → ONNX FP16 i **ONNX INT8** (kwantyzacja statyczna, kalibracja na 200 obrazach z train) → ONNX Runtime Mobile na Androidzie. W teście technicznym: FP32 identyczny z PyTorch, FP16 = ½ rozmiaru, INT8 = ~¼ rozmiaru.
- **Ścieżka LiteRT/TFLite** (`--litert`, tylko Linux: Kaggle/Colab) przez `ai-edge-torch`. API tego pakietu zmienia się między wersjami, więc trzeba je sprawdzić przy pierwszym eksporcie. LiteRT daje dostęp do `benchmark_model` i delegatów GPU/NNAPI.
- **Zgodność po eksporcie:** średnia |różnica| `p_violation` i ROC-AUC przed i po konwersji (`export.json`).
- **Pomiary na telefonach** (rozdz. 5.2): `benchmark_model` (LiteRT) lub `onnxruntime_perf_test` (ONNX), mediana i p90, CPU 1 i 4 wątki, GPU. Skrypty do tego etapu dopiszemy po wyborze formatu.

## 9. Instrukcja: od zbioru do wyników

Kroki w PowerShellu, z `Pawel_Palcar_Praca_Inz\src`:
1. **Po zakończeniu przeglądu:**
   ```
   ..\.venv\Scripts\python.exe split.py
   ..\.venv\Scripts\python.exe stats.py
   ```
2. **Paczka:** `..\.venv\Scripts\python.exe pack_dataset.py` tworzy `data/pack/trail-dataset.zip` (2,7 GB przy JPEG q=95; wysoka jakość zachowuje ślady AI i manipulacji).
3. **Upload na Kaggle** jako prywatny dataset `trail-dataset` (CLI z tokenem albo przeglądarka).
4. **Notebook:** `python kaggle/make_train_kernel.py --session 1 --priority P1`, a następnie `kaggle kernels push -p data/kaggle/train_s1`.
5. **Pobranie** `runs.zip` (`kaggle kernels output …`) i rozpakowanie do `runs/`.
6. **Agregacja:** `python -m train.evaluate runs/E1_effl0_multi_s0 runs/E1_effl0_multi_s1 runs/E1_effl0_multi_s2 --out ../results/models/effl0.csv` (analogicznie dla pozostałych modeli).
7. **Eksport wybranych modeli:** `python -m train.export runs/<najlepszy>` (+ `--litert` na Linuksie).
8. **Odporność:** `python -m train.robustness runs/<najlepszy>`.

Kroki 3–5 mogę wykonać przez API Kaggle, tak jak przy generacji obrazów.

## 10. Pliki

| Plik | Zadanie |
|---|---|
| `src/pack_dataset.py` | paczka zbioru do treningu |
| `src/train/data.py` | Dataset, etykiety, augmentacje, podzbiór AI, wagi klas |
| `src/train/models.py` | architektury, zamrażanie backbone, złożoność |
| `src/train/train.py` | trening jednego runu |
| `src/train/evaluate.py` | metryki, progi, decyzja trójstopniowa, CI, agregacja |
| `src/train/robustness.py` | test odporności na degradację |
| `src/train/export.py` | eksport ONNX/LiteRT, FP16/INT8, zgodność |
| `src/train/experiments.py`, `configs/experiments.yaml` | plan eksperymentów |
| `src/train/runqueue.py` | kolejka runów na 2 GPU |
| `configs/base.yaml` | domyślna konfiguracja |
| `kaggle/make_train_kernel.py` | notebook Kaggle z wklejonym kodem |

## 11. Co zostało sprawdzone (test techniczny, CPU, 160 zdjęć)

- trening EfficientNet-Lite0 (5 klas, 2 etapy), MobileNetV3-Small i CNN (binarnie, od zera): przebiega, zapisuje checkpoint i predykcje;
- ocena: progi, decyzje, bootstrapowe CI;
- eksport ONNX FP32/FP16/INT8 z kontrolą zgodności;
- test odporności (8 degradacji);
- kolejka (rozdział na 2 GPU, limit czasu) i budowa notebooka Kaggle.

Liczby z testu technicznego nie mają znaczenia (40 zdjęć na zbiór, 1–2 epoki).
**Niesprawdzone:** trening na GPU Kaggle (pierwsza sesja to zweryfikuje) i eksport LiteRT (wymaga Linuksa).
