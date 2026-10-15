# Klasa negatywna: plan (co skąd wziąć)

Stan źródeł sprawdzony 2026-10-14. Plan zaakceptowany 2026-10-14.

Ograniczenia, które kształtują plan:
- łącze ~1 MB/s, więc unikamy wielogigabajtowych pobrań i korzystamy z zapytań Range, pojedynczych plików oraz podzbiorów;
- brak GPU NVIDIA, więc generacja AI i inpainting dyfuzyjny idą na **Kaggle Notebooks** (darmowe GPU, ~30 h/tydzień), a reszta działa lokalnie na CPU.

## 1. Zasady

1. **Ta sama normalizacja co dla pozytywów:** bez EXIF, krótszy bok 384–768 px, JPEG z losową jakością 70–95. Ten sam `normalize.py`, uogólniony na obie klasy.
2. **Dopasowanie treści.** Negatywy AI i zmanipulowane przedstawiają przede wszystkim naturę lub szlaki, inaczej model nauczy się „góry = OK” zamiast „autentyczne = OK”. Dlatego trzonem tych podklas są **własna generacja i własne manipulacje na naszych pozytywach**.
3. **Brak przecieku między klasami.** Mem, manipulacja albo obraz img2img zrobione z pozytywu X dziedziczą `group` po X, więc trafiają do tego samego splitu co X.
4. **Część źródeł od razu odkładamy do testu OOD (T2):** generatory, zbiory i style manipulacji, których model nie zobaczy w treningu.
5. **AI i manipulacje dyfuzyjne traktujemy jako scenariusz ataku**, a nie obietnicę wykrywania. Mało przykładów w treningu, osobny test na generatorze znanym i nieznanym (patrz punkt 5).

## 2. Docelowe liczności

| Podklasa | Train/val/test | test_ood (T2) | Trzon |
|---|---|---|---|
| `urban` (miasto, wnętrza, „śmieci”) | ~5 000 | ~500 | Places365 + Mapillary-miasta + COCO |
| `meme` (memy, tekst, zrzuty ekranu) | ~5 000 | ~500 | **syntetyczne na naszych pozytywach** + MemeXGen + Imgflip |
| `ai` (wygenerowane) | 1 000 train/val + 250 test | ~500 | **własna generacja krajobrazów na Kaggle** |
| `manipulated` (zmienione) | ~4 500 | ~1 000 | **własne manipulacje zdjęć szlaków** + DEFACTO / IMD2020 |

Razem ~16–17 tys. negatywów wobec ~19 tys. pozytywów. Nierównowagę wyrównujemy wagami klas w funkcji straty, a nie dokładaniem danych.

## 3. Źródła po podklasach

### 3.1 `urban`: miasto, wnętrza, typowe „śmieci” z telefonu

| Źródło | Co bierzemy | Ile | Jak | Licencja |
|---|---|---|---|---|
| **Places365** | ulica, centrum, parking, autostrada, przejście, centrum handlowe, kuchnia, sypialnia, biuro, salon, restauracja, stacja metra, stadion… | 2 500 | `collect/places365.py --set urban` (zapytania Range, ~100 MB) | badania |
| **Mapillary, miasta** | Kraków, Zakopane, Wrocław, Bratysława, Lublana | 1 500 | ten sam kolektor z **odwrotnym filtrem** (≤20 m od drogi), co daje ten sam typ kamer co pozytywy z Mapillary | CC BY-SA 4.0 |
| **COCO 2017** | jedzenie, zwierzęta domowe, wnętrza, selfie-podobne, sport | 1 000 | pojedyncze pliki z `images.cocodataset.org` wybrane po adnotacjach | CC BY (Flickr) |
| T2: **Open Images** | ujęcia miejskie | 500 | metadane już w cache | CC BY 2.0 |

**Trudne negatywy:** parki miejskie i ogrody (decyzja: negatyw), ulica w górskim mieście (Zakopane, Krupówki), wyciągi narciarskie.

### 3.2 `meme`: memy, nałożony tekst, zrzuty ekranu

| Źródło | Co | Ile | Jak | Licencja |
|---|---|---|---|---|
| **Syntetyczne memy na pozytywach** | napis Impact góra/dół, biały pasek z tekstem (styl Twitter/IG), podpisy „object labeling”, emoji i naklejki; **część z wulgarnymi lub obraźliwymi napisami** (przypadek „ktoś dopisał coś wulgarnego do zdjęcia z gór”) | 2 000 | `synth/make_memes.py` (PIL, lokalnie) | własne |
| **Syntetyczne zrzuty ekranu** | pozytyw w ramce interfejsu galerii, Instagrama lub komunikatora (decyzja: zrzut ekranu to negatyw) | 500 | `synth/make_screens.py` (PIL) | własne |
| **MemeXGen** (HF `YZhao09/MemeXGen`) | prawdziwe memy z podpisami, pliki `*_with_caption.jpg` | 1 500 | pojedyncze pliki z HF | CC BY 4.0 |
| **Imgflip** | ~100 najpopularniejszych szablonów (API `get_memes`) z podpisami | 1 000 | `collect/imgflip.py` | szablony: użycie badawcze |
| T2: **PolyMeme** (Zenodo) | lista URL do memów z Reddita | 500 | `img2dataset`; część linków nie działa | badania |

**Pomijamy:** Hateful Memes, MemeXplain, MemeLens i podobne (treści nienawistne i propagandowe, restrykcyjne warunki użycia).

### 3.3 `ai`: obrazy wygenerowane (scenariusz ataku)

| Część | Generatory | Ile | Rola | Czas GPU (T4) |
|---|---|---|---|---|
| trening + walidacja | SD 1.5 + SDXL | 1 000 (850 + 150) | train / val | ~2–3 h |
| test, generator znany | SD 1.5 + SDXL (inne prompty i inne zdjęcia źródłowe) | 250 | test | ~0,5 h |
| test, generator nieznany | **PixArt-Σ** (zamiast FLUX, decyzja 2026-10-14) | 250 | test_ood | ~2–3 h |
| test, generatory komercyjne | **OpenFake** (GPT Image, nano-banana, Flux.2…), sceny plenerowe | 250–500 | test_ood | bez GPU |
| test „z życia” | darmowe generatory webowe, ręcznie | ~50 | test_real (T3) | — |

- **Prompty:** kombinacje {warunek: błoto / śnieg / lód / powalone drzewo / mgła / zalana ścieżka / sucho} × {teren: las bukowy / hala / grań / kamienny szlak / kosówka} × {styl: „phone photo”, „amateur snapshot”, „overcast”, „blurry”}.
- **~1/3 obrazów w każdej części to img2img z naszych pozytywów** (strength 0,3–0,6), czyli najtrudniejszy przypadek. Dziedziczą `group` po zdjęciu wejściowym.
- **Łącznie ~1 500 własnych obrazów, ~5–6 h GPU,** czyli 1–2 sesje na Kaggle.

### 3.4 `manipulated`: zdjęcia zmienione

**Trzon: własne manipulacje zdjęć szlaków.** Każda dziedziczy `group` po oryginale.

| Technika | Narzędzie | Gdzie | Ile | Co symuluje |
|---|---|---|---|---|
| **Usuwanie obiektu** | LaMa (`simple-lama-inpainting`) | CPU, lokalnie | 1 000 | „magiczna gumka” w telefonie |
| **Splicing** | maski obiektów z COCO wklejone na szlak; połowa z wygładzeniem krawędzi i korekcją kolorów | CPU | 1 000 | doklejony obiekt |
| **Copy-move** | OpenCV, powielenie fragmentu (kamienie, kałuże, drzewa) | CPU | 500 | klasyczne fałszerstwo |
| **Inpainting dyfuzyjny** (scenariusz ataku) | SD/SDXL-inpainting: „deep snow”, „fallen tree blocking path”, „flooded trail”, „dry path” | Kaggle GPU | 300 train + 300 test | fałszywy raport warunków |

**Uzupełnienie i T2**

| Zbiór | Co | Ile | Rola |
|---|---|---|---|
| **DEFACTO inpainting** (HF `bahakirbasoglu/DefactoInpaintingDataset`) | automatyczny inpainting na COCO | 1 000 | train |
| **CASIA v2** (Kaggle, token) | splicing i copy-move | 500 | train |
| T2: **IMD2020** (HF `KhangTruong/IMD2020`, tylko `test`) | ręczne manipulacje „z internetu” | 500 | test_ood |
| T2: **CocoGlide** (repo TruFor) | inpainting dyfuzyjny (GLIDE) | 500 | test_ood |

**Polityka etykiet:**
- jasność, kontrast, filtr, kadrowanie, HDR z telefonu → **autentyczne**;
- jakakolwiek zmiana treści sceny, w tym telefonowa „magiczna gumka” → **manipulated**.

## 4. Treści wulgarne: kaskada z gotowym filtrem NSFW

Blokowanie treści wulgarnych to wymaganie krytyczne aplikacji, ale **nie dodajemy zdjęć pornograficznych do zbioru**. Powody:
- ryzyko prawne: zbiory zebrane z sieci mogą zawierać materiały z udziałem nieletnich lub publikowane bez zgody;
- RODO: dane szczególnej kategorii;
- wymagałoby to zgody etycznej uczelni;
- większość takich zdjęć i tak nie przedstawia szlaku, więc klasyfikator semantyczny je odrzuca.

Zamiast tego aplikacja stosuje **kaskadę**:

```
zdjęcie ─▶ [1] gotowy mobilny filtr NSFW ─▶ [2] nasz model: szlak / autentyczność ─▶ akceptuj / moderacja / odrzuć
```

- **Kandydaci na etap 1:** NSFWJS (MobileNetV2, kilka MB), NudeNet. Wybór i sprawdzenie licencji przy implementacji.
- **W pracy mierzymy:**
  1. rozmiar i czas działania całej kaskady na telefonie;
  2. **fałszywe alarmy filtra NSFW na naszych ~19 tys. pozytywów** (np. turyści w strojach kąpielowych nad jeziorem), do czego nie potrzeba wulgarnych zdjęć;
  3. skuteczność wykrywania nagości cytujemy za autorami modelu.
- **Wulgarny tekst** nałożony na zdjęcie szlaku obsługuje nasz model w ramach podklasy `meme` (punkt 3.2).

## 5. Eksperymenty z AI (do rozdziału wyników)

**Krzywa uczenia:** trening z 0 / 250 / 500 / 750 / 1 000 obrazami AI. Wszystko inne pozostaje stałe: zbiór testowy, pozostałe negatywy i ustawienia treningu. Dla każdego punktu mierzymy:
- wykrywalność AI, osobno dla generatora znanego (test) i nieznanego (FLUX, OpenFake);
- **odsetek odrzuconych prawdziwych zdjęć**, czyli koszt dodania AI;
- wykrywalność pozostałych podklas, żeby sprawdzić, czy AI nie psuje reszty.

**Budżet treningów:**

| Model | Punkty krzywej | Powtórzenia | Treningów |
|---|---|---|---|
| EfficientNet-Lite0 | 0 / 250 / 500 / 750 / 1 000 | 3 | 15 |
| MobileNetV3, własny CNN | 0 / 1 000 | 3 | 12 |

Razem ~27 treningów zamiast 45.

Analogicznie można sprawdzić manipulacje dyfuzyjne (0 vs 300 w treningu).

## 6. Zmiany w kodzie

| Plik | Zmiana |
|---|---|
| `common.py` | `label` i `subclass` w każdym rekordzie |
| `collect/places365.py` | zestawy klas (`--set nature` / `--set urban`) |
| `collect/mapillary.py` | tryb `--mode city`: odwrotny filtr OSM, regiony miejskie |
| nowe `collect/coco.py`, `collect/memexgen.py`, `collect/imgflip.py`, `collect/openfake.py`, `collect/forgery.py` | pobieranie źródeł z tabel powyżej |
| nowe `synth/make_memes.py`, `synth/make_screens.py`, `synth/manipulate_cpu.py` | syntetyczne negatywy liczone lokalnie |
| nowe `kaggle/generate_ai.ipynb`, `kaggle/inpaint.ipynb` | generacja na GPU; wynik to zip z obrazami i `meta.jsonl` |
| `normalize.py`, `dedup.py`, `split.py`, `stats.py` | wspólne dla obu klas, z jednym manifestem `dataset.parquet` (`label`, `subclass`, `group`, `split`) |
| `review.py` | jeden przegląd obu klas, z widokami per podklasa |

## 7. Kolejność

1. **Lokalnie:** urban (Places365, Mapillary-miasta, COCO), memy (MemeXGen, Imgflip, synteza), manipulacje CPU (LaMa, splicing, copy-move).
2. **Kaggle:** generacja AI (~1 500) i inpainting dyfuzyjny (~600).
3. **Publiczne zbiory na T2:** OpenFake, IMD2020, CocoGlide, PolyMeme.
4. **Wspólny pipeline:** CLIP, normalizacja, deduplikacja, podział.
5. **Ręczny przegląd obu klas na raz** (Ty).

## 7a. Stan realizacji i zmiany względem planu (2026-10-14)

| Podklasa | Źródło | Skrypt | Uwagi |
|---|---|---|---|
| urban | Places365 (63 kategorie, 4 670 surowych) | `collect/places365.py --set urban` | trudne negatywy oznaczone `hard=True` |
| urban | Mapillary, 5 miast | `collect/mapillary.py --mode city` | ≤20 m od drogi; **Lublana = test_ood** (zamiast Open Images) |
| urban | COCO val2017 (1 000) | `collect/coco.py` | adnotacje wyciągnięte z zipa zapytaniami Range (20 MB z 250 MB) |
| meme | syntetyczne (3 000) | `synth/make_memes.py` | 2 000 na pozytywach (impact, bar, labels, emoji, vulgar) + 1 000 na 100 szablonach Imgflip |
| meme | zrzuty ekranu (500) | `synth/make_screens.py` | galeria, kanał społecznościowy, komunikator |
| meme | MemeXGen | `collect/memexgen.py` | **tylko oryginały** (wersje `2_flux.png` są wygenerowane przez AI); EN 1 500 → trening, ZH 500 → test_ood (zamiast PolyMeme) |
| manipulated | własne CPU (2 500) | `synth/manipulate_cpu.py` | LaMa, splicing z masek COCO, copy-move |
| manipulated | DEFACTO / CocoGlide / IMD2020 | `collect/forgery.py` | archiwa i parquet czytane fragmentami; CocoGlide i IMD2020 → test_ood |
| ai | OpenFake (test) | `collect/openfake.py` | indeks samych promptów, potem tylko wybrane grupy wierszy |
| ai + manipulated | Kaggle (2 100 zadań) | `kaggle_prep.py` → `kaggle/generate.ipynb` → `collect/kaggle_import.py` | instrukcja: `03_kaggle_instrukcja.md` |

Pipeline (`normalize.py`, `dedup.py`, `split.py`, `stats.py`, `review.py`) obsługuje teraz obie klasy, z jednym manifestem `data/manifests/dataset.parquet`. Zmiany:
- **stabilny wybór kwot** (kolejność według hasha `id` zamiast losowania całej tabeli), więc dodanie danych nie zmienia wyboru w innych źródłach;
- **duplikaty usuwane tylko w obrębie etykiety i tylko wśród oryginałów**, bo przeróbki są celowo prawie identyczne z oryginałem;
- **klucz podziału = najmniejsza nazwa grupy w komponencie**, więc jest stabilny, gdy dochodzą przeróbki;
- **`force_split` dla zadań z Kaggle**, żeby role train/test generatorów były zachowane;
- **negatywy `urban`, które CLIP ocenia jako szlak (`trail ≥ 0,5`), są odrzucane** jako sprzeczne etykiety.

Nie zrealizowano: CASIA v2 (wymaga tokenu Kaggle; opcjonalne, bo DEFACTO i własne manipulacje wystarczają).

### Stan końcowy (2026-10-14, przed przeglądem ręcznym)

Kaggle: 4 sesje (test, 1, 2, 3), łącznie ~7 h GPU. Inpainting SD 1.5: filtr NSFW wbudowany w potok zwracał czasem czarne obrazy dla krajobrazów (fałszywe alarmy). `run_gen.py` ponawia wtedy próbę z innym ziarnem, a po 3 nieudanych próbach pomija zadanie. Pominięto 2 z 450.

| | train | val | test | test_ood | razem |
|---|---|---|---|---|---|
| pozytywy (nature) | 12 225 | 2 562 | 2 599 | 1 897 | 19 283 |
| urban | 3 253 | 662 | 665 | 387 | 4 967 |
| meme | 3 245 | 622 | 717 | 779 | 5 363 |
| manipulated | 2 501 | 526 | 520 | 1 438 | 4 985 |
| ai | 852 | 147 | 226 | 510 | 1 735 |
| **razem** | 22 076 | 4 519 | 4 727 | 5 011 | **36 333** |

- **AI w test_ood:** PixArt-Σ (250), OpenFake (~235, 20 generatorów komercyjnych i otwartych) oraz 25 obrazów img2img SD/SDXL, których zdjęcie źródłowe leży w test_ood.
- **Kontrola skrótów:** średnia jakość JPEG 82,6 w obu klasach; mediana krótszego boku 458 px (negatywy) vs 432 px (pozytywy).
- Pełne tabele: `results/dataset/*.csv`.

## 8. Decyzje (podjęte 2026-10-14)

1. **Park miejski i ogród → negatyw `urban` (trudny negatyw).** Ryzyko odrzucenia zgłoszenia z leśnego parku na skraju miasta opisujemy w dyskusji.
2. **Zrzut ekranu prawdziwego zdjęcia szlaku → negatyw `meme`.** Zgłoszenie ma być zdjęciem z aparatu.
3. **GPU → Kaggle Notebooks.** Token API Kaggle odblokuje też CASIA v2.
4. **AI jako scenariusz ataku:** 1 000 train/val + 250 test + 250 generator nieznany + OpenFake, z krzywą uczenia (punkt 5).
7. **Generator nieznany: PixArt-Σ zamiast FLUX.1-schnell.** FLUX wymaga konta i licencji na HF; PixArt-Σ jest bez logowania i to inna rodzina modeli (transformer + T5). Jego zadania img2img wykonujemy jako text2img.
8. **OpenFake ograniczony do ~260 obrazów** (trafienia rozproszone, pełny plan = ~5 GB pobierania); bez generatora Illustrious (styl anime, nie udaje zdjęcia).
5. **Treści wulgarne → kaskada z gotowym filtrem NSFW,** bez zdjęć pornograficznych w zbiorze (punkt 4).
6. **Licencje NC** (OpenFake CC BY-NC): w pracy dyplomowej są w porządku, ale zbioru nie redystrybuujemy. Publikujemy manifest i skrypty, nie obrazy.
