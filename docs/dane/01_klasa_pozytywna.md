# Klasa pozytywna: skąd są dane, jak je zbieramy, co dokładnie zapisujemy

Stan źródeł sprawdzony 2026-10-08.

## 1. Czym jest „klasa pozytywna”

Pozytyw (`label=1`, `subclass="nature"`) to **autentyczne, niezmienione zdjęcie zrobione w plenerze górskim, leśnym lub na szlaku**, takie, jakie realnie wyśle użytkownik aplikacji do zgłaszania warunków na szlaku.

Do pozytywów celowo należą też **trudne pozytywy**, inaczej model odrzucałby prawdziwych użytkowników:
ludzie na szlaku, schroniska i chatki, drogowskazy, mgła, zmierzch, śnieg, błoto, kałuże, powalone drzewa, mostki, zdjęcia pod słońce lub lekko poruszone.

Pozytywem **nie jest**: miasto, wnętrze, plaża, mapa, zrzut ekranu, obraz/rysunek, zdjęcie z drona, stare zdjęcie czarno-białe, zdjęcie z watermarkiem lub nałożonym tekstem (to ostatnie jest ważne, bo tekst ma być sygnałem mema).

## 2. Źródła w skrócie

| Źródło | Kto tworzy / skąd zdjęcia | Licencja | Cel (po filtrach) | Rola |
|---|---|---|---|---|
| Places365 | MIT CSAIL; zdjęcia z wyszukiwarek, kategorie sceny oznaczone przez ludzi | badania, niekomercyjnie | ~8 000 | trzon zbioru (train/val/test) |
| Mapillary | crowdsourcing: użytkownicy wgrywają zdjęcia z telefonów i kamerek | CC BY-SA 4.0 | ~5 000 | najbardziej „telefonowe” zdjęcia szlaków; Bieszczady → test OOD |
| Unsplash Lite | fotografowie wgrywający zdjęcia na Unsplash | Unsplash License (Lite: komercyjnie i niekomercyjnie) | ~4 000 | różnorodność, ale estetyczny bias |
| Open Images V7 | Google; zdjęcia z Flickra z etykietami | CC BY 2.0 (zdjęcia), CC BY 4.0 (etykiety) | ~2 000 | amatorskie zdjęcia, zastępstwo Flickr API |
| Wikimedia Commons | wolontariusze Wikipedii | CC BY / BY-SA / CC0 per plik | ~1 500 | **w całości test OOD** (źródło niewidziane w treningu) |
| Własne zdjęcia | Ty i znajomi, różne telefony | własne, za zgodą | 250–300 | **w całości test_real (T3)** |

Liczby to cele. Faktyczne liczności po każdym etapie zapisuje plik `results/dataset/pos_funnel.csv` (lejek selekcji).

## 3. Przepływ danych

```
collect/*.py ─▶ data/raw/<źródło>/img/*.jpg + meta.jsonl      (surowe, z metadanymi)
clip_filter.py ─▶ data/interim/clip.parquet                    (ocena treści + tagi)
normalize.py  ─▶ data/processed/positive/*.jpg                 (ujednolicone JPEG-i)
review.py     ─▶ data/interim/rejected.txt                     (ręcznie odrzucone)
dedup.py      ─▶ data/interim/positive_dedup.parquet           (duplikaty i klastry)
split.py      ─▶ data/manifests/positive.parquet / .csv        (MANIFEST, finalny wynik)
stats.py      ─▶ results/dataset/*.csv                         (tabele do rozdz. 3)
```

Każdy skrypt zbierający dopisuje do swojego `meta.jsonl` i po przerwaniu wznawia pracę od miejsca, w którym skończył (pomija już zapisane `id`).

## 4. Źródła po kolei

### 4.1 Places365-Standard

**Co to jest.** Zbiór do rozpoznawania scen z MIT (Zhou i in., *Places: A 10 Million Image Database for Scene Recognition*, IEEE TPAMI 2018). Wersja Standard ma 365 kategorii scen i ~1,8 mln zdjęć treningowych (3 068–5 000 na kategorię).

**Skąd zdjęcia.** Autorzy zbierali je z wyszukiwarek obrazów, wpisując nazwy kategorii scen z przymiotnikami. Potem pracownicy Amazon Mechanical Turk weryfikowali, czy zdjęcie pasuje do kategorii. Prawa autorskie należą do autorów oryginałów, a zbiór udostępniono do badań.

**Jak pobieramy** (`collect/places365.py`):
1. Otwieramy strumień HTTP do archiwum `train_256_places365standard.tar` (26 GB, serwer MIT).
2. Czytamy tar „w locie”, plik po pliku, bez zapisywania całego archiwum.
3. Z nazwy pliku (`data_256/m/mountain_path/00000123.jpg`) odczytujemy kategorię.
4. Zapisujemy tylko wybrane kategorie, losowo ~70% plików, do limitu na kategorię. Na dysku zostaje ~0,5 GB.

Plan awaryjny: `collect/places365_hf.py` czyta mirror na Hugging Face (parquet, ~19 GB strumienia).

**Co bierzemy.** 25 kategorii:
- szlaki i teren: `mountain_path`, `forest_path`, `mountain`, `mountain_snowy`, `forest/broadleaf`, `forest_road`, `valley`, `field/wild`, `creek`, `cliff`, `snowfield`, `river`, `waterfall`, `pasture`, `glacier`, `canyon`, `marsh`, `swamp`, `rock_arch`, `crevasse`;
- infrastruktura szlakowa jako trudne pozytywy: `cabin/outdoor`, `chalet`, `hunting_lodge/outdoor`, `rope_bridge`, `campsite`.

**Co zapisujemy:** `source_id` (ścieżka w archiwum), `places_class`, `license`, `group` (= samo zdjęcie, bo brak informacji o autorze).

**Wady do opisania w pracy.** Rozdzielczość 256×256 (niższa niż w innych źródłach). Kategoria jest przypisana do sceny, a nie do szlaku, więc np. `valley` bywa widokiem z samochodu. Odfiltrowuje to CLIP i przegląd ręczny.

### 4.2 Mapillary

**Co to jest.** Platforma zdjęć ulicznych (od 2020 r. należy do Meta) z ponad 2 mld zdjęć. Każdy może wgrywać serie zdjęć z telefonu, kamerki sportowej albo kamery na samochodzie. Zdjęcia mają pozycję GPS i są łączone w **sekwencje** (jeden przejazd lub spacer).

**Skąd zdjęcia.** Od zwykłych użytkowników, więc to zdjęcia „brzydkie”, z telefonów, bez kadrowania. Najbliżej im do tego, co przyśle użytkownik aplikacji.

**Jak pobieramy** (`collect/mapillary.py`):
1. Dzielimy region (np. Tatry) na kafelki mapy (zoom 14). Jeśli w kafelku jest ≥2000 zdjęć (limit API), dzielimy go dalej na mniejsze.
2. Dla każdego kafelka API `graph.mapillary.com/images` zwraca listę zdjęć: id, pozycję, datę, sekwencję, typ kamery i autora.
3. Z OpenStreetMap (przez Overpass API) pobieramy geometrię ścieżek (`highway=path/footway/track/bridleway/steps`) i dróg.
4. **Zostawiamy zdjęcie, jeśli leży ≤20 m od ścieżki i ≥60 m od drogi.** To główny filtr „to jest szlak, a nie droga”.
5. Odrzucamy panoramy 360° i obiektywy rybie oko.
6. Z każdej sekwencji bierzemy maksymalnie 6 klatek, równomiernie w czasie, żeby jeden spacer nie zdominował zbioru.
7. Pobieramy miniaturę 1024 px.

**Regiony:** Tatry, Beskid Żywiecki, Gorce, Pieniny, Karkonosze. **Bieszczady są holdoutem geograficznym**: trafiają w całości do `test_ood`, co pozwala sprawdzić, czy model działa w górach, których nie widział w treningu.

**Co zapisujemy:** `source_id`, `group` = sekwencja (cała sekwencja zawsze w jednym splicie), `region`, `lon`/`lat`, `camera_type`, `captured_at`, `author`, `license`.

**Licencja.** CC BY-SA 4.0, z atrybucją (autor w manifeście). Dane OSM są na licencji ODbL („© OpenStreetMap contributors”); używamy ich tylko do filtrowania.

**Wymaga:** darmowego tokenu Mapillary (patrz punkt 7).

### 4.3 Unsplash Lite

**Co to jest.** Oficjalny, otwarty zbiór od Unsplash: 25 000 zdjęć o tematyce głównie przyrodniczej plus metadane (słowa kluczowe, autor, aparat z EXIF, lokalizacja). Wersję Lite wolno używać komercyjnie i niekomercyjnie.

**Skąd zdjęcia.** Fotografowie, często półprofesjonalni, wgrywają je na Unsplash.

**Jak pobieramy** (`collect/unsplash_lite.py`):
1. Skrypt sam pobiera archiwum z metadanymi (~320 MB, pliki TSV) i je rozpakowuje.
2. Każde zdjęcie dostaje punkty ze słów kluczowych: +1 za każde „górskie” (mountain, trail, forest, hiking…), −2 za każde „miejskie” (city, street, beach, aerial…). Bierzemy zdjęcia z wynikiem > 0.
3. Maksymalnie 15 zdjęć od jednego fotografa, łącznie 6 000 kandydatów.
4. Obraz pobieramy z CDN Unsplash w szerokości 1024 px.

**Co zapisujemy:** `source_id`, `url`, `author`, `group` = fotograf, `camera` (marka i model aparatu), `lat`/`lon`, `license`.

**Wada (ważne).** Zdjęcia są „ładne” i często zrobione lustrzankami. Dlatego Unsplash to maksymalnie ~20% pozytywów, a kolumna `camera` pozwala policzyć, ile zdjęć pochodzi z telefonów.

### 4.4 Open Images V7

**Co to jest.** Zbiór Google (Kuznetsova i in., *The Open Images Dataset V4*, IJCV 2020) z ~9 mln zdjęć i etykietami na poziomie obrazu z ~20 tys. klas.

**Skąd zdjęcia.** Z Flickra, wybrane z tych, które autorzy udostępnili na licencji CC BY 2.0. To w dużej części amatorskie zdjęcia. Open Images zastępuje Flickr API, które od 2025 r. wydaje klucze tylko płatnym kontom Pro.

**Jak pobieramy** (`collect/openimages.py`). Biblioteka FiftyOne pobiera pliki etykiet, wybiera zdjęcia mające co najmniej jedną z klas: `Hiking`, `Trail`, `Footpath`, `Path`, `Mountain`, `Mountain range`, `Mountain pass`, `Ridge`, `Summit`, `Highland`, `Hill`, `Forest`, `Old-growth forest`, `Wilderness`, `Valley`, `Waterfall`, `Glacier`, `Nature reserve` (nazwy sprawdzone w `oidv7-class-descriptions.csv`), i ściąga 6 000 losowych zdjęć.

**Co zapisujemy:** `source_id` (ID Open Images), `oi_labels` (wszystkie etykiety zdjęcia), `group` (= samo zdjęcie), `license`. Pliki zostają w cache FiftyOne (`~/fiftyone/`).

**Wada.** Klasa `Path` obejmuje też chodniki miejskie, więc filtr CLIP jest tu potrzebny.

### 4.5 Wikimedia Commons (cały → test OOD)

**Co to jest.** Repozytorium mediów Wikipedii. Każdy plik ma własną licencję (CC BY, CC BY-SA, CC0, domena publiczna).

**Jak pobieramy** (`collect/wikimedia.py`):
1. Startujemy od kategorii (sprawdzonych, że istnieją): *Hiking trails in Poland*, *Hiking trails in Slovakia*, *Hiking in Poland*, *Trails in Poland*, *Tatra Mountains*, *Beskid*, *Bieszczady*.
2. Schodzimy do podkategorii na głębokość 2 i odrzucamy po nazwie mapy, obrazy, kościoły, hotele, narciarstwo, zdjęcia historyczne itp.
3. Bierzemy tylko JPEG z krótszym bokiem ≥600 px i pomijamy licencje ND (bez utworów zależnych).
4. Pobieramy miniaturę o szerokości 960 px, z autorem i licencją.

**Dlaczego całość do testu.** To źródło, którego model nie zobaczy w treningu. Pokazuje, czy model uogólnia na inny styl zdjęć, czy nauczył się cech konkretnych źródeł.

**Co zapisujemy:** `source_id` (tytuł pliku), `url`, `license`, `author`, `group` = autor, `category`, `captured_at`, `holdout="source"`.

### 4.6 Własne zdjęcia (cały → test_real / T3)

**Jak.** Wrzucasz pliki do `data/raw/own/<osoba>/` (JPG, HEIC z iPhone'a, PNG) i uruchamiasz `collect/own_photos.py`.

**Co zapisujemy:** autor (nazwa folderu), aparat (z EXIF), data wykonania. **GPS celowo nie jest zapisywany (RODO).** `group` = osoba + dzień, więc zdjęcia z jednej wycieczki trzymają się razem.

**RODO.** Jeśli na zdjęciach od znajomych są twarze: zbierz zgody albo wyklucz te zdjęcia. Zbioru T3 nie publikujemy; w pracy wystarczy jego opis.

## 5. Co dzieje się z danymi po pobraniu

### 5.1 Filtr i tagowanie CLIP (`clip_filter.py`)
CLIP (Radford i in., 2021), w implementacji OpenCLIP: model ViT-B/32 trenowany na LAION-2B. Porównuje obraz z opisami tekstowymi bez dodatkowego treningu (zero-shot) i daje trzy rodzaje wyników:
- `trail` / `nature` / `other`: prawdopodobieństwo, że zdjęcie to szlak, natura albo „coś innego” (miasto, wnętrze, mapa, mem, dron…);
- tagi trudnych pozytywów: `tag_people`, `tag_fog`, `tag_night`, `tag_snow`, `tag_mud`, `tag_sign`, `tag_hut`;
- `tag_text`: tekst lub watermark na zdjęciu.

Odrzucamy zdjęcia z `other ≥ 0,30` lub `tag_text ≥ 0,5`. Tagi są **przybliżone**: służą do statystyk i ukierunkowania przeglądu, a nie jako prawda. Filtr CLIP sam wprowadza stronniczość selekcji (zostają zdjęcia „typowe” dla CLIP), co trzeba opisać w ograniczeniach.

### 5.2 Normalizacja (`normalize.py`), czyli walka ze skrótami
Model nie może rozpoznawać źródła po formacie, rozdzielczości ani metadanych. Dlatego **każde** zdjęcie (później tak samo negatywy):
- traci EXIF i metadane,
- jest tylko zmniejszane (krótszy bok losowo 384–768 px), nigdy powiększane,
- jest zapisywane jako JPEG z losową jakością 70–95.

Odrzucamy: krótszy bok <256 px, proporcje >2,2:1 (panoramy) i zdjęcia czarno-białe. Następnie stosujemy kwoty na źródło (tabela w punkcie 2) i limity na grupę (Unsplash 15 na fotografa, Wikimedia 20 na autora, Mapillary 6 na sekwencję).

### 5.3 Przegląd ręczny (`review.py`, FiftyOne)
Nie przeglądasz wszystkiego, tylko cztery widoki: najwyższy `tag_text`, najwyższy `other`, najniższy `trail` i **losowe 200 z każdego źródła**. Ten ostatni widok daje szacowany odsetek błędnych etykiet per źródło, czyli liczbę do rozdziału 3.

### 5.4 Deduplikacja (`dedup.py`)
- pHash różniący się o ≤4 bity: duplikat, zostaje wersja o większej rozdzielczości;
- pHash ≤10 lub podobieństwo CLIP ≥0,95: „to samo miejsce/ujęcie”. Oba zdjęcia zostają, ale muszą trafić do tego samego splitu, inaczej test byłby zawyżony.

### 5.5 Podział (`split.py`)
- Zdjęcia połączone grupą (sekwencja, fotograf) lub klastrem duplikatów tworzą komponent, który w całości trafia do jednego splitu.
- Split wyznacza hash, więc jest deterministyczny i stabilny po dodaniu nowych danych: 70% train, 15% val, 15% test.
- Specjalne splity: `test_ood` (Bieszczady z Mapillary i całe Wikimedia) oraz `test_real` (własne zdjęcia).

## 6. Manifest: jeden wiersz = jedno zdjęcie

`data/manifests/positive.parquet` (i `.csv`). Ten plik wersjonujemy w git; obrazów nie (licencje).

| Kolumna | Znaczenie |
|---|---|
| `id` | stały identyfikator (hash źródła i ID w źródle) |
| `path` | ścieżka do znormalizowanego JPEG |
| `label`, `subclass` | `1`, `"nature"` |
| `source`, `source_id`, `url` | skąd dokładnie pochodzi zdjęcie |
| `license`, `author` | licencja i atrybucja |
| `group` | jednostka, której nie wolno rozdzielić między splity |
| `region`, `lon`, `lat`, `camera`, `camera_type`, `captured_at` | metadane źródła (jeśli są) |
| `places_class`, `oi_labels`, `category` | etykieta lub kategoria w oryginalnym źródle |
| `orig_w`, `orig_h`, `w`, `h`, `jpeg_q` | rozmiar przed i po normalizacji, jakość JPEG |
| `trail`, `nature`, `other`, `top_prompt`, `tag_*` | wyniki CLIP |
| `phash`, `dup_cluster` | deduplikacja |
| `holdout`, `split` | `train` / `val` / `test` / `test_ood` / `test_real` |

## 7. Jak uruchomić

Jednorazowo (PowerShell, w katalogu repo):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env      # potem uzupełnij .env
```

Token Mapillary: załóż konto na mapillary.com, wejdź w *Dashboard → Developers → Register application* i skopiuj **Client Token** (`MLY|...`) do `.env`.

Przebieg testowy (kilka zdjęć z każdego źródła), z katalogu `src`:

```powershell
cd src
python -m collect.wikimedia --limit 20
python -m collect.unsplash_lite --limit 20
python -m collect.openimages --limit 20
python -m collect.mapillary --limit 20
```

Pełne zbieranie (kroki 1–5 możesz puścić równolegle w osobnych terminalach):

| # | Komenda | Czas (orientacyjnie) |
|---|---|---|
| 1 | `python -m collect.places365` | 30–90 min (26 GB strumienia) |
| 2 | `python -m collect.unsplash_lite` | 30–60 min |
| 3 | `python -m collect.openimages` | 1–2 h |
| 4 | `python -m collect.mapillary` | 1–3 h |
| 5 | `python -m collect.wikimedia` | 1–2 h |
| 6 | `python -m collect.own_photos` | na bieżąco |
| 7 | `python clip_filter.py` | ~30–60 min na CPU |
| 8 | `python normalize.py` | 15–30 min |
| 9 | `python review.py` | 2–3 h pracy ręcznej |
| 10 | `python dedup.py`, `python split.py`, `python stats.py` | ~15 min |

## 8. Co z tego etapu trafia do rozdziału 3 pracy

- tabela źródeł: źródło, licencja, liczność, rola (train / OOD / real);
- lejek selekcji (`pos_funnel.csv`): surowe → po CLIP → po kwotach → po normalizacji → po przeglądzie i deduplikacji;
- parametry selekcji: progi CLIP, kryteria geograficzne Mapillary (20 m / 60 m), progi pHash i CLIP, parametry normalizacji;
- szacowany szum etykiet per źródło (z losowego przeglądu);
- rozkład cech: tagi, rozdzielczość, sezonowość;
- ograniczenia: przybliżone tagi CLIP, stronniczość filtra CLIP, 256 px w Places365, estetyczny bias Unsplash.

## Bibliografia źródeł

- Zhou B., Lapedriza A., Khosla A., Oliva A., Torralba A. *Places: A 10 Million Image Database for Scene Recognition*. IEEE TPAMI, 2018.
- Kuznetsova A. i in. *The Open Images Dataset V4*. IJCV, 2020.
- Mapillary, https://www.mapillary.com (zdjęcia CC BY-SA 4.0); OpenStreetMap contributors (ODbL).
- Unsplash Dataset, https://github.com/unsplash/datasets
- Wikimedia Commons, https://commons.wikimedia.org
- Radford A. i in. *Learning Transferable Visual Models From Natural Language Supervision*. ICML, 2021.
- Cherti M. i in. *Reproducible Scaling Laws for Contrastive Language-Image Learning* (OpenCLIP). CVPR, 2023.
