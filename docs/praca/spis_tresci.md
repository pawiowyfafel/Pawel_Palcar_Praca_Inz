# Spis treści pracy inżynierskiej z opisem sekcji

**Roboczy tytuł:** *Weryfikacja autentyczności zdjęć zgłoszeń szlakowych na urządzeniu mobilnym z wykorzystaniem konwolucyjnych sieci neuronowych*
(wariant krótszy: *Mobilna weryfikacja zdjęć w crowdsourcingowej aplikacji szlakowej*)

**Docelowa objętość:** ~66–70 stron bez załączników. Strony przy rozdziałach są orientacyjne.
Szablon i wymogi formalne (marginesy, streszczenia, oświadczenia) sprawdź w zarządzeniu wydziału i u promotorki.

| Część | Strony |
|---|---|
| Streszczenie PL + Abstract EN | 2 |
| Wstęp | 3 |
| 1. Przegląd zagadnień | 12 |
| 2. Koncepcja systemu i metodologia | 5 |
| 3. Zbiór danych | 15 |
| 4. Modele i procedura treningu | 7 |
| 5. Środowisko eksperymentalne | 3 |
| 6. Wyniki | 14 |
| 7. Dyskusja | 5 |
| 8. Podsumowanie i kierunki rozwoju | 2 |
| **Razem** | **~68** |
| Bibliografia, spisy rysunków i tabel, załączniki | poza limitem |

---

## Streszczenie / Abstract (2 str.)

Po pół strony PL i EN plus słowa kluczowe. Piszesz na końcu. Struktura: problem (fałszywe i niepożądane zgłoszenia w crowdsourcingu) → co zrobiono (zbiór ~36 tys. zdjęć, 4 typy negatywów, 3–4 architektury mobilne, kwantyzacja, pomiar na telefonach) → 2–3 najważniejsze liczby z wyników → główny wniosek (np. „filtr semantyczny działa, detekcja AI tylko jako sygnał do moderacji”).

## Wstęp (3 str.)

- **Motywacja:** aplikacje z danymi od użytkowników (warunki na szlakach) są wrażliwe na złe zgłoszenia: przypadkowe zdjęcia, memy, zdjęcia nie z gór, podróbki AI, manipulacje „fałszywy raport warunków”, treści wulgarne. Moderacja ręczna nie skaluje się. Weryfikacja na urządzeniu działa offline i przed wysłaniem.
- **Cel pracy:** zbadać, czy lekki model na telefonie może pełnić rolę **asystenta moderatora**: automatycznie przepuszczać autentyczne zdjęcia z górskich i leśnych wycieczek, automatycznie blokować oczywiste naruszenia, a **wszystko niepewne flagować do sprawdzenia przez administrację**. Pytanie brzmi też: jakim kosztem (rozmiar, czas, obciążenie moderacji, błędy automatycznych decyzji).
- **Definicja pozytywu:** autentyczne zdjęcie z wycieczki w góry lub do lasu (krajobraz, szczyty, schronisko, ludzie w górach, szlak). **Widoczna ścieżka nie jest warunkiem.**
- **Zakres:** budowa zbioru danych, porównanie architektur mobilnych, kwantyzacja, trójstopniowa decyzja z dwoma progami, pomiary na urządzeniach, aplikacja demonstracyjna.
- **Pytania badawcze:**
  - PB1: Jaki odsetek zgłoszeń model może rozstrzygnąć automatycznie (przepuść lub zablokuj) przy bardzo małej liczbie błędnych decyzji, czyli jakie jest obciążenie moderacji?
  - PB2: Które typy naruszeń model blokuje pewnie (miasto, memy), a które jedynie flaguje (AI, manipulacje)? Jak to wygląda na danych niewidzianych w treningu?
  - PB3: Jaki jest koszt kwantyzacji (dokładność vs rozmiar i czas)?
  - PB4: Ile przykładów AI w treningu potrzeba i czy pogarszają akceptację prawdziwych zdjęć?
- **Struktura pracy:** po jednym zdaniu na rozdział.

---

## 1. Przegląd zagadnień (12 str.)

Literaturę trzymaj zwięźle i podporządkuj ją temu, co później badasz. Każdy podrozdział kończ zdaniem „co z tego wynika dla tej pracy”.

### 1.1 Crowdsourcing danych i jakość zgłoszeń (1,5)
Przykłady aplikacji z danymi od użytkowników (mapy, warunki, raporty). Typy złych zgłoszeń. Moderacja ręczna, automatyczna i hybrydowa.

### 1.2 Klasyfikacja obrazów sieciami konwolucyjnymi (2)
Splot, pooling, głębokie sieci, transfer learning z ImageNet, fine-tuning, funkcja straty BCE, przeuczenie. Krótko, bez wyprowadzeń.

### 1.3 Architektury mobilne i inferencja na urządzeniu (2,5)
MobileNetV3 (bloki inverted residual, squeeze-excitation, h-swish), EfficientNet i EfficientNet-Lite (compound scaling, czemu „Lite” nie ma SE/swish). Kwantyzacja: FP16, INT8 PTQ, zbiór reprezentatywny. LiteRT/TFLite, delegaty (XNNPACK, GPU, NNAPI).
**Tabela:** porównanie architektur z literatury (parametry, MAC, top-1 ImageNet).

### 1.4 Obrazy generowane przez AI i ich wykrywanie (2,5)
GAN vs modele dyfuzyjne (SD, SDXL, transformery dyfuzyjne: PixArt, FLUX). Artefakty generacji (częstotliwościowe, VAE). Detektory i **problem uogólniania**: skuteczność spada na nowych generatorach, np. VCT² (~58% zero-shot), Community Forensics (więcej generatorów daje lepsze uogólnianie). To uzasadnia potraktowanie AI jako scenariusza ataku.

### 1.5 Manipulacje obrazu i ich wykrywanie (1,5)
Splicing, copy-move, inpainting klasyczny (LaMa) i dyfuzyjny. Zbiory forensyczne (CASIA, DEFACTO, IMD2020, CocoGlide). Detektory lokalizujące (np. TruFor) są zwykle za ciężkie na telefon.

### 1.6 Moderacja treści nieodpowiednich (1)
Detektory NSFW (np. NSFWJS, NudeNet), kaskady moderacji, fałszywe alarmy.

### 1.7 Uczenie na skrótach i stronniczość zbiorów (1)
Shortcut learning: model uczy się formatu, rozdzielczości albo źródła zamiast treści. Przykład znany z literatury: w GenImage obrazy prawdziwe są w JPEG, a generowane w PNG. To uzasadnia normalizację z rozdziału 3.

### 1.8 Podsumowanie: luka badawcza (0,5)
Brakuje badań łączących filtr semantyczny i forensykę na telefonie w konkretnym zastosowaniu. Do tego zbiory forensyczne zwykle mają inną treść niż zdjęcia z gór.

---

## 2. Koncepcja systemu i metodologia (5 str.)

### 2.1 Scenariusz użycia: model jako asystent moderatora (1)
Aplikacja offline-first. Użytkownik dodaje zdjęcie z wycieczki, model na telefonie ocenia je przed wysłaniem i wynik trafia do jednej z trzech ścieżek:
- **przepuść:** zdjęcie publikowane bez udziału człowieka;
- **flaguj:** zdjęcie trafia do kolejki administracji z informacją o powodzie (np. „możliwy obraz AI”);
- **zablokuj:** oczywiste naruszenie (mem, zrzut ekranu, zdjęcie z miasta lub wnętrza; NSFW z etapu 1). Użytkownik może się odwołać, więc decyzja nie jest ostateczna.

Model **nie zastępuje** moderacji, tylko ją odciąża. Miarą sukcesu jest odsetek zgłoszeń rozstrzygniętych automatycznie przy znikomej liczbie błędów, a nie sama dokładność.
**Rysunek:** diagram przepływu zgłoszenia z trzema ścieżkami.

### 2.2 Architektura weryfikacji: kaskada (1)
Etap 1: gotowy mobilny filtr NSFW (wymaganie krytyczne; pewne trafienie blokuje, niepewne flaguje). Etap 2: model z tej pracy. Etap 3: reguła decyzyjna z dwoma progami (2.4), opcjonalnie zależna od podklasy.
**Rysunek:** schemat kaskady.

### 2.3 Definicja problemu, klas i polityka etykiet (1,5)
- **Pozytyw:** autentyczne zdjęcie z wycieczki w góry lub do lasu: krajobraz, szczyty, widoki, schronisko, ludzie w górach (także selfie i zdjęcia grupowe), szlak, drogowskazy. **Widoczna ścieżka jest atutem, ale nie warunkiem.**
- **Negatywy (podklasy):** urban (miasto, wnętrza, inne sceny), meme (memy, nałożony tekst, zrzuty ekranu), manipulated (zmieniona treść), ai (wygenerowane).
- **Dwa rodzaje naruszeń:** *semantyczne*, czyli zdjęcie nie z wycieczki (łatwe, oczywiste, można blokować), oraz *forensyczne*, czyli zdjęcie nieprawdziwe (trudne, raczej do flagowania).
- **Decyzje definicyjne:** **stok i ośrodek narciarski = pozytyw** (raporty warunków na stokach), park miejski = negatyw, zrzut ekranu = negatyw (meme), telefonowa „magiczna gumka” = manipulacja, filtry i HDR = autentyczne.
- **Przypadki niejednoznaczne** (np. makro kwiatka bez tła, dzikie zwierzę) usuwamy ze zbioru, a w aplikacji powinny trafiać do strefy „flaguj”.

**Źródło:** `docs/dane/02_klasa_negatywna_plan.md` (pkt 8), `docs/dane/04_przeglad_reczny.md`.

### 2.4 Wymagania i reguła decyzyjna (1)
Wyjście modelu: prawdopodobieństwo naruszenia *p* (lub rozkład na 5 klas, patrz 4.1). Dwa progi:
- *p* < t_przepuść → przepuść;
- *p* > t_blokuj → zablokuj;
- w przeciwnym razie → flaguj.

Wymagania (do ustalenia z promotorką, przykładowe wartości):
- **błędna blokada autentycznego zdjęcia ≤ 0,5–1%,** bo to najgorszy błąd: frustracja użytkownika;
- **automatyczne przepuszczenie naruszenia ≤ 2–5%** (dla naruszeń oczywistych);
- **obciążenie moderacji** (odsetek flagowanych) jak najmniejsze, to optymalizowana wielkość;
- rozmiar modelu < 10 MB, czas < 100 ms na telefonie ze średniej półki.

### 2.5 Plan eksperymentów i metryki (1)
Mapowanie PB1–PB4 na eksperymenty. Metryki:
- **klasyczne:** ROC-AUC, PR-AUC, F1;
- **specyficzne dla asystenta moderatora:** odsetek decyzji automatycznych (pokrycie), obciążenie moderacji, odsetek błędnych blokad autentycznych, odsetek przepuszczonych naruszeń, każda per podklasa;
- **zasoby:** czas (mediana, p90), rozmiar pliku.

**Tabela:** pytanie → eksperyment → zbiór testowy → metryka.

---

## 3. Zbiór danych (15 str.), najważniejszy rozdział

Pisz go teraz, póki wszystko jest świeże. Materiały: `docs/dane/01_klasa_pozytywna.md`, `02_klasa_negatywna_plan.md`, `03_kaggle_instrukcja.md`, `04_przeglad_reczny.md` i tabele z `results/dataset/`.

### 3.1 Założenia projektowe zbioru (1)
Definicja pozytywu z 2.3 (zdjęcie z wycieczki, ścieżka nie jest wymagana) i jej przełożenie na dobór źródeł: kategorie Places365 obejmują góry, doliny, lasy i schroniska, a nie tylko ścieżki. Trzy zasady: (1) dopasowanie treści, czyli negatywy AI i manipulacje przedstawiają szlaki; (2) kontrola skrótów (format, rozdzielczość, metadane); (3) brak przecieku między zbiorami (grupy, przeróbki dziedziczą grupę po oryginale). Ograniczenia: łącze ~1 MB/s, brak lokalnego GPU, co dało techniki pobierania częściowego i GPU w chmurze.

### 3.2 Klasa pozytywna: źródła i pozyskanie (2,5)
Po akapicie na źródło: Places365 (wybór 25 kategorii, odczyt archiwum zapytaniami HTTP Range, czyli ~0,5 GB zamiast 26 GB), Mapillary (filtr geograficzny z OSM: ≤20 m od ścieżki i ≥60 m od drogi, limit 6 klatek z sekwencji, 12 regionów PL/SK/CZ/SI), Unsplash Lite (słowa kluczowe, estetyczny bias), Open Images (etykiety zweryfikowane przez ludzi, losowanie ważone, Flickr zamiast niedostępnego S3), Wikimedia (kategorie, całe źródło do testu OOD), własne zdjęcia (T3).
**Tabela:** źródło, licencja, liczba surowych zdjęć, rola.
**Rysunek:** mapa regionów Mapillary albo przykład filtru OSM (punkty zdjęć na tle ścieżek i dróg).
**Rysunek:** siatka przykładowych pozytywów z każdego źródła.

### 3.3 Klasa negatywna (4)
- **3.3.1 Miasto i wnętrza:** Places365 (63 kategorie, trudne negatywy), Mapillary w miastach (filtr odwrotny, Lublana do OOD), COCO.
- **3.3.2 Memy i zrzuty ekranu:** synteza (style impact/bar/labels/emoji/vulgar, szablony Imgflip), zrzuty ekranu (galeria, kanał społecznościowy, komunikator), MemeXGen (tylko oryginały, bo wersje FLUX odrzucono jako AI; chińskie do OOD).
- **3.3.3 Manipulacje:** własne na CPU (LaMa, splicing z masek COCO, copy-move, próg widoczności 0,5% kadru), publiczne (DEFACTO; CocoGlide i IMD2020 do OOD), inpainting dyfuzyjny na Kaggle.
- **3.3.4 Obrazy AI:** ujęcie jako scenariusz ataku. Generacja na Kaggle (SD 1.5, SDXL, text2img i img2img z naszych zdjęć, ~200 kombinacji promptów), generator OOD PixArt-Σ (z uzasadnieniem rezygnacji z FLUX), OpenFake (20 generatorów, selekcja promptów przyrodniczych, bez Illustrious).
**Tabela:** podklasa, źródło, liczność, rola.
**Rysunek:** przykłady z każdej podklasy, a dla manipulacji pary oryginał/przeróbka.

### 3.4 Treści nieodpowiednie (NSFW) (1)
Wymaganie aplikacji, ale **świadomie poza zbiorem**: ryzyko prawne (materiały z udziałem nieletnich lub publikowane bez zgody w zbiorach z sieci), RODO, wymóg zgody etycznej, a do tego takie zdjęcia i tak nie przedstawiają szlaku. Rozwiązanie: kaskada z gotowym filtrem NSFW (rozdz. 2.2) i pomiar jego fałszywych alarmów na pozytywach (rozdz. 6.10). Wulgarny tekst na zdjęciach jest pokryty podklasą meme. Wspomnij też, że wbudowany filtr NSFW w SD 1.5 dawał fałszywe alarmy na krajobrazach, co jest argumentem za pomiarem fałszywych alarmów.

### 3.5 Filtrowanie zero-shot modelem CLIP (1,5)
Model OpenCLIP ViT-B/32, grupy promptów trail/nature/other, tagi trudnych pozytywów (porównanie z neutralnym opisem, bo CLIP źle rozumie przeczenia), wykrywanie tekstu. **Iteracyjne strojenie:** prompt „a screenshot” zbierał trudne pozytywy, więc dodano prompty dla drogowskazów i schronisk; dodano prompt asfaltu. Próg 0,30 wybrano na podstawie oglądu strefy granicznej. Dla negatywów urban odrzucono te, które CLIP ocenia jako szlak. **Lekcja metodyczna:** znacznik tekstu ze stałym punktem odniesienia („hiking trail”) błędnie odrzucił ~5 tys. zdjęć gór, schronisk i stoków. Po kalibracji CLIP okazał się słabym detektorem tekstu, co uzasadnia przegląd ręczny (dobry materiał do dyskusji o ograniczeniach filtrowania zero-shot).
**Tabela:** odsetek przechodzących per źródło.
**Rysunek:** przykłady odrzuconych i z pogranicza.

### 3.6 Normalizacja i kontrola skrótów (1)
Usunięcie EXIF, tylko zmniejszanie (krótszy bok 384–768), JPEG z jakością 70–95, identycznie dla obu klas. Stabilny wybór kwot (hash id). **Weryfikacja:** średnia jakość JPEG 82,6 w obu klasach, mediana krótszego boku 432 vs 458 px.
**Tabela:** `results/dataset/jpeg_q.csv`, `resolution.csv`.

### 3.7 Deduplikacja, grupy i podział (1,5)
pHash (≤4 to duplikat, ≤10 to klaster), podobieństwo CLIP ≥0,95. Przeróbki nie są usuwane, a syntetyczne memy z jednego szablonu to warianty. Komponenty (grupa + klaster) trafiają w całości do jednego splitu, a klucz to najmniejsza nazwa grupy. Splity: train/val/test 70/15/15, test_ood (Wikimedia, Bieszczady, Lublana, generatory i zbiory niewidziane), test_real (własne).
**Rysunek:** schemat potoku `collect → CLIP → normalize → review → dedup → split`.

### 3.8 Przegląd ręczny i szum etykiet (0,5)
Widoki przeglądu, reguły (tabela skrócona z `04_przeglad_reczny.md`), szacowany szum etykiet per źródło.
**Tabela:** `results/dataset/label_noise.csv`.

### 3.9 Końcowa baza danych (1,5)
**Tabela:** podklasa × split (`subclass_split.csv`).
**Tabela:** lejek selekcji (`funnel.csv`).
**Wykresy:** udział trudnych pozytywów (tagi), rozkład rozdzielczości, sezonowość, generatory w podklasie AI.
Opis zbiorów testowych T1 (in-distribution), T2 (OOD), T3 (rzeczywiste) i testu odporności.

### 3.10 Ograniczenia zbioru (0,5)
Stronniczość selekcji CLIP, 256 px w Places365, estetyka Unsplash, syntetyczny charakter części negatywów, PixArt zamiast FLUX, mała liczba obrazów OpenFake, brak obrazów NSFW.

---

## 4. Modele i procedura treningu (7 str.)

### 4.1 Wybór architektur i głowicy wyjściowej (1,5)
Drabinka modeli z uzasadnieniem: własny mały CNN (dolna granica: czy wystarczy coś trywialnego?), MobileNetV3-Small/Large (standard mobilny), EfficientNet-Lite0 (kompromis, przyjazny dla INT8), opcjonalnie model serwerowy EfficientNet-B3 lub ConvNeXt-Tiny (górna granica: ile tracimy na mobilności).

**Głowica:** binarna (autentyczne/naruszenie) vs **5-klasowa** (nature, urban, meme, manipulated, ai). Wersja 5-klasowa podaje moderatorowi **powód flagi** i pozwala na regułę zależną od podklasy: oczywiste naruszenia (urban, meme) można blokować, a AI i manipulacje co najwyżej flagować. Porównanie obu głowic to osobny eksperyment.
**Tabela:** parametry, MAC, rozdzielczość wejścia.

### 4.2 Przetwarzanie wejścia i augmentacje (1)
Rozdzielczość 224 (własny CNN 128–160). Augmentacje ostrożne: crop, flip, lekki color jitter, losowa kompresja JPEG i losowe zmniejszanie rozdzielczości (przeciw skrótowi rozdzielczości). **Bez mocnego blur i szumu**, bo niszczą ślady AI i manipulacji.

### 4.3 Procedura treningu (1,5)
Transfer learning z ImageNet w 2 etapach (zamrożony backbone, potem fine-tuning), AdamW, harmonogram cosine, BCE z wagami klas (19 tys. vs 17 tys.), early stopping po val ROC-AUC, 3 ziarna, wynik jako średnia ± odchylenie standardowe.

### 4.4 Eksperyment: krzywa uczenia dla obrazów AI (1)
Trening z 0 / 250 / 500 / 750 / 1 000 obrazami AI, przy stałej reszcie zbioru i stałym teście. Pełna krzywa dla EfficientNet-Lite0, a dla pozostałych modeli tylko 0 vs 1 000. Budżet: ~27 treningów.

### 4.5 Kwantyzacja i eksport (1)
FP32 → FP16 → INT8 (post-training, zbiór reprezentatywny ~200 obrazów), eksport do LiteRT, weryfikacja zgodności wyników przed i po konwersji.

### 4.6 Dobór progów decyzyjnych (1)
Progi wyznaczane na zbiorze walidacyjnym, nigdy na testowym:
- **t_blokuj:** najniższy próg, przy którym błędne blokady autentycznych zdjęć ≤ założonego limitu (np. 0,5%);
- **t_przepuść:** najwyższy próg, przy którym przepuszczone naruszenia oczywiste ≤ limitu (np. 2%).

Między progami jest strefa „flaguj”. W wariancie 5-klasowym blokować mogą tylko podklasy urban i meme, a AI i manipulacje kierowane są najwyżej do flagowania. **Wykres:** obciążenie moderacji w funkcji dopuszczalnego błędu (krzywa „pokrycie a ryzyko”).

---

## 5. Środowisko eksperymentalne (3 str.)

### 5.1 Trening i reprodukowalność (1)
Kaggle (GPU T4), PyTorch + timm, wersje bibliotek. Konfiguracja eksperymentu = config + ziarno + hash manifestu. Repozytorium i skrypty (załącznik E).

### 5.2 Urządzenia i metodyka pomiaru czasu (1)
2–3 telefony z różnych półek, `benchmark_model` (CPU 1 i 4 wątki, delegat GPU), rozgrzewka, 100 przebiegów, mediana i p90.
**Tabela:** specyfikacja urządzeń.

### 5.3 Aplikacja demonstracyjna (1)
Kotlin + LiteRT, kaskada NSFW → model, pomiar end-to-end (dekodowanie, skalowanie, inferencja, wczytanie modelu).
**Rysunek:** zrzut ekranu aplikacji.

---

## 6. Wyniki (14 str.)

Każdy podrozdział: tabela lub wykres, 2–3 zdania obserwacji, odniesienie do pytania badawczego.

| Podrozdział | Strony | Co pokazać |
|---|---|---|
| 6.1 Rozmiar i złożoność | 1 | rozmiar pliku per model × precyzja, parametry, MAC |
| 6.2 Czas inferencji | 1,5 | urządzenia × delegat × precyzja (mediana, p90), czas end-to-end w aplikacji |
| 6.3 Skuteczność na T1 | 1,5 | ROC-AUC, PR-AUC, F1 (średnia ± sd z 3 ziaren), macierz pomyłek, test McNemara lub bootstrapowe 95% CI; głowica binarna vs 5-klasowa |
| 6.4 **Decyzja trójstopniowa: asystent moderatora (PB1, PB2)** | 2 | dla każdej podklasy odsetek przepuszczonych / flagowanych / zablokowanych (wykres skumulowany), błędne blokady autentycznych, przepuszczone naruszenia, łączne obciążenie moderacji; krzywa pokrycie a ryzyko |
| 6.5 Uogólnianie (T2) | 2 | Wikimedia, Bieszczady, Lublana, memy ZH, CocoGlide, IMD2020, PixArt-Σ, OpenFake (per generator) |
| 6.6 Warunki rzeczywiste (T3) | 1 | własne zdjęcia z różnych telefonów, przypadki podchwytliwe |
| 6.7 Krzywa uczenia AI (PB4) | 1,5 | wykrywalność AI (generator znany i nieznany) oraz FRR prawdziwych zdjęć w funkcji 0–1 000 przykładów |
| 6.8 Wpływ kwantyzacji (PB3) | 1 | spadek metryk FP32 → FP16 → INT8 vs zysk rozmiaru i czasu |
| 6.9 Odporność na degradację | 1 | spadek AUC przy JPEG q = 90/70/50, skali 0,5/0,25, rekompresji komunikatora |
| 6.10 Filtr NSFW w kaskadzie | 0,5 | fałszywe alarmy na pozytywach, dodatkowy czas i rozmiar |
| 6.11 Pareto, interpretacja i błędy | 1 | wykres dokładność vs czas (wielkość punktu = rozmiar), Grad-CAM, galeria najczęstszych błędów |

---

## 7. Dyskusja (5 str.)

- **7.1 Odpowiedzi na pytania badawcze (1):** PB1–PB4, każda w 3–5 zdaniach z liczbami.
- **7.2 Rekomendowana konfiguracja (1):** model, precyzja i progi, rola kaskady. Ile pracy moderacji oszczędza system (np. „X% zgłoszeń rozstrzyganych automatycznie przy Y% błędów”). Prawdopodobny wniosek: naruszenia semantyczne blokowane automatycznie, AI i manipulacje kierowane do moderacji z podanym powodem.
- **7.3 Semantyka vs forensyka (1):** co jest łatwe (urban, memy), co trudne (AI z nowych generatorów, subtelne inpaintingi) i dlaczego (rozdzielczość 224, kompresja).
- **7.4 Zagrożenia ważności (1):** syntetyczne negatywy vs prawdziwe ataki, stronniczość CLIP, PixArt zamiast FLUX, mały OpenFake, przegląd ręczny jednej osoby, jedna rozdzielczość.
- **7.5 Bezpieczeństwo i obejścia (0,5):** jak atakujący może oszukać model (rekompresja, kadrowanie, zrzut ekranu zdjęcia AI), potrzeba aktualizacji modelu i moderacji.
- **7.6 Aspekty prawne i etyczne (0,5):** licencje (NC, CC BY-SA), RODO dla zdjęć własnych, decyzja o NSFW.

## 8. Podsumowanie i kierunki rozwoju (2 str.)

Co osiągnięto (zbiór, porównanie, wnioski). Kierunki rozwoju: większa rozdzielczość lub analiza fragmentów (patch-based) dla forensyki, okresowy retrening na nowych generatorach, uczenie z danych z moderacji, model wielozadaniowy (podklasa zamiast binarnej decyzji), detekcja na serwerze jako drugi etap.

---

## Bibliografia

Pełna lista źródeł, zbiorów, modeli i narzędzi z linkami: [bibliografia.md](bibliografia.md).

Ok. 40–60 pozycji. Obowiązkowo: Places365, Open Images, COCO, CLIP/OpenCLIP, MobileNetV3, EfficientNet, LiteRT/TFLite i kwantyzacja, LaMa, Stable Diffusion/SDXL, PixArt-Σ, DEFACTO, IMD2020, CocoGlide/TruFor, Community Forensics, OpenFake, VCT², GenImage (bias formatu), shortcut learning (Geirhos i in.), Grad-CAM.

## Załączniki (poza limitem stron)

- **A.** Karta zbioru danych (datasheet): motywacja, skład, proces zbierania, preprocessing, zastosowania, licencje.
- **B.** Tabela źródeł i licencji (na podstawie manifestu).
- **C.** Parametry potoku: progi CLIP, filtr OSM, progi pHash, normalizacja, prompty.
- **D.** Pełne tabele wyników (wszystkie ziarna i konfiguracje).
- **E.** Instrukcja odtworzenia: repozytorium, kolejność skryptów, notebooki Kaggle.

---

## Kolejność pisania (sugestia)

1. **Rozdział 3** teraz, równolegle z przeglądem ręcznym.
2. Rozdział 2 i szkielet rozdziału 4, zanim zaczniesz trening.
3. Rozdziały 5 i 6 w trakcie eksperymentów, bezpośrednio z plików CSV z wynikami.
4. Rozdział 1 (literatura) w przerwach na trening.
5. Rozdziały 7, 8, wstęp i streszczenia na końcu.
