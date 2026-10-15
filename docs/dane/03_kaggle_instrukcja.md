# Generacja AI i inpaintingu na Kaggle: instrukcja

Lokalnie nie ma GPU, więc 1 500 obrazów AI i 600 manipulacji dyfuzyjnych liczymy na darmowym GPU Kaggle.
Wszystkie zadania (prompty, ziarna, zdjęcia źródłowe, maski) są ustalone z góry lokalnie w `jobs.csv`,
więc wynik jest powtarzalny.

## Jednorazowo

1. Załóż konto na <https://www.kaggle.com>.
2. **Settings → Phone verification:** zweryfikuj numer telefonu. Bez tego GPU i Internet w notebookach są niedostępne.
3. (Opcjonalnie, do pobrania CASIA v2) **Settings → API → Create New Token** i zapisz `kaggle.json` w `C:\Users\pawel\.kaggle\`.

## Wgranie danych wejściowych

1. Plik do wgrania: `data/kaggle/trail-gen-input.zip` (~263 MB), tworzony przez `python -m kaggle_prep` w `src/`.
2. Na Kaggle: **Datasets → New Dataset**, przeciągnij zip, nazwa `trail-gen-input`, widoczność **Private** → *Create*.

## Uruchomienie (2 sesje)

1. **Code → New Notebook → File → Import Notebook**, wybierz `kaggle/generate.ipynb` z repo.
2. Panel po prawej:
   - **Accelerator: GPU T4 x2**
   - **Internet: On**
   - **Add Input:** Twój dataset `trail-gen-input`
3. Kliknij **Run All**. Komórka z logami odświeża się co minutę.
4. Gdy w logu pojawi się `KONIEC`: zakładka **Output** (lub katalog `/kaggle/working`), pobierz pliki `sd15.zip`, `sd15inp.zip`, `sdxl.zip`.
5. Sesja 2: zaimportuj `data/kaggle/kernel_s2/generate.ipynb` (tworzy go `python kaggle/make_kernel.py 2`), a następnie pobierz `pixart.zip` i `sdxlinp.zip`.

| Sesja | GPU 0 | GPU 1 | Szac. czas |
|---|---|---|---|
| 1 | sd15 (625) → sd15inp (450) | sdxl (625) | ~4–5 h |
| 2 | pixart (250) | sdxlinp (150) | ~3–4 h |

Limit Kaggle to ~30 h GPU tygodniowo, a pojedyncza sesja trwa maksymalnie 12 h. Skrypt kończy pracę po 11 h i pakuje to, co zdążył zrobić.

**Jeśli sesja się przerwie:** pobierz częściowy zip, a w kolejnej sesji uruchom tylko brakujący generator (np. `PLAN = {0: ['sdxl']}`). Skrypt nie umie wznawiać pracy między sesjami, ale import lokalny pomija duplikaty, więc nic się nie zdubluje.

**Generator testowy: PixArt-Σ zamiast FLUX (decyzja 2026-10-14).** FLUX.1-schnell wymaga logowania na Hugging Face i akceptacji licencji. PixArt-Σ jest dostępny bez logowania (OpenRAIL++) i należy do innej rodziny modeli niż Stable Diffusion: transformer dyfuzyjny z enkoderem T5. Diffusers nie ma dla niego img2img, więc 81 zadań img2img z puli testowej wykonuje jako text2img z tym samym promptem.

**Automatyzacja przez API (zamiast klikania):** `python kaggle/make_kernel.py <sesja>` buduje notebook z wklejonym aktualnym `run_gen.py`, a `kaggle kernels push -p data/kaggle/kernel_s<sesja>` go uruchamia. Token jest w `~/.kaggle/access_token`. Dzięki temu dataset nie musi być wgrywany ponownie po zmianach w kodzie.

## Import wyników

1. Wrzuć pobrane `*.zip` do `data/kaggle/out/`.
2. W `src/`:
   ```
   python -m collect.kaggle_import
   python clip_filter.py
   python normalize.py
   python dedup.py
   python split.py
   python stats.py
   ```

## Co powstaje

| Generator | Model | Zadanie | Rola |
|---|---|---|---|
| sd15 | stable-diffusion-v1-5 | text2img + img2img | train/val (500) + test (125) |
| sdxl | stable-diffusion-xl-base-1.0 | text2img + img2img | train/val (500) + test (125) |
| pixart | PixArt-Σ XL 1024 | text2img | test_ood (250) |
| sd15inp | stable-diffusion-inpainting (SD 1.5) | inpainting | train/val (300) + test (150) |
| sdxlinp | stable-diffusion-xl-1.0-inpainting | inpainting | test_ood (150) |

img2img i inpainting startują z naszych pozytywów z tego samego splitu, więc dziedziczą ich `group`.
