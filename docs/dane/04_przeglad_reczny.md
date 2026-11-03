# Przegląd ręczny: co odrzucać w poszczególnych widokach

**Co znaczy tag `reject`:** zdjęcie **wypada ze zbioru treningowego i testowego**, bo jego etykieta jest
nieprawdziwa, **niejednoznaczna** albo plik jest bezużyteczny. To *nie* jest decyzja „aplikacja ma to blokować”.
Przypadki niejednoznaczne usuwamy ze zbioru, żeby etykiety były spójne. W działającej aplikacji takie zdjęcia
mają trafiać do moderacji (strefa niepewności modelu).

Nie odrzucamy zdjęć dlatego, że są trudne, jeśli ich etykieta jest jasna. Trudne przypadki są w zbiorze celowo.

**Definicja pozytywu (doprecyzowana 2026-10-25):** autentyczne zdjęcie, jakie ludzie robią **na wycieczce w góry
lub do lasu**: krajobraz, szczyty, widoki, szlak, schronisko, ludzie w górach (także selfie i zdjęcia grupowe na tle gór).
**Widoczna ścieżka jest atutem, ale nie warunkiem.**

Pytanie pomocnicze: *czy to wygląda na zdjęcie zrobione podczas górskiej lub leśnej wycieczki?*

## A. Pozytywy (`nature`)

Widoki: **A1, A2, A3** oraz **C losowe – places365 / places365_ski / mapillary / unsplash / openimages / wikimedia**.

| Odrzuć ✗ | Zostaw ✓ (trudne pozytywy) |
|---|---|
| miasto, ulica, budynki w centrum, wnętrze | schronisko, chatka, wiata, mostek, kładka |
| plaża, morze, pustynia, step bez gór i lasu | ludzie na szlaku, turyści, plecaki |
| droga asfaltowa, autostrada, parking | drogowskaz, znaki szlaku, tablice informacyjne |
| tekst nałożony na zdjęcie, watermark, logo, ramka, podpis autora | mgła, noc, zmierzch, deszcz, pod słońce |
| obraz, rysunek, grafika 3D, kolaż, zrzut ekranu | błoto, kałuże, śnieg, lód, powalone drzewa |
| czarno-białe lub w sepii | lekko poruszone lub nieostre zdjęcie z telefonu |
| zdjęcie z drona lub z lotu ptaka | widok ze szczytu, panorama gór (zrobiona z ziemi) |
| zbliżenie zwierzęcia, kwiatka lub grzyba **bez żadnego tła górskiego/leśnego** (niejednoznaczne → do moderacji, nie do treningu) | las, łąka, potok, wodospad, jezioro górskie, **także bez ścieżki** |
| zwierzę w zoo albo na farmie | zwierzę w górach z widocznym otoczeniem (świstak na skałach, kozica) |
| zdjęcie studyjne, portret w mieście lub wnętrzu | **selfie i zdjęcia grupowe na tle gór lub lasu**, ludzie na szczycie |
| | schronisko, jego wnętrze z widokiem na góry, wiata, ławka na szlaku |
| | **stok narciarski, wyciąg, narciarze, ośrodek narciarski w górach** (decyzja 2026-10-25) |

**A2 (tekst lub watermark):** szukaj napisów, logo, znaków wodnych w rogu i podpisów fotografa. Drogowskaz albo tablica w terenie to *nie* tekst nałożony, więc zostają.

**A3 (mało szlaku):** tu jest dużo krajobrazów bez ścieżki (góry, las, łąka, jezioro). **Zostaw je**, bo to typowe zdjęcia z wycieczki. Odrzucaj tylko to, co pasuje do lewej kolumny tabeli.

## B. Negatywy `urban`

Widoki: **B1** oraz **C losowe – places365_urban / mapillary_city / coco**.

| Odrzuć ✗ | Zostaw ✓ (trudne negatywy) |
|---|---|
| zdjęcie, które wygląda jak szlak, las albo góry, bo to w praktyce pozytyw z błędną etykietą | park miejski, ogród, skwer, trawnik |
| | ulica w górskim mieście (np. Zakopane) |
| | plaża, wybrzeże, basen |

**B1** jest posortowany od zdjęć najbardziej „szlakowych”, więc odrzuceń szukaj głównie na początku listy.

## C. Memy (`meme`)

Widoki: **C losowe – memexgen / synth_meme / synth_screen**.

| Odrzuć ✗ | Zostaw ✓ |
|---|---|
| brak jakiegokolwiek tekstu, napisu czy elementów interfejsu (nieudany mem) | memy w każdym języku, także chińskie |
| tekst całkowicie nieczytelny lub ucięty tak, że nie widać, że to mem | memy wulgarne i obraźliwe (to celowy przypadek) |
| pusty lub uszkodzony obraz | zrzuty ekranu galerii, komunikatora, kanału społecznościowego |

## D. Zmanipulowane (`manipulated`)

Widoki: **C losowe – synth_manip / forgery / kaggle_inpaint**.

| Odrzuć ✗ | Zostaw ✓ |
|---|---|
| pusty, czarny, szary lub uszkodzony obraz (rozsypane piksele, pasy) | **wszystko inne, także gdy manipulacji nie widać** |

Nie odrzucaj zdjęcia dlatego, że nie widzisz zmiany. Dobra manipulacja ma być niewidoczna i to właśnie sprawdzamy. Groteskowe wklejenia (np. krzesło na szlaku) też zostają.

## E. Wygenerowane (`ai`)

Widoki: **C losowe – kaggle_ai / openfake**.

| Odrzuć ✗ | Zostaw ✓ |
|---|---|
| pusty, czarny lub uszkodzony obraz | realistyczne „zdjęcia”, także z widocznymi błędami AI (dziwne drzewa, schody donikąd) |
| ewidentny rysunek, kreskówka, anime, grafika wektorowa | ładne, „pocztówkowe” krajobrazy |
| (openfake) obraz, który zupełnie nie przypomina zdjęcia, np. plakat albo infografika | sceny katastrof naturalnych (powódź, osuwisko) |

## Szum etykiet do rozdziału 3

Widoki **C losowe** to próbki o stałej wielkości (200 dla zdjęć, 100 dla danych syntetycznych). Odsetek odrzuconych w danym widoku to **szacowany szum etykiet źródła**, a skrypt `stats.py` policzy go po przeglądzie. Dlatego w widokach C oceniaj **każde** zdjęcie, a nie tylko to, co rzuci się w oczy.

W razie wątpliwości **zostaw zdjęcie**. Odrzucenie powinno wynikać z jasnej reguły z tabel.
