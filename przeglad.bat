@echo off
rem Przegląd ręczny zbioru w FiftyOne (http://localhost:5151). Dwuklik albo uruchomienie z dowolnego folderu.
rem Po zakończeniu: zamknij kartę przeglądarki albo Ctrl+C - lista odrzuconych zapisze się sama.
cd /d "%~dp0"
".venv\Scripts\python.exe" "src\review.py" %*
pause
