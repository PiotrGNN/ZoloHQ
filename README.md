# ZoL0 Advanced Backtesting Engine

## Opis
Zaawansowany silnik backtestingu z AI, automatyzacją, optymalizacją strategii, integracją z dashboardem i API. Pozwala na testowanie strategii tradingowych na danych demo lub własnych, automatyczną optymalizację parametrów, generowanie raportów i rekomendacji oraz integrację z systemem produkcyjnym.

## Szybki start

1. Zainstaluj wymagane biblioteki:
   ```powershell
   pip install -r requirements.txt
   ```
2. Uruchom interfejs Streamlit:
   ```powershell
   streamlit run ui/app.py
   ```
3. (Opcjonalnie) Uruchom API:
   ```powershell
   python api_backtest.py
   ```

## Funkcje
- Backtest na danych demo lub własnych (CSV)
- AI-driven optymalizacja strategii i parametrów
- Automatyczne raporty i rekomendacje
- Eksport wyników do CSV
- Automatyzacja (scheduler, batch, multi-symbol)
- Powiadomienia e-mail
- Integracja z dashboardem ZoL0
- API REST
- Zaawansowane metryki (rolling Sharpe, max wins/losses)
- Hook do live tradingu

## Przykład użycia API
```powershell
Invoke-RestMethod -Uri http://localhost:8520/api/backtest -Method Post -Body '{"strategy":"Momentum","params":{"fast_period":10,"slow_period":30}}' -ContentType 'application/json'
```

## Dokumentacja kodu

- Każdy moduł zawiera docstring z opisem, autorem i datą.
- Kluczowe klasy i funkcje są opatrzone docstringami (patrz: engine/backtest_engine.py, ai/agent.py, ui/app.py).
- Przykładowe użycie API, eksportu, automatyzacji i powiadomień znajduje się w sekcji Funkcje i Przykład użycia API.
- Kod jest modularny i gotowy do dalszej rozbudowy.

## Bezpieczeństwo

- Przed wdrożeniem produkcyjnym:
  - Skonfiguruj ochronę API (autoryzacja, whitelisty, logowanie dostępu).
  - Zmień dane SMTP na własne i nie przechowuj haseł w kodzie.
  - Rozważ wdrożenie HTTPS dla API i dashboardu.

## Współpraca zespołowa

- Możesz komentować kod, dodawać adnotacje i korzystać z powiadomień e-mail.
- Integracja z systemem powiadomień i chatem możliwa przez dashboard ZoL0.
- Współdzielone raporty i eksporty można przesyłać przez e-mail lub system plików.

## FAQ

- Jeśli napotkasz problem z importami, uruchom `pip install -r requirements.txt`.
- Jeśli chcesz dodać własną strategię, utwórz nowy plik w katalogu `strategies/` i zaimplementuj klasę dziedziczącą po `BaseStrategy`.
- Jeśli chcesz zintegrować z brokerem, użyj hooka w UI lub rozbuduj `api_backtest.py`.

## CI/CD Workflow i Edge-Case Checklist

### Automatyzacja CI/CD

Repozytorium zawiera kompletny workflow GitHub Actions (`.github/workflows/ci-cd.yml`), który automatycznie uruchamia:
- Testy regresyjne (dashboard, API, core)
- Testy edge-case i jednostkowe (pytest, katalog `tests/`)
- Testy optymalizacji parametrów
- Linting i sprawdzanie typów (black, ruff, mypy)
- Automatyczny upload artefaktów optymalizacji

Workflow uruchamia się na każdym pushu i pull requeście do gałęzi `main` i `develop`.

### Edge-Case Checklist

- Testy na błędne dane wejściowe, brak pliku, złe uprawnienia, awarie sieci
- Testy na nietypowe parametry, błędy importu, permission errors
- Testy obsługi pamięci, wycieków, nietypowych stanów systemu
- Pokrycie testami wszystkich kluczowych modułów (patrz katalog `tests/`)

### Uruchamianie testów lokalnie

```bash
pip install -r requirements.txt pytest pytest-cov
pytest --cov=.
```

### Usuwanie TODO

Po wdrożeniu workflow CI/CD i testów edge-case, komentarze TODO dotyczące automatyzacji mogą zostać usunięte lub zaktualizowane.

## Kontakt
- Zespół ZoL0: [Twój e-mail lub kontakt]
# codex-zol0
# codex-zol0
