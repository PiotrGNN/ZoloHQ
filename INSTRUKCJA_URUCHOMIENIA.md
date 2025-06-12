# Instrukcja uruchomienia i zatrzymania bota

## Uruchomienie bota (Windows, staging lub produkcja)

1. Upewnij się, że plik `.env` zawiera odpowiednie klucze API i ustawienia środowiska (staging/produkcyjne).
2. W terminalu PowerShell przejdź do katalogu projektu:

```
cd C:\Users\piotr\Desktop\Zol0
```

3. Uruchom bota:

```
python ZoL0-master/main.py --env staging
```

lub dla produkcji:

```
python ZoL0-master/main.py --env production
```

4. Monitoruj logi w katalogu `logs/` oraz powiadomienia na Telegramie.

## Automatyczny restart po awarii (PowerShell)

Utwórz plik `run_bot.ps1` z zawartością:

```
while ($true) {
    Write-Host "[INFO] Uruchamiam bota..."
    python ZoL0-master/main.py --env staging
    Write-Host "[WARN] Bot zakończył działanie. Restart za 10 sekund..."
    Start-Sleep -Seconds 10
}
```

## Zatrzymanie bota

- Jeśli uruchomiono w zwykłym terminalu: naciśnij `Ctrl+C`.
- Jeśli uruchomiono przez skrypt: zamknij okno PowerShell lub przerwij proces.

---

# Najczęstsze problemy i rozwiązania

- **Brak połączenia z giełdą:** sprawdź klucze API, połączenie internetowe, limity API.
- **Brak alertów na Telegramie:** sprawdź konfigurację w `.env` (`TELEGRAM_TOKEN`, `TELEGRAM_CHAT_ID`).
- **Błąd importu lub zależności:** uruchom `pip install -r requirements.txt` lub sprawdź komunikaty błędów.
- **Bot nie wykonuje transakcji:** sprawdź tryb demo/produkcyjny, saldo konta, uprawnienia API.

# Plan awaryjny

1. W przypadku awarii bota: sprawdź logi w katalogu `logs/`.
2. Jeśli problem krytyczny – zatrzymaj bota, napraw błąd, uruchom ponownie.
3. W razie awarii giełdy – wyłącz bota do czasu przywrócenia usług.
4. Regularnie wykonuj backupy plików konfiguracyjnych i bazy danych.

---

W razie poważnych problemów – skontaktuj się z administratorem lub sprawdź dokumentację projektu.
