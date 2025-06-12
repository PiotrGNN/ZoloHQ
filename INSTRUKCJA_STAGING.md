# Instrukcja uruchomienia bota na środowisku staging

1. Skopiuj plik .env i ustaw klucze API do trybu testowego (sandbox/demo) giełdy.
2. Upewnij się, że w konfiguracji bota wybrany jest tryb testowy (np. TESTNET=True lub odpowiedni endpoint API).
3. Uruchom bota poleceniem:

```
python ZoL0-master/main.py --env staging
```

4. Monitoruj logi w katalogu `logs/` oraz sprawdzaj powiadomienia na Telegramie.
5. Przeprowadź testowe transakcje i sprawdź, czy alerty oraz monitoring działają poprawnie.
6. Po minimum 24h testów i pozytywnej weryfikacji działania, przejdź do wdrożenia na produkcję.

---

# Przykładowy skrypt do automatycznego uruchamiania i restartu bota (Windows)

Utwórz plik `run_bot.ps1` z zawartością:

```
while ($true) {
    Write-Host "[INFO] Uruchamiam bota..."
    python ZoL0-master/main.py --env staging
    Write-Host "[WARN] Bot zakończył działanie. Restart za 10 sekund..."
    Start-Sleep -Seconds 10
}
```

---

# Dodatkowe zalecenia
- Po testach stagingowych zmień klucze API na produkcyjne i ustaw tryb produkcyjny.
- Zabezpiecz plik .env i ogranicz dostęp do serwera.
- Upewnij się, że backupy i monitoring są aktywne.

W razie pytań lub problemów – sprawdź checklistę lub skontaktuj się z administratorem systemu.
