# ZoL0 Disaster Recovery (DR) Checklist

## 1. Backup & Retencja
- [ ] Skrypt `backup_to_s3.ps1` uruchamiany automatycznie (Task Scheduler, min. 1x/dzień)
- [ ] Backup bazy danych (`trading.db`, `bot_monitor.db`, `cache_analytics.db`, `compliance.db`) do S3 bucket `zol0-backups`
- [ ] Backup katalogu modeli AI (`saved_models/`) do S3 bucket `zol0-backups`
- [ ] Retencja backupów w S3: 90 dni (lifecycle policy lub skrypt)
- [ ] Logowanie backupów do pliku (`backup_log.txt`)

## 2. Testy odtwarzania (Recovery)
- [ ] Procedura przywracania backupu z S3 (test co najmniej raz na kwartał)
- [ ] Weryfikacja integralności backupu po przywróceniu (checksum, test startu systemu)
- [ ] Dokumentacja ścieżki przywracania krok po kroku

## 3. Automatyzacja & Monitoring
- [ ] Powiadomienia o nieudanym backupie (Slack/email)
- [ ] Alert, jeśli backup nie pojawił się w S3 przez >24h
- [ ] Monitoring miejsca w S3 i lokalnie

## 4. Dokumentacja
- [ ] Aktualny kontakt do osób DR (telefon, email)
- [ ] Lokalizacja i dostęp do kluczy AWS/S3 (secrets manager, nie w repo!)
- [ ] Instrukcja uruchomienia środowiska awaryjnego (np. nowy serwer, docker-compose, restore DB)

## 5. Scenariusze DR
- [ ] Utrata bazy danych – przywróć z najnowszego backupu
- [ ] Utrata modeli AI – przywróć katalog `saved_models/` z backupu
- [ ] Disaster totalny (serwer, storage) – odtwórz całość z S3 na nowej maszynie

---

> **Uwaga:** Regularnie testuj DR! Brak testu = brak backupu.
> 
> Skrypt backupu: `backup_to_s3.ps1` (PowerShell, Windows, AWS CLI wymagane)
