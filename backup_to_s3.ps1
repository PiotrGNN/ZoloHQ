#requires -Version 5.1
<##
.SYNOPSIS
    Profesjonalny skrypt PowerShell do backupu bazy danych i modeli AI do AWS S3 z retencją 90 dni.
.DESCRIPTION
    - Tworzy backup bazy danych (np. trading.db, bot_monitor.db, cache_analytics.db, compliance.db)
    - Tworzy backup katalogu modeli AI (np. saved_models/)
    - Kompresuje backupy do archiwum ZIP z datą
    - Wysyła backup do wskazanego bucketu S3 (np. zol0-backups)
    - Usuwa backupy starsze niż 90 dni z S3 (retencja)
    - Loguje przebieg operacji do pliku
    - Wspiera Disaster Recovery (DR)
.NOTES
    Wymaga AWS CLI oraz uprawnień do S3. Zalecane uruchamianie z harmonogramu zadań (Task Scheduler).
#>

param(
    [string]$S3Bucket = "zol0-backups",
    [string]$BackupRoot = "$PSScriptRoot\backups",
    [string]$DbFiles = "trading.db,bot_monitor.db,cache_analytics.db,compliance.db",
    [string]$AiModelsDir = "saved_models",
    [int]$RetentionDays = 90,
    [string]$LogFile = "$PSScriptRoot\backup_log.txt"
)

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "$timestamp $Message" | Out-File -FilePath $LogFile -Append -Encoding utf8
}

Write-Log "=== [START] Backup run ==="

# 1. Przygotuj katalog backupu
$now = Get-Date -Format 'yyyyMMdd_HHmmss'
$backupDir = Join-Path $BackupRoot "backup_$now"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

# 2. Kopiuj pliki baz danych
$DbFiles.Split(',') | ForEach-Object {
    $src = Join-Path $PSScriptRoot $_
    if (Test-Path $src) {
        Copy-Item $src $backupDir -Force
        Write-Log "Skopiowano $_ do backupu."
    } else {
        Write-Log "[WARN] Plik $_ nie istnieje."
    }
}

# 3. Kopiuj modele AI
$modelsSrc = Join-Path $PSScriptRoot $AiModelsDir
if (Test-Path $modelsSrc) {
    $modelsDest = Join-Path $backupDir "models"
    Copy-Item $modelsSrc $modelsDest -Recurse -Force
    Write-Log "Skopiowano modele AI ($AiModelsDir) do backupu."
} else {
    Write-Log "[WARN] Katalog modeli $AiModelsDir nie istnieje."
}

# 4. Kompresuj backup
$zipPath = "$backupDir.zip"
Compress-Archive -Path $backupDir\* -DestinationPath $zipPath -Force
Write-Log "Backup skompresowany do $zipPath."

# 5. Wyślij do S3
try {
    aws s3 cp $zipPath "s3://$S3Bucket/" | Out-Null
    Write-Log "Backup wysłany do S3: $S3Bucket."
} catch {
    Write-Log "[ERROR] Błąd wysyłania do S3: $_"
}

# 6. Retencja: usuń backupy starsze niż $RetentionDays dni z S3
try {
    $cutoff = (Get-Date).AddDays(-$RetentionDays)
    $objects = aws s3 ls "s3://$S3Bucket/" | Where-Object { $_ -match "backup_\\d{8}_\\d{6}\\.zip" }
    foreach ($obj in $objects) {
        $parts = $obj -split '\s+'
        $dateStr = $parts[0] + ' ' + $parts[1]
        $fileDate = Get-Date $dateStr
        $fileName = $parts[-1]
        if ($fileDate -lt $cutoff) {
            aws s3 rm "s3://$S3Bucket/$fileName" | Out-Null
            Write-Log "Usunięto stary backup z S3: $fileName"
        }
    }
} catch {
    Write-Log "[ERROR] Błąd retencji S3: $_"
}

# 7. Sprzątanie lokalne
Remove-Item $backupDir -Recurse -Force
Remove-Item $zipPath -Force
Write-Log "Backup lokalny usunięty."

Write-Log "=== [END] Backup run ==="
