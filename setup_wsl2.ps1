# Skrypt PowerShell do automatycznej instalacji i testów projektu ZoL0 w WSL2
# Uruchom ten plik w PowerShell: .\setup_wsl2.ps1

# 1. Sprawdź, czy WSL2 jest dostępny
if (-not (wsl.exe -l | Select-String -Pattern 'Ubuntu')) {
    Write-Host 'Nie znaleziono dystrybucji Ubuntu w WSL2. Zainstaluj Ubuntu z Microsoft Store.' -ForegroundColor Red
    exit 1
}

# 2. Przekopiuj skrypt setup_wsl2.sh do katalogu projektu w Ubuntu
wsl.exe mkdir -p /mnt/c/Users/piotr/Desktop/Zol0
wsl.exe cp /mnt/c/Users/piotr/Desktop/Zol0/setup_wsl2.sh /mnt/c/Users/piotr/Desktop/Zol0/

# 3. Uruchom skrypt instalacyjny w WSL2 w katalogu projektu
wsl.exe bash /mnt/c/Users/piotr/Desktop/Zol0/setup_wsl2.sh

Write-Host 'Instalacja i testy w WSL2 zakończone.' -ForegroundColor Green
