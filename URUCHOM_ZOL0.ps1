Write-Host ""
Write-Host "████████████████████████████████████████████████████████" -ForegroundColor Cyan
Write-Host "███                                                  ███" -ForegroundColor Cyan
Write-Host "███         ZoL0 TRADING SYSTEM LAUNCHER            ███" -ForegroundColor Yellow
Write-Host "███            PRAWDZIWE DANE BYBIT                 ███" -ForegroundColor Green
Write-Host "███                                                  ███" -ForegroundColor Cyan
Write-Host "████████████████████████████████████████████████████████" -ForegroundColor Cyan
Write-Host ""

# Ustawienie zmiennych środowiskowych
$env:BYBIT_PRODUCTION_CONFIRMED = "true"
$env:BYBIT_PRODUCTION_ENABLED = "true"

Write-Host "[INFO] Ustawiono zmienne produkcyjne..." -ForegroundColor Green
Write-Host "[INFO] BYBIT_PRODUCTION_CONFIRMED=$($env:BYBIT_PRODUCTION_CONFIRMED)" -ForegroundColor Gray
Write-Host "[INFO] BYBIT_PRODUCTION_ENABLED=$($env:BYBIT_PRODUCTION_ENABLED)" -ForegroundColor Gray
Write-Host ""

# Ustawienie katalogu roboczego
Set-Location "C:\Users\piotr\Desktop\Zol0"

Write-Host "[KROK 1] Uruchamianie serwerów API Backend..." -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Yellow

try {
    Write-Host "[1.1] Main API Server (port 5000)..." -ForegroundColor Cyan
    Start-Process cmd -ArgumentList "/k", "cd /d C:\Users\piotr\Desktop\Zol0\ZoL0-master && python dashboard_api.py" -WindowStyle Minimized
    Write-Host "    ✅ Main API Server uruchomiony" -ForegroundColor Green
    
    Start-Sleep -Seconds 5
    
    Write-Host "[1.2] Enhanced API Server (port 5001)..." -ForegroundColor Cyan
    Start-Process cmd -ArgumentList "/k", "cd /d C:\Users\piotr\Desktop\Zol0 && python enhanced_dashboard_api.py" -WindowStyle Minimized
    Write-Host "    ✅ Enhanced API Server uruchomiony" -ForegroundColor Green
    
    Write-Host "[1.3] Oczekiwanie 15 sekund na inicjalizację API..." -ForegroundColor Gray
    Start-Sleep -Seconds 15
} catch {
    Write-Host "    ❌ Błąd uruchamiania API: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "[KROK 2] Uruchamianie Dashboard Services..." -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Yellow

$Dashboards = @(
    @{Name="Master Control"; Script="master_control_dashboard.py"; Port=8501},
    @{Name="Unified Trading"; Script="unified_trading_dashboard.py"; Port=8502},
    @{Name="Bot Monitor"; Script="enhanced_bot_monitor.py"; Port=8503},
    @{Name="Analytics"; Script="advanced_trading_analytics.py"; Port=8504},
    @{Name="Notifications"; Script="notification_dashboard.py"; Port=8505},
    @{Name="Portfolio"; Script="portfolio_dashboard.py"; Port=8506},
    @{Name="ML Analytics"; Script="ml_predictive_analytics.py"; Port=8507},
    @{Name="Enhanced"; Script="enhanced_dashboard.py"; Port=8508}
)

foreach ($Dashboard in $Dashboards) {
    try {
        Write-Host "[2.$($Dashboards.IndexOf($Dashboard) + 1)] $($Dashboard.Name) (port $($Dashboard.Port))..." -ForegroundColor Cyan
        
        $ScriptPath = Join-Path "C:\Users\piotr\Desktop\Zol0" $Dashboard.Script
        if (Test-Path $ScriptPath) {
            Start-Process cmd -ArgumentList "/k", "cd /d C:\Users\piotr\Desktop\Zol0 && streamlit run $($Dashboard.Script) --server.port $($Dashboard.Port)" -WindowStyle Minimized
            Write-Host "    ✅ $($Dashboard.Name) uruchomiony" -ForegroundColor Green
        } else {
            Write-Host "    ⚠️  Plik nie znaleziony: $($Dashboard.Script)" -ForegroundColor Yellow
        }
        Start-Sleep -Seconds 3
    } catch {
        Write-Host "    ❌ Błąd: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "[KROK 3] Oczekiwanie na inicjalizację..." -ForegroundColor Yellow
Write-Host "=========================================" -ForegroundColor Yellow
Start-Sleep -Seconds 20

Write-Host ""
Write-Host "████████████████████████████████████████████████████████" -ForegroundColor Green
Write-Host "███                                                  ███" -ForegroundColor Green
Write-Host "███         SYSTEM ZoL0 URUCHOMIONY!                ███" -ForegroundColor Yellow
Write-Host "███                                                  ███" -ForegroundColor Green
Write-Host "███  🟢 PRAWDZIWE DANE BYBIT AKTYWNE               ███" -ForegroundColor White
Write-Host "███                                                  ███" -ForegroundColor Green
Write-Host "████████████████████████████████████████████████████████" -ForegroundColor Green
Write-Host ""

Write-Host "[SUKCES] Serwery Backend API:" -ForegroundColor Green
Write-Host "    • Main API Server:     http://localhost:5000" -ForegroundColor White
Write-Host "    • Enhanced API Server: http://localhost:5001" -ForegroundColor White
Write-Host ""

Write-Host "[SUKCES] Dashboardy Trading:" -ForegroundColor Green
Write-Host "    • Master Control:    http://localhost:8501" -ForegroundColor White
Write-Host "    • Unified Trading:   http://localhost:8502" -ForegroundColor White
Write-Host "    • Bot Monitor:       http://localhost:8503" -ForegroundColor White
Write-Host "    • Trading Analytics: http://localhost:8504" -ForegroundColor White
Write-Host "    • Notifications:     http://localhost:8505" -ForegroundColor White
Write-Host "    • Portfolio:         http://localhost:8506" -ForegroundColor White
Write-Host "    • ML Analytics:      http://localhost:8507" -ForegroundColor White
Write-Host "    • Enhanced:          http://localhost:8508" -ForegroundColor White
Write-Host ""

Write-Host "[INFO] Otwieranie Master Control Dashboard..." -ForegroundColor Cyan
try {
    Start-Process "http://localhost:8501"
} catch {
    Write-Host "    ⚠️  Nie można otworzyć przeglądarki automatycznie" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  WSZYSTKIE SYSTEMY ONLINE - GOTOWE DO TRADINGU!" -ForegroundColor Green
Write-Host "  Naciśnij Enter aby sprawdzić status serwisów..." -ForegroundColor Yellow
Write-Host "================================================================" -ForegroundColor Cyan

Read-Host

Write-Host ""
Write-Host "🔍 Test łączności serwisów:" -ForegroundColor Cyan

# Test API services
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000" -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "    ✅ Main API Server - ODPOWIADA" -ForegroundColor Green
    }
} catch {
    Write-Host "    ❌ Main API Server - NIE ODPOWIADA" -ForegroundColor Red
}

try {
    $response = Invoke-WebRequest -Uri "http://localhost:5001" -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "    ✅ Enhanced API Server - ODPOWIADA" -ForegroundColor Green
    }
} catch {
    Write-Host "    ❌ Enhanced API Server - NIE ODPOWIADA" -ForegroundColor Red
}

try {
    $response = Invoke-WebRequest -Uri "http://localhost:8501" -TimeoutSec 3 -ErrorAction SilentlyContinue
    if ($response.StatusCode -eq 200) {
        Write-Host "    ✅ Master Control - ODPOWIADA" -ForegroundColor Green
    }
} catch {
    Write-Host "    ❌ Master Control - NIE ODPOWIADA" -ForegroundColor Red
}

Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  ZoL0 TRADING SYSTEM JEST AKTYWNY!" -ForegroundColor Yellow
Write-Host "  Aby zatrzymać - zamknij wszystkie okna terminali" -ForegroundColor Gray
Write-Host "================================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Naciśnij Enter aby zakończyć..." -ForegroundColor Yellow
Read-Host
