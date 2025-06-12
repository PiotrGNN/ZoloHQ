# Uruchomienie systemu ZoL0 - wszystkie komponenty
Write-Host "🚀 URUCHAMIANIE KOMPLETNEGO SYSTEMU ZoL0" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Krok 1: Backend API
Write-Host "`n🔧 KROK 1: Uruchamianie Backend API..." -ForegroundColor Yellow
Write-Host "Uruchamianie głównego API (port 5000)..." -ForegroundColor Green
Start-Process cmd -ArgumentList "/k", "cd C:\Users\piotr\Desktop\Zol0\ZoL0-master && python dashboard_api.py"
Start-Sleep 3

Write-Host "Uruchamianie rozszerzonego API (port 4001)..." -ForegroundColor Green  
Start-Process cmd -ArgumentList "/k", "cd C:\Users\piotr\Desktop\Zol0 && python enhanced_dashboard_api.py"
Start-Sleep 5

# Krok 2: Dashboardy
Write-Host "`n🔧 KROK 2: Uruchamianie Dashboardów..." -ForegroundColor Yellow

$dashboards = @(
    @{name="Master Control"; file="master_control_dashboard.py"; port=4501},
    @{name="Unified Trading"; file="unified_trading_dashboard.py"; port=4502},
    @{name="Enhanced Bot Monitor"; file="enhanced_bot_monitor.py"; port=4503},
    @{name="Advanced Analytics"; file="advanced_trading_analytics.py"; port=4504},
    @{name="Notifications"; file="notification_dashboard.py"; port=4505},
    @{name="Alert Management"; file="advanced_alert_management.py"; port=4506},
    @{name="Portfolio Optimization"; file="portfolio_optimization.py"; port=4507},
    @{name="ML Analytics"; file="ml_predictive_analytics.py"; port=4508},
    @{name="Enhanced Dashboard"; file="enhanced_dashboard.py"; port=4509}
)

foreach ($dashboard in $dashboards) {
    Write-Host "Uruchamianie $($dashboard.name) (port $($dashboard.port))..." -ForegroundColor Green
    Start-Process cmd -ArgumentList "/k", "cd C:\Users\piotr\Desktop\Zol0 && python -m streamlit run $($dashboard.file) --server.port=$($dashboard.port)"
    Start-Sleep 2
}

Write-Host "`n✅ SYSTEM URUCHOMIONY!" -ForegroundColor Green
Write-Host "`n🌐 Dostępne endpointy:" -ForegroundColor Cyan
Write-Host "Backend APIs:" -ForegroundColor White
Write-Host "  • Główne API:      http://localhost:5000" -ForegroundColor Gray
Write-Host "  • Rozszerzone API: http://localhost:4001" -ForegroundColor Gray
Write-Host "`nDashboardy:" -ForegroundColor White
foreach ($dashboard in $dashboards) {
    Write-Host "  • $($dashboard.name): http://localhost:$($dashboard.port)" -ForegroundColor Gray
}

Write-Host "`n🟢 Wszystkie dashboardy używają teraz PRAWDZIWYCH DANYCH z Bybit!" -ForegroundColor Green
Write-Host "💰 Połączenie z produkcyjnym kontem Bybit" -ForegroundColor Yellow
Write-Host "📡 Dane w czasie rzeczywistym" -ForegroundColor Yellow
