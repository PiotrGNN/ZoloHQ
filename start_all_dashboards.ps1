# ZoL0 Platform Launcher - Wszystkie Dashboardy z Prawdziwymi Danymi Bybit
# ========================================================================

Write-Host "🚀 URUCHAMIANIE PLATFORMY ZoL0 - PRAWDZIWE DANE BYBIT" -ForegroundColor Green
Write-Host "=" * 60

# Sprawdź czy wszystkie pliki istnieją
$dashboards = @{
    "Master Control Dashboard" = @{file="master_control_dashboard.py"; port=8501}
    "Unified Trading Dashboard" = @{file="unified_trading_dashboard.py"; port=8502}
    "Enhanced Bot Monitor" = @{file="enhanced_bot_monitor.py"; port=8503}
    "Advanced Trading Analytics" = @{file="advanced_trading_analytics.py"; port=8504}
    "Notification Dashboard" = @{file="notification_dashboard.py"; port=8505}
    "Advanced Alert Management" = @{file="advanced_alert_management.py"; port=8506}
    "Portfolio Optimization" = @{file="portfolio_optimization.py"; port=8507}
    "ML Predictive Analytics" = @{file="ml_predictive_analytics.py"; port=8508}
    "Enhanced Dashboard" = @{file="enhanced_dashboard.py"; port=8509}
}

Write-Host "📋 SPRAWDZANIE DASHBOARDÓW:" -ForegroundColor Yellow
foreach ($name in $dashboards.Keys) {
    $file = $dashboards[$name].file
    if (Test-Path $file) {
        Write-Host "✅ $name - GOTOWY" -ForegroundColor Green
    } else {
        Write-Host "❌ $name - BRAK PLIKU" -ForegroundColor Red
    }
}

Write-Host "`n🚀 URUCHAMIANIE DASHBOARDÓW:" -ForegroundColor Cyan

# Uruchom wszystkie dashboardy w osobnych oknach
foreach ($name in $dashboards.Keys) {
    $dashboard = $dashboards[$name]
    $file = $dashboard.file
    $port = $dashboard.port
    
    if (Test-Path $file) {
        Write-Host "🌐 $name - http://localhost:$port" -ForegroundColor Green
        
        # Uruchom w nowym oknie PowerShell
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '🚀 Uruchamianie $name na porcie $port' -ForegroundColor Green; streamlit run $file --server.port=$port"
        
        Start-Sleep -Seconds 2
    }
}

Write-Host "`n🎉 WSZYSTKIE DASHBOARDY URUCHOMIONE!" -ForegroundColor Green
Write-Host "📡 Platforma ZoL0 używa prawdziwych danych z Bybit Production API" -ForegroundColor Yellow
Write-Host "`n📋 DOSTĘPNE DASHBOARDY:" -ForegroundColor Cyan
foreach ($name in $dashboards.Keys) {
    $port = $dashboards[$name].port
    Write-Host "🌐 $name - http://localhost:$port" -ForegroundColor White
}

Write-Host "`nAby zatrzymać wszystkie dashboardy, zamknij to okno lub naciśnij Ctrl+C" -ForegroundColor Yellow
Read-Host "Naciśnij Enter, aby kontynuować..."
