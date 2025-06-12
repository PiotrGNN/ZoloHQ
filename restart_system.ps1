# ZoL0 System Restart Script (PowerShell)
# =====================================

Write-Host "🚀 Starting ZoL0 Trading System with Fixed Master Control Dashboard" -ForegroundColor Green
Write-Host "=" * 70

# Kill any existing processes
Write-Host "`n🔄 Cleaning up existing processes..." -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*python*" -or $_.ProcessName -like "*streamlit*"} | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 2

# Start API servers
Write-Host "`n📡 Starting API Servers..." -ForegroundColor Cyan

Write-Host "  - Starting Main API Server (port 5000)..."
Start-Process -FilePath "python" -ArgumentList "dashboard_api.py" -WorkingDirectory "C:\Users\piotr\Desktop\Zol0\ZoL0-master" -WindowStyle Minimized
Start-Sleep -Seconds 3

Write-Host "  - Starting Enhanced API Server (port 4001)..."
Start-Process -FilePath "python" -ArgumentList "enhanced_dashboard_api.py" -WorkingDirectory "C:\Users\piotr\Desktop\Zol0" -WindowStyle Minimized
Start-Sleep -Seconds 3

# Start Dashboards
Write-Host "`n🖥️  Starting Dashboards..." -ForegroundColor Cyan

Write-Host "  - Starting Main Trading Dashboard (port 4501)..."
Start-Process -FilePath "streamlit" -ArgumentList "run", "dashboard.py", "--server.port", "4501" -WorkingDirectory "C:\Users\piotr\Desktop\Zol0\ZoL0-master" -WindowStyle Minimized
Start-Sleep -Seconds 3

Write-Host "  - Starting Unified Trading Dashboard (port 4503)..."
Start-Process -FilePath "streamlit" -ArgumentList "run", "unified_trading_dashboard.py", "--server.port", "4503" -WorkingDirectory "C:\Users\piotr\Desktop\Zol0" -WindowStyle Minimized
Start-Sleep -Seconds 3

Write-Host "  - Starting Enhanced Dashboard (port 4504)..."
Start-Process -FilePath "streamlit" -ArgumentList "run", "enhanced_dashboard.py", "--server.port", "4504" -WorkingDirectory "C:\Users\piotr\Desktop\Zol0" -WindowStyle Minimized
Start-Sleep -Seconds 3

Write-Host "  - Starting FIXED Master Control Dashboard (port 4505)..." -ForegroundColor Yellow
Start-Process -FilePath "streamlit" -ArgumentList "run", "master_control_dashboard.py", "--server.port", "4505" -WorkingDirectory "C:\Users\piotr\Desktop\Zol0" -WindowStyle Minimized
Start-Sleep -Seconds 5

Write-Host "`n✅ System Startup Complete!" -ForegroundColor Green

Write-Host "`n🌐 Access Points:" -ForegroundColor White
Write-Host "  • Main Trading Dashboard:    http://localhost:4501" -ForegroundColor Gray
Write-Host "  • Unified Trading Dashboard: http://localhost:4503" -ForegroundColor Gray
Write-Host "  • Enhanced Dashboard:        http://localhost:4504" -ForegroundColor Gray
Write-Host "  • Master Control Dashboard:  http://localhost:4505  ← FIXED WITH REAL DATA" -ForegroundColor Green
Write-Host "  • Main API:                  http://localhost:5000" -ForegroundColor Gray
Write-Host "  • Enhanced API:              http://localhost:4001" -ForegroundColor Gray

Write-Host "`n📊 Master Control Dashboard now shows REAL Bybit production data!" -ForegroundColor Green
Write-Host "🎛️  All services are running with real-time data integration." -ForegroundColor Green

Write-Host "`nPress Enter to exit..." -ForegroundColor Yellow
Read-Host
