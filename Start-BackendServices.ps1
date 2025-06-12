# ZoL0 Trading System - Backend API Services Startup
# PowerShell script to start backend APIs for real data access

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "         ZoL0 TRADING SYSTEM - BACKEND API STARTUP" -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting backend API services for real Bybit data access..." -ForegroundColor Yellow
Write-Host ""

# Set working directory
Set-Location "C:\Users\piotr\Desktop\Zol0"

# Start Main API Server (Port 5000)
Write-Host "🔧 Step 1: Starting Main API Server (Port 5000)..." -ForegroundColor Green
Write-Host "    File: ZoL0-master\dashboard_api.py" -ForegroundColor Gray
Start-Process -FilePath "cmd" -ArgumentList "/k", "cd ZoL0-master && python dashboard_api.py" -WindowStyle Normal
Start-Sleep -Seconds 3

# Start Enhanced API Server (Port 5001)
Write-Host ""
Write-Host "🔧 Step 2: Starting Enhanced API Server (Port 5001)..." -ForegroundColor Green
Write-Host "    File: enhanced_dashboard_api.py" -ForegroundColor Gray
Start-Process -FilePath "cmd" -ArgumentList "/k", "python enhanced_dashboard_api.py" -WindowStyle Normal
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "✅ Backend API services are starting..." -ForegroundColor Green
Write-Host ""
Write-Host "🌐 API Service Status:" -ForegroundColor Cyan
Write-Host "   • Main API Server:     http://localhost:5000" -ForegroundColor White
Write-Host "   • Enhanced API Server: http://localhost:5001" -ForegroundColor White
Write-Host ""
Write-Host "📊 Available Dashboard Endpoints:" -ForegroundColor Cyan
Write-Host "   • Master Control:      http://localhost:8501" -ForegroundColor White
Write-Host "   • Unified Trading:     http://localhost:8502" -ForegroundColor White
Write-Host "   • Enhanced Bot:        http://localhost:8503" -ForegroundColor White
Write-Host "   • Analytics:           http://localhost:8504" -ForegroundColor White
Write-Host "   • Notifications:       http://localhost:8505" -ForegroundColor White
Write-Host "   • Alerts:              http://localhost:8506" -ForegroundColor White
Write-Host "   • Portfolio:           http://localhost:8507" -ForegroundColor White
Write-Host "   • ML Analytics:        http://localhost:8508" -ForegroundColor White
Write-Host "   • Enhanced:            http://localhost:8509" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Wait 10 seconds for APIs to fully initialize"
Write-Host "   2. Run: .\launch_all_dashboards.bat"
Write-Host "   3. Or run: python launch_all_dashboards.py"
Write-Host ""
Write-Host "💡 API servers are running in separate windows." -ForegroundColor Yellow
Write-Host "   Keep them open for dashboard functionality."
Write-Host "   Close API windows to stop services."
Write-Host ""

Read-Host "Press Enter to continue..."
