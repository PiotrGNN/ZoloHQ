@echo off
title ZoL0 Trading System - Backend API Services

echo.
echo ================================================================
echo         ZoL0 TRADING SYSTEM - BACKEND API STARTUP
echo ================================================================
echo.
echo This script will start the backend API services required for
echo real Bybit data access in all dashboards.
echo.

cd /d "C:\Users\piotr\Desktop\Zol0"

echo 🔧 Step 1: Starting Main API Server (Port 5000)...
echo    File: ZoL0-master\dashboard_api.py
start "ZoL0-Main-API-5000" cmd /k "cd ZoL0-master && python dashboard_api.py"
timeout /t 3 /nobreak >nul

echo.
echo 🔧 Step 2: Starting Enhanced API Server (Port 5001)...
echo    File: enhanced_dashboard_api.py
start "ZoL0-Enhanced-API-5001" cmd /k "python enhanced_dashboard_api.py"
timeout /t 5 /nobreak >nul

echo.
echo ✅ Backend API services are starting...
echo.
echo 🌐 API Service Status:
echo   • Main API Server:     http://localhost:5000
echo   • Enhanced API Server: http://localhost:5001
echo.
echo 📊 Available Dashboard Endpoints:
echo   • Master Control:      http://localhost:8501
echo   • Unified Trading:     http://localhost:8502
echo   • Enhanced Bot:        http://localhost:8503
echo   • Analytics:           http://localhost:8504
echo   • Notifications:       http://localhost:8505
echo   • Alerts:              http://localhost:8506
echo   • Portfolio:           http://localhost:8507
echo   • ML Analytics:        http://localhost:8508
echo   • Enhanced:            http://localhost:8509
echo.
echo 🚀 Next Steps:
echo   1. Wait 10 seconds for APIs to fully initialize
echo   2. Run: launch_all_dashboards.bat
echo   3. Or run: python launch_all_dashboards.py
echo.
echo 💡 API servers are running in separate windows.
echo    Keep them open for dashboard functionality.
echo    Close API windows to stop services.
echo.

pause
