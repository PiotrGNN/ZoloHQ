@echo off
REM filepath: c:\Users\piotr\Desktop\Zol0\START_ALL.bat
REM ZoL0 System Launcher - Starts all APIs and dashboards automatically (minimized windows)

title ZoL0 System Launcher

echo.
echo ================================================================
echo                   ZoL0 SYSTEM LAUNCHER
echo ================================================================
echo.

cd /d "C:\Users\piotr\Desktop\Zol0"

REM Start Backend APIs (minimized)
echo 🚀 Starting Backend APIs...
start /min "Main-API-5000" cmd /k "cd ZoL0-master && python dashboard_api.py"
timeout /t 3 /nobreak >nul

start /min "Enhanced-API-4001" cmd /k "python enhanced_dashboard_api.py"
timeout /t 5 /nobreak >nul

REM Start Dashboards (minimized)
echo 🚀 Starting Dashboards...
start /min "Master-Control-4501" cmd /k "python -m streamlit run master_control_dashboard.py --server.port=4501"
timeout /t 2 /nobreak >nul

start /min "Unified-Trading-4502" cmd /k "python -m streamlit run unified_trading_dashboard.py --server.port=4502"
timeout /t 2 /nobreak >nul

start /min "Bot-Monitor-4503" cmd /k "python -m streamlit run enhanced_bot_monitor.py --server.port=4503"
timeout /t 2 /nobreak >nul

start /min "Analytics-4504" cmd /k "python -m streamlit run advanced_trading_analytics.py --server.port=4504"
timeout /t 2 /nobreak >nul

start /min "Notifications-4505" cmd /k "python -m streamlit run notification_dashboard.py --server.port=4505"
timeout /t 2 /nobreak >nul

start /min "Alerts-4506" cmd /k "python -m streamlit run advanced_alert_management.py --server.port=4506"
timeout /t 2 /nobreak >nul

start /min "Portfolio-4507" cmd /k "python -m streamlit run portfolio_optimization.py --server.port=4507"
timeout /t 2 /nobreak >nul

start /min "ML-Analytics-4508" cmd /k "python -m streamlit run ml_predictive_analytics.py --server.port=4508"
timeout /t 2 /nobreak >nul

start /min "Enhanced-4509" cmd /k "python -m streamlit run enhanced_dashboard.py --server.port=4509"

echo.
echo ✅ SYSTEM LAUNCHED!
echo.
echo 🌐 URLs:
echo   Backend APIs:
echo     http://localhost:5000 (Main API)
echo     http://localhost:4001 (Enhanced API)
echo.
echo   Dashboards:
echo     http://localhost:4501 (Master Control)
echo     http://localhost:4502 (Unified Trading) 
echo     http://localhost:4503 (Bot Monitor)
echo     http://localhost:4504 (Analytics)
echo     http://localhost:4505 (Notifications)
echo     http://localhost:4506 (Alerts)
echo     http://localhost:4507 (Portfolio)
echo     http://localhost:4508 (ML Analytics)
echo     http://localhost:4509 (Enhanced)
echo.
echo 🟢 All systems using REAL BYBIT DATA!
echo.
pause
