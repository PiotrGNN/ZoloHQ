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

start /min "Enhanced-API-5001" cmd /k "python enhanced_dashboard_api.py"
timeout /t 5 /nobreak >nul

REM Start Dashboards (minimized)
echo 🚀 Starting Dashboards...
start /min "Master-Control-8501" cmd /k "python -m streamlit run master_control_dashboard.py --server.port=8501"
timeout /t 2 /nobreak >nul

start /min "Unified-Trading-8502" cmd /k "python -m streamlit run unified_trading_dashboard.py --server.port=8502"
timeout /t 2 /nobreak >nul

start /min "Bot-Monitor-8503" cmd /k "python -m streamlit run enhanced_bot_monitor.py --server.port=8503"
timeout /t 2 /nobreak >nul

start /min "Analytics-8504" cmd /k "python -m streamlit run advanced_trading_analytics.py --server.port=8504"
timeout /t 2 /nobreak >nul

start /min "Notifications-8505" cmd /k "python -m streamlit run notification_dashboard.py --server.port=8505"
timeout /t 2 /nobreak >nul

start /min "Alerts-8506" cmd /k "python -m streamlit run advanced_alert_management.py --server.port=8506"
timeout /t 2 /nobreak >nul

start /min "Portfolio-8507" cmd /k "python -m streamlit run portfolio_optimization.py --server.port=8507"
timeout /t 2 /nobreak >nul

start /min "ML-Analytics-8508" cmd /k "python -m streamlit run ml_predictive_analytics.py --server.port=8508"
timeout /t 2 /nobreak >nul

start /min "Enhanced-8509" cmd /k "python -m streamlit run enhanced_dashboard.py --server.port=8509"

echo.
echo ✅ SYSTEM LAUNCHED!
echo.
echo 🌐 URLs:
echo   Backend APIs:
echo     http://localhost:5000 (Main API)
echo     http://localhost:5001 (Enhanced API)
echo.
echo   Dashboards:
echo     http://localhost:8501 (Master Control)
echo     http://localhost:8502 (Unified Trading) 
echo     http://localhost:8503 (Bot Monitor)
echo     http://localhost:8504 (Analytics)
echo     http://localhost:8505 (Notifications)
echo     http://localhost:8506 (Alerts)
echo     http://localhost:8507 (Portfolio)
echo     http://localhost:8508 (ML Analytics)
echo     http://localhost:8509 (Enhanced)
echo.
echo 🟢 All systems using REAL BYBIT DATA!
echo.
pause
