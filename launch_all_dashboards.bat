@echo off
echo ===============================================
echo    ZoL0 PLATFORM LAUNCHER - WINDOWS BATCH
echo ===============================================
echo.
echo [INFO] Starting all ZoL0 dashboards...
echo [INFO] Using real Bybit Production API data
echo.

cd /d "%~dp0"

echo [INFO] Starting Master Control Dashboard (port 8501)...
start "ZoL0-Master-Control" cmd /k "python -m streamlit run master_control_dashboard.py --server.port=8501"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Unified Trading Dashboard (port 8502)...
start "ZoL0-Trading" cmd /k "python -m streamlit run unified_trading_dashboard.py --server.port=8502"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Enhanced Bot Monitor (port 8503)...
start "ZoL0-Bot-Monitor" cmd /k "python -m streamlit run enhanced_bot_monitor.py --server.port=8503"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Advanced Trading Analytics (port 8504)...
start "ZoL0-Analytics" cmd /k "python -m streamlit run advanced_trading_analytics.py --server.port=8504"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Notification Dashboard (port 8505)...
start "ZoL0-Notifications" cmd /k "python -m streamlit run notification_dashboard.py --server.port=8505"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Advanced Alert Management (port 8506)...
start "ZoL0-Alerts" cmd /k "python -m streamlit run advanced_alert_management.py --server.port=8506"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Portfolio Dashboard (port 8507)...
start "ZoL0-Portfolio" cmd /k "python -m streamlit run portfolio_dashboard.py --server.port=8507"
timeout /t 3 /nobreak >nul

echo [INFO] Starting ML Predictive Analytics (port 8508)...
start "ZoL0-ML-Analytics" cmd /k "python -m streamlit run ml_predictive_analytics.py --server.port=8508"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Enhanced Dashboard (port 8509)...
start "ZoL0-Enhanced" cmd /k "python -m streamlit run enhanced_dashboard.py --server.port=8509"
timeout /t 5 /nobreak >nul

echo.
echo ===============================================
echo [SUCCESS] ALL DASHBOARDS LAUNCHED!
echo ===============================================
echo.
echo 💡 GŁÓWNE LINKI:
echo    🎛️  Master Control: http://localhost:8501
echo    📊 Trading Dashboard: http://localhost:8502
echo    🤖 Bot Monitor: http://localhost:8503
echo    📈 Analytics: http://localhost:8504
echo    🔔 Notifications: http://localhost:8505
echo    🚨 Alerts: http://localhost:8506
echo    📊 Portfolio: http://localhost:8507
echo    🤖 ML Analytics: http://localhost:8508
echo    ✨ Enhanced: http://localhost:8509
echo.
echo 📡 Wszystkie dashboardy używają prawdziwych danych z Bybit!
echo.

rem Otwórz główny dashboard w przeglądarce
start http://localhost:8501

echo 🔧 Zamknij to okno, aby zatrzymać launcher (dashboardy będą działać dalej)
pause
