@echo off
chcp 65001 >nul
title ZoL0 Dashboard Launcher

echo ===============================================
echo    ZoL0 PLATFORM LAUNCHER - FIXED VERSION
echo ===============================================
echo.
echo [INFO] Starting all ZoL0 dashboards...
echo [INFO] Using real Bybit Production API data
echo.

cd /d "%~dp0"

echo [INFO] Starting Master Control Dashboard (port 8501)...
start "ZoL0-Master-Control" cmd /k "cd /d \"%~dp0\" && streamlit run master_control_dashboard.py --server.port 8501"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Enhanced Bot Monitor (port 8502)...
start "ZoL0-Bot-Monitor" cmd /k "cd /d \"%~dp0\" && streamlit run enhanced_bot_monitor.py --server.port 8502"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Advanced Trading Analytics (port 8503)...
start "ZoL0-Analytics" cmd /k "cd /d \"%~dp0\" && streamlit run advanced_trading_analytics.py --server.port 8503"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Notification Dashboard (port 8504)...
start "ZoL0-Notifications" cmd /k "cd /d \"%~dp0\" && streamlit run notification_dashboard.py --server.port 8504"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Portfolio Dashboard (port 8505)...
start "ZoL0-Portfolio" cmd /k "cd /d \"%~dp0\" && streamlit run portfolio_dashboard.py --server.port 8505"
timeout /t 3 /nobreak >nul

echo [INFO] Starting ML Analytics Dashboard (port 8506)...
start "ZoL0-ML-Analytics" cmd /k "cd /d \"%~dp0\" && streamlit run ml_predictive_analytics.py --server.port 8506"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Risk Management Dashboard (port 8507)...
start "ZoL0-Risk-Management" cmd /k "cd /d \"%~dp0\" && streamlit run advanced_risk_management.py --server.port 8507"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Enhanced Dashboard (port 8508)...
start "ZoL0-Enhanced" cmd /k "cd /d \"%~dp0\" && streamlit run enhanced_dashboard.py --server.port 8508"
timeout /t 3 /nobreak >nul

echo [INFO] Starting Compliance Dashboard (port 8509)...
start "ZoL0-Compliance" cmd /k "cd /d \"%~dp0\" && streamlit run regulatory_compliance_dashboard.py --server.port 8509"
timeout /t 3 /nobreak >nul

echo.
echo ===============================================
echo    ALL DASHBOARDS LAUNCHED SUCCESSFULLY!
echo ===============================================
echo.
echo Dashboard URLs:
echo   Master Control:     http://localhost:8501
echo   Bot Monitor:        http://localhost:8502
echo   Trading Analytics:  http://localhost:8503
echo   Notifications:      http://localhost:8504
echo   Portfolio:          http://localhost:8505
echo   ML Analytics:       http://localhost:8506
echo   Risk Management:    http://localhost:8507
echo   Enhanced:           http://localhost:8508
echo   Compliance:         http://localhost:8509
echo.
echo Press any key to exit...
pause >nul
