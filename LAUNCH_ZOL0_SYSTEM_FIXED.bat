@echo off
chcp 65001 >nul
title ZoL0 Trading System Launcher - FIXED VERSION

echo.
echo ================================================================
echo                 ZoL0 TRADING SYSTEM LAUNCHER
echo                    REAL BYBIT DATA MODE  
echo ================================================================
echo.

REM Set production environment variables
set BYBIT_PRODUCTION_CONFIRMED=true
set BYBIT_PRODUCTION_ENABLED=true
set TRADING_MODE=production

echo [INFO] Setting production environment variables...
echo [INFO] BYBIT_PRODUCTION_CONFIRMED=%BYBIT_PRODUCTION_CONFIRMED%
echo [INFO] BYBIT_PRODUCTION_ENABLED=%BYBIT_PRODUCTION_ENABLED%
echo [INFO] TRADING_MODE=%TRADING_MODE%
echo.

cd /d "%~dp0"

echo [STEP 1] Starting Backend API Services...
echo ==========================================

echo [1.1] Starting Main API Server on port 5000...
start "ZoL0-Main-API" cmd /k "cd /d \"%~dp0\ZoL0-master\" && echo Starting Main API Server... && python dashboard_api.py"

echo [1.2] Waiting 5 seconds...
timeout /t 5 /nobreak >nul

echo [1.3] Starting Enhanced API Server on port 5001...
start "ZoL0-Enhanced-API" cmd /k "cd /d \"%~dp0\" && echo Starting Enhanced API Server... && python enhanced_dashboard_api.py"

echo [1.4] Waiting 15 seconds for APIs to initialize...
timeout /t 15 /nobreak >nul

echo.
echo [STEP 2] Starting Dashboard Services...
echo ========================================

echo [2.1] Starting Master Control Dashboard (port 8501)...
start "ZoL0-Master-Control" cmd /k "cd /d \"%~dp0\" && streamlit run master_control_dashboard.py --server.port 8501"

echo [2.2] Starting Enhanced Bot Monitor (port 8502)...
start "ZoL0-Bot-Monitor" cmd /k "cd /d \"%~dp0\" && streamlit run enhanced_bot_monitor.py --server.port 8502"

echo [2.3] Starting Trading Analytics (port 8503)...
start "ZoL0-Analytics" cmd /k "cd /d \"%~dp0\" && streamlit run advanced_trading_analytics.py --server.port 8503"

echo [2.4] Starting Notification Dashboard (port 8504)...
start "ZoL0-Notifications" cmd /k "cd /d \"%~dp0\" && streamlit run notification_dashboard.py --server.port 8504"

echo [2.5] Starting Portfolio Dashboard (port 8505)...
start "ZoL0-Portfolio" cmd /k "cd /d \"%~dp0\" && streamlit run portfolio_dashboard.py --server.port 8505"

echo [2.6] Starting ML Analytics (port 8506)...
start "ZoL0-ML-Analytics" cmd /k "cd /d \"%~dp0\" && streamlit run ml_predictive_analytics.py --server.port 8506"

echo [2.7] Starting Risk Management (port 8507)...
start "ZoL0-Risk-Management" cmd /k "cd /d \"%~dp0\" && streamlit run advanced_risk_management.py --server.port 8507"

echo [2.8] Starting Enhanced Dashboard (port 8508)...
start "ZoL0-Enhanced" cmd /k "cd /d \"%~dp0\" && streamlit run enhanced_dashboard.py --server.port 8508"

echo [2.9] Starting Compliance Dashboard (port 8509)...
start "ZoL0-Compliance" cmd /k "cd /d \"%~dp0\" && streamlit run regulatory_compliance_dashboard.py --server.port 8509"

echo.
echo [STEP 3] System Status
echo ======================

echo [INFO] Waiting 20 seconds for all services to initialize...
timeout /t 20 /nobreak >nul

echo.
echo ================================================================
echo                ZoL0 SYSTEM LAUNCH COMPLETE!
echo                                                  
echo   REAL BYBIT PRODUCTION DATA ACTIVE           
echo                                                  
echo ================================================================
echo.

echo [SUCCESS] Backend API Services:
echo    - Main API Server:     http://localhost:5000
echo    - Enhanced API Server: http://localhost:5001
echo.

echo [SUCCESS] Trading Dashboards:
echo    - Master Control:      http://localhost:8501
echo    - Bot Monitor:         http://localhost:8502
echo    - Trading Analytics:   http://localhost:8503
echo    - Notifications:       http://localhost:8504
echo    - Portfolio:           http://localhost:8505
echo    - ML Analytics:        http://localhost:8506
echo    - Risk Management:     http://localhost:8507
echo    - Enhanced Dashboard:  http://localhost:8508
echo    - Compliance:          http://localhost:8509
echo.

echo [INFO] Opening Master Control Dashboard...
start http://localhost:8501

echo.
echo ========================================================
echo  ALL SYSTEMS ONLINE - READY FOR TRADING!
echo  Press any key to close this launcher window...
echo ========================================================
pause >nul
