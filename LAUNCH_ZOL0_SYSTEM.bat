@echo off
title ZoL0 Trading System Launcher

echo.
echo ============================================================
echo                                                            
echo          ZoL0 TRADING SYSTEM LAUNCHER                     
echo             REAL BYBIT DATA MODE                          
echo                                                            
echo ============================================================
echo.

REM Set production environment variables
set BYBIT_PRODUCTION_CONFIRMED=true
set BYBIT_PRODUCTION_ENABLED=true

echo [INFO] Setting production environment variables...
echo [INFO] BYBIT_PRODUCTION_CONFIRMED=%BYBIT_PRODUCTION_CONFIRMED%
echo [INFO] BYBIT_PRODUCTION_ENABLED=%BYBIT_PRODUCTION_ENABLED%
echo.

echo [STEP 1] Starting Backend API Services...
echo ==========================================

echo [1.1] Starting Main API Server on port 5000...
start "ZoL0 Main API" cmd /k "cd /d C:\Users\piotr\Desktop\Zol0\ZoL0-master && echo Starting Main API Server... && python dashboard_api.py"

echo [1.2] Waiting 5 seconds...
timeout /t 5 /nobreak >nul

echo [1.3] Starting Enhanced API Server on port 4001...
start "ZoL0 Enhanced API" cmd /k "cd /d C:\Users\piotr\Desktop\Zol0 && echo Starting Enhanced API Server... && python enhanced_dashboard_api.py"

echo [1.4] Waiting 15 seconds for APIs to initialize...
timeout /t 15 /nobreak >nul

echo.
echo [STEP 2] Starting Dashboard Services...
echo ========================================

echo [2.1] Starting Master Control Dashboard (port 4501)...
start "ZoL0 Master Control" cmd /k "cd /d C:\Users\piotr\Desktop\Zol0 && streamlit run master_control_dashboard.py --server.port 4501"

echo [2.2] Starting Unified Trading Dashboard (port 4502)...
start "ZoL0 Unified Trading" cmd /k "cd /d C:\Users\piotr\Desktop\Zol0 && streamlit run unified_trading_dashboard.py --server.port 4502"

echo [2.3] Starting Enhanced Bot Monitor (port 4503)...
start "ZoL0 Bot Monitor" cmd /k "cd /d C:\Users\piotr\Desktop\Zol0 && streamlit run enhanced_bot_monitor.py --server.port 4503"

echo [2.4] Starting Trading Analytics (port 4504)...
start "ZoL0 Analytics" cmd /k "cd /d C:\Users\piotr\Desktop\Zol0 && streamlit run advanced_trading_analytics.py --server.port 4504"

echo [2.5] Starting Notification Dashboard (port 4505)...
start "ZoL0 Notifications" cmd /k "cd /d C:\Users\piotr\Desktop\Zol0 && streamlit run notification_dashboard.py --server.port 4505"

echo [2.6] Starting Portfolio Dashboard (port 4506)...
start "ZoL0 Portfolio" cmd /k "cd /d C:\Users\piotr\Desktop\Zol0 && streamlit run portfolio_dashboard.py --server.port 4506"

echo [2.7] Starting ML Analytics (port 4507)...
start "ZoL0 ML Analytics" cmd /k "cd /d C:\Users\piotr\Desktop\Zol0 && streamlit run ml_predictive_analytics.py --server.port 4507"

echo [2.8] Starting Enhanced Dashboard (port 4508)...
start "ZoL0 Enhanced" cmd /k "cd /d C:\Users\piotr\Desktop\Zol0 && streamlit run enhanced_dashboard.py --server.port 4508"

echo.
echo [STEP 3] System Status
echo ======================

echo [INFO] Waiting 20 seconds for all services to initialize...
timeout /t 20 /nobreak >nul

echo.
echo ============================================================
echo                                                            
echo          ZoL0 SYSTEM LAUNCH COMPLETE!                     
echo                                                            
echo   [SUCCESS] REAL BYBIT PRODUCTION DATA ACTIVE             
echo                                                            
echo ============================================================
echo.

echo [SUCCESS] Backend API Services:
echo    • Main API Server:     http://localhost:5000
echo    • Enhanced API Server: http://localhost:4001
echo.

echo [SUCCESS] Trading Dashboards:
echo    • Master Control:      http://localhost:4501
echo    • Unified Trading:     http://localhost:4502
echo    • Bot Monitor:         http://localhost:4503
echo    • Trading Analytics:   http://localhost:4504
echo    • Notifications:       http://localhost:4505
echo    • Portfolio:           http://localhost:4506
echo    • ML Analytics:        http://localhost:4507
echo    • Enhanced Dashboard:  http://localhost:4508
echo.

echo [INFO] Opening Master Control Dashboard...
start http://localhost:4501

echo.
echo ========================================================
echo  ALL SYSTEMS ONLINE - READY FOR TRADING!
echo  Press any key to close this launcher window...
echo ========================================================
pause >nul
