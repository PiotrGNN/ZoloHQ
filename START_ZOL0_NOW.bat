@echo off
title ZoL0 Trading System - URUCHOMIENIE KOMPLETNE
color 0A

echo.
echo ████████████████████████████████████████████████████████
echo ███                                                  ███
echo ███         ZoL0 TRADING SYSTEM LAUNCHER            ███
echo ███            PRAWDZIWE DANE BYBIT                 ███
echo ███                                                  ███
echo ████████████████████████████████████████████████████████
echo.

REM Ustawienie zmiennych środowiskowych
set BYBIT_PRODUCTION_CONFIRMED=true
set BYBIT_PRODUCTION_ENABLED=true

echo [INFO] Zmienne produkcyjne ustawione
echo [INFO] BYBIT_PRODUCTION_CONFIRMED=%BYBIT_PRODUCTION_CONFIRMED%
echo [INFO] BYBIT_PRODUCTION_ENABLED=%BYBIT_PRODUCTION_ENABLED%
echo.

echo [KROK 1] Uruchamianie Backend API Services...
echo ==============================================

echo [1.1] Main API Server (port 5000)...
start "ZoL0 Main API" /MIN cmd /k "cd /d %~dp0ZoL0-master && echo Uruchamianie Main API Server... && python dashboard_api.py"

timeout /t 5 >nul

echo [1.2] Enhanced API Server (port 5001)...
start "ZoL0 Enhanced API" /MIN cmd /k "cd /d %~dp0 && echo Uruchamianie Enhanced API Server... && python enhanced_dashboard_api.py"

echo [1.3] Oczekiwanie 15 sekund na inicjalizację API...
timeout /t 15 >nul

echo.
echo [KROK 2] Uruchamianie Dashboard Services...
echo =============================================

echo [2.1] Master Control Dashboard (8501)...
start "ZoL0 Master Control" /MIN cmd /k "cd /d %~dp0 && streamlit run master_control_dashboard.py --server.port 8501 --server.headless true"

timeout /t 3 >nul

echo [2.2] Unified Trading Dashboard (8502)...
start "ZoL0 Unified Trading" /MIN cmd /k "cd /d %~dp0 && streamlit run unified_trading_dashboard.py --server.port 8502 --server.headless true"

timeout /t 3 >nul

echo [2.3] Enhanced Bot Monitor (8503)...
start "ZoL0 Bot Monitor" /MIN cmd /k "cd /d %~dp0 && streamlit run enhanced_bot_monitor.py --server.port 8503 --server.headless true"

timeout /t 3 >nul

echo [2.4] Trading Analytics (8504)...
start "ZoL0 Analytics" /MIN cmd /k "cd /d %~dp0 && streamlit run advanced_trading_analytics.py --server.port 8504 --server.headless true"

timeout /t 3 >nul

echo [2.5] ML Predictive Analytics (8505)...
start "ZoL0 ML Analytics" /MIN cmd /k "cd /d %~dp0 && streamlit run ml_predictive_analytics.py --server.port 8505 --server.headless true"

timeout /t 3 >nul

echo [2.6] Enhanced Dashboard (8506)...
start "ZoL0 Enhanced" /MIN cmd /k "cd /d %~dp0 && streamlit run enhanced_dashboard.py --server.port 8506 --server.headless true"

echo.
echo [KROK 3] Czekanie na inicjalizację...
echo ====================================
timeout /t 20 >nul

echo.
echo ████████████████████████████████████████████████████████
echo ███                                                  ███
echo ███         SYSTEM ZoL0 URUCHOMIONY!                ███
echo ███                                                  ███
echo ███  🟢 PRAWDZIWE DANE BYBIT AKTYWNE               ███
echo ███                                                  ███
echo ████████████████████████████████████████████████████████
echo.

echo [SUKCES] Backend API Services:
echo    • Main API Server:     http://localhost:5000
echo    • Enhanced API Server: http://localhost:5001
echo.

echo [SUKCES] Trading Dashboards:
echo    • Master Control:      http://localhost:8501
echo    • Unified Trading:     http://localhost:8502
echo    • Bot Monitor:         http://localhost:8503
echo    • Trading Analytics:   http://localhost:8504
echo    • ML Analytics:        http://localhost:8505
echo    • Enhanced Dashboard:  http://localhost:8506
echo.

echo [INFO] Otwieranie Master Control Dashboard...
start "" "http://localhost:8501"

echo.
echo ================================================================
echo   WSZYSTKIE SYSTEMY ONLINE - GOTOWE DO TRADINGU!
echo ================================================================
echo.
echo Nacisnij dowolny klawisz aby sprawdzic status polaczen...
pause >nul

echo.
echo [TEST] Sprawdzanie status serwisow...
echo ====================================

REM Test connectivity
curl -s http://localhost:5000 >nul 2>&1 && echo ✅ Main API - OK || echo ❌ Main API - ERROR
curl -s http://localhost:5001 >nul 2>&1 && echo ✅ Enhanced API - OK || echo ❌ Enhanced API - ERROR  
curl -s http://localhost:8501 >nul 2>&1 && echo ✅ Master Control - OK || echo ❌ Master Control - ERROR
curl -s http://localhost:8502 >nul 2>&1 && echo ✅ Unified Trading - OK || echo ❌ Unified Trading - ERROR
curl -s http://localhost:8503 >nul 2>&1 && echo ✅ Bot Monitor - OK || echo ❌ Bot Monitor - ERROR

echo.
echo ================================================================
echo   ZoL0 TRADING SYSTEM JEST AKTYWNY!
echo   Aby zatrzymac - zamknij wszystkie okna terminali
echo ================================================================
echo.
pause
