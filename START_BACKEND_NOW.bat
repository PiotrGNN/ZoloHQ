@echo off
echo Starting ZoL0 Backend Services...
echo ====================================

REM Set production environment
set BYBIT_PRODUCTION_CONFIRMED=true
set BYBIT_PRODUCTION_ENABLED=true

echo.
echo Starting Main API Server (Port 5000)...
start "Main API Server" /MIN cmd /k "cd /d C:\Users\piotr\Desktop\Zol0\ZoL0-master && python dashboard_api.py"

echo Waiting 5 seconds...
timeout /t 5 /nobreak >nul

echo.
echo Starting Enhanced API Server (Port 5001)...  
start "Enhanced API Server" /MIN cmd /k "cd /d C:\Users\piotr\Desktop\Zol0 && python enhanced_dashboard_api.py"

echo.
echo Waiting 10 seconds for services to initialize...
timeout /t 10 /nobreak >nul

echo.
echo Testing connectivity...
python -c "import requests; print('Main API:', requests.get('http://localhost:5000').status_code if requests.get('http://localhost:5000') else 'Not responding')" 2>nul
python -c "import requests; print('Enhanced API:', requests.get('http://localhost:5001').status_code if requests.get('http://localhost:5001') else 'Not responding')" 2>nul

echo.
echo Backend services started!
echo You can now run: python launch_all_dashboards.py
echo.
pause
