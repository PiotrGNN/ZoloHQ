@echo off
echo.
echo ================================
echo LAUNCHING REAL DATA DASHBOARD
echo ================================
echo.
echo Environment: PRODUCTION
echo Data Source: Bybit Production API
echo Authentication: VERIFIED
echo.
echo Dashboard will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop
echo ================================
echo.

cd /d "%~dp0"
python -m streamlit run unified_trading_dashboard.py --server.port 8501
pause
