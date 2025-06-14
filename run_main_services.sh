#!/bin/bash
# Skrypt do aktywacji środowiska głównego i uruchomienia serwisów
if [ ! -d venv-main ]; then
  python3 -m venv venv-main
fi
source venv-main/bin/activate
pip install --upgrade pip
pip install -r requirements-main.txt
# Automatyczne uruchamianie głównych serwisów
if [ -f unified_trading_dashboard.py ]; then
  echo "Uruchamiam Unified Trading Dashboard (Streamlit)..."
  nohup venv-main/bin/streamlit run unified_trading_dashboard.py --server.port 8502 > unified_trading_dashboard.log 2>&1 &
fi
if [ -f enhanced_dashboard_api.py ]; then
  echo "Uruchamiam Enhanced Dashboard API..."
  nohup venv-main/bin/python enhanced_dashboard_api.py > enhanced_dashboard_api.log 2>&1 &
fi
if [ -f data_export_import_system.py ]; then
  echo "Uruchamiam Data Export/Import API..."
  nohup venv-main/bin/python data_export_import_system.py > data_export_import_system.log 2>&1 &
fi
if [ -f advanced_alert_management.py ]; then
  echo "Uruchamiam Advanced Alert Management..."
  nohup venv-main/bin/python advanced_alert_management.py > advanced_alert_management.log 2>&1 &
fi
if [ -f advanced_performance_monitor.py ]; then
  echo "Uruchamiam Advanced Performance Monitor..."
  nohup venv-main/bin/python advanced_performance_monitor.py > advanced_performance_monitor.log 2>&1 &
fi
if [ -f enhanced_bot_monitor.py ]; then
  echo "Uruchamiam Enhanced Bot Monitor (Streamlit)..."
  nohup venv-main/bin/streamlit run enhanced_bot_monitor.py --server.port 8507 > enhanced_bot_monitor.log 2>&1 &
fi
if [ -f dashboard_api.py ]; then
  echo "Uruchamiam Dashboard API..."
  nohup venv-main/bin/python dashboard_api.py > dashboard_api.log 2>&1 &
fi
wait
# deactivate
