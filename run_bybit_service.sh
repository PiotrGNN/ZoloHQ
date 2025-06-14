#!/bin/bash
# Skrypt do aktywacji środowiska bybit i uruchomienia serwisu
if [ ! -d venv-bybit ]; then
  python3 -m venv venv-bybit
fi
source venv-bybit/bin/activate
pip install --upgrade pip
pip install -r requirements-bybit.txt
# Automatyczne uruchomienie serwisu wymagającego bybit
if [ -f bybit_connector.py ]; then
  echo "Uruchamiam bybit_connector.py..."
  nohup venv-bybit/bin/python bybit_connector.py > bybit_connector.log 2>&1 &
  echo "bybit_connector.py uruchomiony w tle (log: bybit_connector.log)"
else
  echo "Plik bybit_connector.py nie istnieje. Dodaj własny serwis do uruchomienia."
fi
# deactivate
