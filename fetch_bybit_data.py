import sys

sys.path.append("ZoL0-master")
import pandas as pd

from data.execution.bybit_connector import BybitConnector

# Ustaw parametry API (możesz pobrać z .env lub ustawić tu)
api_key = None  # lub os.getenv('BYBIT_API_KEY')
api_secret = None  # lub os.getenv('BYBIT_API_SECRET')

symbol = "BTCUSDT"
interval = "60"  # Bybit API expects "60" for 1h, not "1h"
limit = 500  # liczba świec

connector = BybitConnector(api_key=api_key, api_secret=api_secret)
klines = connector.get_klines(symbol=symbol, interval=interval, limit=limit)

if "result" in klines and "list" in klines["result"]:
    df = pd.DataFrame(klines["result"]["list"])
    print("Podgląd danych zwróconych przez Bybit API:")
    print(df.head())
    # Bybit API: [timestamp, open, high, low, close, volume, turnover]
    df.columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
    df["close"] = df["close"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df[["timestamp", "close"]]
    df.to_csv("data/real_bybit_data.csv", index=False)
    print("Dane zapisane do data/real_bybit_data.csv")
else:
    print("Błąd pobierania danych z Bybit:", klines)
