# Placeholder for demo data generator
import numpy as np
import pandas as pd


def generate_demo_data(symbol: str, start: str = "2023-01-01", end: str = "2024-01-01"):
    dates = pd.date_range(start=start, end=end, freq="h")
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, len(dates))
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    df = pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            "low": [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            "close": prices,
            "volume": np.random.uniform(1000, 10000, len(dates)),
        }
    )
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df
