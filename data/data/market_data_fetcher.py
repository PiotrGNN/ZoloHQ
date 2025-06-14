"""
Dummy MarketDataFetcher for compatibility.
Replace with real implementation for production use.
"""
import pandas as pd

class MarketDataFetcher:
    @staticmethod
    def get_historical_data(symbol, interval, limit):
        # Return a dummy DataFrame for now
        dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq="1H")
        return pd.DataFrame({
            "timestamp": dates,
            "open": [1.0]*limit,
            "high": [1.1]*limit,
            "low": [0.9]*limit,
            "close": [1.0]*limit,
            "volume": [100]*limit,
        }).set_index("timestamp")
