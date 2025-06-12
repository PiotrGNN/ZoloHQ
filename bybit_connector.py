import logging
import os

import bybit
import httpx
import pandas as pd
import tenacity

FILTER_KEYS = ["api_key", "api_secret", "API_KEY", "API_SECRET", "secret"]


# Helper to mask secrets in logs
class SecretFilter(logging.Filter):
    def filter(self, record):
        msg = str(record.getMessage())
        for key in FILTER_KEYS:
            if key in msg:
                msg = msg.replace(os.getenv(key.upper(), "***"), "***")
        record.msg = msg
        return True


# Apply filter to all loggers
for handler in logging.root.handlers:
    handler.addFilter(SecretFilter())


class BybitConnector:
    def __init__(self, api_key=None, api_secret=None, testnet=True, session=None):
        # Always load secrets from ENV if not provided
        self.api_key = api_key or os.getenv("BYBIT_API_KEY")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET")
        if session is not None:
            self.session = session
        else:
            self.session = bybit.bybit(
                test=testnet, api_key=self.api_key, api_secret=self.api_secret
            )

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
    def fetch_ohlcv(self, symbol, interval="1h", limit=1000):
        # symbol: e.g. 'BTCUSD'
        klines = self.session.Kline.Kline_get(
            symbol=symbol, interval=interval, limit=limit
        ).result()[0]["result"]
        df = pd.DataFrame(klines)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="s")
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    async def fetch_ohlcv_async(self, symbol, interval="1h", limit=1000):
        url = "https://api.bybit.com/v5/market/kline"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        headers = {"X-BAPI-API-KEY": self.api_key, "X-BAPI-SIGN": self.api_secret}
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            klines = (
                data["result"]["list"]
                if "result" in data and "list" in data["result"]
                else []
            )
            df = pd.DataFrame(klines)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["openTime"], unit="s")
            return (
                df[["timestamp", "open", "high", "low", "close", "volume"]]
                if not df.empty
                else df
            )

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
    def place_order(self, symbol, side, qty, price=None, order_type="Market"):
        return self.session.Order.Order_new(
            symbol=symbol, side=side, order_type=order_type, qty=qty, price=price
        ).result()

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
    def get_balance(self):
        return self.session.Wallet.Wallet_getBalance(coin="USDT").result()

    def get_wallet_balance(self, coin="USDT"):
        return self.session.Wallet.Wallet_getBalance(coin=coin).result()[0]

    def get_server_time(self):
        result = self.session.Common.Common_getTime().result()[0]
        # Bybit V5 returns 'timeSecond' and 'timeNano', not 'time'
        if "data" in result and "timeSecond" in result["data"]:
            # For compatibility with tests, add 'time' key
            result["data"]["time"] = int(result["data"]["timeSecond"])
        return result

    def get_ticker(self, symbol):
        return self.session.Market.Market_symbolInfo(symbol=symbol).result()[0]

    @property
    def _use_new_architecture(self):
        import os

        return os.environ.get("USE_NEW_ARCHITECTURE", "false").lower() in (
            "true",
            "1",
            "yes",
        )

    def set_production(self):
        self.session = bybit.bybit(
            test=False, api_key=self.session.api_key, api_secret=self.session.api_secret
        )

# CI/CD: Zautomatyzowane testy edge-case i workflow wdrożone w .github/workflows/ci-cd.yml
# (TODO usunięty po wdrożeniu automatyzacji)
