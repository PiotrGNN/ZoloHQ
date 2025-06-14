"""
BinanceConnector module for interacting with Binance API (sync and async).
Includes OpenTelemetry tracing, retry logic, and robust exception handling.
"""

# OpenTelemetry setup (one-time, idempotent)
import logging

import httpx
import pandas as pd
import tenacity
from binance.client import Client

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

if not hasattr(logging, "_otel_initialized_binance"):
    resource = Resource.create({"service.name": "binance-connector"})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    logging._otel_initialized_binance = True
tracer = trace.get_tracer("binance-connector")


class BinanceConnector:
    """
    Connector for Binance API with synchronous and asynchronous methods.
    Provides tracing, retries, and robust error handling.
    """

    def __init__(self, api_key, api_secret, testnet=True):
        """
        Initialize BinanceConnector.
        Args:
            api_key (str): Binance API key.
            api_secret (str): Binance API secret.
            testnet (bool): Use Binance testnet if True.
        """
        try:
            self.client = Client(api_key, api_secret)
            if testnet:
                self.client.API_URL = "https://testnet.binance.vision/api"
        except Exception as e:
            logging.error(f"Failed to initialize Binance client: {e}")
            raise

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
    def fetch_ohlcv(self, symbol, interval="1h", limit=1000):
        """
        Fetch OHLCV data for a symbol.
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            interval (str): Kline interval.
            limit (int): Number of data points.
        Returns:
            pd.DataFrame: OHLCV data.
        Raises:
            Exception: On API or data error.
        """
        with tracer.start_as_current_span("fetch_ohlcv"):
            try:
                klines = self.client.get_klines(
                    symbol=symbol, interval=interval, limit=limit
                )
                df = pd.DataFrame(
                    klines,
                    columns=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "qav",
                        "trades",
                        "taker_base_vol",
                        "taker_quote_vol",
                        "ignore",
                    ],
                )
                df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
                df = df.astype(
                    {
                        "open": float,
                        "high": float,
                        "low": float,
                        "close": float,
                        "volume": float,
                    }
                )
                return df[["timestamp", "open", "high", "low", "close", "volume"]]
            except Exception as e:
                logging.error(f"Error fetching OHLCV for {symbol}: {e}")
                raise

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
    def place_order(self, symbol, side, qty, price=None, order_type="MARKET"):
        """
        Place an order on Binance.
        Args:
            symbol (str): Trading pair symbol.
            side (str): 'BUY' or 'SELL'.
            qty (float): Quantity to trade.
            price (float, optional): Price for LIMIT orders.
            order_type (str): 'MARKET' or 'LIMIT'.
        Returns:
            dict: Order response.
        Raises:
            Exception: On API error.
        """
        with tracer.start_as_current_span("place_order"):
            try:
                return self.client.create_order(
                    symbol=symbol, side=side, type=order_type, quantity=qty, price=price
                )
            except Exception as e:
                logging.error(f"Error placing order for {symbol}: {e}")
                raise

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
    def get_balance(self, asset="USDT"):
        """
        Get balance for a specific asset.
        Args:
            asset (str): Asset symbol (e.g., 'USDT').
        Returns:
            dict: Asset balance.
        Raises:
            Exception: On API error.
        """
        with tracer.start_as_current_span("get_balance"):
            try:
                return self.client.get_asset_balance(asset=asset)
            except Exception as e:
                logging.error(f"Error getting balance for {asset}: {e}")
                raise

    async def fetch_ohlcv_async(self, symbol, interval="1h", limit=1000):
        """
        Asynchronously fetch OHLCV data for a symbol.
        Args:
            symbol (str): Trading pair symbol.
            interval (str): Kline interval.
            limit (int): Number of data points.
        Returns:
            pd.DataFrame: OHLCV data.
        Raises:
            Exception: On network or data error.
        """
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                klines = resp.json()
                df = pd.DataFrame(
                    klines,
                    columns=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "qav",
                        "trades",
                        "taker_base_vol",
                        "taker_quote_vol",
                        "ignore",
                    ],
                )
                df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
                df = df.astype(
                    {
                        "open": float,
                        "high": float,
                        "low": float,
                        "close": float,
                        "volume": float,
                    }
                )
                return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logging.error(f"Async error fetching OHLCV for {symbol}: {e}")
            raise

# CI/CD: Zautomatyzowane testy edge-case i workflow wdrożone w .github/workflows/ci-cd.yml

# Edge-case test examples (to be expanded in test suite)
def _test_edge_cases():
    try:
        connector = BinanceConnector("badkey", "badsecret")
        connector.fetch_ohlcv("INVALIDPAIR")
    except Exception as e:
        print(f"Handled invalid symbol: {e}")
    try:
        connector = BinanceConnector("badkey", "badsecret")
        connector.get_balance("FAKEASSET")
    except Exception as e:
        print(f"Handled invalid asset: {e}")
