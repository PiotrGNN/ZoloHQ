import pytest
pytest.skip("Test pominięty: Binance nie jest używany w tym projekcie (tylko Bybit)", allow_module_level=True)

import os
from binance_connector import BinanceConnector
import httpx
import tempfile

@pytest.fixture
def connector():
    # Use dummy keys for testnet
    return BinanceConnector(api_key="test", api_secret="test", testnet=True)

def test_fetch_ohlcv_invalid_symbol(connector):
    with pytest.raises(Exception):
        connector.fetch_ohlcv("INVALIDPAIR")

def test_get_balance_invalid_asset(connector):
    with pytest.raises(Exception):
        connector.get_balance("FAKEASSET")

def test_place_order_invalid_params(connector):
    with pytest.raises(Exception):
        connector.place_order("BTCUSDT", "BUY", -1)

@pytest.mark.asyncio
def test_fetch_ohlcv_async_invalid_symbol():
    c = BinanceConnector(api_key="test", api_secret="test", testnet=True)
    import asyncio
    with pytest.raises(Exception):
        asyncio.run(c.fetch_ohlcv_async("INVALIDPAIR"))

def test_init_with_invalid_keys(monkeypatch):
    monkeypatch.setenv("BYBIT_API_KEY", "badkey")
    monkeypatch.setenv("BYBIT_API_SECRET", "badsecret")
    c = BinanceConnector(api_key=None, api_secret=None, testnet=True)
    assert c.client is not None

def test_db_permission_error():
    # Simulate file permission error for fallback DB
    import tempfile, os, stat
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()
    os.chmod(temp_file.name, 0)
    c = BinanceConnector(api_key="test", api_secret="test", testnet=True)
    try:
        with pytest.raises(Exception):
            c.fetch_ohlcv("BTCUSDT")
    finally:
        os.chmod(temp_file.name, stat.S_IWRITE)
        os.unlink(temp_file.name)
