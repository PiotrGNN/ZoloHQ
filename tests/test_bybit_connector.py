import pytest
pytest.skip("Testy Bybit wymagają realnego API lub mockowania HTTP. Pominięte w CI.", allow_module_level=True)

import os
from bybit_connector import BybitConnector
import httpx

@pytest.fixture
def connector():
    return BybitConnector(api_key="test", api_secret="test", testnet=True)

def test_fetch_ohlcv_invalid_symbol(connector):
    with pytest.raises(Exception):
        connector.fetch_ohlcv("INVALIDPAIR")

def test_get_balance_invalid(connector):
    with pytest.raises(Exception):
        connector.get_balance()

def test_place_order_invalid_params(connector):
    with pytest.raises(Exception):
        connector.place_order("BTCUSDT", "BUY", -1)

@pytest.mark.asyncio
def test_fetch_ohlcv_async_invalid_symbol():
    c = BybitConnector(api_key="test", api_secret="test", testnet=True)
    import asyncio
    with pytest.raises(Exception):
        asyncio.run(c.fetch_ohlcv_async("INVALIDPAIR"))

def test_set_production_switch():
    c = BybitConnector(api_key="test", api_secret="test", testnet=True)
    c.set_production()
    assert c.session is not None
