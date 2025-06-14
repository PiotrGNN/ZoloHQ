import pytest
import os
from binance_connector import BinanceConnector
import httpx
import tempfile
import asyncio

BINANCE_API_URL = os.getenv("BINANCE_API_URL", "http://localhost:8510")
API_KEY = os.getenv("BINANCE_API_KEY", "test")

@pytest.fixture
def connector():
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
    with pytest.raises(Exception):
        asyncio.run(c.fetch_ohlcv_async("INVALIDPAIR"))

def test_init_with_invalid_keys(monkeypatch):
    monkeypatch.setenv("BYBIT_API_KEY", "badkey")
    monkeypatch.setenv("BYBIT_API_SECRET", "badsecret")
    c = BinanceConnector(api_key=None, api_secret=None, testnet=True)
    assert c.client is not None

def test_db_permission_error():
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

@pytest.mark.asyncio
def test_api_ohlcv():
    async def inner():
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(f"{BINANCE_API_URL}/api/ohlcv", json={"symbol": "BTCUSDT", "interval": "1h", "limit": 10}, headers={"X-API-KEY": API_KEY})
                assert resp.status_code == 200
                data = resp.json()
                assert isinstance(data, list)
        except (httpx.ConnectError, httpx.RequestError, httpx.TimeoutException) as e:
            pytest.skip(f"Binance API not running or connection error: {e}")
    asyncio.run(inner())

@pytest.mark.asyncio
async def test_api_recommendations():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BINANCE_API_URL}/api/recommendations", headers={"X-API-KEY": API_KEY})
            assert resp.status_code == 200
            assert "recommendations" in resp.json()
    except (httpx.ConnectError, httpx.RequestError, httpx.TimeoutException) as e:
        pytest.skip(f"Binance API not running or connection error: {e}")

@pytest.mark.asyncio
async def test_api_analytics():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BINANCE_API_URL}/api/analytics", headers={"X-API-KEY": API_KEY})
            assert resp.status_code == 200
            assert "anomaly_score" in resp.json()
    except (httpx.ConnectError, httpx.RequestError, httpx.TimeoutException) as e:
        pytest.skip(f"Binance API not running or connection error: {e}")

@pytest.mark.asyncio
async def test_api_audit_trail():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BINANCE_API_URL}/api/audit/trail", headers={"X-API-KEY": API_KEY})
            assert resp.status_code == 200
            assert "audit_trail" in resp.json()
    except (httpx.ConnectError, httpx.RequestError, httpx.TimeoutException) as e:
        pytest.skip(f"Binance API not running or connection error: {e}")

@pytest.mark.asyncio
async def test_api_compliance_status():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BINANCE_API_URL}/api/compliance/status", headers={"X-API-KEY": API_KEY})
            assert resp.status_code == 200
            assert "compliance" in resp.json()
    except (httpx.ConnectError, httpx.RequestError, httpx.TimeoutException) as e:
        pytest.skip(f"Binance API not running or connection error: {e}")

@pytest.mark.asyncio
async def test_api_edge_case():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BINANCE_API_URL}/api/test/edge-case")
            assert resp.status_code == 200
            assert "edge_case" in resp.json()
    except (httpx.ConnectError, httpx.RequestError, httpx.TimeoutException) as e:
        pytest.skip(f"Binance API not running or connection error: {e}")
