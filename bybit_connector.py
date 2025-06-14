import logging
import os

import httpx
import pandas as pd
import tenacity
import asyncio
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette_exporter import PrometheusMiddleware, handle_metrics

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


import time
import hmac
import hashlib
import base64
import json

class BybitConnector:
    BASE_URL = "https://api.bybit.com"

    def __init__(self, api_key=None, api_secret=None, testnet=True, session=None):
        self.api_key = api_key or os.getenv("BYBIT_API_KEY")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET")
        self.base_url = self.BASE_URL
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"

    def _sign(self, params: dict) -> str:
        # Bybit v5 signature (API key, timestamp, recvWindow, sign)
        param_str = ""
        if params:
            param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        timestamp = str(int(time.time() * 1000))
        to_sign = timestamp + self.api_key + param_str
        sign = hmac.new(self.api_secret.encode(), to_sign.encode(), hashlib.sha256).hexdigest()
        return timestamp, sign

    async def fetch_ohlcv_async(self, symbol, interval="1h", limit=1000):
        url = f"{self.base_url}/v5/market/kline"
        params = {"symbol": symbol, "interval": interval, "limit": limit, "category": "linear"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            klines = (
                data["result"]["list"]
                if "result" in data and "list" in data["result"]
                else []
            )
            df = pd.DataFrame(klines)
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df[0], unit="s")
                df.columns = [
                    "openTime", "open", "high", "low", "close", "volume", "turnover"
                ]
            return (
                df[["timestamp", "open", "high", "low", "close", "volume"]]
                if not df.empty
                else df
            )

    async def fetch_ohlcv(self, symbol, interval="1h", limit=1000):
        return await self.fetch_ohlcv_async(symbol, interval, limit)

    async def place_order(self, symbol, side, qty, price=None, order_type="Market"):
        url = f"{self.base_url}/v5/order/create"
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": order_type,
            "qty": qty,
        }
        if price:
            params["price"] = price
        timestamp, sign = self._sign(params)
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": sign,
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, headers=headers, json=params)
            resp.raise_for_status()
            return resp.json()

    async def get_balance(self, coin="USDT"):
        url = f"{self.base_url}/v5/account/wallet-balance"
        params = {"accountType": "UNIFIED", "coin": coin}
        timestamp, sign = self._sign(params)
        headers = {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": sign,
        }
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            return resp.json()

    async def get_wallet_balance(self, coin="USDT"):
        return await self.get_balance(coin)

    async def get_server_time(self):
        url = f"{self.base_url}/v5/market/time"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def get_ticker(self, symbol):
        url = f"{self.base_url}/v5/market/tickers"
        params = {"symbol": symbol, "category": "linear"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

    @property
    def _use_new_architecture(self):
        return True

    def set_production(self):
        self.base_url = self.BASE_URL

# CI/CD: Zautomatyzowane testy edge-case i workflow wdrożone w .github/workflows/ci-cd.yml

API_KEYS = {
    "admin-key": "admin",
    "bybit-key": "bybit",
    "partner-key": "partner",
    "premium-key": "premium",
    "saas-key": "saas"
}
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)

def get_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key in API_KEYS:
        return API_KEYS[api_key]
    raise HTTPException(status_code=401, detail="Invalid or missing API key")

bybit_api = FastAPI(title="ZoL0 Bybit Connector API", version="2.0")
bybit_api.add_middleware(PrometheusMiddleware)
bybit_api.add_route("/metrics", handle_metrics)

class OHLCVQuery(BaseModel):
    symbol: str
    interval: str = "1h"
    limit: int = 1000

class OrderQuery(BaseModel):
    symbol: str
    side: str
    qty: float
    price: float = None
    order_type: str = "Market"

class BatchOHLCVQuery(BaseModel):
    queries: list[OHLCVQuery]

@bybit_api.get("/")
async def root():
    return {"status": "ok", "service": "ZoL0 Bybit Connector API", "version": "2.0"}

@bybit_api.get("/api/health")
async def api_health():
    return {"status": "ok", "service": "ZoL0 Bybit Connector API", "version": "2.0"}

# --- Global connector instance ---
bybit_connector = BybitConnector()

@bybit_api.post("/api/ohlcv", dependencies=[Depends(get_api_key)])
async def api_ohlcv(req: OHLCVQuery, role: str = Depends(get_api_key)):
    df = await bybit_connector.fetch_ohlcv(req.symbol, req.interval, req.limit)
    return df.to_dict(orient="records") if not df.empty else []

@bybit_api.post("/api/ohlcv/async", dependencies=[Depends(get_api_key)])
async def api_ohlcv_async(req: OHLCVQuery, role: str = Depends(get_api_key)):
    df = await bybit_connector.fetch_ohlcv_async(req.symbol, req.interval, req.limit)
    return df.to_dict(orient="records") if not df.empty else []

@bybit_api.post("/api/ohlcv/batch", dependencies=[Depends(get_api_key)])
async def api_ohlcv_batch(req: BatchOHLCVQuery, role: str = Depends(get_api_key)):
    results = []
    for q in req.queries:
        df = await bybit_connector.fetch_ohlcv(q.symbol, q.interval, q.limit)
        results.append({"symbol": q.symbol, "data": df.to_dict(orient="records") if not df.empty else []})
    return {"results": results}

@bybit_api.post("/api/order", dependencies=[Depends(get_api_key)])
async def api_order(req: OrderQuery, role: str = Depends(get_api_key)):
    result = await bybit_connector.place_order(req.symbol, req.side, req.qty, req.price, req.order_type)
    return {"result": result}

@bybit_api.get("/api/balance", dependencies=[Depends(get_api_key)])
async def api_balance(role: str = Depends(get_api_key)):
    result = await bybit_connector.get_balance()
    return {"result": result}

@bybit_api.get("/api/wallet_balance", dependencies=[Depends(get_api_key)])
async def api_wallet_balance(coin: str = "USDT", role: str = Depends(get_api_key)):
    result = await bybit_connector.get_wallet_balance(coin)
    return {"result": result}

@bybit_api.get("/api/server_time", dependencies=[Depends(get_api_key)])
async def api_server_time(role: str = Depends(get_api_key)):
    result = await bybit_connector.get_server_time()
    return {"result": result}

@bybit_api.get("/api/ticker", dependencies=[Depends(get_api_key)])
async def api_ticker(symbol: str, role: str = Depends(get_api_key)):
    result = await bybit_connector.get_ticker(symbol)
    return {"result": result}

@bybit_api.get("/api/analytics", dependencies=[Depends(get_api_key)])
async def api_analytics(role: str = Depends(get_api_key)):
    # Placeholder for analytics
    return {"status": "analytics stub"}

@bybit_api.get("/api/export/prometheus", dependencies=[Depends(get_api_key)])
async def api_export_prometheus(role: str = Depends(get_api_key)):
    # Placeholder for Prometheus export
    return PlainTextResponse("# HELP bybit_connector_requests Number of requests\nbybit_connector_requests 1", media_type="text/plain")

@bybit_api.get("/api/report", dependencies=[Depends(get_api_key)])
async def api_report(role: str = Depends(get_api_key)):
    # Placeholder for PDF/CSV/email integration
    return {"status": "report generated (stub)"}

@bybit_api.get("/api/recommendations", dependencies=[Depends(get_api_key)])
async def api_recommendations(role: str = Depends(get_api_key)):
    # Placeholder for recommendations
    return {"recommendations": ["Monitor API rate limits and optimize batch requests."]}

@bybit_api.get("/api/premium/score", dependencies=[Depends(get_api_key)])
async def api_premium_score(role: str = Depends(get_api_key)):
    # Placeholder for premium scoring
    return {"score": 100}

@bybit_api.get("/api/saas/tenant/{tenant_id}/report", dependencies=[Depends(get_api_key)])
async def api_saas_tenant_report(tenant_id: str, role: str = Depends(get_api_key)):
    # Multi-tenant stub
    return {"tenant_id": tenant_id, "report": "stub"}

@bybit_api.get("/api/partner/webhook", dependencies=[Depends(get_api_key)])
async def api_partner_webhook(payload: dict, role: str = Depends(get_api_key)):
    return {"status": "received", "payload": payload}

@bybit_api.get("/api/test/edge-case")
async def api_edge_case():
    try:
        raise RuntimeError("Simulated bybit connector edge-case error")
    except Exception as e:
        return {"edge_case": str(e)}

@bybit_api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

# --- CI/CD test suite ---
import unittest
class TestBybitConnectorAPI(unittest.IsolatedAsyncioTestCase):
    async def test_server_time(self):
        result = await bybit_connector.get_server_time()
        assert "retCode" in result or "result" in result

if __name__ == "__main__":
    import sys
    if "test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        uvicorn.run("bybit_connector:bybit_api", host="0.0.0.0", port=8509, reload=True)
