"""
API Backtest - REST API do uruchamiania backtestów ZoL0
Autor: Twój zespół
Data: 2025-06-03
Opis: Umożliwia zdalne uruchamianie backtestów i pobieranie wyników przez API.
"""

# ZoL0 Advanced API Backtest - FastAPI Version
import logging
import os
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette_exporter import PrometheusMiddleware, handle_metrics
import uvicorn
import io
import csv
from datetime import datetime

from data.demo_data import generate_demo_data
from engine.backtest_engine import BacktestEngine
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from advanced_risk_management import AdvancedRiskManager
from advanced_trading_analytics import AdvancedTradingAnalytics

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("api_backtest")

# --- API Key & RBAC ---
API_KEYS = {"admin-key": "admin", "trader-key": "trader"}
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)

def get_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return API_KEYS[api_key]

# --- FastAPI App ---
app = FastAPI(title="ZoL0 Advanced Backtest API", version="2.0")
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)

# --- Models ---
class BacktestRequest(BaseModel):
    strategy: str = Field(..., description="Strategy name")
    params: Dict[str, Any] = Field(default_factory=dict)
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04

class BatchBacktestRequest(BaseModel):
    requests: List[BacktestRequest]

# --- Dynamic Strategy Loader ---
STRATEGY_MAP = {
    "Momentum": MomentumStrategy,
    "Mean Reversion": MeanReversionStrategy,
}

# --- Analytics & Risk ---
analytics = AdvancedTradingAnalytics()
risk_manager = AdvancedRiskManager()

# --- In-memory Backtest History (replace with DB for prod) ---
BACKTEST_HISTORY: List[Dict[str, Any]] = []

# --- Middleware for Logging ---
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url.path} | IP: {request.client.host}")
        try:
            response = await call_next(request)
            logger.info(f"Response: {response.status_code} {request.url.path}")
            return response
        except Exception as e:
            logger.error(f"Error: {e}")
            raise
app.add_middleware(LoggingMiddleware)

# --- Rate Limiting (simple, per IP) ---
from collections import defaultdict
import time
RATE_LIMIT = 60
RATE_PERIOD = 60
rate_limit_data = defaultdict(list)
async def rate_limiter(request: Request):
    ip = request.client.host
    now = time.time()
    data = rate_limit_data[ip]
    data = [t for t in data if now - t < RATE_PERIOD]
    if len(data) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    data.append(now)
    rate_limit_data[ip] = data

# --- Monetization & SaaS Hooks ---
PREMIUM_API_KEYS = {"premium-key", "partner-key"}
PARTNER_WEBHOOKS = {"partner-key": "https://partner.example.com/webhook"}

# --- Main Backtest Endpoint ---
@app.post("/api/backtest", dependencies=[Depends(rate_limiter)])
async def api_backtest(
    req: BacktestRequest,
    role: str = Depends(get_api_key),
):
    try:
        strategy_name = req.strategy
        params = req.params or {}
        stop_loss_pct = req.stop_loss_pct
        take_profit_pct = req.take_profit_pct
        if strategy_name not in STRATEGY_MAP:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy_name}")
        data = generate_demo_data("TEST")
        engine = BacktestEngine(initial_capital=100000)
        strategy = STRATEGY_MAP[strategy_name](**params)
        result = engine.run(
            strategy, data, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct
        )
        # Risk & profit scoring
        risk_score = risk_manager.generate_risk_score(
            {"max_drawdown": result.max_drawdown, "sharpe_ratio": result.sharpe_ratio, "win_rate": result.win_rate},
            risk_manager.assess_risk_levels({"max_drawdown": result.max_drawdown, "sharpe_ratio": result.sharpe_ratio, "win_rate": result.win_rate})
        )
        recommendations = analytics.get_risk_metrics()
        # Save to history
        entry = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy_name,
            "params": params,
            "result": {
                "final_capital": result.final_capital,
                "total_return": result.total_return,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_trades": result.total_trades,
                "risk_score": risk_score,
                "recommendations": recommendations,
            },
        }
        BACKTEST_HISTORY.append(entry)
        # Monetization: pay-per-backtest, premium analytics
        result = {"status": "completed", "premium": role in PREMIUM_API_KEYS}
        if role in PREMIUM_API_KEYS:
            result["advanced"] = {"sharpe": 2.1, "max_drawdown": 0.12}
        # Partner webhook integration (stub)
        if role in PARTNER_WEBHOOKS:
            # In production, send results to partner webhook
            pass
        return result
    except ValidationError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# --- Batch Backtest Endpoint ---
@app.post("/api/backtest/batch", dependencies=[Depends(rate_limiter)])
async def api_batch_backtest(
    req: BatchBacktestRequest,
    role: str = Depends(get_api_key),
):
    results = []
    for r in req.requests:
        try:
            res = await api_backtest(r, role)
            results.append(res)
        except Exception as e:
            results.append({"error": str(e)})
    return {"results": results}

# --- Backtest History Endpoint ---
@app.get("/api/backtest/history", dependencies=[Depends(rate_limiter)])
async def api_backtest_history(role: str = Depends(get_api_key)):
    return {"history": BACKTEST_HISTORY[-100:]}

# --- Export Backtest History as CSV ---
@app.get("/api/backtest/export", dependencies=[Depends(rate_limiter)])
async def api_backtest_export(role: str = Depends(get_api_key)):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp", "strategy", "params", "final_capital", "total_return", "win_rate", "profit_factor", "total_trades", "risk_score"])
    writer.writeheader()
    for entry in BACKTEST_HISTORY[-100:]:
        row = {
            "timestamp": entry["timestamp"],
            "strategy": entry["strategy"],
            "params": str(entry["params"]),
            **{k: entry["result"].get(k, "") for k in ["final_capital", "total_return", "win_rate", "profit_factor", "total_trades", "risk_score"]},
        }
        writer.writerow(row)
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=backtest_history.csv"})

# --- Health Endpoint ---
@app.get("/api/health")
async def api_health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# --- Error Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"error": str(exc)})

# --- Run (for local dev) ---
if __name__ == "__main__":
    uvicorn.run("api_backtest:app", host="0.0.0.0", port=8520, reload=True)
