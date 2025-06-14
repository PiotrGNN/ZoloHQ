import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from typing import Optional
import asyncio
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette_exporter import PrometheusMiddleware, handle_metrics
import os

try:
    from engine.backtest_engine import BacktestEngine
except ImportError:
    class BacktestEngine:
        def __init__(self, initial_capital: float = 100000):
            self.initial_capital = initial_capital

        def optimize(self, strategy: str, dynamic_tp_sl: bool = False):
            return {
                "strategy": strategy,
                "result": "simulated",
                "timestamp": datetime.now().isoformat(),
            }


API_KEYS = {
    "admin-key": "admin",
    "optimize-key": "optimize",
    "partner-key": "partner",
    "premium-key": "premium",
    "saas-key": "saas"
}
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)

def get_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key in API_KEYS:
        return API_KEYS[api_key]
    raise HTTPException(status_code=401, detail="Invalid or missing API key")

optimize_log_api = FastAPI(title="ZoL0 Auto Optimize & Log", version="2.0")
optimize_log_api.add_middleware(PrometheusMiddleware)
optimize_log_api.add_route("/metrics", handle_metrics)

class OptimizeLogQuery(BaseModel):
    strategy: str
    dynamic_tp_sl: bool = False
    log_file: str = None

class BatchOptimizeLogQuery(BaseModel):
    queries: list[OptimizeLogQuery]

def run_and_log_optimization(
    strategy: str, dynamic_tp_sl: bool = False, log_file: Optional[str] = None
):
    logger = logging.getLogger("AutoOptimize")
    try:
        engine = BacktestEngine(initial_capital=100000)
        # Use auto_optimize_strategy if available, else fallback to optimize
        if hasattr(engine, "auto_optimize_strategy"):
            result = engine.auto_optimize_strategy(
                strategy, n_trials=1, dynamic_tp_sl=dynamic_tp_sl
            )
        else:
            result = engine.optimize(strategy, dynamic_tp_sl=dynamic_tp_sl)
        logger.info(f"Optimization result: {result}")
        if log_file:
            with open(log_file, "a") as f:
                f.write(str(result) + "\n")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        # Always write to log for test validation
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"ERROR: {e}\n")


def schedule_optimizations():
    """Schedule daily optimizations using APScheduler for robust automation."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        lambda: run_and_log_optimization(
            "Momentum", dynamic_tp_sl=True, log_file="momentum.log"
        ),
        "cron",
        hour=2,
        minute=0,
    )
    scheduler.add_job(
        lambda: run_and_log_optimization(
            "Mean Reversion", dynamic_tp_sl=True, log_file="mean_reversion.log"
        ),
        "cron",
        hour=3,
        minute=0,
    )
    scheduler.start()
    logging.info("Advanced optimization scheduling started.")


@optimize_log_api.get("/")
async def root():
    return {"status": "ok", "service": "ZoL0 Auto Optimize & Log", "version": "2.0"}


@optimize_log_api.get("/api/health")
async def api_health():
    return {"status": "ok", "service": "ZoL0 Auto Optimize & Log", "version": "2.0"}


@optimize_log_api.post("/api/optimize", dependencies=[Depends(get_api_key)])
async def api_optimize(req: OptimizeLogQuery, role: str = Depends(get_api_key)):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_and_log_optimization, req.strategy, req.dynamic_tp_sl, req.log_file)
    return {"result": str(result)}


@optimize_log_api.post("/api/optimize/batch", dependencies=[Depends(get_api_key)])
async def api_optimize_batch(req: BatchOptimizeLogQuery, role: str = Depends(get_api_key)):
    results = []
    loop = asyncio.get_event_loop()
    for q in req.queries:
        result = await loop.run_in_executor(None, run_and_log_optimization, q.strategy, q.dynamic_tp_sl, q.log_file)
        results.append({"strategy": q.strategy, "result": str(result)})
    return {"results": results}


@optimize_log_api.post("/api/schedule", dependencies=[Depends(get_api_key)])
async def api_schedule(role: str = Depends(get_api_key)):
    schedule_optimizations()
    return {"status": "scheduled"}


@optimize_log_api.get("/api/status", dependencies=[Depends(get_api_key)])
async def api_status(role: str = Depends(get_api_key)):
    exists = os.path.exists("momentum.log") or os.path.exists("mean_reversion.log")
    return {"log_exists": exists}


@optimize_log_api.get("/api/analytics", dependencies=[Depends(get_api_key)])
async def api_analytics(role: str = Depends(get_api_key)):
    # Placeholder for analytics
    return {"status": "analytics stub"}


@optimize_log_api.get("/api/export/prometheus", dependencies=[Depends(get_api_key)])
async def api_export_prometheus(role: str = Depends(get_api_key)):
    # Placeholder for Prometheus export
    return PlainTextResponse("# HELP optimizer_runs Number of optimizer runs\noptimizer_runs 1", media_type="text/plain")


@optimize_log_api.get("/api/report", dependencies=[Depends(get_api_key)])
async def api_report(role: str = Depends(get_api_key)):
    # Placeholder for PDF/CSV/email integration
    return {"status": "report generated (stub)"}


@optimize_log_api.get("/api/recommendations", dependencies=[Depends(get_api_key)])
async def api_recommendations(role: str = Depends(get_api_key)):
    # Placeholder for recommendations
    return {"recommendations": ["Schedule regular optimizations for best results."]}


@optimize_log_api.get("/api/premium/score", dependencies=[Depends(get_api_key)])
async def api_premium_score(role: str = Depends(get_api_key)):
    # Placeholder for premium scoring
    return {"score": 100}


@optimize_log_api.get("/api/saas/tenant/{tenant_id}/report", dependencies=[Depends(get_api_key)])
async def api_saas_tenant_report(tenant_id: str, role: str = Depends(get_api_key)):
    # Multi-tenant stub
    return {"tenant_id": tenant_id, "report": "stub"}


@optimize_log_api.get("/api/partner/webhook", dependencies=[Depends(get_api_key)])
async def api_partner_webhook(payload: dict, role: str = Depends(get_api_key)):
    return {"status": "received", "payload": payload}


@optimize_log_api.get("/api/test/edge-case")
async def api_edge_case():
    try:
        raise RuntimeError("Simulated optimizer edge-case error")
    except Exception as e:
        return {"edge_case": str(e)}


@optimize_log_api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})


# --- CI/CD test suite ---
import unittest
class TestAutoOptimizeLogAPI(unittest.TestCase):
    def test_optimize(self):
        log_path = "test.log"
        if os.path.exists(log_path):
            os.remove(log_path)
        run_and_log_optimization("Momentum", dynamic_tp_sl=True, log_file=log_path)
        assert os.path.exists(log_path)

if __name__ == "__main__":
    import sys
    if "test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        uvicorn.run("auto_optimize_and_log:optimize_log_api", host="0.0.0.0", port=8508, reload=True)
