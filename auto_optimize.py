import logging
import sys
import asyncio
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette_exporter import PrometheusMiddleware, handle_metrics
from typing import Literal, Optional
import structlog
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
import sentry_sdk

from engine.backtest_engine import BacktestEngine

# --- Sentry Initialization ---
sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN", ""),
    traces_sample_rate=1.0,
    environment=os.environ.get("SENTRY_ENV", "development"),
)

# --- Structlog Configuration ---
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger("auto_optimize")

# --- OpenTelemetry Tracing ---
tracer_provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "zol0-auto-optimizer"}))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
tracer_provider.add_span_processor(span_processor)

FastAPIInstrumentor.instrument_app(optimize_api)
LoggingInstrumentor().instrument(set_logging_format=True)
optimize_api.add_middleware(SentryAsgiMiddleware)

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

optimize_api = FastAPI(title="ZoL0 Auto Optimizer", version="2.0")
optimize_api.add_middleware(PrometheusMiddleware)
optimize_api.add_route("/metrics", handle_metrics)

# --- Advanced Optimization Logic ---
def run_optimization(strategy_name, n_trials=20, dynamic_tp_sl=False, initial_capital=100000, method: str = "bayesian", progress_id: Optional[str] = None):
    """Run optimization with selectable algorithm and progress tracking."""
    engine = BacktestEngine(initial_capital=initial_capital)
    logger.info("run_optimization_start", strategy=strategy_name, n_trials=n_trials, method=method)
    # Choose algorithm
    if method == "bayesian":
        result = engine.auto_optimize_strategy(strategy_name, n_trials=n_trials, dynamic_tp_sl=dynamic_tp_sl, method="bayesian", progress_id=progress_id)
    elif method == "genetic":
        result = engine.auto_optimize_strategy(strategy_name, n_trials=n_trials, dynamic_tp_sl=dynamic_tp_sl, method="genetic", progress_id=progress_id)
    elif method == "ensemble":
        result = engine.auto_optimize_strategy(strategy_name, n_trials=n_trials, dynamic_tp_sl=dynamic_tp_sl, method="ensemble", progress_id=progress_id)
    else:
        result = engine.auto_optimize_strategy(strategy_name, n_trials=n_trials, dynamic_tp_sl=dynamic_tp_sl, method="random", progress_id=progress_id)
    logger.info("run_optimization_complete", strategy=strategy_name, result=result)
    return result

class OptimizeQuery(BaseModel):
    strategy_name: str
    n_trials: int = 20
    dynamic_tp_sl: bool = False
    initial_capital: float = 100000
    method: Literal["bayesian", "genetic", "ensemble", "random"] = "bayesian"
    progress_id: Optional[str] = None

class BatchOptimizeQuery(BaseModel):
    queries: list[OptimizeQuery]

# --- Real-time Progress Store (simple in-memory for now) ---
import threading
progress_store = {}
progress_lock = threading.Lock()

def update_progress(progress_id, value):
    with progress_lock:
        progress_store[progress_id] = value

def get_progress(progress_id):
    with progress_lock:
        return progress_store.get(progress_id, 0)

# --- API Endpoints ---
@optimize_api.get("/")
async def root():
    return {"status": "ok", "service": "ZoL0 Auto Optimizer", "version": "2.0"}

@optimize_api.get("/api/health")
async def api_health():
    return {"status": "ok", "service": "ZoL0 Auto Optimizer", "version": "2.0"}

@optimize_api.post("/api/optimize", dependencies=[Depends(get_api_key)])
async def api_optimize(req: OptimizeQuery, role: str = Depends(get_api_key)):
    loop = asyncio.get_event_loop()
    progress_id = req.progress_id or f"{req.strategy_name}-{int(time.time())}"
    def progress_callback(val):
        update_progress(progress_id, val)
    result = await loop.run_in_executor(None, run_optimization, req.strategy_name, req.n_trials, req.dynamic_tp_sl, req.initial_capital, req.method, progress_id)
    update_progress(progress_id, 100)
    return {"result": result, "progress_id": progress_id}

@optimize_api.get("/api/optimize/progress/{progress_id}", dependencies=[Depends(get_api_key)])
async def api_optimize_progress(progress_id: str, role: str = Depends(get_api_key)):
    progress = get_progress(progress_id)
    return {"progress": progress}

@optimize_api.post("/api/optimize/batch", dependencies=[Depends(get_api_key)])
async def api_optimize_batch(req: BatchOptimizeQuery, role: str = Depends(get_api_key)):
    results = []
    loop = asyncio.get_event_loop()
    for q in req.queries:
        result = await loop.run_in_executor(None, run_optimization, q.strategy_name, q.n_trials, q.dynamic_tp_sl, q.initial_capital)
        results.append({"strategy": q.strategy_name, "result": result})
    return {"results": results}

@optimize_api.get("/api/analytics", dependencies=[Depends(get_api_key)])
async def api_analytics(role: str = Depends(get_api_key)):
    # Placeholder for analytics
    return {"status": "analytics stub"}

@optimize_api.get("/api/export/prometheus", dependencies=[Depends(get_api_key)])
async def api_export_prometheus(role: str = Depends(get_api_key)):
    # Placeholder for Prometheus export
    return PlainTextResponse("# HELP optimizer_runs Number of optimizer runs\noptimizer_runs 1", media_type="text/plain")

@optimize_api.get("/api/report", dependencies=[Depends(get_api_key)])
async def api_report(role: str = Depends(get_api_key)):
    # Placeholder for PDF/CSV/email integration
    return {"status": "report generated (stub)"}

@optimize_api.get("/api/recommendations", dependencies=[Depends(get_api_key)])
async def api_recommendations(role: str = Depends(get_api_key)):
    # Placeholder for recommendations
    return {"recommendations": ["Try more trials for better optimization."]}

@optimize_api.get("/api/premium/score", dependencies=[Depends(get_api_key)])
async def api_premium_score(role: str = Depends(get_api_key)):
    # Placeholder for premium scoring
    return {"score": 100}

@optimize_api.get("/api/saas/tenant/{tenant_id}/report", dependencies=[Depends(get_api_key)])
async def api_saas_tenant_report(tenant_id: str, role: str = Depends(get_api_key)):
    # Multi-tenant stub
    return {"tenant_id": tenant_id, "report": "stub"}

@optimize_api.get("/api/partner/webhook", dependencies=[Depends(get_api_key)])
async def api_partner_webhook(payload: dict, role: str = Depends(get_api_key)):
    return {"status": "received", "payload": payload}

@optimize_api.get("/api/test/edge-case")
async def api_edge_case():
    try:
        raise RuntimeError("Simulated optimizer edge-case error")
    except Exception as e:
        return {"edge_case": str(e)}

@optimize_api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

# --- CI/CD test suite ---
import unittest
class TestAutoOptimizeAPI(unittest.TestCase):
    def test_optimize(self):
        result = run_optimization("Momentum", n_trials=1)
        assert "best_params" in result

if __name__ == "__main__":
    import sys
    if "test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        uvicorn.run("auto_optimize:optimize_api", host="0.0.0.0", port=8507, reload=True)
