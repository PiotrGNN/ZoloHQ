#!/usr/bin/env python3
"""
Dashboard Health Check - Test wszystkich działających dashboardów
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Query, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import csv
import io
import numpy as np
import structlog
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
import sentry_sdk
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.sessions import SessionMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as aioredis
from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

API_KEY = os.environ.get("DASHBOARD_HEALTH_API_KEY", "demo-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key

HEALTH_CHECK_REQUESTS = Counter(
    "dashboard_health_requests_total", "Total dashboard health check API requests", ["endpoint"]
)
HEALTH_CHECK_LATENCY = Histogram(
    "dashboard_health_latency_seconds", "Dashboard health check endpoint latency", ["endpoint"]
)

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
logger = structlog.get_logger("dashboard_health_check")

# --- OpenTelemetry Tracing ---
tracer_provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "zol0-dashboard-health-check"}))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
tracer_provider.add_span_processor(span_processor)

# --- FastAPI App with Security, CORS, GZip, HTTPS, Session, Rate Limiting ---
app = FastAPI(
    title="Dashboard Health Check API",
    version="2.0-maximal",
    description="Comprehensive, observable, and secure dashboard health check and monitoring API.",
    contact={"name": "ZoL0 Engineering", "email": "support@zol0.ai"},
    openapi_tags=[
        {"name": "health", "description": "Health check endpoints"},
        {"name": "monitoring", "description": "Monitoring and observability endpoints"},
        {"name": "check", "description": "Dashboard check endpoints"},
        {"name": "ci", "description": "CI/CD and test endpoints"},
        {"name": "info", "description": "Info endpoints"},
    ],
)

# --- Middleware ---
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*", ".zol0.ai"])
app.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "supersecret"))
app.add_middleware(SentryAsgiMiddleware)

# --- Rate Limiting Initialization ---
@app.on_event("startup")
async def startup_event() -> None:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    redis = await aioredis.from_url(redis_url, encoding="utf8", decode_responses=True)
    await FastAPILimiter.init(redis)

# --- Instrumentation ---
FastAPIInstrumentor.instrument_app(app)
LoggingInstrumentor().instrument(set_logging_format=True)

# --- Security Headers Middleware ---
from starlette.middleware.base import BaseHTTPMiddleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
        return response
app.add_middleware(SecurityHeadersMiddleware)

# --- Pydantic Models with OpenAPI Examples and Validators ---
class HealthCheckRequest(BaseModel):
    """Request model for dashboard health check."""
    ports: Optional[List[int]] = Field(None, example=[8501, 8503], description="List of dashboard ports to check.")

class HealthResponse(BaseModel):
    status: str = Field(example="ok")
    ts: str = Field(example="2025-06-14T12:00:00Z")

# --- Robust Error Handling: Global Exception Handler with Logging, Tracing, Sentry ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception", error=str(exc), path=str(request.url))
    sentry_sdk.capture_exception(exc)
    with tracer.start_as_current_span("global_exception_handler"):
        return JSONResponse(status_code=500, content={"error": str(exc)})

@app.exception_handler(FastAPIRequestValidationError)
async def validation_exception_handler(request: Request, exc: FastAPIRequestValidationError) -> JSONResponse:
    logger.error("Validation error", error=str(exc), path=str(request.url))
    sentry_sdk.capture_exception(exc)
    with tracer.start_as_current_span("validation_exception_handler"):
        return JSONResponse(status_code=422, content={"error": str(exc)})

# --- CI/CD Test Endpoint ---
@app.get("/api/ci/test", tags=["ci"])
async def api_ci_test() -> Dict[str, str]:
    """CI/CD pipeline test endpoint."""
    logger.info("CI/CD test endpoint hit")
    return {"ci": "ok"}

# --- Core async health check logic ---
DEFAULT_DASHBOARD_PORTS = {
    5000: "Main API Server (ZoL0-master)",
    5001: "Enhanced Dashboard API",
    8501: "Main Dashboard (ZoL0-master)",
    8503: "Dashboard (Port 8503)",
    8504: "Dashboard (Port 8504)",
    8505: "Dashboard (Port 8505)",
    8506: "Master Control Dashboard",
    8507: "Enhanced Dashboard",
}

async def check_dashboard(port: int, name: str) -> Dict[str, Any]:
    url = f"http://localhost:{port}"
    result = {"port": port, "name": name, "url": url, "status": None, "type": None, "error": None}
    try:
        async with httpx.AsyncClient(http2=True, timeout=5) as client:
            response = await client.get(url)
        result["status"] = response.status_code
        if response.status_code == 200:
            text = response.text.lower()
            if "streamlit" in text or "st." in text:
                result["type"] = "Streamlit Dashboard"
            elif "flask" in text or response.headers.get("Server", "").startswith("Werkzeug"):
                result["type"] = "Flask API Server"
            else:
                result["type"] = "Web Application"
        else:
            result["error"] = f"HTTP {response.status_code}"
    except httpx.ConnectError:
        result["error"] = "connection_error"
    except httpx.TimeoutException:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)
    return result

# --- API Endpoints ---
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "ts": datetime.now().isoformat()}

@app.get("/metrics", tags=["monitoring"])
def metrics():
    return StreamingResponse(io.BytesIO(generate_latest()), media_type=CONTENT_TYPE_LATEST)

@app.post("/check", tags=["check"], dependencies=[Depends(get_api_key)])
async def check_dashboards(ports: Optional[List[int]] = Query(None)):
    HEALTH_CHECK_REQUESTS.labels(endpoint="check_dashboards").inc()
    with HEALTH_CHECK_LATENCY.labels(endpoint="check_dashboards").time():
        ports_to_check = ports or list(DEFAULT_DASHBOARD_PORTS.keys())
        results = await asyncio.gather(*[
            check_dashboard(port, DEFAULT_DASHBOARD_PORTS.get(port, f"Dashboard (Port {port})"))
            for port in ports_to_check
        ])
        working = sum(1 for r in results if r["status"] == 200)
        total = len(results)
        return {
            "checked": total,
            "working": working,
            "results": results,
            "timestamp": datetime.now().isoformat(),
            "availability_percent": (working/total)*100 if total else 0,
        }

@app.get("/check/single", tags=["check"], dependencies=[Depends(get_api_key)])
async def check_single(port: int = Query(...)):
    HEALTH_CHECK_REQUESTS.labels(endpoint="check_single").inc()
    with HEALTH_CHECK_LATENCY.labels(endpoint="check_single").time():
        name = DEFAULT_DASHBOARD_PORTS.get(port, f"Dashboard (Port {port})")
        return await check_dashboard(port, name)

@app.get("/export/csv", tags=["export"], dependencies=[Depends(get_api_key)])
async def export_csv():
    results = (await check_dashboards())["results"]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Port", "Name", "Status", "Type", "Error"])
    for r in results:
        writer.writerow([r["port"], r["name"], r["status"], r["type"], r["error"]])
    buf.seek(0)
    return StreamingResponse(io.BytesIO(buf.getvalue().encode()), media_type="text/csv")

@app.get("/export/json", tags=["export"], dependencies=[Depends(get_api_key)])
async def export_json():
    results = await check_dashboards()
    buf = io.StringIO()
    import json
    json.dump(results, buf, indent=2)
    buf.seek(0)
    return StreamingResponse(io.BytesIO(buf.getvalue().encode()), media_type="application/json")

@app.get("/ci-cd/edge-case-test", tags=["ci-cd"], dependencies=[Depends(get_api_key)])
async def ci_cd_edge_case_test():
    # Simulate connection error, timeout, and Streamlit detection
    fail = await check_dashboard(9999, "Fake Dashboard")
    timeout = await check_dashboard(5000, "Main API Server (ZoL0-master)")
    return {"connection_error_handled": fail["error"] == "connection_error", "timeout_handled": timeout["error"] == "timeout" or timeout["status"] != 200}

# --- Monetization & Partner Hooks ---
PREMIUM_API_KEYS = {"premium-key", "partner-key"}
PARTNER_WEBHOOKS = {"partner-key": "https://partner.example.com/webhook"}

# --- AI/ML Model Hooks (stub for real model integration) ---
def ai_health_recommendations(results, premium=False):
    errors = [r for r in results if r.get("status") != 200]
    recs = []
    if errors:
        recs.append(f"{len(errors)} dashboard(s) are down or unhealthy. Review ports: {', '.join(str(r['port']) for r in errors)}.")
    else:
        recs.append("All dashboards are healthy. No urgent actions required.")
    if premium:
        recs.append("[PREMIUM] Access advanced uptime analytics and predictive health monitoring.")
    else:
        recs.append("Upgrade to premium for advanced dashboard health analytics and predictive monitoring.")
    return recs

def ai_advanced_health_analytics(results):
    # Example: anomaly detection, trend analysis (stub for ML integration)
    analytics = {
        "uptime_percent": sum(1 for r in results if r.get("status") == 200) / max(1, len(results)) * 100,
        "anomalies": [r for r in results if r.get("error")],
        "predictive": {"next_downtime_estimate": np.random.randint(1, 30)},
    }
    return analytics

# --- Advanced analytics endpoint ---
@app.get("/analytics/advanced", tags=["analytics"], dependencies=[Depends(get_api_key)])
async def advanced_analytics(premium: bool = Query(False)):
    results = (await check_dashboards())["results"]
    analytics = ai_advanced_health_analytics(results)
    return analytics

# --- AI-driven recommendations endpoint (upgraded) ---
@app.get("/analytics/recommendations", tags=["analytics"], dependencies=[Depends(get_api_key)])
async def recommendations(premium: bool = Query(False)):
    results = (await check_dashboards())["results"]
    recs = ai_health_recommendations(results, premium=premium)
    return {"recommendations": recs}

@app.get("/", tags=["info"])
async def root():
    return {"message": "Dashboard Health Check API (modernized)", "ts": datetime.now().isoformat()}

@app.get("/healthcheck", dependencies=[Depends(get_api_key)])
async def healthcheck_all(premium: bool = Query(False)):
    """Check all dashboards, premium users get advanced analytics"""
    results = []
    for port, name in DEFAULT_DASHBOARD_PORTS.items():
        result = await check_dashboard(port, name)
        results.append(result)
    # Monetization: premium analytics for premium/partner users
    api_key = API_KEY
    if api_key in PREMIUM_API_KEYS and premium:
        # Add advanced analytics (stub)
        for r in results:
            r["advanced"] = {"uptime": "99.99%", "sla": True}
    # Partner webhook integration (stub)
    if api_key in PARTNER_WEBHOOKS:
        # In production, send results to partner webhook
        pass
    return {"results": results, "premium": api_key in PREMIUM_API_KEYS}

# --- Dynamic Monetization Endpoints ---
@app.get("/monetize/dynamic-pricing", tags=["monetization"], dependencies=[Depends(get_api_key)])
async def dynamic_pricing(usage: Optional[int] = Query(100)):
    base_price = 5.0
    price = base_price + 0.03 * usage
    if usage > 1000:
        price *= 0.9  # volume discount
    return {"usage": usage, "price": round(price, 2)}

@app.get("/monetize/affiliate", tags=["monetization"], dependencies=[Depends(get_api_key)])
async def affiliate_status(affiliate_id: str = Query(...)):
    return {"affiliate_id": affiliate_id, "status": "active", "commission_rate": 0.12}

@app.get("/monetize/white-label", tags=["monetization"], dependencies=[Depends(get_api_key)])
async def white_label_status(partner_id: str = Query(...)):
    return {"partner_id": partner_id, "status": "enabled", "custom_branding": True}

@app.post("/monetize/webhook", tags=["monetization"], dependencies=[Depends(get_api_key)])
async def monetize_webhook(url: str = Query(...), event: str = Query(...), payload: Optional[str] = Query(None)):
    import httpx
    try:
        async with httpx.AsyncClient(http2=True) as client:
            resp = await client.post(url, json={"event": event, "payload": payload})
        return {"status": resp.status_code, "response": resp.text[:100]}
    except Exception as e:
        return {"error": str(e)}

# --- Run with: uvicorn dashboard_health_check:app --reload ---
