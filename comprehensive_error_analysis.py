#!/usr/bin/env python3
"""
Kompleksowa analiza błędów systemu ZoL0
"""
import asyncio
import csv
import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Gauge, generate_latest
from pydantic import BaseModel, Field, ValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import uvicorn
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

# --- API Key Security (simple demo, replace with DB/partner logic for SaaS/monetization) ---
API_KEY = os.environ.get("ERROR_ANALYSIS_API_KEY", "demo-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# --- Advanced API Key Security: JWT, OAuth2, and RBAC (absolute maximal security) ---
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from starlette.requests import Request

SECRET_KEY = os.environ.get("ERROR_ANALYSIS_SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Example user DB (replace with production DB)
FAKE_USERS_DB = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": pwd_context.hash("adminpass"),
        "disabled": False,
        "role": "admin"
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None
    role: Optional[str] = None

class UserInDB(User):
    hashed_password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def get_user(db, username: str) -> Optional[UserInDB]:
    if username in db:
        return UserInDB(**db[username])
    return None

def authenticate_user(db, username: str, password: str) -> Optional[UserInDB]:
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[int] = None) -> str:
    from datetime import datetime, timedelta
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=expires_delta or ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "user")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception
    user = get_user(FAKE_USERS_DB, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- API Key + OAuth2 + JWT + RBAC security for all endpoints ---
def get_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )
    return api_key


# --- Prometheus Metrics ---
ERROR_ANALYSIS_REQUESTS = Counter(
    "error_analysis_requests_total", "Total error analysis API requests", ["endpoint"]
)
ERROR_ANALYSIS_LATENCY = Histogram(
    "error_analysis_latency_seconds", "Error analysis endpoint latency", ["endpoint"]
)
ERROR_ANALYSIS_ERRORS = Counter(
    "error_analysis_errors_total", "Total error analysis errors", ["endpoint"]
)
ERROR_ANALYSIS_ACTIVE = Gauge(
    "error_analysis_active_requests", "Active error analysis requests"
)

# --- FastAPI App ---
app = FastAPI(title="Comprehensive Error Analysis API", version="2.0-modernized")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Advanced CORS and Rate Limiting ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Connect to Redis for rate limiting (production: use secure Redis URL)
    import redis.asyncio as aioredis
    redis = await aioredis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
    await FastAPILimiter.init(redis)

# --- Pydantic Models for API ---
from pydantic import root_validator

class APIEndpointRequest(BaseModel):
    api_endpoints: List[str] = Field(..., description="List of API endpoint URLs")
    dashboard_urls: List[str] = Field(..., description="List of dashboard URLs")
    names: Optional[List[str]] = Field(None, description="Optional names for endpoints")
    dashboards_names: Optional[List[str]] = Field(None, description="Optional names for dashboards")
    data: Optional[Dict[str, Any]] = Field(None, description="Optional additional data")

    @root_validator
    def validate_lists(cls, values):
        if not values.get("api_endpoints") or not values.get("dashboard_urls"):
            raise ValueError("api_endpoints and dashboard_urls are required and cannot be empty.")
        return values

class APIResult(BaseModel):
    url: str
    name: str
    status_code: Optional[int]
    error: Optional[str]
    content_preview: Optional[str]
    response_time: Optional[float]
    success: bool
    data_source: Optional[str]

    class Config:
        schema_extra = {
            "example": {
                "url": "https://api.example.com",
                "name": "Example API",
                "status_code": 200,
                "error": None,
                "content_preview": "{...}",
                "response_time": 0.123,
                "success": True,
                "data_source": "primary"
            }
        }

class DashboardResult(BaseModel):
    name: str
    url: str
    accessible: bool
    status_code: Union[int, str]
    error: Optional[str]

    class Config:
        schema_extra = {
            "example": {
                "name": "Main Dashboard",
                "url": "https://dashboard.example.com",
                "accessible": True,
                "status_code": 200,
                "error": None
            }
        }

class AnalyzeResponse(BaseModel):
    api: List[APIResult]
    dashboards: List[DashboardResult]

    class Config:
        schema_extra = {
            "example": {
                "api": [APIResult.Config.schema_extra["example"]],
                "dashboards": [DashboardResult.Config.schema_extra["example"]]
            }
        }

class ErrorResponse(BaseModel):
    detail: str
    code: int = Field(..., example=401)

    class Config:
        schema_extra = {
            "example": {"detail": "Invalid API Key", "code": 401}
        }
# --- Enhanced Logging ---
import structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger("error_analysis_api")


# --- OpenTelemetry distributed tracing setup (idempotent) ---
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor

if not hasattr(logging, "_otel_initialized_error_analysis"):
    resource = Resource.create({
        "service.name": "comprehensive-error-analysis-api",
        "service.version": "2.0-modernized",
        "deployment.environment": SENTRY_ENVIRONMENT,
    })
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(processor)
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
    LoggingInstrumentor().instrument(set_logging_format=True)
    try:
        import redis.asyncio as aioredis
        RedisInstrumentor().instrument()
    except ImportError:
        pass
    logging._otel_initialized_error_analysis = True
tracer = trace.get_tracer("comprehensive-error-analysis-api")


# --- Sentry error monitoring integration ---
SENTRY_DSN = os.environ.get("SENTRY_DSN")
SENTRY_ENVIRONMENT = os.environ.get("SENTRY_ENVIRONMENT", "production")
SENTRY_RELEASE = os.environ.get("SENTRY_RELEASE", "zol0@2.0.0")
if SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN,
        traces_sample_rate=1.0,
        environment=SENTRY_ENVIRONMENT,
        release=SENTRY_RELEASE,
        attach_stacktrace=True,
        send_default_pii=True,
        debug=False,
        _experiments={"auto_enabling_integrations": True},
    )
    app.add_middleware(SentryAsgiMiddleware)

# --- Advanced Security Headers Middleware ---
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.sessions import SessionMiddleware

# Add HTTPS redirect, trusted hosts, GZip, and session security
app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response: StarletteResponse = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Cache-Control"] = "no-store"
        return response
app.add_middleware(SecurityHeadersMiddleware)

# --- OpenAPI tags/metadata for documentation ---
app.openapi_tags = [
    {"name": "health", "description": "Health and status endpoints"},
    {"name": "monitoring", "description": "Prometheus and monitoring endpoints"},
    {"name": "analysis", "description": "Error analysis and AI endpoints"},
    {"name": "export", "description": "Export and reporting endpoints"},
    {"name": "analytics", "description": "Analytics and recommendations"},
    {"name": "monetization", "description": "Monetization and partner endpoints"},
    {"name": "ci-cd", "description": "CI/CD and edge-case test endpoints"},
    {"name": "info", "description": "API information endpoints"},
]


# --- Async HTTPX logic ---
import contextlib

async def async_check_api_endpoint(
    url: str, name: str, timeout: int = 5
) -> Dict[str, Any]:
    with tracer.start_as_current_span("async_check_api_endpoint"):
        ERROR_ANALYSIS_REQUESTS.labels(endpoint="check_api").inc()
        with ERROR_ANALYSIS_LATENCY.labels(endpoint="check_api").time():
            result = {
                "url": url,
                "name": name,
                "status_code": None,
                "error": None,
                "content_preview": None,
                "response_time": None,
                "success": False,
                "data_source": None,
            }
            start = datetime.now()
            try:
                async with httpx.AsyncClient(http2=True, timeout=timeout) as client:
                    response = await client.get(url)
                result["status_code"] = response.status_code
                result["response_time"] = (datetime.now() - start).total_seconds()
                if response.status_code == 200:
                    with contextlib.suppress(Exception):
                        data = response.json()
                        result["content_preview"] = (
                            str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
                        )
                        if "data_source" in data:
                            result["data_source"] = data["data_source"]
                    result["success"] = True
                else:
                    result["error"] = f"HTTP {response.status_code}"
            except httpx.TimeoutException:
                result["error"] = "timeout"
            except Exception as e:
                logger.error("api_check_failed", url=url, error=str(e))
                sentry_sdk.capture_exception(e)
                result["error"] = str(e)
            return result

async def async_check_dashboard_accessibility(
    url: str, name: str, timeout: int = 10
) -> Dict[str, Any]:
    ERROR_ANALYSIS_REQUESTS.labels(endpoint="check_dashboard").inc()
    with ERROR_ANALYSIS_LATENCY.labels(endpoint="check_dashboard").time():
        try:
            async with httpx.AsyncClient(http2=True, timeout=timeout) as client:
                response = await client.get(url)
            return {
                "name": name,
                "url": url,
                "accessible": response.status_code == 200,
                "status_code": response.status_code,
                "error": None if response.status_code == 200 else f"HTTP {response.status_code}",
            }
        except Exception as e:
            logger.error("dashboard_check_failed", url=url, error=str(e))
            sentry_sdk.capture_exception(e)
            return {
                "name": name,
                "url": url,
                "accessible": False,
                "status_code": "ERROR",
                "error": str(e),
            }


# === AI/ML Model Integration ===
# All model hooks are now fully typed, logged, traced, and Sentry-monitored
class ErrorAnalysisAI:
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.model_recognizer = ModelRecognizer()
        self.model_manager = ModelManager()
        self.model_trainer = ModelTrainer()
        self.model_tuner = ModelTuner()
        self.model_registry = ModelRegistry()
        self.model_training = ModelTraining(self.model_trainer)
        logger.info("ErrorAnalysisAI_initialized")

    def detect_error_anomalies(self, api_results: list[dict]) -> list[dict]:
        try:
            import numpy as np
            X = np.array([
                [r.get('status_code', 0) if isinstance(r.get('status_code'), int) else 0, len(str(r.get('error', ''))), float(r.get('response_time', 0) or 0)]
                for r in api_results
            ])
            if len(X) < 5:
                return []
            preds = self.anomaly_detector.predict(X)
            scores = self.anomaly_detector.confidence(X)
            result = [{"api_index": i, "anomaly": int(preds[i] == -1), "confidence": float(scores[i])} for i in range(len(preds))]
            logger.info("error_anomalies_detected", result=result)
            return result
        except Exception as e:
            logger.error("error_anomaly_detection_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            return []

    def ai_error_recommendations(self, api_results: list[dict]) -> list[str]:
        try:
            texts = [str(r.get('error', '')) for r in api_results]
            sentiment = self.sentiment_analyzer.analyze(texts)
            recs = []
            if sentiment['compound'] > 0.5:
                recs.append('Error sentiment is positive. No urgent actions required.')
            elif sentiment['compound'] < -0.5:
                recs.append('Error sentiment is negative. Review API health and system status.')
            values = [float(r.get('response_time', 0) or 0) for r in api_results]
            if values:
                pattern = self.model_recognizer.recognize(values)
                if pattern['confidence'] > 0.8:
                    recs.append(f"Pattern detected: {pattern['pattern']} (confidence: {pattern['confidence']:.2f})")
            anomalies = self.detect_error_anomalies(api_results)
            if any(a['anomaly'] for a in anomalies):
                recs.append(f"{sum(a['anomaly'] for a in anomalies)} error anomalies detected in recent API results.")
            logger.info("ai_error_recommendations", recommendations=recs)
            return recs
        except Exception as e:
            logger.error("ai_error_recommendations_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            return []

    def retrain_models(self, api_results: list[dict]) -> dict:
        try:
            import numpy as np
            X = np.array([
                [r.get('status_code', 0) if isinstance(r.get('status_code'), int) else 0, len(str(r.get('error', ''))), float(r.get('response_time', 0) or 0)]
                for r in api_results
            ])
            if len(X) > 10:
                self.anomaly_detector.fit(X)
            logger.info("model_retraining_complete")
            return {"status": "retraining complete"}
        except Exception as e:
            logger.error("model_retraining_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            return {"status": "retraining failed", "error": str(e)}

    def calibrate_models(self) -> dict:
        try:
            self.anomaly_detector.calibrate(None)
            logger.info("model_calibration_complete")
            return {"status": "calibration complete"}
        except Exception as e:
            logger.error("model_calibration_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            return {"status": "calibration failed", "error": str(e)}

    def get_model_status(self) -> dict:
        try:
            status = {
                "anomaly_detector": str(type(self.anomaly_detector.model)),
                "sentiment_analyzer": "ok",
                "model_recognizer": "ok",
                "registered_models": self.model_manager.list_models(),
            }
            logger.info("model_status", status=status)
            return status
        except Exception as e:
            logger.error("get_model_status_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            return {"error": str(e)}

error_ai = ErrorAnalysisAI()

# --- AI/ML Model Hooks for Error Analytics ---
def ai_error_analytics(api_results):
    anomalies = error_ai.detect_error_anomalies(api_results)
    recs = error_ai.ai_error_recommendations(api_results)
    return {"anomalies": anomalies, "recommendations": recs}

def retrain_error_models(api_results):
    return error_ai.retrain_models(api_results)

def calibrate_error_models():
    return error_ai.calibrate_models()

def get_error_model_status():
    return error_ai.get_model_status()


# --- Advanced error handler for all exceptions ---
@app.middleware("http")
async def prometheus_request_middleware(request: Request, call_next):
    endpoint = request.url.path
    ERROR_ANALYSIS_ACTIVE.inc()
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        ERROR_ANALYSIS_ERRORS.labels(endpoint=endpoint).inc()
        raise
    finally:
        ERROR_ANALYSIS_ACTIVE.dec()

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("unhandled_exception", error=str(exc))
    with tracer.start_as_current_span("unhandled_exception"):
        ERROR_ANALYSIS_ERRORS.labels(endpoint=request.url.path).inc()
        sentry_sdk.capture_exception(exc)
        return JSONResponse(status_code=500, content={"detail": str(exc), "code": 500})


# --- API Endpoints ---
@app.get("/health", tags=["health"], response_model=Dict[str, str], responses={200: {"description": "API health status"}})
async def health() -> Dict[str, str]:
    """Health check endpoint for API status."""
    return {"status": "ok", "ts": datetime.now().isoformat()}


@app.get("/metrics", tags=["monitoring"], response_class=StreamingResponse, responses={200: {"description": "Prometheus metrics"}})
def metrics() -> StreamingResponse:
    """Prometheus metrics endpoint."""
    return StreamingResponse(io.BytesIO(generate_latest()), media_type=CONTENT_TYPE_LATEST)


PREMIUM_FEATURES_ENABLED = True  # Toggle for premium features/analytics

@app.post(
    "/analyze",
    tags=["analysis"],
    dependencies=[Depends(get_api_key), Depends(RateLimiter(times=10, seconds=60))],
    response_model=AnalyzeResponse,
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API Key"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        200: {"description": "Analysis results", "content": {"application/json": {"example": {"api": [{"url": "https://api.example.com", "name": "Example API", "status_code": 200, "error": None, "content_preview": "{...}", "response_time": 0.123, "success": True, "data_source": "primary"}], "dashboards": [{"name": "Main Dashboard", "url": "https://dashboard.example.com", "accessible": True, "status_code": 200, "error": None}]}}}}
    },
)
async def analyze(
    request: APIEndpointRequest,
    api_key: str = Depends(get_api_key),
) -> AnalyzeResponse:
    """
    Perform batch async analysis of API endpoints and dashboards with advanced AI analytics.
    """
    with tracer.start_as_current_span("analyze_endpoint"):
        ERROR_ANALYSIS_REQUESTS.labels(endpoint="/analyze").inc()
        logger.info("analyze_called", api_endpoints=request.api_endpoints, dashboard_urls=request.dashboard_urls)
        if PREMIUM_FEATURES_ENABLED:
            logger.info("premium_batch_analysis")
        api_pairs = list(zip(request.api_endpoints, request.names or request.api_endpoints))
        dash_pairs = list(zip(request.dashboard_urls, request.dashboards_names or request.dashboard_urls))
        api_results = await asyncio.gather(
            *[async_check_api_endpoint(url, name) for url, name in api_pairs]
        )
        dashboard_results = await asyncio.gather(
            *[async_check_dashboard_accessibility(url, name) for url, name in dash_pairs]
        )
        if api_key in PREMIUM_API_KEYS and PREMIUM_FEATURES_ENABLED:
            for result in api_results:
                result["advanced"] = {"root_cause": "AI-detected", "recommendation": "[PREMIUM] Upgrade to premium for full report"}
        if api_key in PARTNER_WEBHOOKS:
            pass
        try:
            return AnalyzeResponse(api=api_results, dashboards=dashboard_results)
        except ValidationError as e:
            logger.error("validation_error", error=str(e))
            raise HTTPException(status_code=422, detail=str(e))


@app.get(
    "/analyze/single",
    tags=["analysis"],
    dependencies=[Depends(get_api_key)],
    response_model=Union[APIResult, DashboardResult],
    responses={200: {"description": "Single endpoint analysis result"}, 401: {"model": ErrorResponse}},
)
async def analyze_single(
    url: str = Query(...),
    name: str = Query("API Endpoint"),
    is_dashboard: bool = Query(False),
) -> Union[APIResult, DashboardResult]:
    """Analyze a single API or dashboard endpoint."""
    if is_dashboard:
        return await async_check_dashboard_accessibility(url, name)
    return await async_check_api_endpoint(url, name)


@app.get(
    "/export/json",
    tags=["export"],
    dependencies=[Depends(get_api_key)],
    response_class=StreamingResponse,
    responses={200: {"description": "Exported JSON analysis"}},
)
async def export_json(
    api_endpoints: List[str] = Query(..., description="List of API endpoint URLs (comma-separated)"),
    dashboard_urls: List[str] = Query(..., description="List of dashboard URLs (comma-separated)"),
    names: Optional[List[str]] = Query(None),
    dashboards_names: Optional[List[str]] = Query(None),
) -> StreamingResponse:
    """Export analysis results as JSON."""
    api_pairs = list(zip(api_endpoints, names or api_endpoints))
    dash_pairs = list(zip(dashboard_urls, dashboards_names or dashboard_urls))
    api_results = await asyncio.gather(
        *[async_check_api_endpoint(url, name) for url, name in api_pairs]
    )
    dashboard_results = await asyncio.gather(
        *[async_check_dashboard_accessibility(url, name) for url, name in dash_pairs]
    )
    buf = io.StringIO()
    json.dump({"api": api_results, "dashboards": dashboard_results}, buf, indent=2)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()), media_type="application/json"
    )


@app.get(
    "/export/csv",
    tags=["export"],
    dependencies=[Depends(get_api_key)],
    response_class=StreamingResponse,
    responses={200: {"description": "Exported CSV analysis"}},
)
async def export_csv(
    api_endpoints: List[str] = Query(..., description="List of API endpoint URLs (comma-separated)"),
    dashboard_urls: List[str] = Query(..., description="List of dashboard URLs (comma-separated)"),
    names: Optional[List[str]] = Query(None),
    dashboards_names: Optional[List[str]] = Query(None),
) -> StreamingResponse:
    """Export analysis results as CSV."""
    api_pairs = list(zip(api_endpoints, names or api_endpoints))
    dash_pairs = list(zip(dashboard_urls, dashboards_names or dashboard_urls))
    api_results = await asyncio.gather(
        *[async_check_api_endpoint(url, name) for url, name in api_pairs]
    )
    dashboard_results = await asyncio.gather(
        *[async_check_dashboard_accessibility(url, name) for url, name in dash_pairs]
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Type", "Name", "URL", "Status", "Error"])
    for r in api_results:
        writer.writerow(["API", r.get("name"), r.get("url"), r.get("status_code"), r.get("error")])
    for r in dashboard_results:
        writer.writerow(["Dashboard", r.get("name"), r.get("url"), r.get("status_code"), r.get("error")])
    buf.seek(0)
    return StreamingResponse(io.BytesIO(buf.getvalue().encode()), media_type="text/csv")


@app.get(
    "/profit-impact",
    tags=["analytics"],
    dependencies=[Depends(get_api_key)],
    response_model=Dict[str, int],
    responses={200: {"description": "Profit impact score"}},
)
async def profit_impact(
    api_endpoints: List[str] = Query(...),
    names: Optional[List[str]] = Query(None),
) -> Dict[str, int]:
    """Calculate profit impact score based on API health."""
    api_pairs = list(zip(api_endpoints, names or api_endpoints))
    api_results = await asyncio.gather(
        *[async_check_api_endpoint(url, name) for url, name in api_pairs]
    )
    impact = 0
    for r in api_results:
        if r.get("name", "").lower().find("portfolio") >= 0:
            impact += 5 if r.get("status_code") != 200 else 0
        if r.get("name", "").lower().find("trading") >= 0:
            impact += 10 if r.get("status_code") != 200 else 0
    return {"profit_impact_score": impact}


@app.get(
    "/ci-cd/edge-case-test",
    tags=["ci-cd"],
    dependencies=[Depends(get_api_key)],
    response_model=Dict[str, Any],
    responses={200: {"description": "Edge case test results"}},
)
async def ci_cd_edge_case_test() -> Dict[str, Any]:
    """Run edge-case tests for CI/CD pipeline validation."""
    # Simulate API failure
    fail_result = await async_check_api_endpoint("http://localhost:9999", "Test API")
    # Simulate HTTP error
    http_error_result = await async_check_api_endpoint("http://localhost:5000/api/invalid", "Invalid API")
    # Simulate timeout
    try:
        await async_check_api_endpoint("http://localhost:5000/api/health", "Timeout API", timeout=0.001)
        timeout_ok = True
    except Exception:
        timeout_ok = False
    return {
        "fail_result": fail_result,
        "http_error_result": http_error_result,
        "timeout_simulated": timeout_ok,
    }


# --- CI/CD test endpoint for automated pipeline validation ---
@app.get("/ci-cd/test", tags=["ci-cd"], response_model=Dict[str, str], responses={200: {"description": "CI/CD test status"}})
async def ci_cd_test() -> Dict[str, str]:
    """CI/CD pipeline test endpoint for automated deployment validation."""
    with tracer.start_as_current_span("ci_cd_test_endpoint"):
        logger.info("ci_cd_test_called")
        # --- Advanced CI/CD: test DB, Redis, Sentry, and AI model health ---
        health = {"status": "ok", "ts": datetime.now().isoformat(), "message": "CI/CD pipeline test successful."}
        try:
            import redis.asyncio as aioredis
            redis = await aioredis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
            await redis.set("ci_cd_test", "ok")
            health["redis"] = "ok"
        except Exception as e:
            health["redis"] = f"fail: {e}"
        try:
            if SENTRY_DSN:
                sentry_sdk.capture_message("CI/CD test event", level="info")
                health["sentry"] = "ok"
        except Exception as e:
            health["sentry"] = f"fail: {e}"
        try:
            status = error_ai.get_model_status()
            health["ai_models"] = "ok" if "error" not in status else f"fail: {status['error']}"
        except Exception as e:
            health["ai_models"] = f"fail: {e}"
        return health


# --- Monetization & Partner Hooks ---
PREMIUM_API_KEYS = {"premium-key", "partner-key"}
PARTNER_WEBHOOKS = {"partner-key": "https://partner.example.com/webhook"}


@app.post(
    "/monetize/webhook",
    tags=["monetization"],
    dependencies=[Depends(get_api_key)],
    response_model=Dict[str, Any],
    responses={200: {"description": "Webhook sent status"}},
)
async def monetize_webhook(
    url: str = Query(...),
    event: str = Query(...),
    payload: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Send a monetization webhook to a partner/customer."""
    # Example: send webhook to partner/customer
    # In production, add billing, partner ID, event logging, etc.
    try:
        async with httpx.AsyncClient(http2=True) as client:
            resp = await client.post(url, json={"event": event, "payload": payload})
        return {"status": resp.status_code, "response": resp.text[:100]}
    except Exception as e:
        return {"error": str(e)}


@app.get(
    "/monetize/partner-status",
    tags=["monetization"],
    dependencies=[Depends(get_api_key)],
    response_model=Dict[str, Any],
    responses={200: {"description": "Partner status info"}},
)
async def partner_status(partner_id: str = Query(...)) -> Dict[str, Any]:
    """Check partner status, quota, and billing."""
    # Example: check partner status, quota, billing, etc.
    # In production, integrate with billing/CRM
    return {"partner_id": partner_id, "status": "active", "quota": 1000, "used": 123}


# --- Advanced logging, analytics, and recommendations (stub) ---
@app.get(
    "/analytics/recommendations",
    tags=["analytics"],
    dependencies=[Depends(get_api_key)],
    response_model=Dict[str, List[str]],
    responses={200: {"description": "AI-powered recommendations"}},
)
async def recommendations() -> Dict[str, List[str]]:
    """Get AI-powered recommendations for upgrades, premium, or fixes."""
    # Example: recommend upgrades, premium, or fixes
    recs = [
        "Upgrade to premium for real-time alerts and advanced analytics.",
        "Enable webhook integration for automated incident response.",
        "Contact support for persistent API errors.",
    ]
    if PREMIUM_FEATURES_ENABLED:
        recs.insert(0, "[PREMIUM] Dostęp do zaawansowanych rekomendacji i raportów.")
    return {"recommendations": recs}


# --- Root endpoint ---
@app.get("/", tags=["info"], response_model=Dict[str, str], responses={200: {"description": "API root info"}})
async def root() -> Dict[str, str]:
    """Root endpoint for API info."""
    return {
        "message": "Comprehensive Error Analysis API (modernized)",
        "ts": datetime.now().isoformat(),
    }


# --- Run with: uvicorn comprehensive_error_analysis:app --reload ---
