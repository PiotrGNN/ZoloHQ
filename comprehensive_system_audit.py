#!/usr/bin/env python3
"""
comprehensive_system_audit.py
=============================
Kompleksowy audyt systemu tradingowego - sprawdza czy wszystkie komponenty używają prawdziwych danych
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
from fastapi import FastAPI, Query, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import csv
import io
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import structlog
from pydantic import BaseModel, Field

API_KEY = os.environ.get("SYSTEM_AUDIT_API_KEY", "demo-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key

AUDIT_REQUESTS = Counter(
    "system_audit_requests_total", "Total system audit API requests", ["endpoint"]
)
AUDIT_LATENCY = Histogram(
    "system_audit_latency_seconds", "System audit endpoint latency", ["endpoint"]
)

app = FastAPI(title="Comprehensive System Audit API", version="2.0-modernized")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- OpenTelemetry distributed tracing setup (idempotent) ---
if not hasattr(logging, "_otel_initialized_system_audit"):
    resource = Resource.create({"service.name": "comprehensive-system-audit-api"})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(OTLPSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    logging._otel_initialized_system_audit = True
tracer = trace.get_tracer("comprehensive-system-audit-api")

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger("system_audit_api")

class ErrorResponse(BaseModel):
    detail: str
    code: int = Field(..., example=500)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("unhandled_exception", error=str(exc))
    with tracer.start_as_current_span("unhandled_exception"):
        return JSONResponse(status_code=500, content={"detail": str(exc), "code": 500})

# --- Import and wrap legacy audit logic ---
from comprehensive_system_audit import SystemAuditor, run_ci_cd_tests

def run_audit_method(method_name: str) -> Dict[str, Any]:
    with tracer.start_as_current_span("run_audit_method"):
        auditor = SystemAuditor()
        method = getattr(auditor, method_name, None)
        if not method:
            raise ValueError(f"No such audit method: {method_name}")
        return method()

# === AI/ML Model Integration ===
from ai.models.AnomalyDetector import AnomalyDetector
from ai.models.SentimentAnalyzer import SentimentAnalyzer
from ai.models.ModelRecognizer import ModelRecognizer
from ai.models.ModelManager import ModelManager
from ai.models.ModelTrainer import ModelTrainer
from ai.models.ModelTuner import ModelTuner
from ai.models.ModelRegistry import ModelRegistry
from ai.models.ModelTraining import ModelTraining

class AuditAI:
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.model_recognizer = ModelRecognizer()
        self.model_manager = ModelManager()
        self.model_trainer = ModelTrainer()
        self.model_tuner = ModelTuner()
        self.model_registry = ModelRegistry()
        self.model_training = ModelTraining(self.model_trainer)

    def detect_audit_anomalies(self, audit_results):
        try:
            import numpy as np
            X = np.array([
                [1 if r.get('status', '').lower() == 'fail' else 0, len(str(r.get('details', '')))]
                for r in audit_results.values()
            ])
            if len(X) < 5:
                return []
            preds = self.anomaly_detector.predict(X)
            scores = self.anomaly_detector.confidence(X)
            return [{"section": s, "anomaly": int(preds[i] == -1), "confidence": float(scores[i])} for i, s in enumerate(audit_results.keys())]
        except Exception as e:
            logger.error(f"Audit anomaly detection failed: {e}")
            return []

    def ai_audit_recommendations(self, audit_results):
        try:
            texts = [str(r.get('details', '')) for r in audit_results.values()]
            sentiment = self.sentiment_analyzer.analyze(texts)
            recs = []
            if sentiment['compound'] > 0.5:
                recs.append('Audit sentiment is positive. No urgent actions required.')
            elif sentiment['compound'] < -0.5:
                recs.append('Audit sentiment is negative. Review failing sections and compliance.')
            # Pattern recognition on section status
            values = [1 if r.get('status', '').lower() == 'fail' else 0 for r in audit_results.values()]
            if values:
                pattern = self.model_recognizer.recognize(values)
                if pattern['confidence'] > 0.8:
                    recs.append(f"Pattern detected: {pattern['pattern']} (confidence: {pattern['confidence']:.2f})")
            # Anomaly detection
            anomalies = self.detect_audit_anomalies(audit_results)
            if any(a['anomaly'] for a in anomalies):
                recs.append(f"{sum(a['anomaly'] for a in anomalies)} audit anomalies detected in recent results.")
            return recs
        except Exception as e:
            logger.error(f"AI audit recommendations failed: {e}")
            return []

    def retrain_models(self, audit_results):
        try:
            import numpy as np
            X = np.array([
                [1 if r.get('status', '').lower() == 'fail' else 0, len(str(r.get('details', '')))]
                for r in audit_results.values()
            ])
            if len(X) > 10:
                self.anomaly_detector.fit(X)
            return {"status": "retraining complete"}
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return {"status": "retraining failed", "error": str(e)}

    def calibrate_models(self):
        try:
            self.anomaly_detector.calibrate(None)
            return {"status": "calibration complete"}
        except Exception as e:
            logger.error(f"Model calibration failed: {e}")
            return {"status": "calibration failed", "error": str(e)}

    def get_model_status(self):
        try:
            return {
                "anomaly_detector": str(type(self.anomaly_detector.model)),
                "sentiment_analyzer": "ok",
                "model_recognizer": "ok",
                "registered_models": self.model_manager.list_models(),
            }
        except Exception as e:
            return {"error": str(e)}

audit_ai = AuditAI()

# --- AI/ML Model Hooks for Audit Analytics ---
def ai_audit_analytics(audit_results):
    anomalies = audit_ai.detect_audit_anomalies(audit_results)
    recs = audit_ai.ai_audit_recommendations(audit_results)
    return {"anomalies": anomalies, "recommendations": recs}

def retrain_audit_models(audit_results):
    return audit_ai.retrain_models(audit_results)

def calibrate_audit_models():
    return audit_ai.calibrate_models()

def get_audit_model_status():
    return audit_ai.get_model_status()

# --- API Endpoints ---
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "ts": datetime.now().isoformat()}

@app.get("/metrics", tags=["monitoring"])
def metrics():
    return StreamingResponse(io.BytesIO(generate_latest()), media_type=CONTENT_TYPE_LATEST)

@app.post("/audit/run", tags=["audit"], dependencies=[Depends(get_api_key)])
async def run_audit(audit_type: str = Query("complete")):
    AUDIT_REQUESTS.labels(endpoint="run_audit").inc()
    with AUDIT_LATENCY.labels(endpoint="run_audit").time():
        valid_types = [
            "complete",
            "environment",
            "bybit_connector",
            "production_manager",
            "configuration_files",
            "dashboard_files",
        ]
        if audit_type not in valid_types:
            raise HTTPException(status_code=400, detail="Invalid audit type")
        if audit_type == "complete":
            auditor = SystemAuditor()
            result = auditor.run_complete_audit()
        else:
            result = run_audit_method(f"audit_{audit_type}")
        # --- Integrate AI/ML analytics ---
        result['ai_analytics'] = ai_audit_analytics(result.get('results', {}))
        return result

@app.get("/audit/export/json", tags=["export"], dependencies=[Depends(get_api_key)])
async def export_json():
    path = Path("comprehensive_audit_report.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audit report not found")
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        content = await f.read()
    return StreamingResponse(io.BytesIO(content.encode()), media_type="application/json")

@app.get("/audit/export/csv", tags=["export"], dependencies=[Depends(get_api_key)])
async def export_csv():
    path = Path("comprehensive_audit_report.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audit report not found")
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        content = await f.read()
    report = json.loads(content)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Audit Section", "Status", "Details"])
    for section, result in report["results"].items():
        writer.writerow([section, result.get("status"), json.dumps(result.get("details"))])
    buf.seek(0)
    return StreamingResponse(io.BytesIO(buf.getvalue().encode()), media_type="text/csv")

@app.get("/ci-cd/edge-case-test", tags=["ci-cd"], dependencies=[Depends(get_api_key)])
async def ci_cd_edge_case_test():
    run_ci_cd_tests()
    return {"status": "edge-case tests completed", "ts": datetime.now().isoformat()}

# --- Monetization, SaaS, Partner, Webhook, Multi-tenant endpoints (stubs for extension) ---
@app.post("/monetize/webhook", tags=["monetization"], dependencies=[Depends(get_api_key)])
async def monetize_webhook(
    url: str = Query(...),
    event: str = Query(...),
    payload: Optional[str] = Query(None),
):
    import httpx
    try:
        async with httpx.AsyncClient(http2=True) as client:
            resp = await client.post(url, json={"event": event, "payload": payload})
        return {"status": resp.status_code, "response": resp.text[:100]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/monetize/partner-status", tags=["monetization"], dependencies=[Depends(get_api_key)])
async def partner_status(partner_id: str = Query(...)):
    return {"partner_id": partner_id, "status": "active", "quota": 1000, "used": 123}

# --- Advanced logging, analytics, and recommendations (stub) ---
@app.get("/analytics/recommendations", tags=["analytics"], dependencies=[Depends(get_api_key)])
async def recommendations():
    return {
        "recommendations": [
            "Upgrade to premium for advanced audit automation and compliance.",
            "Enable webhook integration for automated incident response.",
            "Contact support for persistent audit failures.",
        ]
    }

@app.get("/", tags=["info"])
async def root():
    return {"message": "Comprehensive System Audit API (modernized)", "ts": datetime.now().isoformat()}

# --- Run with: uvicorn comprehensive_system_audit:app --reload ---
