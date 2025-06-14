#!/usr/bin/env python3
"""
Kompleksowy test systemu - sprawdzenie wszystkich funkcjonalności
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import aiofiles
from fastapi import FastAPI, Query, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import csv
import io

API_KEY = os.environ.get("SYSTEM_TEST_API_KEY", "demo-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key

TEST_REQUESTS = Counter(
    "system_test_requests_total", "Total system test API requests", ["endpoint"]
)
TEST_LATENCY = Histogram(
    "system_test_latency_seconds", "System test endpoint latency", ["endpoint"]
)

app = FastAPI(title="Comprehensive System Test API", version="2.0-modernized")
logger = logging.getLogger("system_test_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Import and wrap legacy test logic ---
from system_tests_logic import (
    test_environment,
    test_production_data_manager,
    test_bybit_connector,
    test_api_service,
    test_configuration_files,
)

def run_test_method(method_name: str) -> Dict[str, Any]:
    # Run and capture output for a single test
    import io, sys
    buf = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buf
    try:
        result = getattr(sys.modules[__name__], method_name)()
        output = buf.getvalue()
        sys.stdout = sys_stdout
        return {"output": output, "success": True}
    except Exception as e:
        output = buf.getvalue()
        sys.stdout = sys_stdout
        return {"output": output, "success": False, "error": str(e)}

# --- API Endpoints ---
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "ts": datetime.now().isoformat()}

@app.get("/metrics", tags=["monitoring"])
def metrics():
    return StreamingResponse(io.BytesIO(generate_latest()), media_type=CONTENT_TYPE_LATEST)

@app.post("/test/run", tags=["test"], dependencies=[Depends(get_api_key)])
async def run_test(test_type: str = Query("complete")):
    TEST_REQUESTS.labels(endpoint="run_test").inc()
    with TEST_LATENCY.labels(endpoint="run_test").time():
        valid_types = [
            "complete",
            "environment",
            "production_data_manager",
            "bybit_connector",
            "api_service",
            "configuration_files",
        ]
        if test_type not in valid_types:
            raise HTTPException(status_code=400, detail="Invalid test type")
        results = {}
        if test_type == "complete":
            for t in valid_types[1:]:
                results[t] = run_test_method(f"test_{t}")
        else:
            results[test_type] = run_test_method(f"test_{test_type}")
        # Save report
        with open("system_test_report.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

@app.get("/test/export/json", tags=["export"], dependencies=[Depends(get_api_key)])
async def export_json():
    path = "system_test_report.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Test report not found")
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        content = await f.read()
    return StreamingResponse(io.BytesIO(content.encode()), media_type="application/json")

@app.get("/test/export/csv", tags=["export"], dependencies=[Depends(get_api_key)])
async def export_csv():
    path = "system_test_report.json"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Test report not found")
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        content = await f.read()
    report = json.loads(content)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Test Name", "Success", "Output", "Error"])
    for name, result in report.items():
        writer.writerow([
            name,
            result.get("success"),
            result.get("output", "")[:100],
            result.get("error", "")
        ])
    buf.seek(0)
    return StreamingResponse(io.BytesIO(buf.getvalue().encode()), media_type="text/csv")

@app.get("/ci-cd/edge-case-test", tags=["ci-cd"], dependencies=[Depends(get_api_key)])
async def ci_cd_edge_case_test():
    # Simulate missing env var, import error, API failure
    missing_env = os.getenv("FAKE_ENV_VAR") is None
    import_error = False
    api_failure = False
    try:
        import not_a_real_module
    except ImportError:
        import_error = True
    try:
        import requests
        requests.get("http://localhost:9999", timeout=1)
    except Exception:
        api_failure = True
    return {
        "missing_env_handled": missing_env,
        "import_error_handled": import_error,
        "api_failure_handled": api_failure,
    }

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
            "Upgrade to premium for advanced system test automation.",
            "Enable webhook integration for automated incident response.",
            "Contact support for persistent test failures.",
        ]
    }

@app.get("/", tags=["info"])
async def root():
    return {"message": "Comprehensive System Test API (modernized)", "ts": datetime.now().isoformat()}

# --- Run with: uvicorn comprehensive_system_test:app --reload ---
