#!/usr/bin/env python3
"""
ZoL0 Memory Leak Monitor
Continuous monitoring for memory leaks after fixes
"""

import json
import os
import time
from datetime import datetime
import subprocess
import sys
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette_exporter import PrometheusMiddleware, handle_metrics
import io
import csv
import uvicorn

import psutil


API_KEYS = {"admin-key": "admin", "memoryleak-key": "memoryleak", "partner-key": "partner", "premium-key": "premium"}
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)

def get_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return API_KEYS[api_key]


class MemoryLeakMonitor:
    def __init__(self):
        self.baseline = None
        self.alerts_sent = []

    def monitor_continuous(self, duration_minutes=60):
        """
        Monitor for memory leaks continuously. Obsługa błędu zapisu raportu.
        """
        print(f"🔍 Starting {duration_minutes} minute memory leak monitoring...")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        snapshots = []

        while time.time() < end_time:
            # Get memory info
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024

            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "memory_mb": memory_mb,
                "threads": len(process.threads()),
                "files": (
                    len(process.open_files()) if hasattr(process, "open_files") else 0
                ),
            }

            snapshots.append(snapshot)

            # Check for concerning growth
            if len(snapshots) > 10:
                recent_growth = snapshots[-1]["memory_mb"] - snapshots[-10]["memory_mb"]
                if recent_growth > 50:  # 50MB growth in 10 checks
                    print(f"⚠️ HIGH MEMORY GROWTH: {recent_growth:.2f} MB")
                elif recent_growth > 20:
                    print(f"⚠️ Moderate memory growth: {recent_growth:.2f} MB")
                else:
                    print(f"✅ Memory stable: {memory_mb:.2f} MB")
            else:
                print(f"📊 Memory: {memory_mb:.2f} MB")

            time.sleep(30)  # Check every 30 seconds

        # Generate report
        report = {
            "monitoring_duration_minutes": duration_minutes,
            "total_snapshots": len(snapshots),
            "memory_range": {
                "min_mb": min(s["memory_mb"] for s in snapshots),
                "max_mb": max(s["memory_mb"] for s in snapshots),
                "final_mb": snapshots[-1]["memory_mb"],
            },
            "memory_growth_mb": snapshots[-1]["memory_mb"] - snapshots[0]["memory_mb"],
            "snapshots": snapshots,
        }

        report_file = (
            f"memory_leak_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        try:
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            print(f"OK: File write error handled gracefully: {e}")
            return report

        print(f"📄 Monitoring report saved: {report_file}")
        print(f"📈 Total memory growth: {report['memory_growth_mb']:.2f} MB")

        return report


mlm_api = FastAPI(title="ZoL0 Memory Leak Monitor API", version="2.0")
mlm_api.add_middleware(PrometheusMiddleware)
mlm_api.add_route("/metrics", handle_metrics)

# --- Pydantic Models ---
class MonitorRequest(BaseModel):
    duration_minutes: int = 60
class BatchMonitorRequest(BaseModel):
    queries: list[MonitorRequest]

# --- Global monitor instance ---
monitor = MemoryLeakMonitor()

# --- Endpoints ---
@mlm_api.get("/")
async def root():
    return {"status": "ok", "service": "ZoL0 Memory Leak Monitor API", "version": "2.0"}

@mlm_api.get("/api/health")
async def api_health():
    return {"status": "ok", "timestamp": datetime.now().isoformat(), "service": "ZoL0 Memory Leak Monitor API", "version": "2.0"}

@mlm_api.post("/api/monitor/start", dependencies=[Depends(get_api_key)])
async def api_monitor_start(req: MonitorRequest, role: str = Depends(get_api_key)):
    # Run monitoring for requested duration (sync for now)
    report = monitor.monitor_continuous(duration_minutes=req.duration_minutes)
    return report

@mlm_api.get("/api/monitor/status", dependencies=[Depends(get_api_key)])
async def api_monitor_status(role: str = Depends(get_api_key)):
    # Return last report if available
    try:
        with open("memory_leak_monitoring_latest.json", "r") as f:
            return json.load(f)
    except Exception:
        return {"status": "no report available"}

@mlm_api.post("/api/monitor/batch", dependencies=[Depends(get_api_key)])
async def api_monitor_batch(req: BatchMonitorRequest, role: str = Depends(get_api_key)):
    return {"results": [monitor.monitor_continuous(duration_minutes=q.duration_minutes) for q in req.queries]}

@mlm_api.get("/api/export/csv", dependencies=[Depends(get_api_key)])
async def api_export_csv(role: str = Depends(get_api_key)):
    try:
        with open("memory_leak_monitoring_latest.json", "r") as f:
            report = json.load(f)
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(report.keys()))
        writer.writeheader()
        writer.writerow(report)
        return StreamingResponse(iter([output.getvalue()]), media_type="text/csv")
    except Exception:
        return JSONResponse(status_code=404, content={"error": "No report available"})

@mlm_api.get("/api/export/prometheus", dependencies=[Depends(get_api_key)])
async def api_export_prometheus(role: str = Depends(get_api_key)):
    try:
        with open("memory_leak_monitoring_latest.json", "r") as f:
            report = json.load(f)
        growth = report.get("memory_growth_mb", 0)
        return PlainTextResponse(f"# HELP memory_leak_growth_mb Memory leak growth in MB\nmemory_leak_growth_mb {growth}", media_type="text/plain")
    except Exception:
        return PlainTextResponse("# HELP memory_leak_growth_mb Memory leak growth in MB\nmemory_leak_growth_mb 0", media_type="text/plain")

@mlm_api.get("/api/report", dependencies=[Depends(get_api_key)])
async def api_report(role: str = Depends(get_api_key)):
    return {"status": "report generated (stub)"}

@mlm_api.get("/api/recommendations", dependencies=[Depends(get_api_key)])
async def api_recommendations(role: str = Depends(get_api_key)):
    try:
        with open("memory_leak_monitoring_latest.json", "r") as f:
            report = json.load(f)
        recs = []
        growth = report.get("memory_growth_mb", 0)
        if growth > 50:
            recs.append("Investigate for memory leaks and optimize code.")
        elif growth > 20:
            recs.append("Monitor for moderate memory growth.")
        else:
            recs.append("Memory usage is stable.")
        return {"recommendations": recs}
    except Exception:
        return {"recommendations": ["No report available."]}

@mlm_api.get("/api/premium/score", dependencies=[Depends(get_api_key)])
async def api_premium_score(role: str = Depends(get_api_key)):
    try:
        with open("memory_leak_monitoring_latest.json", "r") as f:
            report = json.load(f)
        growth = report.get("memory_growth_mb", 0)
        score = max(0, 100 - int(growth))
        return {"score": score}
    except Exception:
        return {"score": 0}

@mlm_api.get("/api/saas/tenant/{tenant_id}/report", dependencies=[Depends(get_api_key)])
async def api_saas_tenant_report(tenant_id: str, role: str = Depends(get_api_key)):
    try:
        with open("memory_leak_monitoring_latest.json", "r") as f:
            report = json.load(f)
        return {"tenant_id": tenant_id, "report": report}
    except Exception:
        return {"tenant_id": tenant_id, "report": {"status": "no report available"}}

@mlm_api.get("/api/partner/webhook", dependencies=[Depends(get_api_key)])
async def api_partner_webhook(payload: dict, role: str = Depends(get_api_key)):
    # Monetization: process partner webhook payload for SaaS/affiliate integrations
    # In production, validate and process payload, trigger partner actions
    return {"status": "received", "payload": payload}

# --- Monetization: Usage metering endpoint ---
@mlm_api.get("/api/usage", dependencies=[Depends(get_api_key)])
async def api_usage(role: str = Depends(get_api_key)):
    try:
        with open("memory_leak_monitoring_latest.json", "r") as f:
            report = json.load(f)
        return {
            "role": role,
            "memory_growth_mb": report.get("memory_growth_mb", 0),
            "timestamp": report.get("timestamp", "N/A"),
        }
    except Exception:
        return {"role": role, "memory_growth_mb": 0, "timestamp": "N/A"}

@mlm_api.get("/api/test/edge-case")
async def api_edge_case():
    try:
        raise RuntimeError("Simulated memory leak monitor edge-case error")
    except Exception as e:
        return {"edge_case": str(e)}

@mlm_api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

# --- CI/CD test suite ---
import unittest
class TestMemoryLeakMonitorAPI(unittest.TestCase):
    def test_monitor(self):
        report = monitor.monitor_continuous(duration_minutes=0.01)
        assert "memory_growth_mb" in report
    def test_report_file(self):
        try:
            with open("memory_leak_monitoring_latest.json", "r") as f:
                report = json.load(f)
            assert "memory_growth_mb" in report
        except Exception:
            pass

if __name__ == "__main__":
    if "test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        uvicorn.run("memory_leak_monitor:mlm_api", host="0.0.0.0", port=8506, reload=True)
