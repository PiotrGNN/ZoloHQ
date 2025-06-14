#!/usr/bin/env python3
"""
Comprehensive syntax analyzer for unified_trading_dashboard.py
This script will identify ALL syntax issues at once for batch fixing.
"""

import ast
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import FastAPI, UploadFile, File, Query, Depends, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import csv
import io

API_KEY = os.environ.get("SYNTAX_ANALYZER_API_KEY", "demo-key")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key")
    return api_key

SYNTAX_ANALYSIS_REQUESTS = Counter(
    "syntax_analysis_requests_total", "Total syntax analysis API requests", ["endpoint"]
)
SYNTAX_ANALYSIS_LATENCY = Histogram(
    "syntax_analysis_latency_seconds", "Syntax analysis endpoint latency", ["endpoint"]
)

app = FastAPI(title="Comprehensive Syntax Analyzer API", version="2.0-modernized")
logger = logging.getLogger("syntax_analyzer_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Core analysis logic (refactored for async/file upload) ---
def analyze_dashboard_syntax_content(content: str) -> Dict[str, Any]:
    issues = []
    lines = content.split("\n")
    # 1. Check for concatenated lines (missing newlines)
    concatenated_patterns = [
        r"\)\s*[a-zA-Z_]", r"}\s*[a-zA-Z_]", r"]\s*[a-zA-Z_]", r'"\s*[a-zA-Z_]', r"'\s*[a-zA-Z_]",
    ]
    for i, line in enumerate(lines, 1):
        if line.strip():
            for pattern in concatenated_patterns:
                if re.search(pattern, line):
                    if not any(x in line for x in ['f"', "f'", '"""', "'''"]):
                        issues.append({"line": i, "type": "concat", "msg": f"Possible concatenated line: {line[:80]}..."})
    # 2. Indentation
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"): continue
        current_indent = len(line) - len(line.lstrip())
        if current_indent % 4 != 0 and current_indent > 0:
            issues.append({"line": i, "type": "indent", "msg": f"Irregular indentation ({current_indent} spaces)"})
    # 3. AST parse
    try:
        ast.parse(content)
    except SyntaxError as e:
        issues.append({"line": e.lineno, "type": "syntax", "msg": f"Syntax Error: {e.msg}"})
    except Exception as e:
        issues.append({"line": None, "type": "ast", "msg": f"AST Error: {str(e)}"})
    # 4. Try/except
    try_blocks = 0
    except_blocks = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("try:"): try_blocks += 1
        elif stripped.startswith("except"): except_blocks += 1
    if try_blocks != except_blocks:
        issues.append({"line": None, "type": "try_except", "msg": f"Try/except mismatch: {try_blocks} try, {except_blocks} except"})
    # 5. Brackets
    brackets = {"(": 0, "[": 0, "{": 0}
    closing_to_opening = {")": "(", "]": "[", "}": "{"}
    for i, line in enumerate(lines, 1):
        for char in line:
            if char in "([{": brackets[char] += 1
            elif char in ")]}":
                corresponding = closing_to_opening[char]
                brackets[corresponding] -= 1
                if brackets[corresponding] < 0:
                    issues.append({"line": i, "type": "bracket", "msg": f"Unmatched closing {char}"})
    for bracket, count in brackets.items():
        if count != 0:
            issues.append({"line": None, "type": "bracket", "msg": f"Unmatched {bracket}: {count} remaining"})
    return {
        "issues": issues,
        "summary": {
            "total_issues": len(issues),
            "file_size": len(lines),
            "try_blocks": try_blocks,
            "except_blocks": except_blocks,
        },
    }

# --- API Endpoints ---
@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "ts": datetime.now().isoformat()}

@app.get("/metrics", tags=["monitoring"])
def metrics():
    return StreamingResponse(io.BytesIO(generate_latest()), media_type=CONTENT_TYPE_LATEST)

@app.post("/analyze", tags=["analysis"], dependencies=[Depends(get_api_key)])
async def analyze_file(file: UploadFile = File(...)):
    SYNTAX_ANALYSIS_REQUESTS.labels(endpoint="analyze_file").inc()
    with SYNTAX_ANALYSIS_LATENCY.labels(endpoint="analyze_file").time():
        content = (await file.read()).decode()
        result = analyze_dashboard_syntax_content(content)
        return result

@app.post("/analyze/batch", tags=["analysis"], dependencies=[Depends(get_api_key)])
async def analyze_batch(files: List[UploadFile] = File(...)):
    SYNTAX_ANALYSIS_REQUESTS.labels(endpoint="analyze_batch").inc()
    with SYNTAX_ANALYSIS_LATENCY.labels(endpoint="analyze_batch").time():
        results = []
        for file in files:
            content = (await file.read()).decode()
            result = analyze_dashboard_syntax_content(content)
            results.append({"filename": file.filename, **result})
        return results

@app.get("/analyze/path", tags=["analysis"], dependencies=[Depends(get_api_key)])
async def analyze_path(path: str = Query(...)):
    SYNTAX_ANALYSIS_REQUESTS.labels(endpoint="analyze_path").inc()
    with SYNTAX_ANALYSIS_LATENCY.labels(endpoint="analyze_path").time():
        if not Path(path).exists():
            raise HTTPException(status_code=404, detail="File not found")
        async with aiofiles.open(path, "r", encoding="utf-8") as f:
            content = await f.read()
        result = analyze_dashboard_syntax_content(content)
        return result

@app.get("/export/csv", tags=["export"], dependencies=[Depends(get_api_key)])
async def export_csv(path: str = Query(...)):
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        content = await f.read()
    result = analyze_dashboard_syntax_content(content)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Line", "Type", "Message"])
    for issue in result["issues"]:
        writer.writerow([issue["line"], issue["type"], issue["msg"]])
    buf.seek(0)
    return StreamingResponse(io.BytesIO(buf.getvalue().encode()), media_type="text/csv")

@app.get("/export/json", tags=["export"], dependencies=[Depends(get_api_key)])
async def export_json(path: str = Query(...)):
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    async with aiofiles.open(path, "r", encoding="utf-8") as f:
        content = await f.read()
    result = analyze_dashboard_syntax_content(content)
    buf = io.StringIO()
    import json
    json.dump(result, buf, indent=2)
    buf.seek(0)
    return StreamingResponse(io.BytesIO(buf.getvalue().encode()), media_type="application/json")

@app.get("/ci-cd/edge-case-test", tags=["ci-cd"], dependencies=[Depends(get_api_key)])
async def ci_cd_edge_case_test():
    # Simulate missing file
    try:
        await analyze_path("nonexistent_dashboard.py")
        missing_ok = False
    except HTTPException:
        missing_ok = True
    # Simulate syntax error
    bad_code = "def foo(:\n    pass"
    result = analyze_dashboard_syntax_content(bad_code)
    syntax_error_ok = any(i["type"] == "syntax" for i in result["issues"])
    return {"missing_file_handled": missing_ok, "syntax_error_detected": syntax_error_ok}

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
            "Upgrade to premium for advanced syntax analysis and batch automation.",
            "Enable webhook integration for automated code review alerts.",
            "Contact support for persistent syntax issues.",
        ]
    }

@app.get("/", tags=["info"])
async def root():
    return {"message": "Comprehensive Syntax Analyzer API (modernized)", "ts": datetime.now().isoformat()}

# --- Run with: uvicorn comprehensive_syntax_analyzer:app --reload ---
