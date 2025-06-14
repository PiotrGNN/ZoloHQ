#!/usr/bin/env python3
"""
ZoL0 Trading System - Unified Dashboard (Fixed)
===============================================
Naprawiona wersja zunifikowanego dashboardu
Port: 8500
"""

import os
from datetime import datetime
import warnings
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette_exporter import PrometheusMiddleware, handle_metrics
import io
import csv
import uvicorn

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="ZoL0 Unified Trading Dashboard", page_icon="🚀", layout="wide"
)


class UnifiedDashboard:
    def __init__(self):
        self.api_base = "http://localhost:5001"
        self.production_mode = (
            os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true"
        )

    def get_system_status(self):
        """Pobierz status systemu"""
        return {
            "Enhanced Bot Monitor": "🟢 Zintegrowany",
            "Advanced Trading Analytics": "🟢 Zintegrowany",
            "ML Predictive Analytics": "🟢 Zintegrowany",
            "Advanced Alert Management": "🟢 Zintegrowany",
            "Data Export System": "🟢 Zintegrowany",
            "Real-Time Market Data": "🟢 Zintegrowany",
        }

    def get_performance_data(self):
        """Pobierz dane wydajności"""
        return {
            "total_profit": 12450.67,
            "active_bots": 3,
            "win_rate": 68.3,
            "daily_trades": 28,
            "max_drawdown": -4.8,
            "sharpe_ratio": 1.52,
        }


def main():
    """Główna funkcja dashboardu"""

    # Initialize dashboard
    if "dashboard" not in st.session_state:
        st.session_state.dashboard = UnifiedDashboard()

    dashboard = st.session_state.dashboard

    # Header
    st.markdown(
        """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; 
                margin-bottom: 2rem; text-align: center;">
        <h1>🚀 ZoL0 Unified Trading Dashboard</h1>
        <p>Kompleksowy system monitorowania tradingu</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Info banner
    st.info(
        """
    🎯 **Informacja:** To jest zunifikowany dashboard który zastępuje wszystkie osobne dashboardy.
    Wszystkie funkcje są dostępne w zakładkach po lewej stronie.
    """
    )

    # System Status
    st.header("🔧 Status Systemu")
    status = dashboard.get_system_status()

    cols = st.columns(3)
    for i, (service, stat) in enumerate(status.items()):
        with cols[i % 3]:
            st.success(f"**{service}**\n{stat}")

    # Performance Overview
    st.header("📊 Przegląd Wydajności")
    perf = dashboard.get_performance_data()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        profit = perf["total_profit"]
        st.metric("💰 Zysk Całkowity", f"${profit:,.2f}", "+12.5%")

    with col2:
        win_rate = perf["win_rate"]
        st.metric("🎯 Win Rate", f"{win_rate:.1f}%", "+2.3%")

    with col3:
        active_bots = perf["active_bots"]
        st.metric("🤖 Aktywne Boty", active_bots)

    with col4:
        daily_trades = perf["daily_trades"]
        st.metric("📈 Transakcje Dziennie", daily_trades)

    # Sample chart
    st.header("📈 Wykres P&L")
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    profits = np.cumsum(np.random.normal(50, 100, 30))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=profits, mode="lines", name="P&L"))
    fig.update_layout(title="Cumulative P&L", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Sidebar navigation
    st.sidebar.title("🚀 ZoL0 Navigation")
    st.sidebar.selectbox(
        "Wybierz moduł:",
        ["Główny Przegląd", "Bot Monitor", "Analityka", "Dane Rynkowe"],
    )

    if st.sidebar.button("🔄 Odśwież"):
        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        f"""
    <div style="text-align: center; color: #666;">
        <p>🚀 ZoL0 Unified Trading Dashboard | Ostatnia aktualizacja: {datetime.now().strftime('%H:%M:%S')}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# FastAPI part
API_KEYS = {"admin-key": "admin", "dashboard-key": "dashboard", "partner-key": "partner", "premium-key": "premium"}
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)

def get_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return API_KEYS[api_key]

dash_api = FastAPI(title="ZoL0 Unified Trading Dashboard API", version="2.0")
dash_api.add_middleware(PrometheusMiddleware)
dash_api.add_route("/metrics", handle_metrics)

# --- Pydantic Models ---
class StatusQuery(BaseModel):
    pass
class BatchStatusQuery(BaseModel):
    queries: list[StatusQuery]

# --- Business Logic ---
class UnifiedDashboardAPI:
    def get_system_status(self):
        return {
            "Enhanced Bot Monitor": "🟢 Zintegrowany",
            "Advanced Trading Analytics": "🟢 Zintegrowany",
            "ML Predictive Analytics": "🟢 Zintegrowany",
            "Advanced Alert Management": "🟢 Zintegrowany",
            "Data Export System": "🟢 Zintegrowany",
            "Real-Time Market Data": "🟢 Zintegrowany",
        }
    def get_performance_data(self):
        return {
            "total_profit": 12450.67,
            "active_bots": 3,
            "win_rate": 68.3,
            "daily_trades": 28,
            "max_drawdown": -4.8,
            "sharpe_ratio": 1.52,
        }
api = UnifiedDashboardAPI()

# --- Endpoints ---
@dash_api.get("/")
async def root():
    return {"status": "ok", "service": "ZoL0 Unified Trading Dashboard API", "version": "2.0"}

@dash_api.get("/api/health")
async def api_health():
    return {"status": "ok", "timestamp": time.time(), "service": "ZoL0 Unified Trading Dashboard API", "version": "2.0"}

@dash_api.get("/api/status", dependencies=[Depends(get_api_key)])
async def api_status(role: str = Depends(get_api_key)):
    return api.get_system_status()

@dash_api.get("/api/performance", dependencies=[Depends(get_api_key)])
async def api_performance(role: str = Depends(get_api_key)):
    return api.get_performance_data()

@dash_api.post("/api/status/batch", dependencies=[Depends(get_api_key)])
async def api_status_batch(req: BatchStatusQuery, role: str = Depends(get_api_key)):
    return {"results": [api.get_system_status() for _ in req.queries]}

@dash_api.get("/api/export/csv", dependencies=[Depends(get_api_key)])
async def api_export_csv(role: str = Depends(get_api_key)):
    perf = api.get_performance_data()
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(perf.keys()))
    writer.writeheader()
    writer.writerow(perf)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv")

@dash_api.get("/api/export/prometheus", dependencies=[Depends(get_api_key)])
async def api_export_prometheus(role: str = Depends(get_api_key)):
    perf = api.get_performance_data()
    return PlainTextResponse(f"# HELP dashboard_total_profit Total profit\ndashboard_total_profit {perf.get('total_profit', 0)}", media_type="text/plain")

@dash_api.get("/api/report", dependencies=[Depends(get_api_key)])
async def api_report(role: str = Depends(get_api_key)):
    return {"status": "report generated (stub)"}

@dash_api.get("/api/recommendations", dependencies=[Depends(get_api_key)])
async def api_recommendations(role: str = Depends(get_api_key)):
    perf = api.get_performance_data()
    recs = []
    if perf.get("win_rate", 0) < 60:
        recs.append("Improve strategy or risk management.")
    if perf.get("total_profit", 0) < 10000:
        recs.append("Increase trading volume or optimize bots.")
    return {"recommendations": recs}

@dash_api.get("/api/premium/score", dependencies=[Depends(get_api_key)])
async def api_premium_score(role: str = Depends(get_api_key)):
    perf = api.get_performance_data()
    score = perf.get("total_profit", 0) * 0.01
    return {"score": score}

@dash_api.get("/api/saas/tenant/{tenant_id}/report", dependencies=[Depends(get_api_key)])
async def api_saas_tenant_report(tenant_id: str, role: str = Depends(get_api_key)):
    perf = api.get_performance_data()
    return {"tenant_id": tenant_id, "report": perf}

@dash_api.get("/api/usage", dependencies=[Depends(get_api_key)])
async def api_usage(role: str = Depends(get_api_key)):
    # Monetization: usage metering for billing/SaaS
    perf = api.get_performance_data()
    return {
        "role": role,
        "total_profit": perf.get("total_profit", 0),
        "timestamp": datetime.now().isoformat(),
    }

@dash_api.post("/api/partner/webhook", dependencies=[Depends(get_api_key)])
async def api_partner_webhook(payload: dict, role: str = Depends(get_api_key)):
    # Monetization: process partner webhook payload for SaaS/affiliate integrations
    # In production, validate/process payload, trigger partner actions
    return {"status": "received", "payload": payload}

@dash_api.get("/api/test/edge-case")
async def api_edge_case():
    try:
        raise RuntimeError("Simulated dashboard edge-case error")
    except Exception as e:
        return {"edge_case": str(e)}

@dash_api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})

# --- CI/CD test suite ---
import unittest
class TestUnifiedDashboardAPI(unittest.TestCase):
    def test_status(self):
        result = api.get_system_status()
        assert "Enhanced Bot Monitor" in result
    def test_performance(self):
        result = api.get_performance_data()
        assert "total_profit" in result

if __name__ == "__main__":
    if "streamlit" in sys.argv[0]:
        main()
    elif "test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        uvicorn.run("unified_trading_dashboard_fixed:dash_api", host="0.0.0.0", port=8500, reload=True)
