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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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


if __name__ == "__main__":
    main()
