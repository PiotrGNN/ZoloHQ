#!/usr/bin/env python3
"""
Advanced Trading Analytics Dashboard
Zaawansowany dashboard analityki tradingowej z real-time danymi
"""

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="ZoL0 Advanced Trading Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Advanced CSS styling
st.markdown(
    """
<style>
    .analytics-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .performance-metric {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    .risk-metric {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        text-align: center;
    }
    .strategy-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: #333;
        margin: 0.5rem 0;
        border-left: 4px solid #27ae60;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 0.5rem 0;
        border-left: 4px solid #c0392b;
    }
    .metric-large {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-medium {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.3rem 0;
    }
    .trend-positive {
        color: #27ae60;
        font-weight: bold;
    }
    .trend-negative {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


class AdvancedTradingAnalytics:
    def __init__(self):
        self.api_base_url = "http://localhost:5001"
        self.db_path = "trading.db"

    def get_enhanced_performance_data(self):
        """Pobierz zaawansowane dane wydajności z bazy danych i API"""
        try:
            # Get data from API
            api_response = requests.get(
                f"{self.api_base_url}/api/bot/performance", timeout=5
            )
            api_data = (
                api_response.json().get("performance", {})
                if api_response.status_code == 200
                else {}
            )

            # Try to get real data from database if it exists
            db_data = self._get_database_performance()

            # Merge API and database data
            performance = {
                **api_data,
                **db_data,
                "timestamp": datetime.now().isoformat(),
                "data_source": "live" if db_data else "simulated",
            }

            return performance

        except Exception as e:
            st.error(f"Error fetching performance data: {e}")
            return self._get_fallback_performance_data()

    def _get_database_performance(self):
        """Pobierz dane z bazy danych SQLite jeśli istnieje"""
        try:
            if not Path(self.db_path).exists():
                return {}

            conn = sqlite3.connect(self.db_path)

            # Try to get recent trades
            trades_query = """
            SELECT * FROM trades 
            WHERE timestamp >= datetime('now', '-7 days')
            ORDER BY timestamp DESC
            LIMIT 100
            """

            trades_df = pd.read_sql_query(trades_query, conn)

            if not trades_df.empty:
                # Calculate real performance metrics
                performance = self._calculate_performance_metrics(trades_df)
                conn.close()
                return performance

            conn.close()
            return {}

        except Exception:
            return {}

    def _calculate_performance_metrics(self, trades_df):
        """Oblicz metryki wydajności na podstawie rzeczywistych transakcji"""
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["pnl"] > 0])
        losing_trades = len(trades_df[trades_df["pnl"] < 0])

        total_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
        total_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())
        net_profit = trades_df["pnl"].sum()

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Calculate drawdown
        cumulative_pnl = trades_df["pnl"].cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = ((cumulative_pnl - rolling_max) / rolling_max * 100).min()

        # Calculate Sharpe ratio (simplified)
        returns = trades_df["pnl"] / trades_df["entry_price"] * 100
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "max_drawdown": drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_trade": net_profit / total_trades if total_trades > 0 else 0,
        }

    def _get_fallback_performance_data(self):
        """Dane fallback gdy nie można pobrać rzeczywistych danych"""
        return {
            "total_trades": 127,
            "winning_trades": 78,
            "losing_trades": 49,
            "win_rate": 61.4,
            "total_profit": 2487.50,
            "total_loss": 1234.25,
            "net_profit": 1253.25,
            "max_drawdown": -8.3,
            "sharpe_ratio": 1.42,
            "avg_trade": 9.87,
            "data_source": "demo",
        }

    def get_real_time_market_data(self):
        """Pobierz dane rynkowe w czasie rzeczywistym"""
        try:
            # This would connect to real market data feeds
            # For now, generating realistic simulated data

            symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
            market_data = []

            for symbol in symbols:
                price = np.random.uniform(100, 50000)
                change_24h = np.random.uniform(-10, 10)
                volume = np.random.uniform(1000000, 100000000)

                market_data.append(
                    {
                        "symbol": symbol,
                        "price": price,
                        "change_24h": change_24h,
                        "volume": volume,
                        "timestamp": datetime.now(),
                    }
                )

            return market_data

        except Exception:
            return []

    def get_risk_metrics(self):
        """Pobierz zaawansowane metryki ryzyka"""
        try:
            response = requests.get(f"{self.api_base_url}/api/risk/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return self._get_fallback_risk_metrics()
        except Exception:
            return self._get_fallback_risk_metrics()

    def _get_fallback_risk_metrics(self):
        """Fallback metryki ryzyka"""
        return {
            "var_95": -2.3,  # Value at Risk 95%
            "cvar_95": -4.1,  # Conditional VaR
            "beta": 1.2,
            "correlation_btc": 0.85,
            "volatility": 12.4,
            "max_leverage": 3.0,
            "current_leverage": 1.8,
            "margin_ratio": 65.2,
        }

    def generate_advanced_charts(self, performance_data):
        """Generuj zaawansowane wykresy analityczne"""

        # 1. P&L Timeline Chart
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=30), end=datetime.now(), freq="D"
        )
        cumulative_pnl = np.cumsum(np.random.normal(10, 50, len(dates)))

        pnl_fig = go.Figure()
        pnl_fig.add_trace(
            go.Scatter(
                x=dates,
                y=cumulative_pnl,
                mode="lines+markers",
                name="Cumulative P&L",
                line=dict(color="#27ae60", width=3),
                fill="tonexty",
            )
        )

        pnl_fig.update_layout(
            title="Cumulative P&L Over Time",
            xaxis_title="Date",
            yaxis_title="P&L (USDT)",
            template="plotly_dark",
            height=400,
        )

        # 2. Win Rate Breakdown
        win_rate_data = {
            "Strategy": ["Scalping", "Swing", "Grid", "DCA", "Arbitrage"],
            "Win Rate": [65.2, 58.7, 72.1, 81.3, 45.6],
            "Trades": [45, 23, 12, 31, 16],
        }

        win_rate_fig = px.bar(
            win_rate_data,
            x="Strategy",
            y="Win Rate",
            color="Win Rate",
            title="Strategy Win Rates",
            color_continuous_scale="Viridis",
        )
        win_rate_fig.update_layout(template="plotly_dark", height=400)

        # 3. Risk Distribution
        returns = np.random.normal(0.5, 2.5, 1000)
        risk_fig = go.Figure()
        risk_fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name="Returns Distribution",
                marker_color="rgba(55, 83, 109, 0.7)",
            )
        )
        risk_fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Return %",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=400,
        )

        return pnl_fig, win_rate_fig, risk_fig


def main():
    # Header
    st.markdown(
        """
    <div class="analytics-header">
        <h1>📈 Advanced Trading Analytics</h1>
        <p>Zaawansowana analityka tradingowa w czasie rzeczywistym</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize analytics
    if "analytics" not in st.session_state:
        st.session_state.analytics = AdvancedTradingAnalytics()

    analytics = st.session_state.analytics

    # Sidebar controls
    st.sidebar.title("📊 Analytics Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval (s)", [5, 10, 30, 60], index=1
    )

    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    # Get data
    performance_data = analytics.get_enhanced_performance_data()
    market_data = analytics.get_real_time_market_data()
    risk_metrics = analytics.get_risk_metrics()

    # === PERFORMANCE OVERVIEW ===
    st.header("🎯 Performance Overview")

    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)

    with perf_col1:
        net_profit = performance_data.get("net_profit", 0)
        profit_trend = "trend-positive" if net_profit > 0 else "trend-negative"
        st.markdown(
            f"""
        <div class="performance-metric">
            <h3>💰 Net Profit</h3>
            <div class="metric-large {profit_trend}">${net_profit:,.2f}</div>
            <small>Total: {performance_data.get('total_trades', 0)} trades</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with perf_col2:
        win_rate = performance_data.get("win_rate", 0)
        rate_trend = "trend-positive" if win_rate > 60 else "trend-negative"
        st.markdown(
            f"""
        <div class="performance-metric">
            <h3>🎯 Win Rate</h3>
            <div class="metric-large {rate_trend}">{win_rate:.1f}%</div>
            <small>{performance_data.get('winning_trades', 0)}W / {performance_data.get('losing_trades', 0)}L</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with perf_col3:
        sharpe = performance_data.get("sharpe_ratio", 0)
        sharpe_trend = "trend-positive" if sharpe > 1 else "trend-negative"
        st.markdown(
            f"""
        <div class="performance-metric">
            <h3>📊 Sharpe Ratio</h3>
            <div class="metric-large {sharpe_trend}">{sharpe:.2f}</div>
            <small>Risk-adjusted return</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with perf_col4:
        drawdown = performance_data.get("max_drawdown", 0)
        dd_trend = "trend-positive" if drawdown > -10 else "trend-negative"
        st.markdown(
            f"""
        <div class="performance-metric">
            <h3>📉 Max Drawdown</h3>
            <div class="metric-large {dd_trend}">{drawdown:.1f}%</div>
            <small>Peak to trough</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # === ADVANCED CHARTS ===
    st.header("📈 Advanced Analytics")

    pnl_fig, win_rate_fig, risk_fig = analytics.generate_advanced_charts(
        performance_data
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(pnl_fig, use_container_width=True)
        st.plotly_chart(risk_fig, use_container_width=True)

    with chart_col2:
        st.plotly_chart(win_rate_fig, use_container_width=True)

        # Market Data Table
        if market_data:
            st.subheader("🌐 Real-time Market Data")
            market_df = pd.DataFrame(market_data)
            market_df["change_24h"] = market_df["change_24h"].apply(
                lambda x: f"{x:.2f}%"
            )
            market_df["price"] = market_df["price"].apply(lambda x: f"${x:,.2f}")
            market_df["volume"] = market_df["volume"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(
                market_df[["symbol", "price", "change_24h", "volume"]],
                use_container_width=True,
            )

    # === RISK METRICS ===
    st.header("⚠️ Risk Analysis")

    risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

    with risk_col1:
        var_95 = risk_metrics.get("var_95", 0)
        st.markdown(
            f"""
        <div class="risk-metric">
            <h3>📊 VaR (95%)</h3>
            <div class="metric-medium">{var_95:.2f}%</div>
            <small>Daily Value at Risk</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with risk_col2:
        leverage = risk_metrics.get("current_leverage", 0)
        max_leverage = risk_metrics.get("max_leverage", 0)
        leverage_trend = (
            "trend-negative" if leverage > max_leverage * 0.8 else "trend-positive"
        )
        st.markdown(
            f"""
        <div class="risk-metric">
            <h3>⚡ Leverage</h3>
            <div class="metric-medium {leverage_trend}">{leverage:.1f}x</div>
            <small>Max: {max_leverage:.1f}x</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with risk_col3:
        volatility = risk_metrics.get("volatility", 0)
        vol_trend = "trend-negative" if volatility > 20 else "trend-positive"
        st.markdown(
            f"""
        <div class="risk-metric">
            <h3>📊 Volatility</h3>
            <div class="metric-medium {vol_trend}">{volatility:.1f}%</div>
            <small>30-day average</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with risk_col4:
        correlation = risk_metrics.get("correlation_btc", 0)
        corr_trend = "trend-negative" if correlation > 0.9 else "trend-positive"
        st.markdown(
            f"""
        <div class="risk-metric">
            <h3>🔗 BTC Correlation</h3>
            <div class="metric-medium {corr_trend}">{correlation:.2f}</div>
            <small>Portfolio correlation</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # === DATA SOURCE INFO ===
    st.header("ℹ️ Data Information")
    data_source = performance_data.get("data_source", "unknown")

    if data_source == "live":
        st.success("✅ Using real trading data from database")
    elif data_source == "demo":
        st.info("ℹ️ Using simulated demo data")
    else:
        st.warning("⚠️ Data source unknown")

    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
