#!/usr/bin/env python3
"""
Portfolio Dashboard for ZoL0 Trading System
Real-time portfolio monitoring with production data integration
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Add ZoL0-master to path for imports
sys.path.append(str(Path(__file__).parent / "ZoL0-master"))

# Configure page
st.set_page_config(
    page_title="ZoL0 Portfolio Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styling
st.markdown(
    """
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class PortfolioDashboard:
    """Main Portfolio Dashboard Class"""

    def __init__(self):
        self.api_base_url = "http://localhost:5001"
        self.production_manager = None
        self.production_mode = (
            os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true"
        )
        self._initialize_production_data()

    def _initialize_production_data(self):
        """Initialize production data manager if available"""
        try:
            from production_data_manager import get_production_data

            self.production_manager = get_production_data()
            if self.production_mode:
                st.sidebar.success("🟢 Production data enabled")
            else:
                st.sidebar.info("🔄 Testnet data enabled")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Production data not available: {e}")

    def get_portfolio_data(self):
        """Get portfolio data from production manager or API"""
        try:
            # Try production manager first
            if self.production_manager and self.production_mode:
                data = self.production_manager.get_enhanced_portfolio_details()
                if data and data.get("success"):
                    return data

            # Fallback to API
            response = requests.get(f"{self.api_base_url}/api/portfolio", timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching portfolio data: {e}")

        # Final fallback
        return self._get_fallback_data()

    def _get_fallback_data(self):
        """Fallback portfolio data"""
        return {
            "success": True,
            "data_source": "fallback",
            "portfolio_summary": {
                "total_equity": 10000.0,
                "total_available": 9500.0,
                "total_wallet_balance": 10000.0,
                "unrealized_pnl": 250.0,
                "active_positions": 3,
            },
            "coin_details": {
                "USDT": {
                    "equity": 8000.0,
                    "available_balance": 7500.0,
                    "wallet_balance": 8000.0,
                    "percentage_of_portfolio": 80.0,
                },
                "BTC": {
                    "equity": 1500.0,
                    "available_balance": 1500.0,
                    "wallet_balance": 1500.0,
                    "percentage_of_portfolio": 15.0,
                },
                "ETH": {
                    "equity": 500.0,
                    "available_balance": 500.0,
                    "wallet_balance": 500.0,
                    "percentage_of_portfolio": 5.0,
                },
            },
            "environment": "demo",
        }

    def render_header(self):
        """Render dashboard header"""
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.title("💼 Portfolio Dashboard")
            st.markdown("**Real-time portfolio monitoring and analysis**")

        with col2:
            if st.button("🔄 Refresh Data"):
                st.rerun()

        with col3:
            data_source = "Production" if self.production_mode else "Demo"
            st.markdown(
                f"""
            <div class="info-card">
                <h4>Data Source</h4>
                <h3>{data_source}</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

    def render_portfolio_summary(self, portfolio_data):
        """Render portfolio summary metrics"""
        st.subheader("📊 Portfolio Summary")

        summary = portfolio_data.get("portfolio_summary", {})

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_equity = summary.get("total_equity", 0)
            st.markdown(
                f"""
            <div class="metric-card">
                <h4>💰 Total Equity</h4>
                <h2>${total_equity:,.2f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            available = summary.get("total_available", 0)
            st.markdown(
                f"""
            <div class="success-card">
                <h4>💵 Available Balance</h4>
                <h2>${available:,.2f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            pnl = summary.get("unrealized_pnl", 0)
            card_class = "success-card" if pnl >= 0 else "warning-card"
            st.markdown(
                f"""
            <div class="{card_class}">
                <h4>📈 Unrealized P&L</h4>
                <h2>${pnl:,.2f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            positions = summary.get("active_positions", 0)
            st.markdown(
                f"""
            <div class="info-card">
                <h4>🎯 Active Positions</h4>
                <h2>{positions}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

    def render_coin_breakdown(self, portfolio_data):
        """Render coin breakdown charts"""
        st.subheader("🪙 Asset Allocation")

        coin_details = portfolio_data.get("coin_details", {})
        if not coin_details:
            st.warning("No coin data available")
            return

        # Prepare data for charts
        coins = []
        values = []
        percentages = []

        for coin, details in coin_details.items():
            coins.append(coin)
            values.append(details.get("equity", 0))
            percentages.append(details.get("percentage_of_portfolio", 0))

        col1, col2 = st.columns(2)

        with col1:
            # Pie chart
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=coins, values=values, textinfo="label+percent", hole=0.3
                    )
                ]
            )
            fig_pie.update_layout(title="Portfolio Distribution", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar chart
            fig_bar = px.bar(
                x=coins,
                y=values,
                title="Asset Values (USD)",
                labels={"x": "Asset", "y": "Value (USD)"},
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)

    def render_detailed_breakdown(self, portfolio_data):
        """Render detailed asset breakdown table"""
        st.subheader("📋 Detailed Asset Breakdown")

        coin_details = portfolio_data.get("coin_details", {})
        if not coin_details:
            st.warning("No detailed coin data available")
            return

        # Create DataFrame
        rows = []
        for coin, details in coin_details.items():
            rows.append(
                {
                    "Asset": coin,
                    "Equity": f"${details.get('equity', 0):,.4f}",
                    "Available": f"${details.get('available_balance', 0):,.4f}",
                    "Wallet Balance": f"${details.get('wallet_balance', 0):,.4f}",
                    "Locked": f"${details.get('locked_balance', 0):,.4f}",
                    "Portfolio %": f"{details.get('percentage_of_portfolio', 0):.2f}%",
                }
            )

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

    def render_position_info(self, portfolio_data):
        """Render position information"""
        st.subheader("🎯 Position Information")

        positions_summary = portfolio_data.get("positions_summary", {})
        positions = positions_summary.get("positions", [])

        if not positions:
            st.info("No active positions")
            return

        # Display positions in expandable format
        for i, position in enumerate(positions[:5]):  # Show first 5 positions
            with st.expander(f"Position {i+1}: {position.get('symbol', 'Unknown')}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Size", position.get("size", 0))

                with col2:
                    unrealized_pnl = float(position.get("unrealisedPnl", 0))
                    st.metric("Unrealized P&L", f"${unrealized_pnl:,.2f}")

                with col3:
                    side = position.get("side", "Unknown")
                    st.metric("Side", side)

    def render_market_context(self, portfolio_data):
        """Render market context"""
        st.subheader("🌍 Market Context")

        market_context = portfolio_data.get("market_context", {})

        col1, col2, col3 = st.columns(3)

        with col1:
            btc_price = market_context.get("btc_price", 0)
            st.markdown(
                f"""
            <div class="metric-card">
                <h4>₿ BTC Price</h4>
                <h3>${btc_price:,.2f}</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            eth_price = market_context.get("eth_price", 0)
            st.markdown(
                f"""
            <div class="metric-card">
                <h4>Ξ ETH Price</h4>
                <h3>${eth_price:,.2f}</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            last_updated = market_context.get("last_updated", "Unknown")
            if last_updated != "Unknown":
                try:
                    update_time = datetime.fromisoformat(
                        last_updated.replace("Z", "+00:00")
                    )
                    formatted_time = update_time.strftime("%H:%M:%S")
                except Exception:
                    logging.exception(
                        "Exception occurred in portfolio_dashboard at line 335"
                    )
                    formatted_time = "Unknown"
            else:
                formatted_time = "Unknown"

            st.markdown(
                f"""
            <div class="info-card">
                <h4>🕐 Last Updated</h4>
                <h3>{formatted_time}</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

    def render_connection_status(self, portfolio_data):
        """Render connection status"""
        st.subheader("🔗 Connection Status")

        data_source = portfolio_data.get("data_source", "unknown")
        environment = portfolio_data.get("environment", "unknown")
        connection_status = portfolio_data.get("connection_status", {})

        col1, col2 = st.columns(2)

        with col1:
            status_color = (
                "success-card" if data_source == "production_api" else "warning-card"
            )
            st.markdown(
                f"""
            <div class="{status_color}">
                <h4>📡 Data Source</h4>
                <h3>{data_source}</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            env_color = "success-card" if environment == "production" else "info-card"
            st.markdown(
                f"""
            <div class="{env_color}">
                <h4>🌐 Environment</h4>
                <h3>{environment}</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

        if isinstance(connection_status, dict) and connection_status:
            st.json(connection_status)


def main():
    """Main dashboard function"""
    dashboard = PortfolioDashboard()

    # Render header
    dashboard.render_header()

    # Get portfolio data
    with st.spinner("Loading portfolio data..."):
        portfolio_data = dashboard.get_portfolio_data()

    if portfolio_data and portfolio_data.get("success"):
        # Render main sections
        dashboard.render_portfolio_summary(portfolio_data)

        st.divider()

        # Two column layout for charts
        dashboard.render_coin_breakdown(portfolio_data)

        st.divider()

        # Detailed breakdown
        dashboard.render_detailed_breakdown(portfolio_data)

        st.divider()

        # Position and market info
        col1, col2 = st.columns(2)

        with col1:
            dashboard.render_position_info(portfolio_data)

        with col2:
            dashboard.render_market_context(portfolio_data)

        st.divider()

        # Connection status
        dashboard.render_connection_status(portfolio_data)

    else:
        st.error("Failed to load portfolio data")
        st.json(portfolio_data)

    # Auto-refresh
    time.sleep(1)


if __name__ == "__main__":
    main()
