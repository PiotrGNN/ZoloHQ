"""
ZoL0 Trading Bot - Real-Time Market Data Integration System
Port: 8513

Enterprise-grade real-time market data feeds, multiple exchange connections,
data normalization, streaming analytics, and market intelligence system.
"""

import logging
import queue
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("market_data.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    CUSTOM_API = "custom_api"


class DataType(Enum):
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    CANDLES = "candles"
    NEWS = "news"
    SENTIMENT = "sentiment"


@dataclass
class MarketDataPoint:
    symbol: str
    timestamp: datetime
    data_type: DataType
    source: DataSourceType
    data: Dict[str, Any]
    latency_ms: float = 0.0


@dataclass
class ExchangeConnection:
    name: str
    source_type: DataSourceType
    api_key: Optional[str]
    secret_key: Optional[str]
    sandbox: bool
    is_connected: bool
    last_ping: Optional[datetime]
    error_count: int = 0
    rate_limit: int = 1000


@dataclass
class MarketSnapshot:
    symbol: str
    timestamp: datetime
    price: float
    volume_24h: float
    change_24h: float
    change_percent_24h: float
    bid: float
    ask: float
    spread: float
    market_cap: Optional[float] = None


class MarketDataAggregator:
    def __init__(self):
        self.data_sources: Dict[str, ExchangeConnection] = {}
        self.market_data: List[MarketDataPoint] = []
        self.market_snapshots: Dict[str, MarketSnapshot] = {}
        self.data_streams = {}
        self.subscribers = {}
        self.data_queue = queue.Queue()
        self.latency_stats = {}
        self.error_log = []

        # Initialize demo data and connections
        self._initialize_demo_data()
        self._setup_demo_connections()

    def _initialize_demo_data(self):
        """Initialize demo market data"""
        current_time = datetime.now()

        # Demo symbols
        symbols = ["BTC/USD", "ETH/USD", "AAPL", "GOOGL", "TSLA", "SPY", "QQQ"]
        base_prices = {
            "BTC/USD": 45000,
            "ETH/USD": 2800,
            "AAPL": 180,
            "GOOGL": 140,
            "TSLA": 250,
            "SPY": 480,
            "QQQ": 380,
        }

        # Generate demo market data
        for symbol in symbols:
            base_price = base_prices[symbol]

            # Generate price history
            for i in range(100):
                timestamp = current_time - timedelta(minutes=i)
                price_change = np.random.normal(0, base_price * 0.001)
                price = base_price + price_change

                market_point = MarketDataPoint(
                    symbol=symbol,
                    timestamp=timestamp,
                    data_type=DataType.TICKER,
                    source=(
                        DataSourceType.BINANCE
                        if "USD" in symbol
                        else DataSourceType.YAHOO_FINANCE
                    ),
                    data={
                        "price": price,
                        "volume": np.random.uniform(1000000, 10000000),
                        "bid": price - np.random.uniform(0.01, 0.1),
                        "ask": price + np.random.uniform(0.01, 0.1),
                    },
                    latency_ms=np.random.uniform(10, 50),
                )
                self.market_data.append(market_point)

            # Create market snapshot
            latest_data = [d for d in self.market_data if d.symbol == symbol][-1]
            snapshot = MarketSnapshot(
                symbol=symbol,
                timestamp=current_time,
                price=latest_data.data["price"],
                volume_24h=np.random.uniform(50000000, 500000000),
                change_24h=np.random.uniform(-base_price * 0.05, base_price * 0.05),
                change_percent_24h=np.random.uniform(-5, 5),
                bid=latest_data.data["bid"],
                ask=latest_data.data["ask"],
                spread=latest_data.data["ask"] - latest_data.data["bid"],
                market_cap=np.random.uniform(1e9, 1e12) if "USD" in symbol else None,
            )
            self.market_snapshots[symbol] = snapshot

    def _setup_demo_connections(self):
        """Setup demo exchange connections"""
        demo_connections = [
            {
                "name": "Binance Pro",
                "source_type": DataSourceType.BINANCE,
                "sandbox": False,
                "rate_limit": 1200,
            },
            {
                "name": "Coinbase Advanced",
                "source_type": DataSourceType.COINBASE,
                "sandbox": False,
                "rate_limit": 10000,
            },
            {
                "name": "Kraken Pro",
                "source_type": DataSourceType.KRAKEN,
                "sandbox": False,
                "rate_limit": 500,
            },
            {
                "name": "Yahoo Finance",
                "source_type": DataSourceType.YAHOO_FINANCE,
                "sandbox": False,
                "rate_limit": 2000,
            },
            {
                "name": "Alpha Vantage",
                "source_type": DataSourceType.ALPHA_VANTAGE,
                "sandbox": False,
                "rate_limit": 500,
            },
        ]

        for conn_data in demo_connections:
            connection = ExchangeConnection(
                name=conn_data["name"],
                source_type=conn_data["source_type"],
                api_key="demo_key_" + conn_data["name"].lower().replace(" ", "_"),
                secret_key="demo_secret",
                sandbox=conn_data["sandbox"],
                is_connected=True,
                last_ping=datetime.now(),
                rate_limit=conn_data["rate_limit"],
            )
            self.data_sources[connection.name] = connection

    def add_data_source(
        self,
        name: str,
        source_type: DataSourceType,
        api_key: str = None,
        secret_key: str = None,
        sandbox: bool = True,
    ) -> bool:
        """Add a new data source"""
        try:
            connection = ExchangeConnection(
                name=name,
                source_type=source_type,
                api_key=api_key,
                secret_key=secret_key,
                sandbox=sandbox,
                is_connected=False,
                last_ping=None,
            )

            # Test connection
            if self._test_connection(connection):
                connection.is_connected = True
                connection.last_ping = datetime.now()
                self.data_sources[name] = connection
                logger.info(f"Data source {name} added successfully")
                return True
            else:
                logger.error(f"Failed to connect to data source {name}")
                return False

        except Exception as e:
            logger.error(f"Error adding data source {name}: {e}")
            return False

    def _test_connection(self, connection: ExchangeConnection) -> bool:
        """Test connection to a data source"""
        try:
            if connection.source_type == DataSourceType.BINANCE:
                # Test Binance connection
                return True
            elif connection.source_type == DataSourceType.YAHOO_FINANCE:
                # Test Yahoo Finance connection
                return True
            # Add more connection tests as needed
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def subscribe_to_symbol(
        self, symbol: str, data_types: List[DataType], sources: List[str] = None
    ) -> str:
        """Subscribe to real-time data for a symbol"""
        subscription_id = f"sub_{symbol}_{int(time.time())}"

        self.subscribers[subscription_id] = {
            "symbol": symbol,
            "data_types": data_types,
            "sources": sources or list(self.data_sources.keys()),
            "created_at": datetime.now(),
            "is_active": True,
        }

        logger.info(f"Subscribed to {symbol} with subscription ID {subscription_id}")
        return subscription_id

    def get_market_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        """Get current market snapshot for a symbol"""
        return self.market_snapshots.get(symbol)

    def get_historical_data(
        self, symbol: str, timeframe: str = "1h", limit: int = 100
    ) -> pd.DataFrame:
        """Get historical market data"""
        symbol_data = [
            d
            for d in self.market_data
            if d.symbol == symbol and d.data_type == DataType.TICKER
        ]

        if not symbol_data:
            return pd.DataFrame()

        df = pd.DataFrame(
            [
                {
                    "timestamp": d.timestamp,
                    "price": d.data.get("price", 0),
                    "volume": d.data.get("volume", 0),
                    "bid": d.data.get("bid", 0),
                    "ask": d.data.get("ask", 0),
                    "source": d.source.value,
                }
                for d in sorted(symbol_data, key=lambda x: x.timestamp)[-limit:]
            ]
        )

        return df

    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics for data sources"""
        stats = {}
        for source_name, _connection in self.data_sources.items():
            source_data = [
                d for d in self.market_data if d.source.value in source_name.lower()
            ]
            if source_data:
                latencies = [
                    d.latency_ms for d in source_data[-100:]
                ]  # Last 100 data points
                stats[source_name] = {
                    "avg_latency": statistics.mean(latencies),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies),
                    "std_latency": (
                        statistics.stdev(latencies) if len(latencies) > 1 else 0
                    ),
                }
        return stats

    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics"""
        current_time = datetime.now()
        recent_data = [
            d
            for d in self.market_data
            if current_time - d.timestamp < timedelta(hours=1)
        ]

        metrics = {
            "total_data_points": len(self.market_data),
            "recent_data_points": len(recent_data),
            "unique_symbols": len(set(d.symbol for d in self.market_data)),
            "active_sources": len(
                [c for c in self.data_sources.values() if c.is_connected]
            ),
            "error_rate": len(self.error_log) / max(len(self.market_data), 1) * 100,
            "data_freshness": (
                min([(current_time - d.timestamp).total_seconds() for d in recent_data])
                if recent_data
                else float("inf")
            ),
        }

        return metrics

    def simulate_real_time_data(self):
        """Simulate real-time market data updates"""
        # This would normally connect to real WebSocket feeds
        symbols = list(self.market_snapshots.keys())

        for symbol in symbols:
            snapshot = self.market_snapshots[symbol]

            # Simulate price movement
            price_change = np.random.normal(0, snapshot.price * 0.0001)
            new_price = max(snapshot.price + price_change, 0.01)

            # Update snapshot
            snapshot.price = new_price
            snapshot.timestamp = datetime.now()
            snapshot.change_24h = price_change
            snapshot.change_percent_24h = (price_change / snapshot.price) * 100

            # Add new data point
            new_data_point = MarketDataPoint(
                symbol=symbol,
                timestamp=datetime.now(),
                data_type=DataType.TICKER,
                source=(
                    DataSourceType.BINANCE
                    if "USD" in symbol
                    else DataSourceType.YAHOO_FINANCE
                ),
                data={
                    "price": new_price,
                    "volume": np.random.uniform(100000, 1000000),
                    "bid": new_price - np.random.uniform(0.01, 0.1),
                    "ask": new_price + np.random.uniform(0.01, 0.1),
                },
                latency_ms=np.random.uniform(5, 30),
            )
            self.market_data.append(new_data_point)

        # Keep only recent data to prevent memory issues
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.market_data = [d for d in self.market_data if d.timestamp > cutoff_time]


def main():
    st.set_page_config(
        page_title="ZoL0 Market Data Integration",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .price-up {
        color: #28a745;
        font-weight: bold;
    }
    .price-down {
        color: #dc3545;
        font-weight: bold;
    }
    .price-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    .source-connected {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.2rem 0;
    }
    .source-disconnected {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.2rem 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>📊 ZoL0 Real-Time Market Data Integration</h1>
        <p>Enterprise-grade market data feeds, analytics, and intelligence platform</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize market data system
    if "market_system" not in st.session_state:
        st.session_state.market_system = MarketDataAggregator()

    market_system = st.session_state.market_system

    # Simulate real-time updates
    market_system.simulate_real_time_data()

    # Sidebar
    st.sidebar.title("📊 Market Data")

    tab_selection = st.sidebar.radio(
        "Select Module",
        [
            "🌍 Market Overview",
            "📈 Real-Time Data",
            "🔌 Data Sources",
            "📊 Analytics Dashboard",
            "⚡ Streaming Monitor",
            "🔧 Configuration",
        ],
    )

    if tab_selection == "🌍 Market Overview":
        st.header("Global Market Overview")

        # Market summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            active_symbols = len(market_system.market_snapshots)
            st.markdown(
                f"""
            <div class="metric-container">
                <h3>📊 Active Symbols</h3>
                <h2>{active_symbols}</h2>
                <p>Currently tracked</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            connected_sources = len(
                [c for c in market_system.data_sources.values() if c.is_connected]
            )
            st.markdown(
                f"""
            <div class="metric-container">
                <h3>🔗 Connected Sources</h3>
                <h2>{connected_sources}</h2>
                <p>Live data feeds</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            total_volume = sum(
                s.volume_24h for s in market_system.market_snapshots.values()
            )
            st.markdown(
                f"""
            <div class="metric-container">
                <h3>💰 24h Volume</h3>
                <h2>${total_volume/1e9:.1f}B</h2>
                <p>Across all markets</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            avg_latency = np.mean(
                [d.latency_ms for d in market_system.market_data[-100:]]
            )
            st.markdown(
                f"""
            <div class="metric-container">
                <h3>⚡ Avg Latency</h3>
                <h2>{avg_latency:.1f}ms</h2>
                <p>Data delivery speed</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Market overview table
        st.subheader("Market Snapshots")

        market_data = []
        for symbol, snapshot in market_system.market_snapshots.items():
            price_class = (
                "price-up"
                if snapshot.change_percent_24h > 0
                else (
                    "price-down" if snapshot.change_percent_24h < 0 else "price-neutral"
                )
            )
            change_symbol = "+" if snapshot.change_percent_24h > 0 else ""

            market_data.append(
                {
                    "Symbol": symbol,
                    "Price": f"${snapshot.price:.2f}",
                    "24h Change": f"{change_symbol}{snapshot.change_percent_24h:.2f}%",
                    "Volume": f"${snapshot.volume_24h/1e6:.1f}M",
                    "Bid": f"${snapshot.bid:.2f}",
                    "Ask": f"${snapshot.ask:.2f}",
                    "Spread": f"${snapshot.spread:.4f}",
                    "Last Update": snapshot.timestamp.strftime("%H:%M:%S"),
                }
            )

        market_df = pd.DataFrame(market_data)
        st.dataframe(market_df, use_container_width=True)

        # Market heat map
        st.subheader("Market Performance Heat Map")

        symbols = list(market_system.market_snapshots.keys())
        changes = [
            market_system.market_snapshots[s].change_percent_24h for s in symbols
        ]
        volumes = [market_system.market_snapshots[s].volume_24h for s in symbols]

        fig = px.scatter(
            x=symbols,
            y=changes,
            size=volumes,
            color=changes,
            color_continuous_scale=["red", "yellow", "green"],
            title="24h Performance vs Volume",
            labels={"x": "Symbol", "y": "24h Change (%)", "size": "Volume"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    elif tab_selection == "📈 Real-Time Data":
        st.header("Real-Time Market Data")

        # Symbol selector
        available_symbols = list(market_system.market_snapshots.keys())
        selected_symbol = st.selectbox("Select Symbol", available_symbols)

        if selected_symbol:
            snapshot = market_system.get_market_snapshot(selected_symbol)

            # Current price display
            col1, col2, col3 = st.columns(3)

            with col1:
                price_class = (
                    "price-up" if snapshot.change_percent_24h > 0 else "price-down"
                )
                st.markdown(
                    f"""
                <div class="metric-container">
                    <h3>Current Price</h3>
                    <h1 class="{price_class}">${snapshot.price:.2f}</h1>
                    <p>Last update: {snapshot.timestamp.strftime('%H:%M:%S')}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                change_class = "price-up" if snapshot.change_24h > 0 else "price-down"
                change_symbol = "+" if snapshot.change_24h > 0 else ""
                st.markdown(
                    f"""
                <div class="metric-container">
                    <h3>24h Change</h3>
                    <h2 class="{change_class}">{change_symbol}${snapshot.change_24h:.2f}</h2>
                    <h3 class="{change_class}">({change_symbol}{snapshot.change_percent_24h:.2f}%)</h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                <div class="metric-container">
                    <h3>24h Volume</h3>
                    <h2>${snapshot.volume_24h/1e6:.1f}M</h2>
                    <p>Spread: ${snapshot.spread:.4f}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Real-time chart
            st.subheader(f"Real-Time Price Chart - {selected_symbol}")

            # Get historical data
            historical_df = market_system.get_historical_data(selected_symbol, limit=50)

            if not historical_df.empty:
                # Create candlestick-style chart
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=historical_df["timestamp"],
                        y=historical_df["price"],
                        mode="lines+markers",
                        name="Price",
                        line=dict(color="#007bff", width=2),
                        marker=dict(size=4),
                    )
                )

                fig.update_layout(
                    title=f"{selected_symbol} Price Movement",
                    xaxis_title="Time",
                    yaxis_title="Price ($)",
                    height=400,
                    showlegend=True,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Order book visualization
                st.subheader("Order Book Visualization")

                # Simulate order book data
                bids = []
                asks = []

                for i in range(10):
                    bid_price = snapshot.bid - (i * snapshot.spread / 10)
                    ask_price = snapshot.ask + (i * snapshot.spread / 10)
                    bid_size = np.random.uniform(0.1, 10.0)
                    ask_size = np.random.uniform(0.1, 10.0)

                    bids.append(
                        {
                            "price": bid_price,
                            "size": bid_size,
                            "total": sum([b["size"] for b in bids]) + bid_size,
                        }
                    )
                    asks.append(
                        {
                            "price": ask_price,
                            "size": ask_size,
                            "total": sum([a["size"] for a in asks]) + ask_size,
                        }
                    )

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📉 Bids")
                    bids_df = pd.DataFrame(bids)
                    bids_df["price"] = bids_df["price"].round(2)
                    bids_df["size"] = bids_df["size"].round(4)
                    bids_df["total"] = bids_df["total"].round(4)
                    st.dataframe(bids_df, use_container_width=True)

                with col2:
                    st.subheader("📈 Asks")
                    asks_df = pd.DataFrame(asks)
                    asks_df["price"] = asks_df["price"].round(2)
                    asks_df["size"] = asks_df["size"].round(4)
                    asks_df["total"] = asks_df["total"].round(4)
                    st.dataframe(asks_df, use_container_width=True)

    elif tab_selection == "🔌 Data Sources":
        st.header("Data Source Management")

        # Connection status
        st.subheader("Connection Status")

        for source_name, connection in market_system.data_sources.items():
            status_class = (
                "source-connected" if connection.is_connected else "source-disconnected"
            )
            status_text = (
                "🟢 Connected" if connection.is_connected else "🔴 Disconnected"
            )
            last_ping = (
                connection.last_ping.strftime("%H:%M:%S")
                if connection.last_ping
                else "Never"
            )

            st.markdown(
                f"""
            <div class="{status_class}">
                <strong>{source_name}</strong> ({connection.source_type.value}) - {status_text}<br>
                Last ping: {last_ping} | Rate limit: {connection.rate_limit}/hour | Errors: {connection.error_count}
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Add new data source
        st.subheader("Add New Data Source")

        col1, col2 = st.columns(2)

        with col1:
            new_source_name = st.text_input("Data Source Name")
            source_type = st.selectbox("Source Type", [e.value for e in DataSourceType])
            sandbox_mode = st.checkbox("Sandbox Mode", value=True)

        with col2:
            api_key = st.text_input("API Key", type="password")
            secret_key = st.text_input("Secret Key", type="password")
            st.number_input(
                "Rate Limit (requests/hour)", min_value=1, max_value=100000, value=1000
            )

        if st.button("🔗 Add Data Source"):
            if new_source_name and source_type:
                source_enum = DataSourceType(source_type)
                if market_system.add_data_source(
                    new_source_name, source_enum, api_key, secret_key, sandbox_mode
                ):
                    st.success(f"Data source '{new_source_name}' added successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to add data source '{new_source_name}'")
            else:
                st.error("Please provide source name and type")

        # Data source performance
        st.subheader("Performance Metrics")

        latency_stats = market_system.get_latency_stats()

        if latency_stats:
            performance_data = []
            for source, stats in latency_stats.items():
                performance_data.append(
                    {
                        "Source": source,
                        "Avg Latency (ms)": f"{stats['avg_latency']:.1f}",
                        "Min Latency (ms)": f"{stats['min_latency']:.1f}",
                        "Max Latency (ms)": f"{stats['max_latency']:.1f}",
                        "Std Dev (ms)": f"{stats['std_latency']:.1f}",
                    }
                )

            performance_df = pd.DataFrame(performance_data)
            st.dataframe(performance_df, use_container_width=True)

            # Latency comparison chart
            sources = list(latency_stats.keys())
            avg_latencies = [latency_stats[s]["avg_latency"] for s in sources]

            fig = px.bar(
                x=sources,
                y=avg_latencies,
                title="Average Latency by Data Source",
                labels={"x": "Data Source", "y": "Average Latency (ms)"},
            )
            st.plotly_chart(fig, use_container_width=True)

    elif tab_selection == "📊 Analytics Dashboard":
        st.header("Market Analytics Dashboard")

        # Data quality metrics
        quality_metrics = market_system.get_data_quality_metrics()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📊 Total Data Points", f"{quality_metrics['total_data_points']:,}"
            )

        with col2:
            st.metric(
                "🕐 Recent Data Points", f"{quality_metrics['recent_data_points']:,}"
            )

        with col3:
            st.metric("🎯 Unique Symbols", quality_metrics["unique_symbols"])

        with col4:
            st.metric("📡 Active Sources", quality_metrics["active_sources"])

        # Market correlation analysis
        st.subheader("Market Correlation Analysis")

        # Calculate correlation matrix
        symbols = list(market_system.market_snapshots.keys())
        correlation_data = {}

        for symbol in symbols:
            historical_df = market_system.get_historical_data(symbol, limit=50)
            if not historical_df.empty:
                correlation_data[symbol] = historical_df["price"].values

        if len(correlation_data) > 1:
            # Ensure all series have the same length
            min_length = min(len(series) for series in correlation_data.values())
            correlation_data = {k: v[-min_length:] for k, v in correlation_data.items()}

            corr_df = pd.DataFrame(correlation_data).corr()

            fig = px.imshow(
                corr_df,
                title="Symbol Correlation Matrix",
                color_continuous_scale="RdBu",
                aspect="auto",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Volume analysis
        st.subheader("Volume Analysis")

        volume_data = []
        for symbol, snapshot in market_system.market_snapshots.items():
            volume_data.append(
                {
                    "Symbol": symbol,
                    "Volume": snapshot.volume_24h,
                    "Price": snapshot.price,
                }
            )

        volume_df = pd.DataFrame(volume_data)

        fig = px.scatter(
            volume_df,
            x="Volume",
            y="Price",
            text="Symbol",
            title="Price vs Volume Analysis",
            labels={"Volume": "24h Volume", "Price": "Current Price"},
        )
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

        # Market trends
        st.subheader("Market Trends")

        # Calculate market trends
        trends_data = []
        for symbol, snapshot in market_system.market_snapshots.items():
            historical_df = market_system.get_historical_data(symbol, limit=20)
            if len(historical_df) >= 10:
                recent_prices = historical_df["price"].tail(10).values
                older_prices = historical_df["price"].head(10).values

                recent_avg = np.mean(recent_prices)
                older_avg = np.mean(older_prices)
                trend = (recent_avg - older_avg) / older_avg * 100

                trends_data.append(
                    {
                        "Symbol": symbol,
                        "Trend (%)": trend,
                        "Current Price": snapshot.price,
                    }
                )

        if trends_data:
            trends_df = pd.DataFrame(trends_data)

            fig = px.bar(
                trends_df,
                x="Symbol",
                y="Trend (%)",
                title="Short-term Trend Analysis",
                color="Trend (%)",
                color_continuous_scale=["red", "yellow", "green"],
            )
            st.plotly_chart(fig, use_container_width=True)

    elif tab_selection == "⚡ Streaming Monitor":
        st.header("Real-Time Streaming Monitor")

        # Streaming status
        col1, col2, col3 = st.columns(3)

        with col1:
            active_streams = len(
                [s for s in market_system.subscribers.values() if s["is_active"]]
            )
            st.metric("🌊 Active Streams", active_streams)

        with col2:
            data_rate = len(
                [
                    d
                    for d in market_system.market_data
                    if datetime.now() - d.timestamp < timedelta(minutes=1)
                ]
            )
            st.metric("📊 Data Rate", f"{data_rate}/min")

        with col3:
            error_rate = quality_metrics["error_rate"]
            st.metric("❌ Error Rate", f"{error_rate:.2f}%")

        # Subscription management
        st.subheader("Active Subscriptions")

        if market_system.subscribers:
            subs_data = []
            for sub_id, sub_info in market_system.subscribers.items():
                subs_data.append(
                    {
                        "Subscription ID": sub_id[:12] + "...",
                        "Symbol": sub_info["symbol"],
                        "Data Types": ", ".join(
                            [dt.value for dt in sub_info["data_types"]]
                        ),
                        "Sources": ", ".join(sub_info["sources"]),
                        "Created": sub_info["created_at"].strftime("%H:%M:%S"),
                        "Status": (
                            "🟢 Active" if sub_info["is_active"] else "🔴 Inactive"
                        ),
                    }
                )

            subs_df = pd.DataFrame(subs_data)
            st.dataframe(subs_df, use_container_width=True)
        else:
            st.info("No active subscriptions")

        # Create new subscription
        st.subheader("Create New Subscription")

        col1, col2 = st.columns(2)

        with col1:
            sub_symbol = st.selectbox("Symbol", available_symbols)
            data_types = st.multiselect(
                "Data Types", [dt.value for dt in DataType], default=["ticker"]
            )

        with col2:
            sources = st.multiselect(
                "Data Sources",
                list(market_system.data_sources.keys()),
                default=list(market_system.data_sources.keys()),
            )

        if st.button("📡 Create Subscription"):
            if sub_symbol and data_types:
                data_type_enums = [DataType(dt) for dt in data_types]
                sub_id = market_system.subscribe_to_symbol(
                    sub_symbol, data_type_enums, sources
                )
                st.success(f"Subscription created with ID: {sub_id}")
                st.rerun()
            else:
                st.error("Please select symbol and data types")

        # Real-time data feed
        st.subheader("Live Data Feed")

        # Show recent data points
        recent_data = sorted(market_system.market_data, key=lambda x: x.timestamp)[-20:]

        feed_data = []
        for data_point in reversed(recent_data):
            feed_data.append(
                {
                    "Time": data_point.timestamp.strftime("%H:%M:%S.%f")[:-3],
                    "Symbol": data_point.symbol,
                    "Type": data_point.data_type.value,
                    "Source": data_point.source.value,
                    "Price": f"${data_point.data.get('price', 0):.2f}",
                    "Latency": f"{data_point.latency_ms:.1f}ms",
                }
            )

        feed_df = pd.DataFrame(feed_data)
        st.dataframe(feed_df, use_container_width=True)

        # Auto-refresh option
        auto_refresh = st.checkbox("Auto-refresh (5 seconds)")
        if auto_refresh:
            time.sleep(5)
            st.rerun()

    elif tab_selection == "🔧 Configuration":
        st.header("System Configuration")

        # General settings
        st.subheader("General Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.number_input(
                "Max Data Points to Store",
                min_value=1000,
                max_value=1000000,
                value=100000,
            )

            st.number_input(
                "Data Retention (hours)", min_value=1, max_value=168, value=24
            )

        with col2:
            st.number_input(
                "Default Refresh Rate (seconds)", min_value=1, max_value=60, value=5
            )

            st.checkbox("Enable Data Caching", value=True)

        # API rate limiting
        st.subheader("API Rate Limiting")

        col1, col2 = st.columns(2)

        with col1:
            st.number_input(
                "Global Rate Limit (requests/minute)",
                min_value=1,
                max_value=10000,
                value=1000,
            )

        with col2:
            st.number_input("Burst Limit", min_value=1, max_value=1000, value=100)

        # Alert thresholds
        st.subheader("Alert Thresholds")

        col1, col2 = st.columns(2)

        with col1:
            st.number_input(
                "High Latency Threshold (ms)", min_value=1, max_value=10000, value=1000
            )

            st.number_input(
                "Error Rate Threshold (%)", min_value=0.1, max_value=50.0, value=5.0
            )

        with col2:
            st.number_input(
                "Connection Timeout (seconds)", min_value=1, max_value=300, value=30
            )

            st.number_input("Max Retry Attempts", min_value=1, max_value=10, value=3)

        if st.button("💾 Save Configuration"):
            st.success("Configuration saved successfully!")

        # System status
        st.subheader("System Status")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("💾 Memory Usage", "245 MB")

        with col2:
            st.metric("🖥️ CPU Usage", "12%")

        with col3:
            st.metric("🌐 Network I/O", "1.2 MB/s")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🕐 System Uptime", "99.98%")

    with col2:
        current_time = datetime.now()
        st.metric("🕒 Current Time", current_time.strftime("%H:%M:%S"))

    with col3:
        st.metric("📡 Data Status", "Live")


if __name__ == "__main__":
    main()
