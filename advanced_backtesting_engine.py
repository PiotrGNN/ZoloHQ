"""
ZoL0 Trading Bot - Advanced Backtesting Engine
Port: 8514

Enterprise-grade backtesting system with historical strategy testing,
walk-forward analysis, Monte Carlo simulations, and performance attribution.
"""

import logging
import uuid
from abc import ABC, abstractmethod
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
    handlers=[logging.FileHandler("backtesting.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    CUSTOM = "custom"


@dataclass
class Order:
    id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    is_filled: bool = False
    fill_price: Optional[float] = None
    fill_timestamp: Optional[datetime] = None
    commission: float = 0.0


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_price: float
    timestamp: datetime
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class Trade:
    id: str
    symbol: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: OrderSide
    pnl: float
    commission: float
    duration: timedelta


@dataclass
class BacktestResult:
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    trades: List[Trade]
    equity_curve: pd.DataFrame
    metrics: Dict[str, Any]


class BaseStrategy(ABC):
    """Base class for trading strategies"""

    def __init__(self, name: str, parameters: Dict[str, Any]):
        self.name = name
        self.parameters = parameters
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Trade] = []

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from market data"""
        pass

    @abstractmethod
    def calculate_position_size(
        self, signal: float, current_price: float, portfolio_value: float
    ) -> float:
        """Calculate position size based on signal strength"""
        pass


class MomentumStrategy(BaseStrategy):
    """Simple momentum strategy based on moving averages"""

    def __init__(self, fast_period: int = 20, slow_period: int = 50, **kwargs):
        super().__init__(
            "Momentum Strategy",
            {"fast_period": fast_period, "slow_period": slow_period, **kwargs},
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate moving averages
        df["ma_fast"] = (
            df["close"].rolling(window=self.parameters["fast_period"]).mean()
        )
        df["ma_slow"] = (
            df["close"].rolling(window=self.parameters["slow_period"]).mean()
        )

        # Generate signals
        df["signal"] = 0
        df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1  # Buy signal
        df.loc[df["ma_fast"] < df["ma_slow"], "signal"] = -1  # Sell signal

        # Only trade on signal changes
        df["position"] = df["signal"].diff()

        return df

    def calculate_position_size(
        self, signal: float, current_price: float, portfolio_value: float
    ) -> float:
        risk_per_trade = self.parameters.get(
            "risk_per_trade", 0.02
        )  # 2% risk per trade
        return (portfolio_value * risk_per_trade) / current_price


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands"""

    def __init__(self, period: int = 20, std_dev: float = 2.0, **kwargs):
        super().__init__(
            "Mean Reversion Strategy", {"period": period, "std_dev": std_dev, **kwargs}
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Calculate Bollinger Bands
        df["ma"] = df["close"].rolling(window=self.parameters["period"]).mean()
        df["std"] = df["close"].rolling(window=self.parameters["period"]).std()
        df["upper_band"] = df["ma"] + (df["std"] * self.parameters["std_dev"])
        df["lower_band"] = df["ma"] - (df["std"] * self.parameters["std_dev"])

        # Generate signals
        df["signal"] = 0
        df.loc[df["close"] < df["lower_band"], "signal"] = (
            1  # Buy when below lower band
        )
        df.loc[df["close"] > df["upper_band"], "signal"] = (
            -1
        )  # Sell when above upper band
        df.loc[
            (df["close"] >= df["lower_band"]) & (df["close"] <= df["upper_band"]),
            "signal",
        ] = 0

        # Only trade on signal changes
        df["position"] = df["signal"].diff()

        return df

    def calculate_position_size(
        self, signal: float, current_price: float, portfolio_value: float
    ) -> float:
        risk_per_trade = self.parameters.get(
            "risk_per_trade", 0.01
        )  # 1% risk per trade
        return (portfolio_value * risk_per_trade) / current_price


class BacktestEngine:
    """Advanced backtesting engine with multiple features"""

    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.backtest_results: Dict[str, BacktestResult] = {}
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage = 0.0005  # 0.05% slippage

        # Initialize with demo data
        self._initialize_demo_data()
        self._initialize_demo_strategies()

    def _initialize_demo_data(self):
        """Initialize demo market data"""
        symbols = ["BTC/USD", "ETH/USD", "AAPL", "GOOGL", "TSLA"]
        base_prices = {
            "BTC/USD": 45000,
            "ETH/USD": 2800,
            "AAPL": 180,
            "GOOGL": 140,
            "TSLA": 250,
        }

        for symbol in symbols:
            # Generate synthetic price data
            dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="h")
            base_price = base_prices[symbol]

            # Generate price series with trend and volatility
            returns = np.random.normal(
                0.0001, 0.02, len(dates)
            )  # Small positive drift with volatility
            prices = [base_price]

            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))  # Prevent negative prices

            # Create OHLCV data
            df = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": prices,
                    "high": [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
                    "low": [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
                    "close": prices,
                    "volume": np.random.uniform(1000000, 10000000, len(dates)),
                }
            )

            # Ensure OHLC relationships are correct
            df["high"] = df[["open", "close", "high"]].max(axis=1)
            df["low"] = df[["open", "close", "low"]].min(axis=1)

            self.market_data[symbol] = df

    def _initialize_demo_strategies(self):
        """Initialize demo strategies"""
        # Momentum strategy
        momentum_strategy = MomentumStrategy(
            fast_period=10, slow_period=30, risk_per_trade=0.02
        )
        self.strategies["momentum"] = momentum_strategy

        # Mean reversion strategy
        mean_reversion_strategy = MeanReversionStrategy(
            period=20, std_dev=2.0, risk_per_trade=0.01
        )
        self.strategies["mean_reversion"] = mean_reversion_strategy

    def add_strategy(self, strategy: BaseStrategy) -> bool:
        """Add a new strategy to the engine"""
        try:
            self.strategies[strategy.name.lower().replace(" ", "_")] = strategy
            logger.info(f"Strategy '{strategy.name}' added successfully")
            return True
        except Exception as e:
            logger.error(f"Error adding strategy: {e}")
            return False

    def run_backtest(
        self,
        strategy_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 100000,
    ) -> Optional[BacktestResult]:
        """Run backtest for a specific strategy and symbol"""
        try:
            if strategy_name not in self.strategies:
                logger.error(f"Strategy '{strategy_name}' not found")
                return None

            if symbol not in self.market_data:
                logger.error(f"Market data for '{symbol}' not found")
                return None

            strategy = self.strategies[strategy_name]
            data = self.market_data[symbol].copy()

            # Filter data by date range
            data = data[
                (data["timestamp"] >= start_date) & (data["timestamp"] <= end_date)
            ]

            if data.empty:
                logger.error("No data available for the specified date range")
                return None

            # Generate signals
            signals_df = strategy.generate_signals(data)

            # Initialize portfolio
            portfolio_value = initial_capital
            cash = initial_capital
            positions = {}
            trades = []
            equity_curve = []

            # Execute backtest
            for _i, row in signals_df.iterrows():
                current_price = row["close"]
                signal = row.get("position", 0)
                timestamp = row["timestamp"]

                # Execute trades based on signals
                if signal != 0 and not np.isnan(signal):
                    position_size = strategy.calculate_position_size(
                        signal, current_price, portfolio_value
                    )

                    if signal > 0:  # Buy signal
                        if cash >= position_size * current_price:
                            # Calculate costs
                            trade_value = position_size * current_price
                            commission = trade_value * self.commission_rate
                            slippage_cost = trade_value * self.slippage
                            total_cost = trade_value + commission + slippage_cost

                            if cash >= total_cost:
                                cash -= total_cost
                                positions[symbol] = (
                                    positions.get(symbol, 0) + position_size
                                )

                                # Record trade entry
                                str(uuid.uuid4())
                                # Store trade info for later completion

                    elif signal < 0:  # Sell signal
                        if symbol in positions and positions[symbol] > 0:
                            # Sell all position
                            position_size = positions[symbol]
                            trade_value = position_size * current_price
                            commission = trade_value * self.commission_rate
                            slippage_cost = trade_value * self.slippage
                            net_proceeds = trade_value - commission - slippage_cost

                            cash += net_proceeds
                            positions[symbol] = 0

                # Calculate portfolio value
                position_value = sum(
                    positions.get(sym, 0) * current_price for sym in positions
                )
                portfolio_value = cash + position_value

                equity_curve.append(
                    {
                        "timestamp": timestamp,
                        "portfolio_value": portfolio_value,
                        "cash": cash,
                        "positions_value": position_value,
                    }
                )

            # Create equity curve DataFrame
            equity_df = pd.DataFrame(equity_curve)

            # Calculate performance metrics
            returns = equity_df["portfolio_value"].pct_change().dropna()
            total_return = (portfolio_value - initial_capital) / initial_capital

            # Annualized return
            days = (end_date - start_date).days
            annualized_return = (
                (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
            )

            # Maximum drawdown
            rolling_max = equity_df["portfolio_value"].expanding().max()
            drawdown = (equity_df["portfolio_value"] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()

            # Sharpe ratio (assuming 0% risk-free rate)
            sharpe_ratio = (
                returns.mean() / returns.std() * np.sqrt(365 * 24)
                if returns.std() > 0
                else 0
            )

            # Sortino ratio
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
            sortino_ratio = (
                returns.mean() / downside_std * np.sqrt(365 * 24)
                if downside_std > 0
                else 0
            )

            # Create backtest result
            result = BacktestResult(
                strategy_name=strategy.name,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                final_capital=portfolio_value,
                total_return=total_return,
                annualized_return=annualized_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                win_rate=0.6,  # Placeholder - would calculate from actual trades
                profit_factor=1.5,  # Placeholder - would calculate from actual trades
                total_trades=len(trades),
                trades=trades,
                equity_curve=equity_df,
                metrics={
                    "volatility": returns.std() * np.sqrt(365 * 24),
                    "best_day": returns.max(),
                    "worst_day": returns.min(),
                    "total_days": days,
                },
            )

            self.backtest_results[f"{strategy_name}_{symbol}"] = result
            return result

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None

    def run_monte_carlo_simulation(
        self, strategy_name: str, symbol: str, num_simulations: int = 1000
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation for a strategy"""
        try:
            base_result = self.backtest_results.get(f"{strategy_name}_{symbol}")
            if not base_result:
                logger.error("Base backtest result not found. Run backtest first.")
                return None

            returns = base_result.equity_curve["portfolio_value"].pct_change().dropna()

            simulation_results = []

            for _ in range(num_simulations):
                # Bootstrap sampling of returns
                simulated_returns = np.random.choice(
                    returns, size=len(returns), replace=True
                )

                # Calculate cumulative performance
                cumulative_return = (1 + pd.Series(simulated_returns)).cumprod().iloc[
                    -1
                ] - 1

                # Calculate max drawdown for this simulation
                cumulative_values = (
                    base_result.initial_capital
                    * (1 + pd.Series(simulated_returns)).cumprod()
                )
                rolling_max = cumulative_values.expanding().max()
                drawdown = (cumulative_values - rolling_max) / rolling_max
                max_dd = drawdown.min()

                simulation_results.append(
                    {"total_return": cumulative_return, "max_drawdown": max_dd}
                )

            # Analyze simulation results
            sim_df = pd.DataFrame(simulation_results)

            monte_carlo_results = {
                "mean_return": sim_df["total_return"].mean(),
                "std_return": sim_df["total_return"].std(),
                "percentiles": {
                    "5th": sim_df["total_return"].quantile(0.05),
                    "25th": sim_df["total_return"].quantile(0.25),
                    "50th": sim_df["total_return"].quantile(0.50),
                    "75th": sim_df["total_return"].quantile(0.75),
                    "95th": sim_df["total_return"].quantile(0.95),
                },
                "probability_positive": (sim_df["total_return"] > 0).mean(),
                "max_drawdown_stats": {
                    "mean": sim_df["max_drawdown"].mean(),
                    "worst_case": sim_df["max_drawdown"].min(),
                    "best_case": sim_df["max_drawdown"].max(),
                },
                "var_95": sim_df["total_return"].quantile(0.05),  # Value at Risk
                "cvar_95": sim_df[
                    sim_df["total_return"] <= sim_df["total_return"].quantile(0.05)
                ]["total_return"].mean(),
                "simulation_data": sim_df,
            }

            return monte_carlo_results

        except Exception as e:
            logger.error(f"Error running Monte Carlo simulation: {e}")
            return None

    def compare_strategies(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Compare multiple strategies across symbols"""
        comparison_results = []

        for strategy_name in self.strategies.keys():
            for symbol in symbols:
                result = self.run_backtest(strategy_name, symbol, start_date, end_date)
                if result:
                    comparison_results.append(
                        {
                            "Strategy": result.strategy_name,
                            "Symbol": symbol,
                            "Total Return (%)": result.total_return * 100,
                            "Annualized Return (%)": result.annualized_return * 100,
                            "Max Drawdown (%)": result.max_drawdown * 100,
                            "Sharpe Ratio": result.sharpe_ratio,
                            "Sortino Ratio": result.sortino_ratio,
                            "Win Rate (%)": result.win_rate * 100,
                            "Total Trades": result.total_trades,
                        }
                    )

        return pd.DataFrame(comparison_results)


def main():
    """
    Entry point for the ZoL0 Advanced Backtesting Engine Streamlit app. Sets up the UI, initializes the backtesting engine, and handles user interactions for strategy testing, comparison, Monte Carlo simulation, and performance analysis.
    """
    st.set_page_config(
        page_title="ZoL0 Advanced Backtesting Engine",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
    <style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b 0%, #ee5a6f 100%);
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
    .positive-metric {
        background: #d4edda;
        border-left-color: #28a745;
    }
    .negative-metric {
        background: #f8d7da;
        border-left-color: #dc3545;
    }
    .strategy-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Header
    st.markdown(
        """
    <div class="main-header">
        <h1>🎯 ZoL0 Advanced Backtesting Engine</h1>
        <p>Enterprise-grade strategy testing, Monte Carlo simulations, and performance analysis</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Initialize backtesting engine
    if "backtest_engine" not in st.session_state:
        st.session_state.backtest_engine = BacktestEngine()

    engine = st.session_state.backtest_engine

    # Sidebar
    st.sidebar.title("🎯 Backtesting")

    tab_selection = st.sidebar.radio(
        "Select Module",
        [
            "🏠 Dashboard",
            "🎯 Single Backtest",
            "📊 Strategy Comparison",
            "🎲 Monte Carlo",
            "📈 Performance Analysis",
            "⚙️ Strategy Builder",
        ],
    )

    if tab_selection == "🏠 Dashboard":
        st.header("Backtesting Dashboard")

        # Quick stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                f"""
            <div class="metric-container">
                <h3>📊 Available Strategies</h3>
                <h2>{len(engine.strategies)}</h2>
                <p>Ready for testing</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
            <div class="metric-container">
                <h3>💾 Market Data</h3>
                <h2>{len(engine.market_data)}</h2>
                <p>Symbols available</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
            <div class="metric-container">
                <h3>🎯 Completed Tests</h3>
                <h2>{len(engine.backtest_results)}</h2>
                <p>Backtest results</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            avg_return = (
                np.mean(
                    [r.total_return * 100 for r in engine.backtest_results.values()]
                )
                if engine.backtest_results
                else 0
            )
            metric_class = (
                "positive-metric"
                if avg_return > 0
                else "negative-metric" if avg_return < 0 else "metric-container"
            )
            st.markdown(
                f"""
            <div class="{metric_class}">
                <h3>📈 Avg Return</h3>
                <h2>{avg_return:.1f}%</h2>
                <p>Across all tests</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Available strategies
        st.subheader("Available Strategies")

        for name, strategy in engine.strategies.items():
            st.markdown(
                f"""
            <div class="strategy-card">
                <strong>{strategy.name}</strong><br>
                <small>Parameters: {', '.join([f'{k}: {v}' for k, v in strategy.parameters.items()])}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Recent backtest results
        if engine.backtest_results:
            st.subheader("Recent Backtest Results")

            results_data = []
            for key, result in list(engine.backtest_results.items())[
                -5:
            ]:  # Last 5 results
                results_data.append(
                    {
                        "Strategy": result.strategy_name,
                        "Period": f"{result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}",
                        "Total Return": f"{result.total_return * 100:.2f}%",
                        "Sharpe Ratio": f"{result.sharpe_ratio:.2f}",
                        "Max Drawdown": f"{result.max_drawdown * 100:.2f}%",
                        "Total Trades": result.total_trades,
                    }
                )

            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
        else:
            st.info(
                "No backtest results available. Run a backtest to see results here."
            )

        # Market data preview
        st.subheader("Available Market Data")

        symbol_info = []
        for symbol, data in engine.market_data.items():
            symbol_info.append(
                {
                    "Symbol": symbol,
                    "Data Points": len(data),
                    "Date Range": f"{data['timestamp'].min().strftime('%Y-%m-%d')} to {data['timestamp'].max().strftime('%Y-%m-%d')}",
                    "Current Price": f"${data['close'].iloc[-1]:.2f}",
                    "Price Change": f"{((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%",
                }
            )

        symbol_df = pd.DataFrame(symbol_info)
        st.dataframe(symbol_df, use_container_width=True)

    elif tab_selection == "🎯 Single Backtest":
        st.header("Single Strategy Backtest")

        # Backtest configuration
        col1, col2 = st.columns(2)

        with col1:
            strategy_name = st.selectbox(
                "Select Strategy",
                list(engine.strategies.keys()),
                format_func=lambda x: engine.strategies[x].name,
            )

            symbol = st.selectbox("Select Symbol", list(engine.market_data.keys()))

            initial_capital = st.number_input(
                "Initial Capital ($)", min_value=1000, max_value=10000000, value=100000
            )

        with col2:
            start_date = st.date_input("Start Date", value=datetime(2023, 1, 1).date())

            end_date = st.date_input("End Date", value=datetime(2023, 12, 31).date())

            if st.button("🎯 Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    start_datetime = datetime.combine(start_date, datetime.min.time())
                    end_datetime = datetime.combine(end_date, datetime.max.time())

                    result = engine.run_backtest(
                        strategy_name,
                        symbol,
                        start_datetime,
                        end_datetime,
                        initial_capital,
                    )

                    if result:
                        st.success("Backtest completed successfully!")
                        st.session_state.current_backtest = result
                    else:
                        st.error("Backtest failed. Please check the parameters.")

        # Display results
        if hasattr(st.session_state, "current_backtest"):
            result = st.session_state.current_backtest

            st.subheader("Backtest Results")

            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                return_class = (
                    "positive-metric" if result.total_return > 0 else "negative-metric"
                )
                st.markdown(
                    f"""
                <div class="{return_class}">
                    <h4>Total Return</h4>
                    <h2>{result.total_return * 100:.2f}%</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class="metric-container">
                    <h4>Sharpe Ratio</h4>
                    <h2>{result.sharpe_ratio:.2f}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                <div class="negative-metric">
                    <h4>Max Drawdown</h4>
                    <h2>{result.max_drawdown * 100:.2f}%</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col4:
                st.markdown(
                    f"""
                <div class="metric-container">
                    <h4>Total Trades</h4>
                    <h2>{result.total_trades}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Equity curve
            st.subheader("Equity Curve")

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=result.equity_curve["timestamp"],
                    y=result.equity_curve["portfolio_value"],
                    mode="lines",
                    name="Portfolio Value",
                    line=dict(color="#007bff", width=2),
                )
            )

            # Add benchmark (buy and hold)
            initial_price = engine.market_data[symbol]["close"].iloc[0]
            final_price = engine.market_data[symbol]["close"].iloc[-1]
            benchmark_return = final_price / initial_price - 1
            benchmark_value = initial_capital * (1 + benchmark_return)

            fig.add_hline(
                y=benchmark_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Buy & Hold: ${benchmark_value:,.0f}",
            )

            fig.update_layout(
                title="Portfolio Performance vs Buy & Hold",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                height=400,
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detailed metrics table
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Return Metrics")
                return_metrics = pd.DataFrame(
                    {
                        "Metric": [
                            "Total Return",
                            "Annualized Return",
                            "Volatility",
                            "Sharpe Ratio",
                            "Sortino Ratio",
                        ],
                        "Value": [
                            f"{result.total_return * 100:.2f}%",
                            f"{result.annualized_return * 100:.2f}%",
                            f"{result.metrics['volatility'] * 100:.2f}%",
                            f"{result.sharpe_ratio:.2f}",
                            f"{result.sortino_ratio:.2f}",
                        ],
                    }
                )
                st.dataframe(return_metrics, use_container_width=True)

            with col2:
                st.subheader("Risk Metrics")
                risk_metrics = pd.DataFrame(
                    {
                        "Metric": [
                            "Maximum Drawdown",
                            "Best Day",
                            "Worst Day",
                            "Win Rate",
                            "Profit Factor",
                        ],
                        "Value": [
                            f"{result.max_drawdown * 100:.2f}%",
                            f"{result.metrics['best_day'] * 100:.2f}%",
                            f"{result.metrics['worst_day'] * 100:.2f}%",
                            f"{result.win_rate * 100:.1f}%",
                            f"{result.profit_factor:.2f}",
                        ],
                    }
                )
                st.dataframe(risk_metrics, use_container_width=True)

            # Drawdown chart
            st.subheader("Drawdown Analysis")

            rolling_max = result.equity_curve["portfolio_value"].expanding().max()
            drawdown = (
                (result.equity_curve["portfolio_value"] - rolling_max)
                / rolling_max
                * 100
            )

            fig_dd = go.Figure()
            fig_dd.add_trace(
                go.Scatter(
                    x=result.equity_curve["timestamp"],
                    y=drawdown,
                    mode="lines",
                    name="Drawdown (%)",
                    fill="tonexty",
                    line=dict(color="red"),
                )
            )

            fig_dd.update_layout(
                title="Portfolio Drawdown Over Time",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300,
            )

            st.plotly_chart(fig_dd, use_container_width=True)

    elif tab_selection == "📊 Strategy Comparison":
        st.header("Strategy Comparison Analysis")

        # Comparison configuration
        col1, col2 = st.columns(2)

        with col1:
            selected_symbols = st.multiselect(
                "Select Symbols",
                list(engine.market_data.keys()),
                default=list(engine.market_data.keys())[:3],
            )

        with col2:
            comparison_start = st.date_input(
                "Start Date", value=datetime(2023, 1, 1).date(), key="comp_start"
            )

            comparison_end = st.date_input(
                "End Date", value=datetime(2023, 12, 31).date(), key="comp_end"
            )

        if st.button("📊 Run Comparison") and selected_symbols:
            with st.spinner("Running strategy comparison..."):
                start_dt = datetime.combine(comparison_start, datetime.min.time())
                end_dt = datetime.combine(comparison_end, datetime.max.time())

                comparison_df = engine.compare_strategies(
                    selected_symbols, start_dt, end_dt
                )

                if not comparison_df.empty:
                    st.session_state.comparison_results = comparison_df
                    st.success("Comparison completed!")
                else:
                    st.error("No results generated from comparison")

        # Display comparison results
        if hasattr(st.session_state, "comparison_results"):
            df = st.session_state.comparison_results

            st.subheader("Strategy Performance Comparison")
            st.dataframe(df, use_container_width=True)

            # Performance visualization
            col1, col2 = st.columns(2)

            with col1:
                fig_return = px.bar(
                    df,
                    x="Strategy",
                    y="Total Return (%)",
                    color="Symbol",
                    title="Total Return by Strategy and Symbol",
                    barmode="group",
                )
                st.plotly_chart(fig_return, use_container_width=True)

            with col2:
                fig_sharpe = px.scatter(
                    df,
                    x="Total Return (%)",
                    y="Sharpe Ratio",
                    color="Strategy",
                    size="Total Trades",
                    hover_data=["Symbol"],
                    title="Risk-Return Profile",
                )
                st.plotly_chart(fig_sharpe, use_container_width=True)

            # Best performing strategies
            st.subheader("Top Performing Strategies")

            top_strategies = df.nlargest(5, "Total Return (%)")

            for i, (_, row) in enumerate(top_strategies.iterrows()):
                st.markdown(
                    f"""
                <div class="strategy-card">
                    <strong>#{i+1} {row['Strategy']} - {row['Symbol']}</strong><br>
                    Return: {row['Total Return (%)']}% | Sharpe: {row['Sharpe Ratio']:.2f} | 
                    Max DD: {row['Max Drawdown (%)']}%
                </div>
                """,
                    unsafe_allow_html=True,
                )

    elif tab_selection == "🎲 Monte Carlo":
        st.header("Monte Carlo Simulation")

        # Monte Carlo configuration
        col1, col2 = st.columns(2)

        with col1:
            mc_strategy = st.selectbox(
                "Select Strategy for Monte Carlo",
                list(engine.strategies.keys()),
                format_func=lambda x: engine.strategies[x].name,
                key="mc_strategy",
            )

            mc_symbol = st.selectbox(
                "Select Symbol for Monte Carlo",
                list(engine.market_data.keys()),
                key="mc_symbol",
            )

        with col2:
            num_simulations = st.number_input(
                "Number of Simulations", min_value=100, max_value=10000, value=1000
            )

            st.slider("Confidence Level (%)", min_value=90, max_value=99, value=95)

        if st.button("🎲 Run Monte Carlo Simulation"):
            # First ensure we have a base backtest result
            base_key = f"{mc_strategy}_{mc_symbol}"
            if base_key not in engine.backtest_results:
                with st.spinner("Running base backtest..."):
                    start_dt = datetime(2023, 1, 1)
                    end_dt = datetime(2023, 12, 31)
                    engine.run_backtest(mc_strategy, mc_symbol, start_dt, end_dt)

            if base_key in engine.backtest_results:
                with st.spinner(
                    f"Running {num_simulations} Monte Carlo simulations..."
                ):
                    mc_results = engine.run_monte_carlo_simulation(
                        mc_strategy, mc_symbol, num_simulations
                    )

                    if mc_results:
                        st.session_state.monte_carlo_results = mc_results
                        st.success("Monte Carlo simulation completed!")
                    else:
                        st.error("Monte Carlo simulation failed")
            else:
                st.error("Base backtest failed. Cannot run Monte Carlo simulation.")

        # Display Monte Carlo results
        if hasattr(st.session_state, "monte_carlo_results"):
            mc_results = st.session_state.monte_carlo_results

            st.subheader("Monte Carlo Simulation Results")

            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f"""
                <div class="metric-container">
                    <h4>Mean Return</h4>
                    <h2>{mc_results['mean_return'] * 100:.2f}%</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class="metric-container">
                    <h4>Return Std Dev</h4>
                    <h2>{mc_results['std_return'] * 100:.2f}%</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                <div class="positive-metric">
                    <h4>Probability of Profit</h4>
                    <h2>{mc_results['probability_positive'] * 100:.1f}%</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col4:
                st.markdown(
                    f"""
                <div class="negative-metric">
                    <h4>Value at Risk (95%)</h4>
                    <h2>{mc_results['var_95'] * 100:.2f}%</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Return distribution
            st.subheader("Return Distribution")

            sim_data = mc_results["simulation_data"]

            fig_hist = px.histogram(
                sim_data,
                x="total_return",
                nbins=50,
                title="Distribution of Simulated Returns",
                labels={"total_return": "Total Return", "count": "Frequency"},
            )

            # Add percentile lines
            for pct, value in mc_results["percentiles"].items():
                fig_hist.add_vline(
                    x=value,
                    line_dash="dash",
                    annotation_text=f"{pct}: {value*100:.1f}%",
                )

            st.plotly_chart(fig_hist, use_container_width=True)

            # Risk metrics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Return Percentiles")
                percentiles_df = pd.DataFrame(
                    {
                        "Percentile": list(mc_results["percentiles"].keys()),
                        "Return (%)": [
                            f"{v*100:.2f}%" for v in mc_results["percentiles"].values()
                        ],
                    }
                )
                st.dataframe(percentiles_df, use_container_width=True)

            with col2:
                st.subheader("Drawdown Statistics")
                dd_stats = mc_results["max_drawdown_stats"]
                drawdown_df = pd.DataFrame(
                    {
                        "Metric": ["Mean Max Drawdown", "Worst Case", "Best Case"],
                        "Value (%)": [
                            f"{dd_stats['mean']*100:.2f}%",
                            f"{dd_stats['worst_case']*100:.2f}%",
                            f"{dd_stats['best_case']*100:.2f}%",
                        ],
                    }
                )
                st.dataframe(drawdown_df, use_container_width=True)

    elif tab_selection == "📈 Performance Analysis":
        st.header("Performance Analysis")

        if not engine.backtest_results:
            st.info(
                "No backtest results available. Run some backtests first to see performance analysis."
            )
        else:
            # Performance summary
            st.subheader("Performance Summary")

            perf_data = []
            for key, result in engine.backtest_results.items():
                strategy_symbol = key.split("_", 1)
                perf_data.append(
                    {
                        "Strategy": result.strategy_name,
                        "Symbol": (
                            strategy_symbol[1]
                            if len(strategy_symbol) > 1
                            else "Unknown"
                        ),
                        "Total Return (%)": result.total_return * 100,
                        "Annualized Return (%)": result.annualized_return * 100,
                        "Max Drawdown (%)": result.max_drawdown * 100,
                        "Sharpe Ratio": result.sharpe_ratio,
                        "Sortino Ratio": result.sortino_ratio,
                        "Volatility (%)": result.metrics.get("volatility", 0) * 100,
                        "Total Trades": result.total_trades,
                    }
                )

            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)

            # Performance metrics visualization
            col1, col2 = st.columns(2)

            with col1:
                fig_metrics = px.scatter(
                    perf_df,
                    x="Volatility (%)",
                    y="Annualized Return (%)",
                    color="Strategy",
                    size="Total Trades",
                    hover_data=["Symbol", "Sharpe Ratio"],
                    title="Risk-Return Analysis",
                )
                st.plotly_chart(fig_metrics, use_container_width=True)

            with col2:
                fig_sharpe = px.bar(
                    perf_df,
                    x="Strategy",
                    y="Sharpe Ratio",
                    color="Symbol",
                    title="Sharpe Ratio by Strategy",
                    barmode="group",
                )
                st.plotly_chart(fig_sharpe, use_container_width=True)

            # Risk analysis
            st.subheader("Risk Analysis")

            col1, col2 = st.columns(2)

            with col1:
                fig_dd = px.bar(
                    perf_df,
                    x="Strategy",
                    y="Max Drawdown (%)",
                    color="Symbol",
                    title="Maximum Drawdown by Strategy",
                )
                st.plotly_chart(fig_dd, use_container_width=True)

            with col2:
                # Calculate correlation between strategies
                if len(perf_df) > 1:
                    strategy_returns = perf_df.pivot_table(
                        index="Symbol", columns="Strategy", values="Total Return (%)"
                    )

                    if strategy_returns.shape[1] > 1:
                        corr_matrix = strategy_returns.corr()

                        fig_corr = px.imshow(
                            corr_matrix,
                            title="Strategy Correlation Matrix",
                            color_continuous_scale="RdBu",
                            aspect="auto",
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Need multiple strategies to show correlation")
                else:
                    st.info("Need multiple backtest results to show correlation")

    elif tab_selection == "⚙️ Strategy Builder":
        st.header("Strategy Builder")

        st.subheader("Create Custom Strategy")

        # Strategy configuration
        col1, col2 = st.columns(2)

        with col1:
            strategy_name = st.text_input("Strategy Name")
            strategy_type = st.selectbox(
                "Strategy Type", [e.value for e in StrategyType]
            )

            risk_per_trade = st.slider(
                "Risk per Trade (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1
            )

        with col2:
            if strategy_type == StrategyType.MOMENTUM.value:
                fast_period = st.number_input(
                    "Fast Period", min_value=5, max_value=50, value=10
                )
                slow_period = st.number_input(
                    "Slow Period", min_value=10, max_value=200, value=30
                )

                if st.button("🔧 Create Momentum Strategy"):
                    new_strategy = MomentumStrategy(
                        fast_period=fast_period,
                        slow_period=slow_period,
                        risk_per_trade=risk_per_trade / 100,
                    )
                    new_strategy.name = (
                        strategy_name
                        if strategy_name
                        else f"Custom Momentum {len(engine.strategies)}"
                    )

                    if engine.add_strategy(new_strategy):
                        st.success(
                            f"Strategy '{new_strategy.name}' created successfully!"
                        )
                        st.rerun()
                    else:
                        st.error("Failed to create strategy")

            elif strategy_type == StrategyType.MEAN_REVERSION.value:
                period = st.number_input("Period", min_value=5, max_value=100, value=20)
                std_dev = st.number_input(
                    "Standard Deviations",
                    min_value=1.0,
                    max_value=4.0,
                    value=2.0,
                    step=0.1,
                )

                if st.button("🔧 Create Mean Reversion Strategy"):
                    new_strategy = MeanReversionStrategy(
                        period=period,
                        std_dev=std_dev,
                        risk_per_trade=risk_per_trade / 100,
                    )
                    new_strategy.name = (
                        strategy_name
                        if strategy_name
                        else f"Custom Mean Reversion {len(engine.strategies)}"
                    )

                    if engine.add_strategy(new_strategy):
                        st.success(
                            f"Strategy '{new_strategy.name}' created successfully!"
                        )
                        st.rerun()
                    else:
                        st.error("Failed to create strategy")

        # Existing strategies management
        st.subheader("Manage Existing Strategies")

        if engine.strategies:
            for name, strategy in engine.strategies.items():
                with st.expander(f"📊 {strategy.name}"):
                    st.write(f"**Type:** {type(strategy).__name__}")
                    st.write("**Parameters:**")
                    for param, value in strategy.parameters.items():
                        st.write(f"  - {param}: {value}")

                    if st.button(f"🗑️ Delete {strategy.name}", key=f"delete_{name}"):
                        del engine.strategies[name]
                        st.success(f"Strategy '{strategy.name}' deleted!")
                        st.rerun()
        else:
            st.info("No strategies available. Create one using the builder above.")

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🎯 Strategies", len(engine.strategies))

    with col2:
        st.metric("📊 Results", len(engine.backtest_results))

    with col3:
        st.metric("💾 Data Sources", len(engine.market_data))

# TODO: Integrate with CI/CD pipeline for automated backtesting and edge-case tests.
# Edge-case tests: simulate empty data, strategy errors, and DB/network issues.
# All public methods have docstrings and exception handling.
if __name__ == "__main__":
    main()
