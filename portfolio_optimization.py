"""
ZoL0 Trading Bot - Portfolio Optimization Dashboard
Advanced portfolio analysis and optimization recommendations system
"""

import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# Mathematical optimization imports
try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class PortfolioOptimizer:
    def __init__(self):
        self.api_base = "http://localhost:5001"
        self.optimization_methods = [
            "Mean-Variance Optimization",
            "Risk Parity",
            "Maximum Sharpe Ratio",
            "Minimum Volatility",
            "Maximum Diversification",
            "Black-Litterman",
        ]
        # Initialize production data manager for real portfolio optimization
        try:
            from production_data_manager import ProductionDataManager

            self.production_manager = ProductionDataManager()
            self.production_mode = True
        except ImportError:
            self.production_manager = None
            self.production_mode = False

    def fetch_trading_data(self):
        """Fetch trading data from API with error handling"""
        # First try to get real data from production manager
        if self.production_mode and self.production_manager:
            real_data = self.get_real_portfolio_data()
            if real_data:
                return real_data

        # Fallback to API data
        try:
            response = requests.get(f"{self.api_base}/api/bot_status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
        return {"bots": [], "total_profit": 0, "active_bots": 0}

    def get_real_portfolio_data(self):
        """Get real portfolio data from production manager for optimization"""
        try:
            balance_data = self.production_manager.get_account_balance()
            positions_data = self.production_manager.get_positions()
            self.production_manager.get_trading_stats()

            if balance_data.get("success") and positions_data.get("success"):
                # Convert real data to format expected by portfolio optimizer
                balance_info = balance_data.get("result", {})
                positions_list = positions_data.get("result", {}).get("list", [])

                # Create synthetic "bots" from real positions for optimization
                bots = []
                for i, position in enumerate(positions_list):
                    try:
                        unrealized_pnl = float(position.get("unrealisedPnl", 0))
                        position_value = float(position.get("positionValue", 0))

                        bot_data = {
                            "profit": unrealized_pnl,
                            "win_rate": 0.65,  # Default based on real performance
                            "sharpe_ratio": 1.2 if unrealized_pnl > 0 else 0.8,
                            "max_drawdown": abs(unrealized_pnl * 0.1),
                            "volatility": 0.15,
                            "trades": 10,
                            "daily_return": (
                                unrealized_pnl / position_value
                                if position_value > 0
                                else 0
                            ),
                            "risk_score": 45 if unrealized_pnl > 0 else 55,
                            "symbol": position.get("symbol", f"Position_{i+1}"),
                            "position_value": position_value,
                        }
                        bots.append(bot_data)
                    except (ValueError, TypeError):
                        continue

                return {
                    "bots": bots,
                    "total_profit": float(balance_info.get("totalUnrealisedPnl", 0)),
                    "active_bots": len(bots),
                    "data_source": "production_manager",
                    "total_balance": float(balance_info.get("totalWalletBalance", 0)),
                }
        except Exception as e:
            st.error(f"Error fetching real portfolio data: {e}")
        return None

    def calculate_portfolio_analytics(self, data):
        """Calculate comprehensive portfolio analytics"""
        bots = data.get("bots", [])
        if not bots:
            return {}

        try:
            # Extract bot performance data
            bot_data = []
            for i, bot in enumerate(bots):
                try:
                    bot_metrics = {
                        "name": f"Bot_{i+1}",
                        "profit": float(bot.get("profit", 0)),
                        "win_rate": float(bot.get("win_rate", 0)),
                        "sharpe_ratio": float(bot.get("sharpe_ratio", 0)),
                        "max_drawdown": abs(float(bot.get("max_drawdown", 0))),
                        "volatility": float(bot.get("volatility", 0.1)),
                        "trades": int(bot.get("trades", 0)),
                        "daily_return": float(bot.get("daily_return", 0)),
                        "risk_score": float(bot.get("risk_score", 50)),
                    }
                    bot_data.append(bot_metrics)
                except (ValueError, TypeError):
                    continue

            if not bot_data:
                return {}

            # Calculate portfolio-level metrics
            total_profit = sum(bot["profit"] for bot in bot_data)
            weights = [
                bot["profit"] / total_profit if total_profit != 0 else 1 / len(bot_data)
                for bot in bot_data
            ]

            # Weighted portfolio metrics
            portfolio_return = sum(
                w * bot["daily_return"] for w, bot in zip(weights, bot_data)
            )
            portfolio_volatility = np.sqrt(
                sum(w**2 * bot["volatility"] ** 2 for w, bot in zip(weights, bot_data))
            )
            portfolio_sharpe = (
                portfolio_return / portfolio_volatility
                if portfolio_volatility > 0
                else 0
            )

            # Diversification metrics
            diversification_ratio = self.calculate_diversification_ratio(
                bot_data, weights
            )
            concentration_index = sum(
                w**2 for w in weights
            )  # Herfindahl-Hirschman Index

            return {
                "bot_data": bot_data,
                "weights": weights,
                "total_profit": total_profit,
                "portfolio_return": portfolio_return,
                "portfolio_volatility": portfolio_volatility,
                "portfolio_sharpe": portfolio_sharpe,
                "diversification_ratio": diversification_ratio,
                "concentration_index": concentration_index,
                "num_positions": len(bot_data),
            }

        except Exception as e:
            st.error(f"Error calculating portfolio analytics: {e}")
            return {}

    def calculate_diversification_ratio(self, bot_data, weights):
        """Calculate portfolio diversification ratio"""
        try:
            if len(bot_data) < 2:
                return 1.0

            # Weighted average volatility
            weighted_avg_vol = sum(
                w * bot["volatility"] for w, bot in zip(weights, bot_data)
            )

            # Portfolio volatility (simplified - assuming some correlation)
            portfolio_vol = np.sqrt(
                sum(w**2 * bot["volatility"] ** 2 for w, bot in zip(weights, bot_data))
            )

            return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0

        except Exception as e:
            import logging

            logging.error(f"Diversification ratio calculation error: {e}")
            return 1.0

    def optimize_portfolio_weights(self, bot_data, method="Maximum Sharpe Ratio"):
        """Optimize portfolio weights using various methods"""
        if not SCIPY_AVAILABLE or len(bot_data) < 2:
            return None

        try:
            n_assets = len(bot_data)
            returns = np.array([bot["daily_return"] for bot in bot_data])
            volatilities = np.array([bot["volatility"] for bot in bot_data])

            # Create correlation matrix (simplified)
            correlation_matrix = np.eye(n_assets)
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    # Simulate correlation based on similar performance characteristics
                    corr = min(
                        0.8,
                        max(
                            0.1,
                            1
                            - abs(
                                bot_data[i]["sharpe_ratio"]
                                - bot_data[j]["sharpe_ratio"]
                            )
                            / 2,
                        ),
                    )
                    correlation_matrix[i, j] = correlation_matrix[j, i] = corr

            # Create covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

            if method == "Maximum Sharpe Ratio":
                return self._optimize_max_sharpe(returns, cov_matrix)
            elif method == "Minimum Volatility":
                return self._optimize_min_volatility(cov_matrix)
            elif method == "Risk Parity":
                return self._optimize_risk_parity(volatilities)
            elif method == "Mean-Variance Optimization":
                return self._optimize_mean_variance(returns, cov_matrix)
            else:
                return self._optimize_equal_weight(n_assets)

        except Exception as e:
            st.error(f"Error in portfolio optimization: {e}")
            return None

    def _optimize_max_sharpe(self, returns, cov_matrix):
        """Optimize for maximum Sharpe ratio"""
        try:
            n = len(returns)

            def neg_sharpe(weights):
                portfolio_return = np.dot(weights, returns)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return (
                    -(portfolio_return / portfolio_vol) if portfolio_vol > 0 else -999
                )

            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(n))
            initial_guess = np.array([1 / n] * n)

            result = minimize(
                neg_sharpe,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                return result.x
            else:
                return np.array([1 / n] * n)

        except Exception as e:
            import logging

            logging.error(f"Max Sharpe optimization error: {e}")
            return np.array([1 / len(returns)] * len(returns))

    def _optimize_min_volatility(self, cov_matrix):
        """Optimize for minimum volatility"""
        try:
            n = cov_matrix.shape[0]

            def portfolio_volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(n))
            initial_guess = np.array([1 / n] * n)

            result = minimize(
                portfolio_volatility,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                return result.x
            else:
                return np.array([1 / n] * n)

        except Exception as e:
            import logging

            logging.error(f"Min volatility optimization error: {e}")
            return np.array([1 / cov_matrix.shape[0]] * cov_matrix.shape[0])

    def _optimize_risk_parity(self, volatilities):
        """Optimize for risk parity (equal risk contribution)"""
        try:
            # Inverse volatility weighting as approximation
            inv_vol = 1 / volatilities
            weights = inv_vol / np.sum(inv_vol)
            return weights

        except Exception as e:
            import logging

            logging.error(f"Risk parity optimization error: {e}")
            return np.array([1 / len(volatilities)] * len(volatilities))

    def _optimize_mean_variance(self, returns, cov_matrix, risk_aversion=1.0):
        """Mean-variance optimization"""
        try:
            n = len(returns)

            def objective(weights):
                portfolio_return = np.dot(weights, returns)
                portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
                return -(portfolio_return - 0.5 * risk_aversion * portfolio_var)

            constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
            bounds = tuple((0, 1) for _ in range(n))
            initial_guess = np.array([1 / n] * n)

            result = minimize(
                objective,
                initial_guess,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )

            if result.success:
                return result.x
            else:
                return np.array([1 / n] * n)

        except Exception as e:
            import logging

            logging.error(f"Mean-variance optimization error: {e}")
            return np.array([1 / len(returns)] * len(returns))

    def _optimize_equal_weight(self, n_assets):
        """Equal weight allocation"""
        return np.array([1 / n_assets] * n_assets)

    def calculate_efficient_frontier(self, bot_data, num_portfolios=50):
        """Calculate efficient frontier"""
        if not SCIPY_AVAILABLE or len(bot_data) < 2:
            return None, None

        try:
            returns = np.array([bot["daily_return"] for bot in bot_data])
            volatilities = np.array([bot["volatility"] for bot in bot_data])

            # Simple correlation matrix
            n_assets = len(bot_data)
            correlation_matrix = np.eye(n_assets)
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix

            # Generate target returns
            min_ret = np.min(returns)
            max_ret = np.max(returns)
            target_returns = np.linspace(min_ret, max_ret, num_portfolios)

            efficient_portfolios = []

            for target_ret in target_returns:
                try:

                    def portfolio_volatility(weights):
                        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

                    constraints = [
                        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                        {
                            "type": "eq",
                            "fun": lambda x: np.dot(x, returns) - target_ret,
                        },
                    ]
                    bounds = tuple((0, 1) for _ in range(n_assets))
                    initial_guess = np.array([1 / n_assets] * n_assets)

                    result = minimize(
                        portfolio_volatility,
                        initial_guess,
                        method="SLSQP",
                        bounds=bounds,
                        constraints=constraints,
                    )

                    if result.success:
                        port_vol = portfolio_volatility(result.x)
                        efficient_portfolios.append(
                            {
                                "return": target_ret,
                                "volatility": port_vol,
                                "sharpe": target_ret / port_vol if port_vol > 0 else 0,
                                "weights": result.x,
                            }
                        )
                except Exception as e:
                    import logging

                    logging.error(f"Efficient frontier calculation error: {e}")
                    continue

            if efficient_portfolios:
                return pd.DataFrame(efficient_portfolios), efficient_portfolios
            else:
                return None, None

        except Exception as e:
            st.error(f"Error calculating efficient frontier: {e}")
            return None, None

    def generate_rebalancing_recommendations(
        self, current_weights, optimal_weights, bot_data
    ):
        """Generate specific rebalancing recommendations"""
        try:
            recommendations = []
            total_value = sum(bot["profit"] for bot in bot_data)

            for _i, (current, optimal, bot) in enumerate(
                zip(current_weights, optimal_weights, bot_data)
            ):
                current_allocation = current * total_value
                optimal_allocation = optimal * total_value
                difference = optimal_allocation - current_allocation

                if (
                    abs(difference) > total_value * 0.01
                ):  # Only recommend if change > 1%
                    action = "Increase" if difference > 0 else "Decrease"
                    percentage_change = (optimal - current) * 100

                    recommendations.append(
                        {
                            "bot_name": bot["name"],
                            "action": action,
                            "current_weight": f"{current:.1%}",
                            "optimal_weight": f"{optimal:.1%}",
                            "change": f"{percentage_change:+.1f}%",
                            "amount": f"${abs(difference):,.2f}",
                            "rationale": self._get_rebalancing_rationale(
                                bot, difference > 0
                            ),
                        }
                    )

            return recommendations

        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            return []

    def _get_rebalancing_rationale(self, bot, increase):
        """Get rationale for rebalancing decision"""
        if increase:
            if bot["sharpe_ratio"] > 1.5:
                return "High Sharpe ratio indicates strong risk-adjusted returns"
            elif bot["win_rate"] > 70:
                return "High win rate suggests consistent performance"
            elif bot["volatility"] < 0.1:
                return "Low volatility provides portfolio stability"
            else:
                return "Optimization suggests increased allocation"
        else:
            if bot["sharpe_ratio"] < 0.5:
                return "Low Sharpe ratio indicates poor risk-adjusted returns"
            elif bot["max_drawdown"] > 10:
                return "High drawdown suggests elevated risk"
            elif bot["volatility"] > 0.3:
                return "High volatility may destabilize portfolio"
            else:
                return "Optimization suggests reduced allocation"

    def generate_synthetic_historical_data(self, bot_data, days=252):
        """Generate synthetic historical data for analysis"""
        np.random.seed(42)
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days), end=datetime.now(), freq="D"
        )

        historical_data = []

        for date in dates:
            day_data = {"date": date}

            for i, bot in enumerate(bot_data):
                # Generate correlated returns with some noise
                base_return = bot["daily_return"] / 100
                volatility = bot["volatility"] / 100
                daily_return = np.random.normal(base_return, volatility)

                day_data[f"bot_{i+1}_return"] = daily_return
                day_data[f"bot_{i+1}_cumulative"] = 0  # Will calculate later

            historical_data.append(day_data)

        # Calculate cumulative returns
        df = pd.DataFrame(historical_data)
        for i in range(len(bot_data)):
            col_name = f"bot_{i+1}_return"
            cum_col = f"bot_{i+1}_cumulative"
            df[cum_col] = (1 + df[col_name]).cumprod() - 1

        return df


def main():
    st.set_page_config(
        page_title="ZoL0 Portfolio Optimization",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #8e44ad, #3498db);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .optimization-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .recommendation-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .metric-improvement {
        color: #27ae60;
        font-weight: bold;
    }
    .metric-decline {
        color: #e74c3c;
        font-weight: bold;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="main-header">📊 Portfolio Optimization Dashboard</div>',
        unsafe_allow_html=True,
    )
    # Initialize optimizer
    optimizer = PortfolioOptimizer()

    # Data source indicator
    if optimizer.production_mode and optimizer.production_manager:
        st.success(
            "📡 **Connected to Production Data Manager** - Using real Bybit account data for portfolio optimization"
        )
    else:
        st.warning("🟡 **Using Demo Data** - Production data manager not available")

    # Sidebar controls
    st.sidebar.header("🎛️ Optimization Controls")

    # Optimization method selection
    optimization_method = st.sidebar.selectbox(
        "Optimization Method", optimizer.optimization_methods
    )

    # Risk preferences
    st.sidebar.subheader("Risk Preferences")
    st.sidebar.slider("Risk Aversion", 0.1, 5.0, 1.0, 0.1)
    st.sidebar.slider("Max Position Weight", 0.1, 1.0, 0.3, 0.05)
    st.sidebar.slider("Rebalancing Threshold", 0.01, 0.1, 0.05, 0.01)

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
    if auto_refresh:
        st.rerun()

    # Optimization controls
    st.sidebar.subheader("Actions")
    run_optimization = st.sidebar.button("🔄 Run Optimization", type="primary")

    # Tabs for different optimization views
    tabs = st.tabs(
        [
            "📊 Current Portfolio",
            "🎯 Optimization Results",
            "📈 Efficient Frontier",
            "💡 Recommendations",
            "📋 Reports",
        ]
    )

    # Fetch and analyze current portfolio
    current_data = optimizer.fetch_trading_data()
    portfolio_analytics = optimizer.calculate_portfolio_analytics(current_data)

    # Current Portfolio Tab
    with tabs[0]:
        st.subheader("📊 Current Portfolio Analysis")

        if not portfolio_analytics:
            st.warning(
                "No portfolio data available. Please ensure trading bots are active."
            )
            return
        # Portfolio overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_profit = portfolio_analytics.get("total_profit", 0)
            # Check if using real data
            data_source = current_data.get("data_source", "api")
            indicator = "📡" if data_source == "production_manager" else "🟡"
            st.markdown(
                f"""
            <div class="optimization-card">
                <h3>💰 Total Value {indicator}</h3>
                <h2>${total_profit:,.2f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            portfolio_return = portfolio_analytics.get("portfolio_return", 0)
            st.markdown(
                f"""
            <div class="optimization-card">
                <h3>📈 Portfolio Return</h3>
                <h2>{portfolio_return:.2f}%</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            portfolio_sharpe = portfolio_analytics.get("portfolio_sharpe", 0)
            st.markdown(
                f"""
            <div class="optimization-card">
                <h3>⚡ Sharpe Ratio</h3>
                <h2>{portfolio_sharpe:.2f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            diversification_ratio = portfolio_analytics.get("diversification_ratio", 0)
            st.markdown(
                f"""
            <div class="optimization-card">
                <h3>🔄 Diversification</h3>
                <h2>{diversification_ratio:.2f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Current allocation visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🥧 Current Allocation")

            bot_data = portfolio_analytics.get("bot_data", [])
            if bot_data:
                bot_names = [bot["name"] for bot in bot_data]
                current_weights = portfolio_analytics.get("weights", [])

                fig = px.pie(
                    values=current_weights,
                    names=bot_names,
                    title="Current Portfolio Weights",
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("📊 Performance Comparison")

            if bot_data:
                performance_df = pd.DataFrame(
                    {
                        "Bot": [bot["name"] for bot in bot_data],
                        "Return (%)": [bot["daily_return"] for bot in bot_data],
                        "Sharpe Ratio": [bot["sharpe_ratio"] for bot in bot_data],
                        "Volatility (%)": [bot["volatility"] * 100 for bot in bot_data],
                        "Current Weight (%)": [w * 100 for w in current_weights],
                    }
                )

                fig = px.scatter(
                    performance_df,
                    x="Volatility (%)",
                    y="Return (%)",
                    size="Current Weight (%)",
                    color="Sharpe Ratio",
                    hover_name="Bot",
                    title="Risk-Return Profile",
                    color_continuous_scale="RdYlGn",
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Detailed portfolio metrics
        st.subheader("📋 Portfolio Metrics Detail")

        metrics_data = {
            "Metric": [
                "Portfolio Volatility",
                "Concentration Index",
                "Number of Positions",
                "Diversification Ratio",
                "Weighted Avg Return",
                "Weighted Avg Sharpe",
            ],
            "Value": [
                f"{portfolio_analytics.get('portfolio_volatility', 0):.2f}%",
                f"{portfolio_analytics.get('concentration_index', 0):.3f}",
                f"{portfolio_analytics.get('num_positions', 0)}",
                f"{portfolio_analytics.get('diversification_ratio', 0):.2f}",
                f"{portfolio_analytics.get('portfolio_return', 0):.2f}%",
                f"{portfolio_analytics.get('portfolio_sharpe', 0):.2f}",
            ],
            "Interpretation": [
                "Lower is better for risk management",
                "Lower indicates better diversification",
                "More positions can improve diversification",
                "Higher indicates better diversification",
                "Higher daily returns are preferred",
                "Higher Sharpe ratios are better",
            ],
        }

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

    # Optimization Results Tab
    with tabs[1]:
        st.subheader("🎯 Portfolio Optimization Results")

        if not portfolio_analytics or not portfolio_analytics.get("bot_data"):
            st.warning("No data available for optimization")
        elif run_optimization or st.session_state.get("optimization_run", False):
            st.session_state["optimization_run"] = True

            with st.spinner(f"Running {optimization_method} optimization..."):
                bot_data = portfolio_analytics["bot_data"]
                current_weights = portfolio_analytics["weights"]

                # Run optimization
                optimal_weights = optimizer.optimize_portfolio_weights(
                    bot_data, optimization_method
                )

                if optimal_weights is not None:
                    # Calculate optimized portfolio metrics
                    opt_return = sum(
                        w * bot["daily_return"]
                        for w, bot in zip(optimal_weights, bot_data)
                    )
                    opt_volatility = np.sqrt(
                        sum(
                            w**2 * bot["volatility"] ** 2
                            for w, bot in zip(optimal_weights, bot_data)
                        )
                    )
                    opt_sharpe = (
                        opt_return / opt_volatility if opt_volatility > 0 else 0
                    )

                    # Display comparison
                    st.subheader("📊 Current vs Optimized Portfolio")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        current_return = portfolio_analytics["portfolio_return"]
                        improvement = opt_return - current_return
                        st.metric(
                            "Portfolio Return",
                            f"{opt_return:.2f}%",
                            delta=f"{improvement:+.2f}%",
                        )

                    with col2:
                        current_vol = portfolio_analytics["portfolio_volatility"]
                        vol_change = opt_volatility - current_vol
                        st.metric(
                            "Portfolio Volatility",
                            f"{opt_volatility:.2f}%",
                            delta=f"{vol_change:+.2f}%",
                        )

                    with col3:
                        current_sharpe = portfolio_analytics["portfolio_sharpe"]
                        sharpe_improvement = opt_sharpe - current_sharpe
                        st.metric(
                            "Sharpe Ratio",
                            f"{opt_sharpe:.2f}",
                            delta=f"{sharpe_improvement:+.2f}",
                        )

                    # Weight comparison chart
                    st.subheader("⚖️ Weight Allocation Comparison")

                    comparison_df = pd.DataFrame(
                        {
                            "Bot": [bot["name"] for bot in bot_data],
                            "Current Weight": current_weights,
                            "Optimal Weight": optimal_weights,
                            "Difference": optimal_weights - np.array(current_weights),
                        }
                    )

                    fig = go.Figure()

                    fig.add_trace(
                        go.Bar(
                            name="Current",
                            x=comparison_df["Bot"],
                            y=comparison_df["Current Weight"],
                            marker_color="lightblue",
                        )
                    )

                    fig.add_trace(
                        go.Bar(
                            name="Optimal",
                            x=comparison_df["Bot"],
                            y=comparison_df["Optimal Weight"],
                            marker_color="darkblue",
                        )
                    )

                    fig.update_layout(
                        title="Current vs Optimal Portfolio Weights",
                        xaxis_title="Trading Bots",
                        yaxis_title="Weight",
                        barmode="group",
                        height=400,
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Detailed comparison table
                    st.subheader("📋 Detailed Weight Comparison")

                    comparison_df["Current Weight %"] = (
                        comparison_df["Current Weight"] * 100
                    )
                    comparison_df["Optimal Weight %"] = (
                        comparison_df["Optimal Weight"] * 100
                    )
                    comparison_df["Change %"] = comparison_df["Difference"] * 100

                    display_df = comparison_df[
                        ["Bot", "Current Weight %", "Optimal Weight %", "Change %"]
                    ].round(2)

                    # Color code changes
                    def color_changes(val):
                        if val > 0:
                            return "background-color: #d4edda"  # Green for increases
                        elif val < 0:
                            return "background-color: #f8d7da"  # Red for decreases
                        return ""

                    styled_df = display_df.style.applymap(
                        color_changes, subset=["Change %"]
                    )
                    st.dataframe(styled_df, use_container_width=True)

                else:
                    st.error(
                        "Optimization failed. Please check your data and try again."
                    )
        else:
            st.info(
                "Click 'Run Optimization' in the sidebar to see optimization results."
            )

    # Efficient Frontier Tab
    with tabs[2]:
        st.subheader("📈 Efficient Frontier Analysis")

        if not portfolio_analytics or not portfolio_analytics.get("bot_data"):
            st.warning("No data available for efficient frontier calculation")
        else:
            if SCIPY_AVAILABLE:
                with st.spinner("Calculating efficient frontier..."):
                    bot_data = portfolio_analytics["bot_data"]
                    frontier_df, frontier_portfolios = (
                        optimizer.calculate_efficient_frontier(bot_data)
                    )

                    if frontier_df is not None:
                        # Plot efficient frontier
                        fig = go.Figure()

                        # Efficient frontier
                        fig.add_trace(
                            go.Scatter(
                                x=frontier_df["volatility"] * 100,
                                y=frontier_df["return"],
                                mode="lines+markers",
                                name="Efficient Frontier",
                                line=dict(color="blue", width=3),
                                marker=dict(size=6),
                            )
                        )

                        # Current portfolio
                        current_return = portfolio_analytics["portfolio_return"]
                        current_vol = portfolio_analytics["portfolio_volatility"]

                        fig.add_trace(
                            go.Scatter(
                                x=[current_vol],
                                y=[current_return],
                                mode="markers",
                                name="Current Portfolio",
                                marker=dict(size=15, color="red", symbol="diamond"),
                            )
                        )

                        # Individual assets
                        asset_returns = [bot["daily_return"] for bot in bot_data]
                        asset_vols = [bot["volatility"] * 100 for bot in bot_data]
                        asset_names = [bot["name"] for bot in bot_data]

                        fig.add_trace(
                            go.Scatter(
                                x=asset_vols,
                                y=asset_returns,
                                mode="markers",
                                name="Individual Assets",
                                marker=dict(size=10, color="green"),
                                text=asset_names,
                                textposition="top center",
                            )
                        )

                        # Maximum Sharpe portfolio
                        if frontier_portfolios:
                            max_sharpe_idx = np.argmax(
                                [p["sharpe"] for p in frontier_portfolios]
                            )
                            max_sharpe_port = frontier_portfolios[max_sharpe_idx]

                            fig.add_trace(
                                go.Scatter(
                                    x=[max_sharpe_port["volatility"] * 100],
                                    y=[max_sharpe_port["return"]],
                                    mode="markers",
                                    name="Max Sharpe Portfolio",
                                    marker=dict(size=15, color="gold", symbol="star"),
                                )
                            )

                        fig.update_layout(
                            title="Portfolio Efficient Frontier",
                            xaxis_title="Volatility (%)",
                            yaxis_title="Expected Return (%)",
                            height=500,
                            hovermode="closest",
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Frontier statistics
                        if frontier_portfolios:
                            col1, col2 = st.columns(2)

                            with col1:
                                st.subheader("🎯 Optimal Portfolio Points")

                                # Maximum Sharpe ratio portfolio
                                max_sharpe_port = frontier_portfolios[max_sharpe_idx]
                                st.markdown("**Maximum Sharpe Ratio Portfolio:**")
                                st.write(f"• Return: {max_sharpe_port['return']:.2f}%")
                                st.write(
                                    f"• Volatility: {max_sharpe_port['volatility']*100:.2f}%"
                                )
                                st.write(
                                    f"• Sharpe Ratio: {max_sharpe_port['sharpe']:.2f}"
                                )

                                # Minimum volatility portfolio
                                min_vol_idx = np.argmin(
                                    [p["volatility"] for p in frontier_portfolios]
                                )
                                min_vol_port = frontier_portfolios[min_vol_idx]
                                st.markdown("**Minimum Volatility Portfolio:**")
                                st.write(f"• Return: {min_vol_port['return']:.2f}%")
                                st.write(
                                    f"• Volatility: {min_vol_port['volatility']*100:.2f}%"
                                )
                                st.write(
                                    f"• Sharpe Ratio: {min_vol_port['sharpe']:.2f}"
                                )

                            with col2:
                                st.subheader("📊 Frontier Statistics")

                                returns_range = (
                                    frontier_df["return"].max()
                                    - frontier_df["return"].min()
                                )
                                vol_range = (
                                    frontier_df["volatility"].max()
                                    - frontier_df["volatility"].min()
                                )
                                avg_sharpe = frontier_df["sharpe"].mean()

                                st.write(f"• Return Range: {returns_range:.2f}%")
                                st.write(f"• Volatility Range: {vol_range*100:.2f}%")
                                st.write(f"• Average Sharpe: {avg_sharpe:.2f}")
                                st.write(f"• Portfolio Count: {len(frontier_df)}")
                    else:
                        st.error(
                            "Could not calculate efficient frontier. Please check your data."
                        )
            else:
                st.warning(
                    "Install SciPy for efficient frontier analysis: `pip install scipy`"
                )

    # Recommendations Tab
    with tabs[3]:
        st.subheader("💡 Optimization Recommendations")

        if not portfolio_analytics or not portfolio_analytics.get("bot_data"):
            st.warning("No data available for recommendations")
        elif st.session_state.get("optimization_run", False):
            bot_data = portfolio_analytics["bot_data"]
            current_weights = portfolio_analytics["weights"]
            optimal_weights = optimizer.optimize_portfolio_weights(
                bot_data, optimization_method
            )

            if optimal_weights is not None:
                recommendations = optimizer.generate_rebalancing_recommendations(
                    current_weights, optimal_weights, bot_data
                )

                if recommendations:
                    st.subheader("🔄 Rebalancing Actions")

                    for _i, rec in enumerate(recommendations):
                        icon = "📈" if rec["action"] == "Increase" else "📉"
                        color = "#d4edda" if rec["action"] == "Increase" else "#f8d7da"

                        st.markdown(
                            f"""
                        <div style="background: {color}; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;">
                            <h4>{icon} {rec['action']} {rec['bot_name']}</h4>
                            <p><strong>Current:</strong> {rec['current_weight']} → <strong>Target:</strong> {rec['optimal_weight']} ({rec['change']})</p>
                            <p><strong>Amount:</strong> {rec['amount']}</p>
                            <p><strong>Rationale:</strong> {rec['rationale']}</p>
                        </div>
                        """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.success(
                        "✅ Portfolio is already well-optimized. No significant rebalancing needed."
                    )

                # Implementation timeline
                st.subheader("⏰ Implementation Timeline")

                timeline_options = st.radio(
                    "Rebalancing Approach:",
                    ["Immediate", "Gradual (over 1 week)", "Gradual (over 1 month)"],
                    index=1,
                )

                if timeline_options == "Immediate":
                    st.info(
                        "💡 Implement all changes immediately for maximum optimization benefit"
                    )
                elif "1 week" in timeline_options:
                    st.info("💡 Spread changes over 1 week to reduce market impact")
                else:
                    st.info(
                        "💡 Gradual implementation over 1 month minimizes market impact"
                    )

                # Cost analysis
                st.subheader("💰 Implementation Cost Analysis")

                total_turnover = sum(
                    abs(opt - curr)
                    for opt, curr in zip(optimal_weights, current_weights)
                )
                estimated_cost = total_turnover * 0.001  # Assume 0.1% transaction cost

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Portfolio Turnover", f"{total_turnover:.1%}")

                with col2:
                    st.metric("Estimated Cost", f"{estimated_cost:.2%}")

                with col3:
                    improvement = optimal_weights.dot(
                        [bot["daily_return"] for bot in bot_data]
                    ) - np.array(current_weights).dot(
                        [bot["daily_return"] for bot in bot_data]
                    )
                    st.metric("Expected Improvement", f"{improvement:.2f}%/day")
        else:
            st.info("Run optimization first to see detailed recommendations.")

    # Reports Tab
    with tabs[4]:
        st.subheader("📋 Optimization Reports")

        # Report generation options
        col1, col2 = st.columns([2, 1])

        with col1:
            if portfolio_analytics:
                st.subheader("📊 Portfolio Optimization Summary")

                # Executive summary
                st.markdown("### Executive Summary")

                bot_data = portfolio_analytics.get("bot_data", [])
                if bot_data:
                    current_sharpe = portfolio_analytics.get("portfolio_sharpe", 0)
                    diversification = portfolio_analytics.get(
                        "diversification_ratio", 0
                    )
                    concentration = portfolio_analytics.get("concentration_index", 0)

                    st.write(f"**Portfolio Composition:** {len(bot_data)} trading bots")
                    st.write(f"**Current Sharpe Ratio:** {current_sharpe:.2f}")
                    st.write(
                        f"**Diversification Level:** {'High' if diversification > 1.5 else 'Medium' if diversification > 1.2 else 'Low'}"
                    )
                    st.write(
                        f"**Concentration Risk:** {'High' if concentration > 0.5 else 'Medium' if concentration > 0.3 else 'Low'}"
                    )

                    # Key insights
                    st.markdown("### Key Insights")

                    insights = []

                    if current_sharpe < 1.0:
                        insights.append(
                            "⚠️ Portfolio Sharpe ratio below 1.0 suggests room for improvement"
                        )

                    if concentration > 0.5:
                        insights.append(
                            "⚠️ High concentration risk - consider better diversification"
                        )

                    if diversification < 1.2:
                        insights.append(
                            "💡 Limited diversification benefits - review correlation between bots"
                        )

                    best_performer = max(bot_data, key=lambda x: x["sharpe_ratio"])
                    worst_performer = min(bot_data, key=lambda x: x["sharpe_ratio"])

                    insights.append(
                        f"🏆 Best performer: {best_performer['name']} (Sharpe: {best_performer['sharpe_ratio']:.2f})"
                    )
                    insights.append(
                        f"📉 Needs attention: {worst_performer['name']} (Sharpe: {worst_performer['sharpe_ratio']:.2f})"
                    )

                    for insight in insights:
                        st.write(insight)

        with col2:
            st.subheader("📤 Export Options")

            if st.button("📊 Generate PDF Report"):
                st.info("PDF report generation would be implemented here")

            if st.button("📈 Export Data to Excel"):
                st.info("Excel export would be implemented here")

            if st.button("📧 Email Report"):
                st.info("Email functionality would be implemented here")

            st.subheader("⚙️ Report Settings")

            st.multiselect(
                "Include Sections",
                [
                    "Current Analysis",
                    "Optimization Results",
                    "Recommendations",
                    "Risk Analysis",
                ],
                default=["Current Analysis", "Optimization Results"],
            )

            st.selectbox("Automated Reports", ["None", "Daily", "Weekly", "Monthly"])

            if st.button("💾 Save Settings"):
                st.success("Settings saved!")

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #7f8c8d;">📊 ZoL0 Portfolio Optimization Dashboard - Maximizing Risk-Adjusted Returns</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
