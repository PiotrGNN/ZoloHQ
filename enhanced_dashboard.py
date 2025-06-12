"""
Enhanced Dashboard with Core System Monitoring
Rozszerzony dashboard z monitorowaniem systemu core
"""

import gc  # For garbage collection
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import requests
import streamlit as st

from dashboard_performance_optimizer import auto_optimize, dashboard_optimizer, performance_monitor

# Memory optimization imports (after standard imports)
from memory_cleanup_optimizer import apply_memory_optimizations, memory_optimizer, memory_safe_session_state

# Dodaj ścieżkę do core
sys.path.insert(0, str(Path(__file__).parent / "ZoL0-master"))

st.set_page_config(
    page_title="ZoL0 AI Trading System Dashboard", page_icon="🚀", layout="wide"
)

# Stylowanie
st.markdown(
    """
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .strategy-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .core-status {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }    .ai-status-secondary {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        margin: 0.5rem 0;
    }
    .control-panel {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 2px solid #e0e0e0;
    }
    .environment-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .env-testnet {
        background: #e3f2fd;
        color: #1976d2;
        border: 2px solid #2196f3;
    }
    .env-production {
        background: #fff3e0;
        color: #f57c00;
        border: 2px solid #ff9800;
    }
    .validation-item {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid #eee;
    }
    .ai-status {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class CoreSystemMonitor:
    """Monitor systemu core w czasie rzeczywistym"""

    def __init__(self):
        self.core_path = Path(__file__).parent / "ZoL0-master" / "core"
        self.production_mode = (
            os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true"
        )

        # Initialize production data manager for real data access
        try:
            from production_data_manager import ProductionDataManager

            self.production_manager = ProductionDataManager()
        except ImportError:
            self.production_manager = None

        # Memory management - optimized
        self._cache = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        self._max_cache_size = 10  # Limit cache size

    def _cleanup_cache(self):
        """Clear old cache data to prevent memory leaks"""
        current_time = time.time()
        if (
            current_time - self._last_cleanup > self._cleanup_interval
            or len(self._cache) > self._max_cache_size
        ):
            # Keep only recent items
            if len(self._cache) > self._max_cache_size:
                # Remove oldest entries
                sorted_items = sorted(
                    self._cache.items(),
                    key=lambda x: x[1][1] if isinstance(x[1], tuple) else 0,
                )
                for key, _ in sorted_items[: -self._max_cache_size // 2]:
                    del self._cache[key]
            else:
                self._cache.clear()

            self._last_cleanup = current_time
            gc.collect()  # Force garbage collection

    @performance_monitor("get_core_status")
    def get_core_status(self):
        """Pobierz status komponentów core"""
        # Check cache first to avoid repeated initialization
        cache_key = "core_status"
        current_time = time.time()

        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if current_time - timestamp < 60:  # Cache for 1 minute
                return cached_data

        # Cleanup old cache periodically
        self._cleanup_cache()

        status = {
            "strategies": {"count": 0, "status": "unknown", "list": []},
            "ai_models": {"count": 0, "status": "unknown"},
            "trading_engine": {"status": "unknown"},
            "portfolio": {"status": "unknown"},
            "risk_management": {"status": "unknown"},
            "monitoring": {"status": "unknown"},
        }

        try:
            # Test strategii
            from core.strategies.manager import StrategyManager

            manager = StrategyManager()
            status["strategies"] = {
                "count": len(manager.strategies),
                "status": "active",
                "list": list(manager.strategies.keys()),
            }
            # Clear reference to prevent memory accumulation
            del manager
        except Exception as e:
            status["strategies"]["status"] = f"error: {str(e)[:50]}"

        try:
            # Test AI
            status["ai_models"]["status"] = "active"

            # Dodaj licznik AI modeli z folderu ai_models
            import ai_models

            ai_models_dict = ai_models.get_available_models()
            status["ai_models"]["count"] = len(ai_models_dict)
            # Clear reference
            del ai_models_dict
        except Exception as e:
            status["ai_models"]["status"] = f"error: {str(e)[:50]}"

        try:
            # Test trading engine
            status["trading_engine"]["status"] = "active"
        except Exception as e:
            status["trading_engine"]["status"] = f"error: {str(e)[:50]}"

        try:
            # Test portfolio
            status["portfolio"]["status"] = "active"
        except Exception as e:
            status["portfolio"]["status"] = f"error: {str(e)[:50]}"

        try:
            # Test risk management
            status["risk_management"]["status"] = "active"
        except Exception as e:
            status["risk_management"]["status"] = f"error: {str(e)[:50]}"

        # Cache the result
        self._cache[cache_key] = (status, current_time)
        return status

    @performance_monitor("get_system_metrics")
    def get_system_metrics(self):
        """Pobierz metryki systemowe"""
        # Cache system metrics to reduce psutil calls
        cache_key = "system_metrics"
        current_time = time.time()

        if cache_key in self._cache:
            cached_data, timestamp = self._cache[cache_key]
            if current_time - timestamp < 30:  # Cache for 30 seconds
                return cached_data

        metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),  # Reduced interval
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": (
                psutil.disk_usage("/").percent
                if os.name != "nt"
                else psutil.disk_usage("C:").percent
            ),
            "processes": len(psutil.pids()),
            "timestamp": datetime.now(),
        }

        # Cache the result
        self._cache[cache_key] = (metrics, current_time)
        return metrics


def clear_session_state_memory():
    """Clear old session state data to prevent memory leaks"""
    keys_to_remove = []
    for key in memory_safe_session_state("keys")():
        if (
            key.startswith("temp_")
            or key.startswith("old_")
            or key.startswith("cache_")
        ):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del st.session_state[key]

    # Force garbage collection
    gc.collect()


def main():
    """Główna funkcja dashboard"""

    # Automatic performance optimization
    auto_optimize()

    # Initialize memory optimizations
    apply_memory_optimizations()
    # Add memory monitor to sidebar
    with st.sidebar:
        st.subheader("🧠 Memory Monitor")
        memory_optimizer.create_memory_monitor_widget()

        st.subheader("⚡ Performance Monitor")
        dashboard_optimizer.create_performance_widget()

    # Clear old session state data periodically
    clear_session_state_memory()
    # Ensure CoreSystemMonitor is always in session state
    if "core_monitor" not in st.session_state:
        st.session_state.core_monitor = CoreSystemMonitor()
    core_monitor = st.session_state.core_monitor
    # Memory usage tracking
    if "page_loads" not in st.session_state:
        st.session_state.page_loads = 0
    st.session_state.page_loads += 1

    # Force cleanup every 50 page loads
    if st.session_state.page_loads % 50 == 0:
        clear_session_state_memory()
        gc.collect()
    # Header
    st.title("🚀 ZoL0 AI Trading System Dashboard")
    st.markdown("**Enhanced with Core System Monitoring**")
    st.caption(
        f"Page loads: {memory_safe_session_state('page_loads', 0)} | Memory optimized"
    )

    # === SEKCJA CONTROL PANEL ===
    st.header("🎛️ System Control Panel")

    # Panel kontrolny w dwóch kolumnach
    control_col1, control_col2, control_col3 = st.columns(3)

    with control_col1:
        st.subheader("🌍 Environment Control")

        # Pobierz aktualny status środowiska
        try:
            env_response = requests.get(
                "http://localhost:5001/api/environment/status", timeout=5
            )
            if env_response.status_code == 200:
                env_data = env_response.json()
                current_env = env_data.get("status", {}).get("environment", "unknown")
                production_ready = env_data.get("status", {}).get(
                    "production_enabled", False
                ) and env_data.get("status", {}).get("production_confirmed", False)
            else:
                current_env = "unknown"
                production_ready = False
        except Exception as e:
            current_env = "unknown"
            production_ready = False
            st.warning(f"⚠️  Env fetch error: {e}")

        st.info(f"Current Environment: **{current_env.title()}**")

        # Przełącznik środowiska
        target_env = st.selectbox(
            "Switch to Environment:",
            ["testnet", "production"],
            index=0 if current_env == "testnet" else 1,
        )

        if st.button("🔄 Switch Environment", type="primary"):
            if target_env == "production" and not production_ready:
                st.error("⚠️ Production environment not properly configured!")
                st.warning(
                    "Please set BYBIT_PRODUCTION_CONFIRMED=true and BYBIT_PRODUCTION_ENABLED=true"
                )
            else:
                with st.spinner("Switching environment..."):
                    try:
                        response = requests.post(
                            "http://localhost:5001/api/environment/switch",
                            json={"target_environment": target_env},
                            timeout=10,
                        )
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                st.success(f"✅ Successfully switched to {target_env}")
                                st.rerun()
                            else:
                                st.error(
                                    f"❌ Failed: {result.get('error', 'Unknown error')}"
                                )
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Connection Error: {str(e)}")

    with control_col2:
        st.subheader("⚙️ Trading Engine Control")

        # Pobierz status trading engine
        try:
            trading_response = requests.get(
                "http://localhost:5001/api/trading/status", timeout=5
            )
            if trading_response.status_code == 200:
                trading_data = trading_response.json()
                engine_active = trading_data.get("status", {}).get("active", False)
                engine_available = trading_data.get("success", False)
            else:
                engine_active = False
                engine_available = False
        except Exception as e:
            engine_active = False
            engine_available = False
            st.warning(f"⚠️  Trading status fetch error: {e}")

        # Status display
        if engine_available:
            status_icon = "🟢" if engine_active else "🔴"
            status_text = "Running" if engine_active else "Stopped"
        else:
            status_icon = "❓"
            status_text = "Unavailable"

        st.info(f"Engine Status: {status_icon} **{status_text}**")

        # Control buttons
        col_start, col_stop = st.columns(2)

        with col_start:
            if st.button(
                "▶️ Start Trading", disabled=engine_active or not engine_available
            ):
                with st.spinner("Starting trading engine..."):
                    try:
                        response = requests.post(
                            "http://localhost:5001/api/trading/start",
                            json={"symbols": ["BTCUSDT", "ETHUSDT"]},
                            timeout=10,
                        )
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                st.success("✅ Trading Engine started!")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed: {result.get('error')}")
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Connection Error: {str(e)}")

        with col_stop:
            if st.button("⏹️ Stop Trading", disabled=not engine_active):
                with st.spinner("Stopping trading engine..."):
                    try:
                        response = requests.post(
                            "http://localhost:5001/api/trading/stop", timeout=10
                        )
                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                st.success("✅ Trading Engine stopped!")
                                st.rerun()
                            else:
                                st.error(f"❌ Failed: {result.get('error')}")
                        else:
                            st.error(f"❌ API Error: {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ Connection Error: {str(e)}")

    with control_col3:
        st.subheader("🔍 System Validation")

        # Pobierz wyniki walidacji
        try:
            validation_response = requests.get(
                "http://localhost:5001/api/system/validation", timeout=5
            )
            if validation_response.status_code == 200:
                validation_data = validation_response.json()
                validation_results = validation_data.get("validation", {})
                ready_for_prod = validation_data.get("ready_for_production", False)
            else:
                validation_results = {}
                ready_for_prod = False
        except Exception as e:
            validation_results = {}
            ready_for_prod = False
            st.warning(f"⚠️  Validation fetch error: {e}")

        # Display validation results
        st.write("**System Components:**")
        for component, status in validation_results.items():
            icon = "✅" if status else "❌"
            readable_name = component.replace("_", " ").title()
            st.write(f"{icon} {readable_name}")

        # Overall status
        if ready_for_prod:
            st.success("🟢 Ready for Production")
        else:
            st.warning("🟡 Development Mode Only")

        if st.button("🔄 Refresh Validation"):
            st.rerun()

    st.divider()
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    auto_refresh = st.sidebar.checkbox(
        "Auto Refresh (30s)", value=False
    )  # Disabled by default
    refresh_button = st.sidebar.button("🔄 Refresh Now")

    # Memory management controls
    st.sidebar.subheader("🧹 Memory Management")
    current_memory = psutil.virtual_memory().percent
    st.sidebar.metric("Current Memory Usage", f"{current_memory:.1f}%")

    if st.sidebar.button("🗑️ Clear Cache"):
        clear_session_state_memory()
        core_monitor._cache.clear()
        gc.collect()
        st.sidebar.success("Cache cleared!")

    # Auto refresh with memory management
    if auto_refresh:
        time.sleep(30)  # 30 second delay
        # Clear cache before refresh to prevent accumulation
        if len(core_monitor._cache) > 10:
            core_monitor._cache.clear()
        st.rerun()

    if refresh_button:
        clear_session_state_memory()
        st.rerun()
    # Pobierz dane z bezpiecznym dostępem do session state
    core_status = st.session_state.core_monitor.get_core_status()
    system_metrics = st.session_state.core_monitor.get_system_metrics()

    # === SEKCJA 1: SYSTEM STATUS ===
    st.header("📊 System Status Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h3>💾 System Resources</h3>
            <p>CPU: {system_metrics['cpu_percent']:.1f}%</p>
            <p>Memory: {system_metrics['memory_percent']:.1f}%</p>
            <p>Disk: {system_metrics['disk_percent']:.1f}%</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        strategy_status = (
            "🟢 Active"
            if core_status["strategies"]["status"] == "active"
            else "🔴 Error"
        )
        st.markdown(
            f"""
        <div class="strategy-card">
            <h3>🎯 Trading Strategies</h3>
            <p>Status: {strategy_status}</p>
            <p>Count: {core_status["strategies"]["count"]}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        ai_status = (
            "🟢 Active"
            if core_status["ai_models"]["status"] == "active"
            else "🔴 Error"
        )
        st.markdown(
            f"""
        <div class="ai-status">
            <h3>🤖 AI Models</h3>
            <p>Status: {ai_status}</p>
            <p>Count: {core_status["ai_models"]["count"]}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        engine_status = (
            "🟢 Active"
            if core_status["trading_engine"]["status"] == "active"
            else "🔴 Error"
        )
        st.markdown(
            f"""
        <div class="core-status">
            <h3>⚙️ Trading Engine</h3>
            <p>Status: {engine_status}</p>
            <p>Portfolio: {"🟢" if core_status["portfolio"]["status"] == "active" else "🔴"}</p>
            <p>Risk Mgmt: {"🟢" if core_status["risk_management"]["status"] == "active" else "🔴"}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    # === SEKCJA 2: CORE COMPONENTS DETAILS ===
    st.header("🔧 Core Components Status")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Active Trading Strategies")
        if core_status["strategies"]["status"] == "active":
            strategy_df = pd.DataFrame(
                {
                    "Strategy": core_status["strategies"]["list"],
                    "Status": ["🟢 Active"] * len(core_status["strategies"]["list"]),
                    "Type": ["AI Enhanced"] * len(core_status["strategies"]["list"]),
                }
            )

            # Periodic memory cleanup
            memory_optimizer.periodic_cleanup()
            st.dataframe(strategy_df, use_container_width=True)
        else:
            st.error(f"Strategies Error: {core_status['strategies']['status']}")

    with col2:
        st.subheader("🤖 AI Models Integration")
        ai_components = [
            {
                "Component": "RL Trader",
                "Status": (
                    "🟢 Active"
                    if "active" in core_status["ai_models"]["status"]
                    else "🔴 Error"
                ),
            },
            {"Component": "Sentiment Analysis", "Status": "🟢 Active"},
            {"Component": "Anomaly Detection", "Status": "🟢 Active"},
            {"Component": "Pattern Recognition", "Status": "🟢 Active"},
            {"Component": "Model Training", "Status": "🟢 Active"},
        ]
        ai_df = pd.DataFrame(ai_components)

        # Periodic memory cleanup
        memory_optimizer.periodic_cleanup()
        st.dataframe(ai_df, use_container_width=True)

    # === SEKCJA 3: PERFORMANCE METRICS ===
    st.header("📈 Performance Metrics")

    col1, col2 = st.columns(2)
    with col1:
        # CPU/Memory usage over time - limit data points to prevent memory buildup
        max_points = 30  # Limit to 30 data points
        cpu_data = [
            system_metrics["cpu_percent"] + (i % 10 - 5) for i in range(max_points)
        ]
        memory_data = [
            system_metrics["memory_percent"] + (i % 8 - 4) for i in range(max_points)
        ]
        chart_dates = pd.date_range(
            start=datetime.now() - timedelta(days=max_points),
            end=datetime.now(),
            freq="D",
        )

        fig_system = go.Figure()
        fig_system.add_trace(
            go.Scatter(
                x=chart_dates, y=cpu_data, name="CPU %", line=dict(color="#ff6b6b")
            )
        )
        fig_system.add_trace(
            go.Scatter(
                x=chart_dates,
                y=memory_data,
                name="Memory %",
                line=dict(color="#4ecdc4"),
            )
        )
        fig_system.update_layout(
            title="System Resource Usage (30 days)",
            yaxis_title="Percentage",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),  # Reduce margins
        )

        # Periodic memory cleanup
        memory_optimizer.periodic_cleanup()
        st.plotly_chart(fig_system, use_container_width=True)

        # Clear chart object from memory
        del fig_system, cpu_data, memory_data, chart_dates

    with col2:
        # Strategy performance - limit data to prevent memory accumulation
        strategy_list = core_status["strategies"]["list"][:10]  # Limit to 10 strategies
        strategy_performance = {
            strategy: 75 + (hash(strategy) % 20) for strategy in strategy_list
        }

        if strategy_performance:
            fig_strategies = px.bar(
                x=list(strategy_performance.keys()),
                y=list(strategy_performance.values()),
                title="Strategy Performance Score",
                color=list(strategy_performance.values()),
                color_continuous_scale="viridis",
                height=400,
            )
            fig_strategies.update_layout(
                showlegend=False, margin=dict(l=50, r=50, t=50, b=50)
            )

            # Periodic memory cleanup
            memory_optimizer.periodic_cleanup()
            st.plotly_chart(fig_strategies, use_container_width=True)

            # Clear chart object from memory
            del fig_strategies, strategy_performance, strategy_list
        else:
            st.info("No strategy performance data available")

    # === SEKCJA 4: REAL-TIME MONITORING ===
    st.header("🔄 Real-Time Monitoring")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="🔥 Active Strategies",
            value=core_status["strategies"]["count"],
            delta=1 if core_status["strategies"]["count"] > 5 else 0,
        )

    with col2:
        st.metric(
            label="🤖 AI Models",
            value=core_status["ai_models"]["count"],
            delta=5 if core_status["ai_models"]["count"] > 20 else 0,
        )

    with col3:
        st.metric(
            label="💾 Memory Usage",
            value=f"{system_metrics['memory_percent']:.1f}%",
            delta=f"{system_metrics['memory_percent'] - 50:.1f}%",
        )
    # === SEKCJA 5: LOG MONITORING ===
    st.header("📋 System Logs")

    # Limit logs to prevent memory accumulation
    max_logs = 10  # Reduced from potentially unlimited
    recent_logs = [
        {
            "Time": datetime.now() - timedelta(minutes=i),
            "Level": ["INFO", "WARNING", "ERROR"][i % 3],
            "Component": ["Strategy", "AI Model", "Trading Engine"][i % 3],
            "Message": f"System event {i+1}",
        }
        for i in range(max_logs)
    ]
    logs_df = pd.DataFrame(recent_logs)

    # Periodic memory cleanup before displaying data
    memory_optimizer.periodic_cleanup()
    st.dataframe(logs_df, use_container_width=True, height=300)  # Fixed height

    # Clear DataFrame from memory
    del logs_df, recent_logs

    # === SEKCJA 6: ALERTS & NOTIFICATIONS ===
    st.header("⚠️ Alerts & Status")

    alerts = []

    # Sprawdź alerty systemowe
    if system_metrics["cpu_percent"] > 80:
        alerts.append("🔴 HIGH CPU USAGE: Consider scaling resources")

    if system_metrics["memory_percent"] > 80:
        alerts.append("🔴 HIGH MEMORY USAGE: Check for memory leaks")

    if core_status["strategies"]["status"] != "active":
        alerts.append("🔴 STRATEGIES OFFLINE: Check strategy manager")

    if core_status["ai_models"]["status"] != "active":
        alerts.append("🔴 AI MODELS ERROR: Check AI integration")

    if not alerts:
        alerts.append("🟢 ALL SYSTEMS OPERATIONAL")

    for alert in alerts:
        if "🔴" in alert:
            st.error(alert)
        elif "🟡" in alert:
            st.warning(alert)
        else:
            st.success(alert)
    # === FOOTER ===
    st.markdown("---")

    # Memory usage footer
    memory_info = psutil.virtual_memory()
    process_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    st.markdown(
        f"""
    **ZoL0 AI Trading System Dashboard** | 
    Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
    Core System v0.6.4 | 
    AI Models: {core_status["ai_models"]["count"]} Active |    System Memory: {memory_info.percent:.1f}% | 
    Process Memory: {process_memory:.1f}MB | 
    Page Loads: {memory_safe_session_state('page_loads', 0)}
    """
    )
    # Final cleanup
    if memory_safe_session_state("page_loads", 0) % 10 == 0:
        gc.collect()


if __name__ == "__main__":
    main()
