#!/usr/bin/env python3
"""
Enhanced Performance Dashboard Integration for ZoL0
==================================================

Integrates performance monitoring, caching analytics, and production
usage data into the existing Streamlit dashboard.
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Import our monitoring systems
try:
    from advanced_performance_monitor import PerformanceMonitor
    from api_cache_system import get_cache_instance
    from production_usage_monitor import get_production_monitor

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    st.warning("Performance monitoring modules not available")


class PerformanceDashboard:
    """Enhanced performance dashboard integration"""

    def __init__(self):
        if MONITORING_AVAILABLE:
            self.performance_monitor = PerformanceMonitor()
            self.cache = get_cache_instance()
            self.production_monitor = get_production_monitor()
        else:
            self.performance_monitor = None
            self.cache = None
            self.production_monitor = None

    def render_performance_overview(self):
        """Render performance overview section"""
        st.header("🚀 Performance Monitoring")

        if not MONITORING_AVAILABLE:
            st.error("Performance monitoring system not available")
            return

        # Get real-time data
        dashboard_data = self.production_monitor.get_production_dashboard_data()
        self.cache.get_stats()

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Avg Response Time",
                f"{dashboard_data['realtime_metrics']['avg_response_time']:.2f}s",
                delta=(
                    f"{-0.5:.2f}s"
                    if dashboard_data["realtime_metrics"]["avg_response_time"] < 2.0
                    else f"{+0.3:.2f}s"
                ),
            )

        with col2:
            cache_hit_rate = dashboard_data["realtime_metrics"]["cache_hit_rate"]
            st.metric(
                "Cache Hit Rate",
                f"{cache_hit_rate:.1%}",
                delta=f"{+10:.0f}%" if cache_hit_rate > 0.7 else f"{-5:.0f}%",
            )

        with col3:
            error_rate = dashboard_data["realtime_metrics"]["error_rate"]
            st.metric(
                "Error Rate",
                f"{error_rate:.1%}",
                delta=f"{-2:.1f}%" if error_rate < 0.05 else f"{+1:.1f}%",
            )

        with col4:
            requests_per_min = dashboard_data["realtime_metrics"]["requests_per_minute"]
            st.metric(
                "Requests/Min",
                f"{requests_per_min}",
                delta=f"{+10}" if requests_per_min > 50 else f"{-5}",
            )

    def render_system_health(self):
        """Render system health section"""
        st.subheader("💻 System Health")

        if not MONITORING_AVAILABLE:
            return

        dashboard_data = self.production_monitor.get_production_dashboard_data()
        system_metrics = dashboard_data["system_metrics"]

        # System metrics gauge charts
        col1, col2, col3 = st.columns(3)

        with col1:
            # CPU Usage Gauge
            fig_cpu = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=system_metrics["cpu_usage"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "CPU Usage (%)"},
                    delta={"reference": 50},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 50], "color": "lightgray"},
                            {"range": [50, 80], "color": "yellow"},
                            {"range": [80, 100], "color": "red"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )
            fig_cpu.update_layout(height=300)
            st.plotly_chart(fig_cpu, use_container_width=True)

        with col2:
            # Memory Usage Gauge
            fig_mem = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=system_metrics["memory_usage"],
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Memory Usage (%)"},
                    delta={"reference": 60},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "darkgreen"},
                        "steps": [
                            {"range": [0, 60], "color": "lightgray"},
                            {"range": [60, 85], "color": "yellow"},
                            {"range": [85, 100], "color": "red"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )
            fig_mem.update_layout(height=300)
            st.plotly_chart(fig_mem, use_container_width=True)

        with col3:
            # Cache Efficiency Gauge
            cache_hit_rate = dashboard_data["realtime_metrics"]["cache_hit_rate"] * 100
            fig_cache = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=cache_hit_rate,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Cache Hit Rate (%)"},
                    delta={"reference": 70},
                    gauge={
                        "axis": {"range": [None, 100]},
                        "bar": {"color": "purple"},
                        "steps": [
                            {"range": [0, 50], "color": "red"},
                            {"range": [50, 70], "color": "yellow"},
                            {"range": [70, 100], "color": "lightgray"},
                        ],
                        "threshold": {
                            "line": {"color": "green", "width": 4},
                            "thickness": 0.75,
                            "value": 80,
                        },
                    },
                )
            )
            fig_cache.update_layout(height=300)
            st.plotly_chart(fig_cache, use_container_width=True)

    def render_api_performance_charts(self):
        """Render API performance charts"""
        st.subheader("📊 API Performance Analytics")

        if not MONITORING_AVAILABLE:
            return

        # Get performance data
        perf_summary = self.performance_monitor.get_performance_summary(hours=24)

        if not perf_summary or "endpoints" not in perf_summary:
            st.info("No performance data available yet")
            return

        # Response time trends
        col1, col2 = st.columns(2)

        with col1:
            # Endpoint performance comparison
            endpoints_data = perf_summary["endpoints"]
            if endpoints_data:
                endpoints = list(endpoints_data.keys())
                avg_times = [endpoints_data[ep]["avg_duration"] for ep in endpoints]

                fig_endpoints = px.bar(
                    x=endpoints,
                    y=avg_times,
                    title="Average Response Time by Endpoint",
                    labels={"x": "Endpoint", "y": "Response Time (s)"},
                )
                fig_endpoints.update_layout(height=400)
                st.plotly_chart(fig_endpoints, use_container_width=True)

        with col2:
            # Success rate by endpoint
            if endpoints_data:
                success_rates = [
                    endpoints_data[ep]["success_rate"] * 100 for ep in endpoints
                ]

                fig_success = px.bar(
                    x=endpoints,
                    y=success_rates,
                    title="Success Rate by Endpoint",
                    labels={"x": "Endpoint", "y": "Success Rate (%)"},
                    color=success_rates,
                    color_continuous_scale="RdYlGn",
                )
                fig_success.update_layout(height=400)
                st.plotly_chart(fig_success, use_container_width=True)

    def render_cache_analytics(self):
        """Render cache analytics section"""
        st.subheader("🗄️ Cache Analytics")

        if not MONITORING_AVAILABLE:
            return

        cache_stats = self.cache.get_stats()
        endpoint_stats = self.cache.get_endpoint_stats()

        # Cache overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Hit Rate", f"{cache_stats['hit_rate']:.1%}")

        with col2:
            st.metric("Total Requests", f"{cache_stats['total_requests']:,}")

        with col3:
            st.metric("Cache Size", f"{cache_stats['current_size_mb']:.1f} MB")

        with col4:
            st.metric("Entries", f"{cache_stats['entries_count']:,}")

        # Cache utilization chart
        fig_utilization = go.Figure(
            data=[
                go.Pie(
                    labels=["Used", "Available"],
                    values=[
                        cache_stats["current_size_mb"],
                        cache_stats["max_size_mb"] - cache_stats["current_size_mb"],
                    ],
                    title="Cache Size Utilization",
                )
            ]
        )
        fig_utilization.update_layout(height=300)
        st.plotly_chart(fig_utilization, use_container_width=True)

        # Endpoint cache statistics
        if endpoint_stats:
            st.subheader("Cache by Endpoint")

            endpoint_df = pd.DataFrame(
                [
                    {
                        "Endpoint": endpoint,
                        "Entries": stats["entries"],
                        "Total Size (MB)": stats["total_size"] / 1024 / 1024,
                        "Access Count": stats["hits"],
                    }
                    for endpoint, stats in endpoint_stats.items()
                ]
            )

            st.dataframe(endpoint_df, use_container_width=True)

    def render_alerts_and_recommendations(self):
        """Render alerts and optimization recommendations"""
        st.subheader("⚠️ Alerts & Recommendations")

        if not MONITORING_AVAILABLE:
            return

        dashboard_data = self.production_monitor.get_production_dashboard_data()
        recent_alerts = dashboard_data.get("recent_alerts", [])

        # Alerts section
        if recent_alerts:
            st.write("**Recent Alerts:**")

            for alert in recent_alerts[-5:]:  # Show last 5 alerts
                severity_color = {
                    "low": "🟢",
                    "medium": "🟡",
                    "high": "🟠",
                    "critical": "🔴",
                }.get(alert["severity"], "⚪")

                alert_time = datetime.fromtimestamp(alert["timestamp"]).strftime(
                    "%H:%M:%S"
                )
                st.write(
                    f"{severity_color} **{alert_time}** - {alert['category'].title()}: {alert['message']}"
                )
        else:
            st.success("No recent alerts 🎉")

        # Optimization recommendations
        optimization_report = self.production_monitor.get_optimization_report()
        recommendations = optimization_report.get("recommendations", [])

        if recommendations:
            st.write("**Optimization Recommendations:**")

            for rec in recommendations[:3]:  # Show top 3 recommendations
                priority_icon = {1: "🔵", 2: "🟡", 3: "🔴"}.get(
                    rec[4], "⚪"
                )  # rec[4] is priority
                st.write(
                    f"{priority_icon} **{rec[2]}**: {rec[3]}"
                )  # rec[2] is category, rec[3] is recommendation
                st.write(
                    f"   *Estimated improvement: {rec[5]*100:.0f}%, Effort: {rec[6]}*"
                )  # rec[5] is improvement, rec[6] is effort
        else:
            st.info("No optimization recommendations at this time")

    def render_performance_trends(self):
        """Render performance trends over time"""
        st.subheader("📈 Performance Trends")

        if not MONITORING_AVAILABLE:
            return

        try:
            # Get historical data from database
            db_path = Path("production_monitoring.db")
            if not db_path.exists():
                st.info("No historical data available yet")
                return

            conn = sqlite3.connect(db_path)

            # Query real-time metrics for trends
            query = """
                SELECT timestamp, metric_type, value 
                FROM realtime_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """

            df = pd.read_sql_query(
                query, conn, params=[time.time() - 3600]
            )  # Last hour
            conn.close()

            if df.empty:
                st.info("No trend data available yet")
                return

            # Convert timestamp to datetime
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

            # Create trends chart
            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Response Time",
                    "Cache Hit Rate",
                    "CPU Usage",
                    "Memory Usage",
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                ],
            )

            # Response time trend
            response_time_data = df[df["metric_type"] == "avg_response_time"]
            if not response_time_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=response_time_data["datetime"],
                        y=response_time_data["value"],
                        mode="lines",
                        name="Response Time (s)",
                    ),
                    row=1,
                    col=1,
                )

            # Cache hit rate trend
            cache_data = df[df["metric_type"] == "cache_hit_rate"]
            if not cache_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=cache_data["datetime"],
                        y=cache_data["value"] * 100,
                        mode="lines",
                        name="Cache Hit Rate (%)",
                        line=dict(color="green"),
                    ),
                    row=1,
                    col=2,
                )

            # CPU usage trend
            cpu_data = df[df["metric_type"] == "cpu_usage"]
            if not cpu_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=cpu_data["datetime"],
                        y=cpu_data["value"],
                        mode="lines",
                        name="CPU Usage (%)",
                        line=dict(color="orange"),
                    ),
                    row=2,
                    col=1,
                )

            # Memory usage trend
            memory_data = df[df["metric_type"] == "memory_usage"]
            if not memory_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=memory_data["datetime"],
                        y=memory_data["value"],
                        mode="lines",
                        name="Memory Usage (%)",
                        line=dict(color="red"),
                    ),
                    row=2,
                    col=2,
                )

            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading trend data: {e}")

    def render_complete_dashboard(self):
        """Render the complete performance dashboard"""
        st.title("🎯 ZoL0 Performance Dashboard")

        # Auto-refresh option
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()

        # Manual refresh button
        if st.button("🔄 Refresh Data"):
            st.rerun()

        # Render all sections
        self.render_performance_overview()
        st.divider()

        self.render_system_health()
        st.divider()

        self.render_api_performance_charts()
        st.divider()

        self.render_cache_analytics()
        st.divider()

        self.render_alerts_and_recommendations()
        st.divider()

        self.render_performance_trends()

        # Footer with last update time
        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def add_performance_tab_to_dashboard():
    """Add performance monitoring tab to existing dashboard"""

    # This function would be called from the main dashboard.py
    # to add our performance monitoring as a new tab

    performance_dashboard = PerformanceDashboard()

    with st.container():
        performance_dashboard.render_complete_dashboard()


if __name__ == "__main__":
    # Standalone performance dashboard
    st.set_page_config(
        page_title="ZoL0 Performance Dashboard", page_icon="🚀", layout="wide"
    )

    dashboard = PerformanceDashboard()
    dashboard.render_complete_dashboard()
