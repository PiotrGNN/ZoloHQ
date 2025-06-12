#!/usr/bin/env python3
"""
Memory Monitoring Dashboard
Real-time memory usage monitoring for ZoL0 system
"""

import json
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st

# Add memory optimizer
try:
    from memory_cleanup_optimizer import apply_memory_optimizations, memory_optimizer

    apply_memory_optimizations()
except ImportError:
    st.warning("Memory cleanup optimizer not available")

st.set_page_config(page_title="ZoL0 Memory Monitor", page_icon="🧠", layout="wide")


def get_zol0_processes():
    """Get all ZoL0-related Python processes"""
    processes = []

    zol0_scripts = [
        "enhanced_dashboard.py",
        "master_control_dashboard.py",
        "unified_trading_dashboard.py",
        "enhanced_dashboard_api.py",
        "dashboard.py",
        "dashboard_api.py",
    ]

    for proc in psutil.process_iter(
        ["pid", "name", "memory_info", "cmdline", "create_time"]
    ):
        try:
            if proc.info["name"] == "python.exe":
                cmdline = proc.info["cmdline"]
                if cmdline and len(cmdline) > 1:
                    script_name = os.path.basename(cmdline[1])
                    if any(script in script_name for script in zol0_scripts):
                        memory_mb = proc.info["memory_info"].rss / 1024 / 1024
                        runtime = datetime.now() - datetime.fromtimestamp(
                            proc.info["create_time"]
                        )

                        processes.append(
                            {
                                "PID": proc.info["pid"],
                                "Script": script_name,
                                "Memory (MB)": round(memory_mb, 2),
                                "Runtime": str(runtime).split(".")[0],
                                "Status": (
                                    "🚨 CRITICAL"
                                    if memory_mb > 600
                                    else "⚠️ HIGH" if memory_mb > 300 else "✅ OK"
                                ),
                            }
                        )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return sorted(processes, key=lambda x: x["Memory (MB)"], reverse=True)


def get_system_memory_info():
    """Get overall system memory information"""
    memory = psutil.virtual_memory()
    return {
        "Total": round(memory.total / 1024 / 1024 / 1024, 2),
        "Available": round(memory.available / 1024 / 1024 / 1024, 2),
        "Used": round(memory.used / 1024 / 1024 / 1024, 2),
        "Percentage": memory.percent,
    }


def create_memory_trend_chart(processes_df):
    """Create memory usage trend chart"""
    if processes_df.empty:
        return go.Figure()

    fig = px.bar(
        processes_df,
        x="Script",
        y="Memory (MB)",
        color="Memory (MB)",
        color_continuous_scale="RdYlGn_r",
        title="Memory Usage by Process",
    )

    fig.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)

    # Add threshold lines
    fig.add_hline(
        y=600, line_dash="dash", line_color="red", annotation_text="Critical (600MB)"
    )
    fig.add_hline(
        y=300, line_dash="dash", line_color="orange", annotation_text="High (300MB)"
    )

    return fig


def main():
    st.title("🧠 ZoL0 Memory Monitoring Dashboard")
    st.markdown("Real-time memory usage monitoring for all ZoL0 processes")

    # Add refresh button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()

    with col2:
        auto_refresh = st.checkbox("Auto Refresh (30s)")

    if auto_refresh:
        time.sleep(1)
        st.rerun()

    # System overview
    st.subheader("📊 System Memory Overview")
    system_memory = get_system_memory_info()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total RAM", f"{system_memory['Total']:.1f} GB")
    with col2:
        st.metric(
            "Used RAM",
            f"{system_memory['Used']:.1f} GB",
            f"{system_memory['Percentage']:.1f}%",
        )
    with col3:
        st.metric("Available RAM", f"{system_memory['Available']:.1f} GB")
    with col4:
        status = (
            "🚨 HIGH"
            if system_memory["Percentage"] > 80
            else "⚠️ MEDIUM" if system_memory["Percentage"] > 60 else "✅ OK"
        )
        st.metric("System Status", status)

    # ZoL0 processes
    st.subheader("🐍 ZoL0 Process Memory Usage")
    processes = get_zol0_processes()

    if processes:
        processes_df = pd.DataFrame(processes)

        # Summary metrics
        total_memory = processes_df["Memory (MB)"].sum()
        critical_count = len(processes_df[processes_df["Memory (MB)"] > 600])
        high_count = len(processes_df[processes_df["Memory (MB)"] > 300])

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total ZoL0 Memory", f"{total_memory:.1f} MB")
        with col2:
            st.metric("Active Processes", len(processes))
        with col3:
            st.metric("Critical Processes", critical_count)
        with col4:
            st.metric("High Usage Processes", high_count)

        # Memory chart
        fig = create_memory_trend_chart(processes_df)
        st.plotly_chart(fig, use_container_width=True)

        # Process table
        st.subheader("📋 Detailed Process Information")
        st.dataframe(processes_df, use_container_width=True, hide_index=True)

        # Alerts
        if critical_count > 0:
            st.error(
                f"🚨 **CRITICAL**: {critical_count} processes using >600MB memory!"
            )
            st.markdown("**Recommended Actions:**")
            st.markdown("- Restart critical processes")
            st.markdown("- Check for memory leaks")
            st.markdown("- Monitor data loading patterns")

        elif high_count > 0:
            st.warning(f"⚠️ **HIGH USAGE**: {high_count} processes using >300MB memory")
            st.markdown("**Monitor these processes closely**")

        else:
            st.success(
                "✅ **ALL PROCESSES NORMAL**: Memory usage within acceptable limits"
            )

    else:
        st.info("No ZoL0 processes currently running")

    # Memory optimization tools
    st.subheader("🛠️ Memory Management Tools")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧹 Force Garbage Collection", type="secondary"):
            import gc

            collected = gc.collect()
            st.success(f"Collected {collected} objects")

    with col2:
        if st.button("📊 Generate Memory Report", type="secondary"):
            report = {
                "timestamp": datetime.now().isoformat(),
                "system_memory": system_memory,
                "processes": processes,
                "total_zol0_memory": total_memory if processes else 0,
                "critical_count": critical_count if processes else 0,
            }

            filename = f"memory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(os.getcwd(), filename)

            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)

            st.success(f"Report saved: {filename}")

    with col3:
        if st.button("🔄 Restart High Memory Processes", type="secondary"):
            high_memory_processes = [p for p in processes if p["Memory (MB)"] > 400]
            if high_memory_processes:
                st.warning(f"Would restart {len(high_memory_processes)} processes")
                for proc in high_memory_processes:
                    st.write(
                        f"- {proc['Script']} (PID: {proc['PID']}, {proc['Memory (MB)']}MB)"
                    )
            else:
                st.info("No high memory processes to restart")

    # Memory optimization status
    st.subheader("⚙️ Memory Optimization Status")

    try:
        memory_info = memory_optimizer.check_memory_usage()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Process Memory", f"{memory_info['memory_mb']:.1f} MB")
        with col2:
            st.metric("Memory Percentage", f"{memory_info['memory_percent']:.1f}%")
        with col3:
            status = "🚨 CRITICAL" if memory_info["is_critical"] else "✅ OK"
            st.metric("Optimization Status", status)

        # Memory monitor widget
        memory_optimizer.create_memory_monitor_widget()

    except Exception as e:
        st.error(f"Memory optimizer not available: {e}")

    # Auto-refresh
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
