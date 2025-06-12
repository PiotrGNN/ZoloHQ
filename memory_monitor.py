#!/usr/bin/env python3
"""
Memory Monitor for ZoL0 Trading System
Monitors memory usage of all Python processes and identifies potential leaks
"""

import time
from datetime import datetime

import pandas as pd
import psutil


def get_python_processes():
    """Get all Python processes with their memory usage"""
    python_procs = []

    for proc in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
        try:
            if proc.info["name"] == "python.exe":
                memory_mb = proc.info["memory_info"].rss / 1024 / 1024

                # Try to identify which dashboard/service this is
                cmdline = " ".join(proc.info["cmdline"]) if proc.info["cmdline"] else ""
                service_type = "Unknown"

                if "enhanced_dashboard.py" in cmdline:
                    service_type = "Enhanced Dashboard"
                elif "streamlit run" in cmdline:
                    if "8501" in cmdline:
                        service_type = "Portfolio Dashboard"
                    elif "8502" in cmdline:
                        service_type = "Risk Dashboard"
                    elif "8503" in cmdline:
                        service_type = "Strategy Dashboard"
                    elif "8504" in cmdline:
                        service_type = "AI Analytics Dashboard"
                    elif "8505" in cmdline:
                        service_type = "Technical Analysis Dashboard"
                    elif "8506" in cmdline:
                        service_type = "System Monitor Dashboard"
                    elif "8507" in cmdline:
                        service_type = "News Dashboard"
                    elif "8508" in cmdline:
                        service_type = "Advanced Trading Analytics"
                    elif "8509" in cmdline:
                        service_type = "Enhanced Dashboard (PORT 8509)"
                elif "enhanced_dashboard_api.py" in cmdline:
                    service_type = "Enhanced API Server"
                elif "run_production_api.py" in cmdline:
                    service_type = "Production API Server"

                python_procs.append(
                    {
                        "pid": proc.info["pid"],
                        "memory_mb": memory_mb,
                        "service_type": service_type,
                        "cmdline": cmdline[:100],  # Truncate for readability
                    }
                )

        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return python_procs


def monitor_memory(duration_minutes=10, check_interval=30):
    """Monitor memory usage over time"""
    print(f"Starting memory monitoring for {duration_minutes} minutes...")
    print("Checking every {} seconds\n".format(check_interval))

    monitoring_data = []
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)

    while time.time() < end_time:
        timestamp = datetime.now()
        python_procs = get_python_processes()

        print(f"\n=== Memory Report - {timestamp.strftime('%H:%M:%S')} ===")
        print(f"{'PID':<8} {'Memory (MB)':<12} {'Service Type':<30} {'Status'}")
        print("-" * 80)

        total_memory = 0
        high_memory_procs = []

        for proc in python_procs:
            memory_mb = proc["memory_mb"]
            total_memory += memory_mb

            status = ""
            if memory_mb > 500:
                status = "⚠️ HIGH"
                high_memory_procs.append(proc)
            elif memory_mb > 300:
                status = "🟡 ELEVATED"
            else:
                status = "🟢 NORMAL"

            print(
                f"{proc['pid']:<8} {memory_mb:<12.1f} {proc['service_type']:<30} {status}"
            )

            # Store data for analysis
            monitoring_data.append(
                {
                    "timestamp": timestamp,
                    "pid": proc["pid"],
                    "memory_mb": memory_mb,
                    "service_type": proc["service_type"],
                }
            )

        print(f"\nTotal Python Memory Usage: {total_memory:.1f} MB")

        if high_memory_procs:
            print(f"\n⚠️ HIGH MEMORY PROCESSES ({len(high_memory_procs)}):")
            for proc in high_memory_procs:
                print(
                    f"  PID {proc['pid']}: {proc['memory_mb']:.1f}MB - {proc['service_type']}"
                )

        print(f"\nNext check in {check_interval} seconds...")
        time.sleep(check_interval)

    # Save monitoring data
    df = pd.DataFrame(monitoring_data)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"memory_monitoring_{timestamp_str}.csv"
    df.to_csv(csv_filename, index=False)

    print(f"\n✅ Monitoring complete. Data saved to: {csv_filename}")

    # Generate summary report
    generate_summary_report(df, csv_filename)


def generate_summary_report(df, csv_filename):
    """Generate a summary report of memory usage"""
    print("\n" + "=" * 60)
    print("MEMORY MONITORING SUMMARY REPORT")
    print("=" * 60)

    if df.empty:
        print("No data collected.")
        return

    # Group by service type
    service_stats = (
        df.groupby("service_type")["memory_mb"]
        .agg(["mean", "max", "min", "std"])
        .round(2)
    )

    print("\nMEMORY USAGE BY SERVICE:")
    print(service_stats)

    # Identify potential memory leaks (increasing trend)
    print("\nPOTENTIAL MEMORY LEAK ANALYSIS:")
    for service in df["service_type"].unique():
        service_data = df[df["service_type"] == service].sort_values("timestamp")
        if len(service_data) >= 3:
            # Check if memory is consistently increasing
            memory_values = service_data["memory_mb"].values
            if len(memory_values) >= 3:
                recent_avg = memory_values[-3:].mean()
                initial_avg = memory_values[:3].mean()
                growth_rate = (
                    ((recent_avg - initial_avg) / initial_avg) * 100
                    if initial_avg > 0
                    else 0
                )

                if growth_rate > 10:  # More than 10% growth
                    print(
                        f"⚠️ {service}: {growth_rate:.1f}% memory growth - POTENTIAL LEAK"
                    )
                elif growth_rate > 5:
                    print(
                        f"🟡 {service}: {growth_rate:.1f}% memory growth - Monitor closely"
                    )
                else:
                    print(f"🟢 {service}: {growth_rate:.1f}% memory growth - Normal")

    # High memory services
    high_memory_services = service_stats[service_stats["max"] > 500]
    if not high_memory_services.empty:
        print("\n⚠️ HIGH MEMORY SERVICES (>500MB):")
        for service, stats in high_memory_services.iterrows():
            print(f"  {service}: Max {stats['max']}MB, Avg {stats['mean']}MB")

    print(f"\nDetailed data saved to: {csv_filename}")


def quick_check():
    """Quick memory check of all Python processes"""
    python_procs = get_python_processes()

    print("=== QUICK MEMORY CHECK ===")
    print(f"{'PID':<8} {'Memory (MB)':<12} {'Service Type':<30} {'Status'}")
    print("-" * 80)

    total_memory = 0
    for proc in python_procs:
        memory_mb = proc["memory_mb"]
        total_memory += memory_mb

        status = ""
        if memory_mb > 500:
            status = "⚠️ HIGH"
        elif memory_mb > 300:
            status = "🟡 ELEVATED"
        else:
            status = "🟢 NORMAL"

        print(
            f"{proc['pid']:<8} {memory_mb:<12.1f} {proc['service_type']:<30} {status}"
        )

    print(f"\nTotal Python Memory Usage: {total_memory:.1f} MB")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_check()
    elif len(sys.argv) > 1 and sys.argv[1] == "monitor":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        interval = int(sys.argv[3]) if len(sys.argv) > 3 else 30
        monitor_memory(duration, interval)
    else:
        print("Memory Monitor for ZoL0 Trading System")
        print("\nUsage:")
        print(
            "  python memory_monitor.py quick                    - Quick memory check"
        )
        print(
            "  python memory_monitor.py monitor [min] [interval] - Monitor for [min] minutes"
        )
        print("\nExamples:")
        print("  python memory_monitor.py quick")
        print(
            "  python memory_monitor.py monitor 15 60    # Monitor for 15 min, check every 60s"
        )
        print(
            "  python memory_monitor.py monitor 5 30     # Monitor for 5 min, check every 30s"
        )
