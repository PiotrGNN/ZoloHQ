#!/usr/bin/env python3
"""
Simple Dashboard Launcher
Launches individual dashboards one at a time to test real data integration
"""

import os
import subprocess
import sys


def launch_dashboard(dashboard_file, port):
    """Launch a single dashboard"""
    print(f"\n🚀 Launching {dashboard_file} on port {port}...")
    print(f"📊 Dashboard will be available at: http://localhost:{port}")
    print("💡 Press Ctrl+C to stop this dashboard and return to menu")

    try:
        # Change to the correct directory
        os.chdir(r"C:\Users\piotr\Desktop\Zol0")

        # Launch streamlit
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            dashboard_file,
            "--server.port",
            str(port),
        ]
        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print(f"\n✅ Dashboard {dashboard_file} stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching {dashboard_file}: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def main():
    print("🔥 ZoL0 Trading System Dashboard Launcher")
    print("=" * 50)

    dashboards = [
        ("Master Control Dashboard", "master_control_dashboard.py", 8501),
        ("Enhanced Bot Monitor", "enhanced_bot_monitor.py", 8502),
        ("Unified Trading Dashboard", "unified_trading_dashboard.py", 8503),
        ("Enhanced Dashboard", "enhanced_dashboard.py", 8504),
        ("Advanced Trading Analytics", "advanced_trading_analytics.py", 8505),
        ("Portfolio Optimization", "portfolio_optimization.py", 8506),
        ("Advanced Alert Management", "advanced_alert_management.py", 8507),
        ("ML Predictive Analytics", "ml_predictive_analytics.py", 8508),
        ("Notification Dashboard", "notification_dashboard.py", 8509),
    ]

    while True:
        print("\n📊 Available Dashboards:")
        print("-" * 30)

        for i, (name, file, port) in enumerate(dashboards, 1):
            print(f"{i}. {name} (Port {port})")

        print("0. Exit")

        try:
            choice = input("\n🎯 Select dashboard to launch (0-9): ").strip()

            if choice == "0":
                print("👋 Goodbye!")
                break

            choice_num = int(choice)
            if 1 <= choice_num <= len(dashboards):
                name, file, port = dashboards[choice_num - 1]
                launch_dashboard(file, port)
            else:
                print("❌ Invalid choice. Please select 0-9.")

        except ValueError:
            print("❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main()
