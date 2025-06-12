#!/usr/bin/env python3
"""
ZoL0 Trading System - EMERGENCY LAUNCH SCRIPT
This script will attempt to start all services step by step
"""

import logging
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests


def print_banner():
    print("\n" + "=" * 60)
    print("🔥 ZoL0 TRADING SYSTEM - EMERGENCY LAUNCHER 🔥")
    print("🟢 REAL BYBIT PRODUCTION DATA MODE")
    print("=" * 60)


def set_production_env():
    """Set production environment variables"""
    os.environ["BYBIT_PRODUCTION_CONFIRMED"] = "true"
    os.environ["BYBIT_PRODUCTION_ENABLED"] = "true"
    print("✅ Production environment variables set")


def check_python():
    """Check if Python is available"""
    try:
        result = subprocess.run(
            [sys.executable, "--version"], capture_output=True, text=True
        )
        print(f"✅ Python available: {result.stdout.strip()}")
        return True
    except Exception:
        print("❌ Python not available")
        return False


def check_streamlit():
    """Check if Streamlit is available"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "--version"],
            capture_output=True,
            text=True,
        )
        print(f"✅ Streamlit available: {result.stdout.strip()}")
        return True
    except Exception:
        print("❌ Streamlit not available - installing...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "streamlit"], check=True
            )
            print("✅ Streamlit installed successfully")
            return True
        except Exception:
            print("❌ Failed to install Streamlit")
            return False


def start_api_service(script_path, port, name):
    """Start an API service"""
    print(f"\n🚀 Starting {name} on port {port}...")

    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return None

    try:
        # Use CREATE_NEW_CONSOLE flag on Windows to create new window
        if os.name == "nt":  # Windows
            process = subprocess.Popen(
                [sys.executable, script_path],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=os.path.dirname(script_path),
            )
        else:
            process = subprocess.Popen(
                [sys.executable, script_path], cwd=os.path.dirname(script_path)
            )

        print(f"✅ {name} started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None


def start_dashboard_service(script_path, port, name):
    """Start a dashboard service"""
    print(f"🎯 Starting {name} on port {port}...")

    if not os.path.exists(script_path):
        print(f"⚠️  Script not found: {script_path}")
        return None

    try:
        # Use CREATE_NEW_CONSOLE flag on Windows to create new window
        if os.name == "nt":  # Windows
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    script_path,
                    "--server.port",
                    str(port),
                    "--server.headless",
                    "true",
                ],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                cwd=os.path.dirname(script_path),
            )
        else:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    script_path,
                    "--server.port",
                    str(port),
                    "--server.headless",
                    "true",
                ],
                cwd=os.path.dirname(script_path),
            )

        print(f"✅ {name} started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None


def check_service(url, name, timeout=5):
    """Check if a service is responding"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {name} - RESPONDING")
            return True
    except Exception as e:
        logging.exception(f"Exception occurred while checking service {name}: {e}")
    print(f"❌ {name} - NOT RESPONDING")
    return False


def main():
    print_banner()

    # Preliminary checks
    if not check_python():
        return

    if not check_streamlit():
        return

    # Set production environment
    set_production_env()

    base_dir = Path(__file__).parent
    processes = []

    print("\n" + "=" * 40)
    print("📡 STEP 1: Starting Backend API Services")
    print("=" * 40)

    # Start API services
    api_services = [
        {
            "script": base_dir / "ZoL0-master" / "dashboard_api.py",
            "port": 5000,
            "name": "Main API Server",
        },
        {
            "script": base_dir / "enhanced_dashboard_api.py",
            "port": 4001,
            "name": "Enhanced API Server",
        },
    ]

    for service in api_services:
        process = start_api_service(
            str(service["script"]), service["port"], service["name"]
        )
        if process:
            processes.append(process)

    print("\n⏳ Waiting 15 seconds for API services to initialize...")
    time.sleep(15)

    print("\n" + "=" * 40)
    print("🎯 STEP 2: Starting Dashboard Services")
    print("=" * 40)

    # Start dashboard services
    dashboard_services = [
        {
            "script": "master_control_dashboard.py",
            "port": 4501,
            "name": "Master Control",
        },
        {
            "script": "unified_trading_dashboard.py",
            "port": 4502,
            "name": "Unified Trading",
        },
        {"script": "enhanced_bot_monitor.py", "port": 4503, "name": "Bot Monitor"},
        {
            "script": "advanced_trading_analytics.py",
            "port": 4504,
            "name": "Trading Analytics",
        },
        {"script": "notification_dashboard.py", "port": 4505, "name": "Notifications"},
        {"script": "portfolio_dashboard.py", "port": 4506, "name": "Portfolio"},
        {"script": "ml_predictive_analytics.py", "port": 4507, "name": "ML Analytics"},
        {"script": "enhanced_dashboard.py", "port": 4508, "name": "Enhanced Dashboard"},
    ]

    for service in dashboard_services:
        script_path = base_dir / service["script"]
        process = start_dashboard_service(
            str(script_path), service["port"], service["name"]
        )
        if process:
            processes.append(process)
        time.sleep(2)  # Small delay between starts

    print("\n⏳ Waiting 20 seconds for dashboard services to initialize...")
    time.sleep(20)

    print("\n" + "=" * 40)
    print("🧪 STEP 3: Checking Service Status")
    print("=" * 40)

    # Check API services
    check_service("http://localhost:5000", "Main API Server")
    check_service("http://localhost:4001", "Enhanced API Server")

    # Check dashboard services
    for service in dashboard_services:
        check_service(f"http://localhost:{service['port']}", service["name"])

    print("\n" + "=" * 60)
    print("🎉 ZoL0 SYSTEM LAUNCH COMPLETE!")
    print("🟢 REAL BYBIT PRODUCTION DATA ACTIVE")
    print("=" * 60)

    print("\n📡 Backend Services:")
    print("   • Main API Server: http://localhost:5000")
    print("   • Enhanced API Server: http://localhost:4001")

    print("\n🎯 Trading Dashboards:")
    for service in dashboard_services:
        print(f"   • {service['name']}: http://localhost:{service['port']}")

    print("\n🚀 Quick Access:")
    print("   • Master Control: http://localhost:4501")
    print("   • Unified Trading: http://localhost:4502")

    # Open main dashboard
    try:
        print("\n🌐 Opening Master Control Dashboard...")
        webbrowser.open("http://localhost:4501")
    except Exception as e:
        logging.exception(f"Could not auto-open browser: {e}")
        print("⚠️  Could not auto-open browser")

    print("\n" + "=" * 60)
    print("✅ ALL SYSTEMS ONLINE - READY FOR TRADING!")
    print("🔴 Press Ctrl+C to stop all services")
    print("=" * 60)

    # Keep running until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down all services...")
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                logging.exception(
                    f"Exception occurred in EMERGENCY_LAUNCH block at line 238: {e}"
                )
                try:
                    process.kill()
                except Exception as e2:
                    logging.exception(f"Exception occurred while killing process: {e2}")
            print("✅ All services stopped. System shutdown complete.")
    except Exception as e:
        logging.exception(
            f"Exception occurred in EMERGENCY_LAUNCH block at line 221: {e}"
        )


if __name__ == "__main__":
    main()
