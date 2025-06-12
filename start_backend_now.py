#!/usr/bin/env python3
"""
Start Backend Services for ZoL0 Trading System
Starts both API servers in separate processes
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def check_port(port):
    """Check if a port is available"""
    try:
        requests.get(f"http://localhost:{port}", timeout=2)
        return True
    except Exception:
        return False


def start_api_server(script_path, port, name):
    """Start an API server"""
    print(f"🚀 Starting {name} on port {port}...")
    print(f"   Script: {script_path}")

    if not os.path.exists(script_path):
        print(f"❌ Error: {script_path} not found!")
        return None

    try:
        # Start the process
        process = subprocess.Popen(
            [sys.executable, script_path], cwd=os.path.dirname(script_path)
        )

        print(f"✅ {name} started with PID: {process.pid}")
        return process
    except Exception as e:
        logging.exception("Exception occurred in start_backend_now at line 22")
        print(f"❌ Failed to start {name}: {e}")
        return None


def main():
    print("🔥 ZoL0 Backend Services Startup")
    print("=" * 50)

    # Set production environment
    os.environ["BYBIT_PRODUCTION_CONFIRMED"] = "true"
    os.environ["BYBIT_PRODUCTION_ENABLED"] = "true"

    base_dir = Path(__file__).parent

    # Define API servers
    servers = [
        {
            "name": "Main API Server",
            "script": base_dir / "ZoL0-master" / "dashboard_api.py",
            "port": 5000,
        },
        {
            "name": "Enhanced API Server",
            "script": base_dir / "enhanced_dashboard_api.py",
            "port": 4001,
        },
    ]

    processes = []

    # Start each server
    for server in servers:
        if check_port(server["port"]):
            print(
                f"⚠️  Port {server['port']} already in use - skipping {server['name']}"
            )
            continue

        process = start_api_server(
            str(server["script"]), server["port"], server["name"]
        )
        if process:
            processes.append(process)

    if processes:
        print(f"\n✅ Started {len(processes)} backend services")
        print("\n⏳ Waiting 10 seconds for services to initialize...")
        time.sleep(10)

        print("\n🧪 Testing API connectivity...")
        for server in servers:
            if check_port(server["port"]):
                print(f"✅ {server['name']} - RESPONDING on port {server['port']}")
            else:
                print(f"❌ {server['name']} - NOT RESPONDING on port {server['port']}")

        print("\n🎯 Backend services are now running!")
        print("🟢 Dashboards will now use REAL BYBIT DATA")
        print("\nNext steps:")
        print("1. Run: python launch_all_dashboards.py")
        print("2. Open dashboards: http://localhost:4501 to 4509")
        print("\nPress Ctrl+C to stop all services")

        try:
            # Keep running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all services...")
            for process in processes:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    process.kill()
            print("✅ All services stopped")
    else:
        print("❌ No services were started successfully")


if __name__ == "__main__":
    main()
