#!/usr/bin/env python3
"""
System Restart Script - ZoL0 Trading System
===========================================
This script restarts all ZoL0 services with the fixed Master Control Dashboard
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import List


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("zol0_launcher")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    return logger


def run_command(command: str, cwd: Path = None) -> subprocess.Popen:
    """Run a command in a non-blocking subprocess window."""
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logging.info(f"Started process: {command} (cwd={cwd})")
        return process
    except Exception as e:
        logging.error(f"Failed to start process: {command} (cwd={cwd}) - {e}")
        return None


def start_process(
    command: List[str], cwd: Path, logger: logging.Logger, name: str
) -> subprocess.Popen:
    try:
        proc = subprocess.Popen(
            command, cwd=str(cwd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        logger.info(f"Started {name}: {' '.join(command)} (PID: {proc.pid})")
        return proc
    except Exception as e:
        logger.error(f"Failed to start {name}: {e}")
        return None


def main():
    logger = setup_logger()
    logger.info("🚀 Starting ZoL0 Trading System with Master Control Dashboard")
    logger.info("=" * 70)

    base_dir = Path("C:/Users/piotr/Desktop/Zol0")
    zol0_master_dir = base_dir / "ZoL0-master"

    # Step 1: Start API servers
    logger.info("\n📡 Starting API Servers...")
    api_processes = []
    api_processes.append(
        start_process(
            ["python", "dashboard_api.py"],
            zol0_master_dir,
            logger,
            "Main API Server (port 5000)",
        )
    )
    time.sleep(3)
    api_processes.append(
        start_process(
            ["python", "enhanced_dashboard_api.py"],
            base_dir,
            logger,
            "Enhanced API Server (port 5001)",
        )
    )
    time.sleep(3)

    # Step 2: Start Dashboards
    logger.info("\n🖥️  Starting Dashboards...")
    dashboards = [
        ("Master Control Dashboard", "master_control_dashboard.py", 8501),
        ("Unified Trading Dashboard", "unified_trading_dashboard.py", 8502),
        ("Enhanced Bot Monitor", "enhanced_bot_monitor.py", 8503),
        ("Advanced Trading Analytics", "advanced_trading_analytics.py", 8504),
        ("Notification Dashboard", "notification_dashboard.py", 8505),
        ("Advanced Alert Management", "advanced_alert_management.py", 8506),
        ("Portfolio Optimization", "portfolio_optimization.py", 8507),
        ("ML Predictive Analytics", "ml_predictive_analytics.py", 8508),
        ("Enhanced Dashboard", "enhanced_dashboard.py", 8509),
    ]
    dashboard_processes = []
    for name, script, port in dashboards:
        dashboard_path = base_dir / script
        if not dashboard_path.exists():
            logger.warning(f"Dashboard script not found: {dashboard_path}")
            continue
        dashboard_processes.append(
            start_process(
                ["streamlit", "run", script, "--server.port", str(port)],
                base_dir,
                logger,
                f"{name} (port {port})",
            )
        )
        time.sleep(2)
        logger.info(f"🌐 {name}: http://localhost:{port}")

    logger.info("\nAll APIs and dashboards have been started in the background.")
    logger.info("Endpoints:")
    logger.info("  • Main API:      http://localhost:5000")
    logger.info("  • Enhanced API:  http://localhost:5001")
    for name, _, port in dashboards:
        logger.info(f"  • {name}: http://localhost:{port}")
    logger.info(
        "\nPress Ctrl+C to stop this launcher. Closing this window will NOT stop the background services."
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping all services...")
        for proc in api_processes + dashboard_processes:
            if proc:
                proc.terminate()
        logger.info("✅ All services stopped.")


if __name__ == "__main__":
    main()
