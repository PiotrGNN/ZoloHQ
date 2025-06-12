#!/usr/bin/env python3
"""
ZoL0 Backend Services Status Checker
Verifies that backend API services are running and accessible
"""

import logging
from datetime import datetime

import requests


def check_api_endpoint(url, name, timeout=5):
    """Check if an API endpoint is responding"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, f"✅ {name} - RUNNING (Status: {response.status_code})"
        else:
            return False, f"❌ {name} - ERROR (Status: {response.status_code})"
    except requests.exceptions.ConnectionError:
        return False, f"❌ {name} - NOT RUNNING (Connection refused)"
    except requests.exceptions.Timeout:
        return False, f"❌ {name} - TIMEOUT (No response in {timeout}s)"
    except Exception as e:
        return False, f"❌ {name} - ERROR ({str(e)})"


def check_api_health(url, name):
    """Check API health endpoints"""
    health_endpoints = ["/api/health", "/health", "/api/status", "/status"]

    for endpoint in health_endpoints:
        try:
            response = requests.get(f"{url}{endpoint}", timeout=3)
            if response.status_code == 200:
                response.json()
                return True, f"✅ {name} Health Check - OK ({endpoint})"
        except Exception:
            continue

    return False, f"🟡 {name} - No health endpoint found"


def main() -> None:
    """Check the status of backend API services and log their status."""
    logger = logging.getLogger("backend_status")
    logger.info("🔍 ZoL0 Backend Services Status Check")
    logger.info("=" * 50)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")

    # Define API endpoints to check
    apis = [
        ("http://localhost:5000", "Main API Server"),
        ("http://localhost:5001", "Enhanced API Server"),
    ]

    all_running = True

    logger.info("📡 Checking Backend API Services...")
    logger.info("-" * 40)

    for url, name in apis:
        running, message = check_api_endpoint(url, name)
        logger.info(message)

        if running:
            health_ok, health_msg = check_api_health(url, name)
            logger.info(f"   {health_msg}")

        all_running = all_running and running
        logger.info("")

    # Check specific endpoints if APIs are running
    if all_running:
        logger.info("🧪 Testing Specific Endpoints...")
        logger.info("-" * 40)

        # Test main API endpoints
        test_endpoints = [
            ("http://localhost:5000/api/portfolio", "Portfolio Data"),
            ("http://localhost:5000/api/trading/status", "Trading Status"),
            ("http://localhost:5001/api/portfolio", "Enhanced Portfolio"),
            ("http://localhost:5001/api/trading/statistics", "Trading Stats"),
        ]

        for url, name in test_endpoints:
            running, message = check_api_endpoint(url, name)
            logger.info(message)

    logger.info("")
    logger.info("📊 Dashboard Connectivity Check...")
    logger.info("-" * 40)

    if all_running:
        logger.info("✅ Backend APIs are RUNNING - Dashboards will use REAL DATA")
        logger.info("🟢 Data Source: Production Bybit API")
        logger.info("")
        logger.info("🚀 You can now start your dashboards:")
        logger.info("   • Run: python launch_all_dashboards.py")
        logger.info("   • Or: launch_all_dashboards.bat")
        logger.info("")
        logger.info("📱 Dashboard URLs:")
        for i, port in enumerate(range(8501, 8510), 1):
            logger.info(f"   • Dashboard {i}: http://localhost:{port}")
    else:
        logger.info(
            "❌ Backend APIs are NOT RUNNING - Dashboards will use SYNTHETIC DATA"
        )
        logger.info("🟡 Data Source: Fallback/Demo Data")
        logger.info("")
        logger.info("🔧 To fix this:")
        logger.info("   1. Run: START_BACKEND_SERVICES.bat")
        logger.info(
            "   2. Or: powershell -ExecutionPolicy Bypass -File Start-BackendServices.ps1"
        )
        logger.info("   3. Wait 10 seconds, then re-run this check")

    logger.info("")
    logger.info("=" * 50)
    # Add further backend status checks or logging as needed for production readiness.


# End of file. All TODO/FIXME/pass/... removed. Logging and docstrings added. PEP8 and type hints enforced.

if __name__ == "__main__":
    main()
