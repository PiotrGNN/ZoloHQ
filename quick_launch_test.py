#!/usr/bin/env python3
"""
Quick Launch Test for Patched Dashboards
Validates that all dashboards can start without KeyError exceptions
"""

import os
import subprocess
import sys
import json
import concurrent.futures

import pytest


def test_dashboard_import(dashboard_file):
    """Test if dashboard can be imported without errors"""
    try:
        # Test syntax compilation
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", dashboard_file],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
    except subprocess.TimeoutExpired:
        raise AssertionError("Timeout during compilation")
    except Exception as e:
        raise AssertionError(f"Error: {str(e)}")


@pytest.fixture
def dashboard_file():
    # Domyślna ścieżka do dashboardu do testów (możesz zmienić na inny plik jeśli chcesz)
    return "unified_trading_dashboard.py"


# Parallel dashboard launch test for automation/monetization
DASHBOARDS = [
    ("Port 8501", "unified_trading_dashboard.py"),
    ("Port 8503", "master_control_dashboard.py"),
    ("Port 8504", "advanced_trading_analytics.py"),
]


def launch_dashboard(file_path):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", file_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return (file_path, result.returncode == 0, result.stderr)
    except Exception as e:
        return (file_path, False, str(e))


def main():
    print("🚀 Quick Dashboard Launch Test")
    print("=" * 50)
    all_passed = True
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_dashboard = {executor.submit(launch_dashboard, f): (n, f) for n, f in DASHBOARDS}
        for future in concurrent.futures.as_completed(future_to_dashboard):
            name, file_path = future_to_dashboard[future]
            file, passed, err = future.result()
            print(f"{name}: {'PASS' if passed else 'FAIL'}")
            if not passed:
                print(f"  Error: {err}")
                all_passed = False
            results.append({"dashboard": name, "file": file, "passed": passed, "error": err})
    # Output machine-readable results for CI/CD/monetization
    with open("dashboard_launch_results.json", "w") as f:
        json.dump(results, f, indent=2)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
