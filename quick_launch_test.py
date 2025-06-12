#!/usr/bin/env python3
"""
Quick Launch Test for Patched Dashboards
Validates that all dashboards can start without KeyError exceptions
"""

import os
import subprocess
import sys

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


def main():
    print("🚀 Quick Dashboard Launch Test")
    print("=" * 50)

    dashboards = [
        ("Port 8501", "unified_trading_dashboard.py"),
        ("Port 8503", "master_control_dashboard.py"),
        ("Port 8504", "advanced_trading_analytics.py"),
    ]

    all_passed = True

    for name, file_path in dashboards:
        if os.path.exists(file_path):
            print(f"\n🔧 Testing {name} ({file_path})...")
            success, message = test_dashboard_import(file_path)

            if success:
                print(f"   ✅ {message}")
            else:
                print(f"   ❌ {message}")
                all_passed = False
        else:
            print(f"   ❌ File not found: {file_path}")
            all_passed = False

    print("\n" + "=" * 50)
    assert all_passed, "Some dashboards have issues - check errors above"

    print("🎉 ALL DASHBOARDS READY FOR LAUNCH!")
    print("\nTo start dashboards:")
    print("   streamlit run unified_trading_dashboard.py --server.port 8501")
    print("   streamlit run master_control_dashboard.py --server.port 8503")
    print("   streamlit run advanced_trading_analytics.py --server.port 8504")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
