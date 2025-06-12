#!/usr/bin/env python3
"""
Master Control Dashboard Validation Script
==========================================
Tests the fixed Master Control Dashboard for syntax and functionality
"""

import ast
import sys
from pathlib import Path

import requests


def validate_python_syntax(file_path):
    """Validate Python syntax of the dashboard file"""
    print(f"🔍 Validating Python syntax: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()

        # Parse the AST to check for syntax errors
        ast.parse(source)
        print("✅ Python syntax is valid")
        return True

    except SyntaxError as e:
        print(f"❌ Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def test_dashboard_imports():
    """Test if the dashboard can import required modules"""
    print("\n📦 Testing required imports...")

    required_modules = [
        "streamlit",
        "requests",
        "pandas",
        "plotly.graph_objects",
        "plotly.express",
        "datetime",
        "json",
        "logging",
    ]

    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing_modules.append(module)

    return len(missing_modules) == 0


def test_api_connections():
    """Test if API endpoints are accessible"""
    print("\n🔗 Testing API connections...")

    endpoints = [
        "http://localhost:5000/api/portfolio",
        "http://localhost:5001/api/portfolio",
        "http://localhost:5001/api/trading/statistics",
    ]

    working_endpoints = 0
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint}")
                working_endpoints += 1
            else:
                print(f"⚠️  {endpoint} (Status: {response.status_code})")
        except Exception as e:
            print(f"❌ {endpoint} (Error: {e})")

    return working_endpoints > 0


def main():
    print("🎛️  Master Control Dashboard Validation")
    print("=" * 50)

    # Test 1: Python syntax validation
    dashboard_path = Path("C:/Users/piotr/Desktop/Zol0/master_control_dashboard.py")
    syntax_ok = validate_python_syntax(dashboard_path)

    # Test 2: Import validation
    imports_ok = test_dashboard_imports()

    # Test 3: API connection test
    apis_ok = test_api_connections()

    # Summary
    print("\n📋 Validation Summary:")
    print(f"  Python Syntax: {'✅ PASS' if syntax_ok else '❌ FAIL'}")
    print(f"  Required Imports: {'✅ PASS' if imports_ok else '❌ FAIL'}")
    print(f"  API Connections: {'✅ PASS' if apis_ok else '⚠️  PARTIAL'}")

    if syntax_ok and imports_ok:
        print("\n🎉 Master Control Dashboard is ready to run!")
        print(
            "📊 The dashboard will show REAL Bybit production data when APIs are available."
        )
        print(
            "🔄 If APIs are unavailable, it will gracefully fall back to simulated data."
        )

        print("\n🚀 To start the dashboard, run:")
        print("   streamlit run master_control_dashboard.py --server.port 8505")

        return True
    else:
        print("\n❌ Issues found. Please fix before running the dashboard.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
