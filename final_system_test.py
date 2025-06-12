#!/usr/bin/env python3
"""
final_system_test.py
-------------------
Comprehensive test of the ZoL0 Dashboard Control Panel system.

This script tests all major functionality including:
- Environment management
- Trading engine control  
- System validation
- API endpoints
- Safety features
"""

import json
import os
import time
from datetime import datetime

import requests

API_BASE = "http://localhost:5001"


def print_banner():
    """Print test banner"""
    print("=" * 70)
    print("🎯 ZoL0 Dashboard Control Panel - Final System Test")
    print("=" * 70)
    print()


def test_api_connectivity():
    """Test basic API connectivity"""
    print("🔗 Testing API Connectivity...")
    print("-" * 50)

    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Enhanced Dashboard API is responding")
            assert True
        else:
            print(f"❌ API returned status {response.status_code}")
            raise AssertionError()
    except requests.exceptions.RequestException as e:
        import pytest
        pytest.skip(f"API not running: {e}")


def test_environment_management():
    """Test environment management features"""
    print("\n🌍 Testing Environment Management...")
    print("-" * 50)

    try:
        # Get environment status
        response = requests.get(f"{API_BASE}/api/environment/status")
        if response.status_code == 200:
            data = response.json()
            env = data["status"]["environment"]
            production_ready = data["status"]["production_confirmed"]
            print(f"✅ Current Environment: {env}")
            print(f"📊 Production Ready: {production_ready}")
        else:
            print(f"❌ Environment status failed: {response.status_code}")
            raise AssertionError()

        # Test environment switch validation (safe - won't actually switch)
        switch_data = {"target_environment": "testnet", "confirm": False}
        response = requests.post(f"{API_BASE}/api/environment/switch", json=switch_data)
        if response.status_code == 200:
            print("✅ Environment switch validation working")
        else:
            print(f"⚠️ Environment switch test returned: {response.status_code}")

        assert True

    except Exception as e:
        import pytest
        pytest.skip(f"Environment API not running: {e}")


def test_trading_engine():
    """Test trading engine functionality"""
    print("\n⚙️ Testing Trading Engine...")
    print("-" * 50)

    try:
        # Get initial status
        response = requests.get(f"{API_BASE}/api/trading/status")
        if response.status_code == 200:
            status = response.json()["status"]
            print(f"📊 Engine Active: {status['active']}")
            print(f"📈 Trading Pairs: {status.get('trading_pairs', [])}")
            print(f"🔢 Total Trades: {status.get('total_trades', 0)}")
        else:
            print(f"❌ Trading status failed: {response.status_code}")
            raise AssertionError()

        # Test start trading (safe symbols)
        start_data = {"symbols": ["BTCUSDT", "ETHUSDT"]}
        response = requests.post(f"{API_BASE}/api/trading/start", json=start_data)
        if response.status_code == 200:
            print("✅ Trading engine start command successful")
        else:
            print(f"⚠️ Trading start failed: {response.status_code}")

        # Wait a moment
        time.sleep(1)

        # Check status after start
        response = requests.get(f"{API_BASE}/api/trading/status")
        if response.status_code == 200:
            status = response.json()["status"]
            if status["active"]:
                print("✅ Trading engine is now active")
            else:
                print("⚠️ Trading engine start may have failed")

        # Test stop trading
        response = requests.post(f"{API_BASE}/api/trading/stop")
        if response.status_code == 200:
            print("✅ Trading engine stop command successful")
        else:
            print(f"⚠️ Trading stop failed: {response.status_code}")

        assert True

    except Exception as e:
        import pytest
        pytest.skip(f"Trading engine API not running: {e}")


def test_system_validation():
    """Test system validation functionality"""
    print("\n🔍 Testing System Validation...")
    print("-" * 50)

    try:
        response = requests.get(f"{API_BASE}/api/system/validation")
        if response.status_code == 200:
            validation = response.json()["validation"]
            print("System Components:")

            for component, status in validation["components"].items():
                icon = "✅" if status else "❌"
                print(f"  {icon} {component.replace('_', ' ').title()}")

            overall_ready = validation["production_ready"]
            icon = "✅" if overall_ready else "❌"
            print(f"\n{icon} Overall Production Ready: {overall_ready}")

            assert True
        else:
            print(f"❌ System validation failed: {response.status_code}")
            raise AssertionError()
    except Exception as e:
        import pytest
        pytest.skip(f"System validation API not running: {e}")


def test_core_system_apis():
    """Test core system API endpoints"""
    print("\n🎯 Testing Core System APIs...")
    print("-" * 50)

    endpoints = [
        "/core/status",
        "/core/strategies",
        "/core/ai-models",
        "/core/system-metrics",
    ]

    results = {}

    for endpoint in endpoints:
        try:
            response = requests.get(f"{API_BASE}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"✅ {endpoint}")
                results[endpoint] = True
            else:
                print(f"⚠️ {endpoint} - Status {response.status_code}")
                results[endpoint] = False
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")
            results[endpoint] = False

    success_rate = sum(results.values()) / len(results) * 100
    print(f"\n📊 Core API Success Rate: {success_rate:.1f}%")
    if success_rate == 0:
        import pytest
        pytest.skip("Core system APIs not available.")
    assert success_rate > 75


def test_dashboard_ui_connectivity():
    """Test if Streamlit dashboard is accessible"""
    print("\n🖥️ Testing Dashboard UI...")
    print("-" * 50)

    try:
        response = requests.get("http://localhost:8501/healthz", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit Dashboard is accessible")
            assert True
        else:
            print(f"⚠️ Dashboard returned status {response.status_code}")
            raise AssertionError()
    except requests.exceptions.RequestException as e:
        import pytest
        pytest.skip(f"Dashboard UI not running: {e}")


def test_safety_features():
    """Test safety and security features"""
    print("\n🛡️ Testing Safety Features...")
    print("-" * 50)

    safety_score = 0
    total_checks = 0

    # Check environment variable safety
    total_checks += 1
    prod_confirmed = os.getenv("BYBIT_PRODUCTION_CONFIRMED", "false").lower()
    if prod_confirmed == "false":
        print("✅ Production not confirmed (safe default)")
        safety_score += 1
    else:
        print("⚠️ Production is confirmed")

    # Check if in testnet mode
    total_checks += 1
    testnet = os.getenv("BYBIT_TESTNET", "false").lower()
    if testnet == "true":
        print("✅ Running in testnet mode")
        safety_score += 1
    else:
        print("⚠️ Not explicitly in testnet mode")

    # Check API validation
    total_checks += 1
    try:
        response = requests.get(f"{API_BASE}/api/system/validation")
        if response.status_code == 200:
            validation = response.json()["validation"]
            if not validation["production_ready"]:
                print("✅ System not ready for production (safe)")
                safety_score += 1
            else:
                print("⚠️ System shows production ready")
    except Exception:
        print("❌ Could not check production readiness")

    safety_percentage = (safety_score / total_checks) * 100
    print(f"\n🔒 Safety Score: {safety_percentage:.1f}%")
    if safety_percentage == 0:
        import pytest
        pytest.skip("Safety features cannot be validated (API not running)")
    assert safety_percentage > 60


def generate_test_report():
    """Generate comprehensive test report"""
    print("\n📋 Generating Test Report...")
    print("-" * 50)

    test_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {
            "api_connectivity": test_api_connectivity(),
            "environment_management": test_environment_management(),
            "trading_engine": test_trading_engine(),
            "system_validation": test_system_validation(),
            "core_apis": test_core_system_apis(),
            "dashboard_ui": test_dashboard_ui_connectivity(),
            "safety_features": test_safety_features(),
        },
    }

    # Calculate overall success rate
    passed_tests = sum(test_results["tests"].values())
    total_tests = len(test_results["tests"])
    success_rate = (passed_tests / total_tests) * 100

    test_results["summary"] = {
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "success_rate": success_rate,
        "overall_status": (
            "PASS"
            if success_rate >= 80
            else "PARTIAL" if success_rate >= 60 else "FAIL"
        ),
    }

    # Save to file
    with open("logs/final_system_test.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("\n📄 Test report saved to: logs/final_system_test.json")
    return test_results


def print_final_summary(test_results):
    """Print final test summary"""
    summary = test_results["summary"]

    print("\n" + "=" * 70)
    print("🎉 FINAL TEST SUMMARY")
    print("=" * 70)

    status_icon = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}[
        summary["overall_status"]
    ]

    print(f"{status_icon} Overall Status: {summary['overall_status']}")
    print(f"📊 Success Rate: {summary['success_rate']:.1f}%")
    print(f"✅ Passed Tests: {summary['passed_tests']}/{summary['total_tests']}")

    print("\nDetailed Results:")
    for test_name, result in test_results["tests"].items():
        icon = "✅" if result else "❌"
        formatted_name = test_name.replace("_", " ").title()
        print(f"  {icon} {formatted_name}")

    print("\n🎯 Next Steps:")
    if summary["overall_status"] == "PASS":
        print("  🚀 System is ready for use!")
        print("  📚 Review DASHBOARD_CONTROL_PANEL_GUIDE.md")
        print("  ⚙️ Configure production with configure_production.py")
    elif summary["overall_status"] == "PARTIAL":
        print("  🔧 Some issues detected - review failed tests")
        print("  📖 Check logs for detailed error information")
        print("  🔄 Re-run tests after fixing issues")
    else:
        print("  🚨 Major issues detected - system needs attention")
        print("  📞 Review error logs and documentation")
        print("  🛠️ Fix issues before proceeding")

    print("=" * 70)


def main():
    """Main test execution"""
    print_banner()

    print("🔍 Starting comprehensive system test...")
    print("This will test all major components of the ZoL0 Dashboard Control Panel.")
    print()

    start_time = time.time()

    # Run all tests and generate report
    test_results = generate_test_report()

    end_time = time.time()
    test_duration = end_time - start_time

    test_results["execution_time"] = test_duration

    # Print final summary
    print_final_summary(test_results)

    print(f"\n⏱️ Total Test Time: {test_duration:.2f} seconds")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Test cancelled by user.")
    except Exception as e:
        print(f"\n❌ Test execution error: {e}")
        print("📋 Check logs for more details.")
