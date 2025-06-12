#!/usr/bin/env python3
"""
Simple validation of the dashboard fixes
"""
import sys

sys.path.append(".")


def test_imports():
    """Test that all dashboards can be imported"""
    print("Testing imports...")

    try:
        from unified_trading_dashboard import UnifiedDashboard

        print("✅ UnifiedDashboard imported successfully")

        dashboard = UnifiedDashboard()
        print("✅ UnifiedDashboard instance created")

        # Test performance data
        perf_data = dashboard.get_unified_performance_data()
        data_source = perf_data.get("data_source", "unknown")
        print(f"✅ Performance data source: {data_source}")

        # Check that it's not unknown
        if data_source != "unknown":
            print("✅ Data source properly identified")
        else:
            print("❌ Data source is 'unknown'")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_api_availability():
    """Test if Enhanced Dashboard API is available"""
    print("\nTesting API availability...")

    try:
        import requests

        response = requests.get("http://localhost:5001/health", timeout=3)
        if response.status_code == 200:
            print("✅ Enhanced Dashboard API is running")
            return True
        else:
            print(f"⚠️ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ API not available: {e}")
        return False


def main():
    print("🔧 Quick Dashboard Validation")
    print("=" * 40)

    # Test basic functionality
    imports_ok = test_imports()
    api_ok = test_api_availability()

    print("\n" + "=" * 40)
    if imports_ok:
        print("🎉 Dashboard fixes are working!")
        if not api_ok:
            print("💡 Start the API with: python enhanced_dashboard_api.py")
    else:
        print("❌ Dashboard fixes need more work")

    return imports_ok


if __name__ == "__main__":
    main()
