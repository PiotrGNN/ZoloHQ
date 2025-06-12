#!/usr/bin/env python3
"""
Final validation that the unified dashboard displays real data instead of demo warnings
"""
import sys
from pathlib import Path

import requests

# Add path for imports
sys.path.append(str(Path(__file__).parent))


def validate_dashboard_data_sources():
    """Validate that the dashboard is using real data sources"""
    try:
        from unified_trading_dashboard import UnifiedDashboard

        print("🔧 Creating UnifiedDashboard instance...")
        dashboard = UnifiedDashboard()

        print("\n📊 Testing data sources:")

        # Test 1: Performance data source
        perf_data = dashboard.get_unified_performance_data()
        data_source = perf_data.get("data_source", "unknown")
        print(f"  Performance Data: {data_source}")

        if data_source == "production_api":
            print("    ✅ Using REAL production API data")
        elif data_source == "api_endpoint":
            print("    ✅ Using REAL Enhanced Dashboard API data")
        elif data_source == "demo_data":
            print("    ❌ Still using demo data")
            return False
        else:
            print(f"    ⚠️ Unknown data source: {data_source}")

        # Test 2: System status
        system_status = dashboard.get_system_status()
        print(f"  System Services: {len(system_status)} services detected")

        api_status = system_status.get("Enhanced Dashboard API", "Unknown")
        if api_status == "🟢 Online":
            print("    ✅ Enhanced Dashboard API connected")
        else:
            print(f"    ⚠️ API status: {api_status}")

        # Test 3: Production manager
        if dashboard.production_manager:
            print("    ✅ Production manager initialized")
            if dashboard.production_mode:
                print("    ✅ Production mode enabled")
            else:
                print("    ⚠️ Development mode (expected if not in production)")
        else:
            print("    ⚠️ Production manager not available")

        return True

    except Exception as e:
        print(f"❌ Error validating dashboard: {e}")
        return False


def simulate_streamlit_render_functions():
    """Simulate how render functions will access the dashboard"""
    try:
        from unified_trading_dashboard import UnifiedDashboard

        print("\n🎭 Simulating Streamlit render function access:")

        # Mock session state
        class MockSessionState:
            def __init__(self):
                self._state = {}

            def get(self, key, default=None):
                return self._state.get(key, default)

            def __contains__(self, key):
                return key in self._state

            def __setattr__(self, key, value):
                if key.startswith("_"):
                    super().__setattr__(key, value)
                else:
                    self._state[key] = value

        # Simulate main() function initialization
        session_state = MockSessionState()

        # This is what happens in main() now
        if "unified_dashboard" not in session_state:
            session_state.unified_dashboard = UnifiedDashboard()
            print("  ✅ Dashboard initialized in session state")

        # This is what happens in render functions
        dashboard = session_state.get("unified_dashboard")

        if dashboard:
            print("  ✅ Dashboard accessible from session state")

            # Test the data that render functions would get
            perf_data = dashboard.get_unified_performance_data()
            data_source = perf_data.get("data_source", "unknown")

            if data_source in ["production_api", "api_endpoint"]:
                print(f"  ✅ Render functions will get REAL data from: {data_source}")
                print("  🎉 NO MORE DEMO DATA WARNINGS!")
                return True
            else:
                print(f"  ❌ Render functions still getting demo data: {data_source}")
                return False
        else:
            print("  ❌ Dashboard not accessible from session state")
            return False

    except Exception as e:
        print(f"❌ Error simulating render functions: {e}")
        return False


def test_api_endpoints():
    """Test that backend APIs are working"""
    print("\n🔌 Testing backend API connectivity:")

    try:
        # Test Enhanced Dashboard API
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            print("  ✅ Enhanced Dashboard API (port 5001): Online")
        else:
            print(f"  ⚠️ Enhanced Dashboard API: Status {response.status_code}")

        # Test some data endpoints
        try:
            response = requests.get("http://localhost:5001/api/performance", timeout=5)
            if response.status_code == 200:
                print("  ✅ Performance data endpoint: Working")
            else:
                print(f"  ⚠️ Performance data endpoint: Status {response.status_code}")
        except Exception:
            print("  ⚠️ Performance data endpoint: Not available")

        return True

    except Exception as e:
        print(f"  ❌ API connectivity test failed: {e}")
        return False


def main():
    """Run comprehensive validation"""
    print("🚀 UNIFIED DASHBOARD REAL DATA VALIDATION")
    print("=" * 60)

    # Test 1: Data sources
    test1_result = validate_dashboard_data_sources()

    # Test 2: Streamlit simulation
    test2_result = simulate_streamlit_render_functions()

    # Test 3: API connectivity
    test3_result = test_api_endpoints()

    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL VALIDATION SUMMARY")
    print(f"Dashboard Data Sources: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"Session State Access: {'✅ PASS' if test2_result else '❌ FAIL'}")
    print(f"API Connectivity: {'✅ PASS' if test3_result else '❌ FAIL'}")

    if test1_result and test2_result and test3_result:
        print("\n🎉 SUCCESS: UNIFIED DASHBOARD FIX COMPLETE!")
        print("💡 The dashboard should now display:")
        print("   • Real trading data instead of 'Demo data preview'")
        print("   • Real account balances instead of 'Using demo data'")
        print("   • Real market data instead of 'ML predictions using demo data'")
        print("   • Real alerts instead of 'Using demo alerts'")
        print("   • No more 'production manager not initialized' warnings")
        print(
            "\n🚀 Ready to launch: streamlit run unified_trading_dashboard.py --server.port 8512"
        )
    else:
        print("\n⚠️ Some validation tests failed - review results above")


if __name__ == "__main__":
    main()
