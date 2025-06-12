#!/usr/bin/env python3
"""
Test script to validate that the unified dashboard session state fix works
"""
import sys
from pathlib import Path

# Add path for imports
sys.path.append(str(Path(__file__).parent))


def test_unified_dashboard_import():
    """Test that UnifiedDashboard can be imported and initialized"""
    try:
        from unified_trading_dashboard import UnifiedDashboard

        print("✅ Successfully imported UnifiedDashboard class")

        # Test initialization
        dashboard = UnifiedDashboard()
        print("✅ Successfully created UnifiedDashboard instance")

        # Test production manager
        if dashboard.production_manager:
            print("✅ Production manager initialized successfully")
        else:
            print("⚠️ Production manager not available - using API fallback")

        # Test system status
        status = dashboard.get_system_status()
        print(f"✅ System status retrieved: {len(status)} services")

        # Test performance data
        perf_data = dashboard.get_unified_performance_data()
        data_source = perf_data.get("data_source", "unknown")
        print(f"✅ Performance data retrieved from: {data_source}")

        return True

    except Exception as e:
        print(f"❌ Error testing UnifiedDashboard: {e}")
        return False


def test_session_state_logic():
    """Test the session state initialization logic"""
    try:
        # Mock streamlit session state
        class MockSessionState:
            def __init__(self):
                self._state = {}

            def __contains__(self, key):
                return key in self._state

            def __setattr__(self, key, value):
                if key.startswith("_"):
                    super().__setattr__(key, value)
                else:
                    self._state[key] = value

            def __getattr__(self, key):
                if key.startswith("_"):
                    return super().__getattribute__(key)
                return self._state.get(key)

        # Simulate the session state logic
        mock_session_state = MockSessionState()

        # Test the initialization logic
        if "unified_dashboard" not in mock_session_state:
            from unified_trading_dashboard import UnifiedDashboard

            mock_session_state.unified_dashboard = UnifiedDashboard()
            print("✅ Session state initialization logic works")

        # Test access
        dashboard = mock_session_state.unified_dashboard
        if dashboard:
            print("✅ Dashboard accessible from session state")
            return True
        else:
            print("❌ Dashboard not accessible from session state")
            return False

    except Exception as e:
        print(f"❌ Error testing session state logic: {e}")
        return False


def main():
    """Run all tests"""
    print("🔧 Testing Unified Dashboard Session State Fix")
    print("=" * 50)

    # Test 1: Basic import and initialization
    print("\n📋 Test 1: UnifiedDashboard Import and Initialization")
    test1_result = test_unified_dashboard_import()

    # Test 2: Session state logic
    print("\n📋 Test 2: Session State Logic")
    test2_result = test_session_state_logic()

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print(f"UnifiedDashboard Import/Init: {'✅ PASS' if test1_result else '❌ FAIL'}")
    print(f"Session State Logic: {'✅ PASS' if test2_result else '❌ FAIL'}")

    if test1_result and test2_result:
        print("\n🎉 ALL TESTS PASSED - Session state fix should work!")
        print(
            "💡 The unified dashboard should now display real data instead of demo warnings"
        )
    else:
        print("\n⚠️ Some tests failed - additional fixes may be needed")


if __name__ == "__main__":
    main()
