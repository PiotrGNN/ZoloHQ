#!/usr/bin/env python3
"""
Test script to verify the Master Control Dashboard data source fix
"""

import sys

sys.path.append(".")


def test_master_control_data_source():
    """Test the Master Control Dashboard data source logic"""
    print("🧪 Testing Master Control Dashboard Data Source Fix")
    print("=" * 60)

    try:
        from master_control_dashboard import MasterControlDashboard

        # Initialize dashboard
        dashboard = MasterControlDashboard()
        print("✅ Master Control Dashboard initialized")

        # Get system metrics
        metrics = dashboard.get_system_metrics()
        print("✅ System metrics retrieved")

        # Test the data source logic
        print("\n📊 Testing Data Source Logic:")
        print("Available metrics keys:", list(metrics.keys()))

        # Check for portfolio data
        has_enhanced = "enhanced_portfolio" in metrics
        has_main = "main_portfolio" in metrics

        print(f"Enhanced portfolio present: {has_enhanced}")
        print(f"Main portfolio present: {has_main}")

        # Test the data source determination logic (same as in main dashboard)
        data_source = None
        if has_enhanced:
            data_source = metrics["enhanced_portfolio"].get("data_source", None)
            print(f"Enhanced portfolio data_source: {data_source}")
        elif has_main:
            data_source = metrics["main_portfolio"].get("data_source", None)
            print(f"Main portfolio data_source: {data_source}")

        # Determine what status would be shown
        print("\n🎯 Data Source Status Result:")
        if data_source == "production_api":
            print(
                '✅ SUCCESS: Would display "🟢 Data source: Bybit production API (real)"'
            )
        elif data_source == "api_endpoint":
            print(
                '✅ SUCCESS: Would display "🔵 Data source: Enhanced Dashboard API (real)"'
            )
        elif data_source == "fallback":
            print(
                '⚠️ WARNING: Would display "🟡 Data source: Fallback (API unavailable)"'
            )
        elif data_source:
            print(f'⚠️ WARNING: Would display "🟠 Data source: {data_source}"')
        else:
            print('❌ ISSUE: Would display "🔴 Data source: Unknown"')

        return data_source is not None

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_master_control_data_source()
    if success:
        print("\n🎉 Master Control Dashboard data source fix is working!")
    else:
        print("\n💥 Master Control Dashboard data source fix needs more work")
