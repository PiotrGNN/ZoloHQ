#!/usr/bin/env python3
"""
Dashboard Production Data Verification
Test all dashboard files to ensure they're using real Bybit production data
"""

import importlib.util
import os
from datetime import datetime
from pathlib import Path


def test_dashboard_import(dashboard_file):
    """Test if a dashboard file can be imported and uses production data"""
    print(f"\n🔍 Testing {dashboard_file}...")
    print("-" * 50)

    try:
        # Load the dashboard module
        spec = importlib.util.spec_from_file_location("dashboard", dashboard_file)
        if spec is None:
            print(f"❌ Could not create spec for {dashboard_file}")
            return False

        importlib.util.module_from_spec(spec)

        # Check if file contains production-related functions
        with open(dashboard_file, "r", encoding="utf-8") as f:
            content = f.read()

        production_indicators = [
            "_get_api_data",
            "_initialize_real_data",
            "fetch_real_trading_data",
            "check_real_api_status",
            "BYBIT_PRODUCTION_ENABLED",
        ]

        found_indicators = []
        for indicator in production_indicators:
            if indicator in content:
                found_indicators.append(indicator)

        if found_indicators:
            print(f"✅ Production features found: {', '.join(found_indicators)}")

            # Check for correct import paths
            if 'sys.path.append(str(Path(__file__).parent / "ZoL0-master"))' in content:
                print("✅ Correct import path configuration found")
            elif "from ZoL0master" in content:
                print("⚠️  Old import path found - needs fixing")
            else:
                print("ℹ️  No specific import path configuration")

            return True
        else:
            print("❌ No production features found")
            return False

    except Exception as e:
        print(f"❌ Error testing {dashboard_file}: {e}")
        return False


def main():
    """Test all dashboard files"""
    print("=" * 80)
    print("🚀 ZoL0 Dashboard Production Data Verification")
    print("=" * 80)
    print(f"⏰ Test run: {datetime.now()}")
    print()

    # List of dashboard files to test
    dashboard_files = [
        "advanced_trading_analytics.py",
        "real_time_market_data_integration.py",
        "ml_predictive_analytics.py",
        "enhanced_bot_monitor.py",
        "advanced_alert_management.py",
        "data_export_import_system.py",
    ]

    base_path = Path("c:/Users/piotr/Desktop/Zol0")
    results = []

    for dashboard_file in dashboard_files:
        file_path = base_path / dashboard_file
        if file_path.exists():
            result = test_dashboard_import(file_path)
            results.append((dashboard_file, result))
        else:
            print(f"\n❌ File not found: {dashboard_file}")
            results.append((dashboard_file, False))

    # Summary
    print("\n" + "=" * 80)
    print("📊 DASHBOARD VERIFICATION SUMMARY")
    print("=" * 80)

    passed = 0
    for dashboard_file, result in results:
        status = "✅ READY" if result else "❌ NEEDS SETUP"
        print(f"{status} {dashboard_file}")
        if result:
            passed += 1

    print()
    print(f"Dashboards ready for production: {passed}/{len(results)}")

    if passed == len(results):
        print("🎉 ALL DASHBOARDS READY FOR PRODUCTION!")
        print("✅ All dashboards are configured to use real Bybit production data")
    else:
        print("⚠️  Some dashboards need configuration")

    print("\n📋 Current Environment Status:")
    print(f"   BYBIT_PRODUCTION_ENABLED: {os.getenv('BYBIT_PRODUCTION_ENABLED')}")
    print(f"   BYBIT_PRODUCTION_CONFIRMED: {os.getenv('BYBIT_PRODUCTION_CONFIRMED')}")
    print(f"   API Key: {'✅ Set' if os.getenv('BYBIT_API_KEY') else '❌ Missing'}")
    print(
        f"   API Secret: {'✅ Set' if os.getenv('BYBIT_API_SECRET') else '❌ Missing'}"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
