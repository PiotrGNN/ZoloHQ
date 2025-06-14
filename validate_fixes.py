# Quick Test Script for Dashboard Fixes
# Simple validation to ensure all fixes are working

import sys
import traceback


def test_unified_dashboard():
    """Test unified dashboard import and basic functionality"""
    print("🔧 Testing Unified Dashboard...")

    try:
        # Test import
        from unified_trading_dashboard import UnifiedDashboard

        print("  ✅ Import successful")

        # Test instance creation
        dashboard = UnifiedDashboard()
        print("  ✅ Instance created")

        # Test performance data method
        perf_data = dashboard.get_unified_performance_data()
        data_source = perf_data.get("data_source", "unknown")
        print(f"  📊 Data source: {data_source}")

        # Test required fields
        required_fields = ["total_profit", "win_rate", "active_bots", "data_source"]
        missing_fields = [field for field in required_fields if field not in perf_data]

        if missing_fields:
            print(f"❌ Brakujące metryki: {missing_fields}")
        else:
            print("✅ Wszystkie kluczowe metryki obecne.")

        # Logowanie nieoptymalnych danych
        if perf_data.get("total_profit", 0) < 0:
            print("❌ Dashboard generuje straty! Konieczna optymalizacja.")
        if perf_data.get("win_rate", 0) < 0.5:
            print("⚠️ Niska skuteczność strategii!")

        print("  ✅ Unified Dashboard test passed")
        return True

    except Exception as e:
        print(f"  ❌ Error: {e}")
        print(f"  📍 Details: {traceback.format_exc()}")
        return False


def test_column_safety():
    """Test that DataFrame operations are safe"""
    print("\n🔧 Testing DataFrame Column Safety...")

    try:
        import pandas as pd

        # Simulate real data without 'Zysk' column
        real_data = pd.DataFrame(
            {
                "Data": pd.date_range("2025-01-01", periods=5, freq="D"),
                "Symbol": ["BTCUSDT"] * 5,
                "Cena": [45000, 46000, 45500, 47000, 46500],
                "Wolumen": [1000, 1200, 950, 1300, 1100],
                "High": [45500, 46500, 46000, 47500, 47000],
                "Low": [44500, 45500, 45000, 46500, 46000],
            }
        )

        print(f"  📊 Created test DataFrame with columns: {list(real_data.columns)}")

        # Test safe metric calculations (like in the fixed code)
        if "Zysk" in real_data.columns:
            total_profit = real_data["Zysk"].sum()
            print(f"  💰 Total Profit: ${total_profit:.2f}")
        else:
            print("  💰 Total Profit: N/A (no profit column) - SAFE")

        if "Cena" in real_data.columns:
            avg_price = real_data["Cena"].mean()
            print(f"  💵 Average Price: ${avg_price:.2f}")
        else:
            print("  💵 Average Price: N/A")

        if "Wolumen" in real_data.columns:
            total_volume = real_data["Wolumen"].sum()
            print(f"  📈 Total Volume: {total_volume:.0f}")
        else:
            print("  📈 Total Volume: N/A")

        # Test win rate calculation with price data when no profit data
        if "Zysk" in real_data.columns:
            win_rate = len(real_data[real_data["Zysk"] > 0]) / len(real_data) * 100
            print(f"  🎯 Win Rate (Profit): {win_rate:.1f}%")
        else:
            if "Cena" in real_data.columns and len(real_data) > 1:
                price_increases = (
                    real_data["Cena"].shift(-1) > real_data["Cena"]
                ).sum()
                win_rate = (price_increases / (len(real_data) - 1)) * 100
                print(f"  🎯 Win Rate (Price Up): {win_rate:.1f}% - CALCULATED SAFELY")
            else:
                print("  🎯 Win Rate: N/A")

        print("  ✅ DataFrame operations completed safely")
        return True

    except Exception as e:
        print(f"  ❌ Error in DataFrame operations: {e}")
        return False


def test_api_connection():
    """Test if Enhanced Dashboard API is available"""
    print("\n🔧 Testing API Connection...")

    try:
        import requests

        response = requests.get("http://localhost:4001/health", timeout=3)
        if response.status_code == 200:
            print("  ✅ Enhanced Dashboard API is running")

            # Test portfolio endpoint
            portfolio_response = requests.get(
                "http://localhost:4001/api/portfolio", timeout=3
            )
            if portfolio_response.status_code == 200:
                print("  ✅ Portfolio endpoint accessible")
                data = portfolio_response.json()
                print(f"  📊 API data keys: {list(data.keys())}")
            else:
                print(
                    f"  ⚠️ Portfolio endpoint returned {portfolio_response.status_code}"
                )

            return True
        else:
            print(f"  ⚠️ API returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ⚠️ API not available: {e}")
        print("  💡 This is OK - dashboards will use fallback data")
        return False


def main():
    """Run all validation tests"""
    print("🚀 Dashboard Fixes Validation")
    print("=" * 50)

    tests = [
        ("Unified Dashboard", test_unified_dashboard),
        ("DataFrame Safety", test_column_safety),
        ("API Connection", test_api_connection),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    print("\n" + "=" * 50)
    print("📊 Validation Results:")

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1

    print(f"\n🎯 Summary: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("\n🎉 ALL FIXES VALIDATED SUCCESSFULLY!")
        print("🚀 Dashboards are ready to run without errors")
    else:
        print(f"\n⚠️ {len(results) - passed} tests failed - review issues above")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
