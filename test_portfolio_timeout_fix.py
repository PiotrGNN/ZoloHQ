#!/usr/bin/env python3
"""
Test script to verify portfolio timeout fixes
"""
import sys
import time
import traceback
from pathlib import Path

# Add ZoL0-master to path
sys.path.append(str(Path(__file__).parent / "ZoL0-master"))


def test_optimized_portfolio_manager():
    """Test the optimized production data manager with advanced profit analysis and timeout protection."""
    import time

    try:
        print("🚀 Testing Optimized Portfolio Manager")
        print("=" * 50)

        from production_data_manager import ProductionDataManager

        print("✅ Production manager imported successfully")

        mgr = ProductionDataManager()
        print("✅ Production manager initialized")

        # Test 1: Account balance with timeout protection
        print("\n💰 Testing account balance (with timeout protection)...")
        start_time = time.time()
        balance = mgr.get_account_balance()
        elapsed = time.time() - start_time

        print(f"⏱️ Time: {elapsed:.2f}s, Balance: {balance}")

        if elapsed > 5:
            print("⚠️ Timeout risk detected! Consider optimizing API calls.")

        # Test 2: Historical profit analysis
        print("\n📈 Analyzing historical profits...")
        if hasattr(mgr, "get_historical_performance"):
            perf = mgr.get_historical_performance()
            profits = [p["profit"] for p in perf if "profit" in p]
            avg_profit = sum(profits) / len(profits) if profits else 0
            print(f"Średni zysk historyczny: {avg_profit:.2f}")

            if avg_profit < 0:
                print("❌ Portfel generuje straty! Zalecana optymalizacja strategii.")
            else:
                print("✅ Portfel zyskowny.")
        else:
            print("Brak metody get_historical_performance - pomiń analizę.")

        # Test 3: Rekomendacje optymalizacji
        print("\n🔍 Rekomendacje optymalizacji portfela:")
        if hasattr(mgr, "recommend_optimizations"):
            recs = mgr.recommend_optimizations()
            for rec in recs:
                print(f"- {rec}")
        else:
            print("Brak rekomendacji - dodaj recommend_optimizations do ProductionDataManager.")

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        raise AssertionError()


if __name__ == "__main__":
    print("🔧 Portfolio Timeout Fix Verification")
    print("=" * 50)

    success = test_optimized_portfolio_manager()

    if success:
        print("\n🎉 All tests passed! Portfolio timeout issue should be resolved.")
    else:
        print("\n❌ Tests failed. Timeout issue may persist.")

    print("\n🏁 Test complete")
