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
    """Test the optimized production data manager"""
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
        balance = mgr.get_account_balance(use_cache=False)
        elapsed = time.time() - start_time

        print(f"⏱️ Account balance took: {elapsed:.2f}s")
        print(
            f"📊 Balance success: {balance.get('retCode') == 0 or balance.get('success', False)}"
        )
        print(f"📊 Data source: {balance.get('data_source', 'unknown')}")

        if elapsed > 15:
            print("⚠️ Balance call took longer than expected")
            raise AssertionError()

        # Test 2: Enhanced portfolio details with timeout protection
        print("\n🔍 Testing enhanced portfolio details (with timeout protection)...")
        start_time = time.time()
        try:
            portfolio = mgr.get_enhanced_portfolio_details(use_cache=False)
            elapsed = time.time() - start_time

            print(f"⏱️ Enhanced portfolio took: {elapsed:.2f}s")
            print(f"📊 Portfolio success: {portfolio.get('success', False)}")
            print(f"📊 Data source: {portfolio.get('data_source', 'unknown')}")

            if elapsed > 30:
                print("⚠️ Portfolio call took longer than expected")
                raise AssertionError()

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Enhanced portfolio failed after {elapsed:.2f}s: {e}")
            raise AssertionError()

        # Test 3: Dashboard-compatible portfolio data
        print("\n📊 Testing dashboard portfolio data...")
        start_time = time.time()
        try:
            dashboard_data = mgr.get_portfolio_data(use_cache=False)
            elapsed = time.time() - start_time

            print(f"⏱️ Dashboard portfolio took: {elapsed:.2f}s")
            print(f"📊 Dashboard success: {dashboard_data.get('success', False)}")
            print(f"📊 Total value: {dashboard_data.get('total_value', 'N/A')}")
            print(
                f"📊 Available balance: {dashboard_data.get('available_balance', 'N/A')}"
            )

            if elapsed > 35:
                print("⚠️ Dashboard portfolio call took longer than expected")
                raise AssertionError()

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Dashboard portfolio failed after {elapsed:.2f}s: {e}")
            raise AssertionError()

        # Test 4: Cache effectiveness
        print("\n🗂️ Testing cache effectiveness...")
        start_time = time.time()
        mgr.get_portfolio_data(use_cache=True)  # Should use cache
        elapsed = time.time() - start_time

        print(f"⏱️ Cached portfolio took: {elapsed:.2f}s")
        if elapsed > 1.0:
            print("⚠️ Cached call took too long - cache might not be working")
        else:
            print("✅ Cache working effectively")

        print("\n✅ All timeout tests passed!")
        assert True

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
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
