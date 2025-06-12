#!/usr/bin/env python3
"""
Quick test of the positions API fix
"""

import time

from production_data_manager import ProductionDataManager


def test_positions_fix():
    """Test the positions API fix specifically"""
    print("🔍 TESTING POSITIONS API FIX")
    print("=" * 40)

    try:
        mgr = ProductionDataManager()

        if not mgr.bybit_connector:
            print("❌ No bybit connector available")
            return False

        connector = mgr.bybit_connector

        # Test positions API directly
        print("1️⃣ Testing get_positions() directly...")
        start_time = time.time()
        positions_result = connector.get_positions()
        elapsed = time.time() - start_time

        print(f"   ⏱️ Time: {elapsed:.2f}s")
        print(f"   🔄 Return Code: {positions_result.get('retCode')}")
        print(f"   📝 Message: {positions_result.get('retMsg', 'No message')}")

        if positions_result.get("retCode") == 0:
            print("   ✅ Positions API working correctly!")

            # Check positions data
            result_data = positions_result.get("result", {})
            positions_list = result_data.get("list", [])
            print(f"   📊 Found {len(positions_list)} positions")

            # Show sample position if any
            if positions_list:
                sample_pos = positions_list[0]
                symbol = sample_pos.get("symbol", "Unknown")
                size = sample_pos.get("size", "0")
                side = sample_pos.get("side", "None")
                print(f"   📈 Sample: {symbol} - Size: {size} - Side: {side}")
            else:
                print("   📝 No open positions found (account may be empty)")

            assert True
        else:
            print(f"   ❌ API Error: {positions_result.get('retMsg', 'Unknown error')}")
            raise AssertionError()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise AssertionError()


if __name__ == "__main__":
    success = test_positions_fix()
    if success:
        print("\n🎉 Positions API fix successful!")
    else:
        print("\n⚠️ Positions API still needs work")
