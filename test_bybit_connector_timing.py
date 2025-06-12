#!/usr/bin/env python3
"""
Test individual Bybit connector methods to understand performance bottlenecks
"""

import time

from production_data_manager import ProductionDataManager


def test_bybit_connector_timing():
    """Test timing of individual bybit connector methods"""
    print("🔍 TESTING BYBIT CONNECTOR TIMING")
    print("=" * 50)

    try:
        mgr = ProductionDataManager()
        print("✅ Production Manager loaded")
        print(f"🔗 Bybit connector available: {mgr.bybit_connector is not None}")

        if not mgr.bybit_connector:
            print("❌ No bybit connector available")
            return

        connector = mgr.bybit_connector

        # Test 1: get_account_balance (direct)
        print("\n1️⃣ Testing get_account_balance (direct)...")
        start_time = time.time()
        try:
            balance = connector.get_account_balance()
            elapsed = time.time() - start_time
            print(f"⏱️ get_account_balance: {elapsed:.2f}s")
            print(f"📊 Success: {balance.get('retCode') == 0}")
            print(f"📊 Return code: {balance.get('retCode')}")
            print(f"📊 Message: {balance.get('retMsg')}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"⏱️ get_account_balance: {elapsed:.2f}s (failed)")
            print(f"❌ Error: {e}")

        # Test 2: get_ticker (direct)
        print("\n2️⃣ Testing get_ticker (direct)...")
        start_time = time.time()
        try:
            ticker = connector.get_ticker("BTCUSDT")
            elapsed = time.time() - start_time
            print(f"⏱️ get_ticker: {elapsed:.2f}s")
            print(f"📊 Success: {ticker.get('retCode') == 0}")
            print(f"📊 Return code: {ticker.get('retCode')}")
            if ticker.get("retCode") == 0:
                price = (
                    ticker.get("result", {})
                    .get("list", [{}])[0]
                    .get("lastPrice", "N/A")
                )
                print(f"📊 BTC Price: {price}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"⏱️ get_ticker: {elapsed:.2f}s (failed)")
            print(f"❌ Error: {e}")

        # Test 3: get_positions (direct)
        print("\n3️⃣ Testing get_positions (direct)...")
        start_time = time.time()
        try:
            positions = connector.get_positions()
            elapsed = time.time() - start_time
            print(f"⏱️ get_positions: {elapsed:.2f}s")
            print(f"📊 Success: {positions.get('retCode') == 0}")
            print(f"📊 Return code: {positions.get('retCode')}")
            print(f"📊 Message: {positions.get('retMsg')}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"⏱️ get_positions: {elapsed:.2f}s (failed)")
            print(f"❌ Error: {e}")

        # Test 4: get_server_time (direct)
        print("\n4️⃣ Testing get_server_time (direct)...")
        start_time = time.time()
        try:
            server_time = connector.get_server_time()
            elapsed = time.time() - start_time
            print(f"⏱️ get_server_time: {elapsed:.2f}s")
            print(f"📊 Success: {server_time.get('retCode') == 0}")
            print(f"📊 Return code: {server_time.get('retCode')}")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"⏱️ get_server_time: {elapsed:.2f}s (failed)")
            print(f"❌ Error: {e}")

        # Test rate limiting check
        print("\n5️⃣ Testing multiple quick calls (rate limiting test)...")
        for i in range(3):
            start_time = time.time()
            try:
                server_time = connector.get_server_time()
                elapsed = time.time() - start_time
                print(
                    f"⏱️ Call {i+1}: {elapsed:.2f}s (retCode: {server_time.get('retCode')})"
                )
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"⏱️ Call {i+1}: {elapsed:.2f}s (error: {e})")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_bybit_connector_timing()
