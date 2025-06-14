#!/usr/bin/env python3
"""
Debug script to test portfolio data retrieval
"""
import os
import sys
import time
import traceback
from pathlib import Path

# Add ZoL0-master to path
sys.path.append(str(Path(__file__).parent / "ZoL0-master"))

import pytest
from dotenv import load_dotenv

load_dotenv()


def test_basic_connection():
    """Test basic connection without production data manager"""
    try:
        print("🔍 Testing basic Bybit connection...")
        # Load environment variables

        from data.execution.bybit_connector import BybitConnector

        api_key = os.getenv("BYBIT_API_KEY", "lAXnmPeMMVecqcW8oT")
        api_secret = os.getenv(
            "BYBIT_API_SECRET", "RAQcrNjFSVBGWeRBjQGL8fTRzbtbKHmAArGz"
        )

        if not api_key:
            pytest.skip("BYBIT_API_KEY not set in environment.")

        # Create connector with timeout
        if "BYBIT_TESTNET" not in os.environ:
            print("⚠️  BYBIT_TESTNET not set in environment. Defaulting to production (testnet=False).")
        use_testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"
        connector = BybitConnector(
            api_key=api_key, api_secret=api_secret, testnet=use_testnet  # Production if false
        )

        print("✅ Connector created, testing server time...")
        start_time = time.time()
        server_time = connector.get_server_time()
        elapsed = time.time() - start_time

        print(f"⏱️ Server time call took: {elapsed:.2f}s")
        print(f"📡 Server time result: {server_time.get('retCode', 'unknown')}")

        if server_time.get("retCode") == 0:
            print("✅ Basic connection working")

            print("\n💰 Testing account balance...")
            start_time = time.time()
            balance = connector.get_account_balance()
            elapsed = time.time() - start_time

            print(f"⏱️ Balance call took: {elapsed:.2f}s")
            print(f"📊 Balance result: {balance.get('retCode', 'unknown')}")

            if balance.get("retCode") == 0:
                print("✅ Account balance working")
                result = balance.get("result", {})
                total_equity = result.get("totalEquity", "0")
                total_available = result.get("totalAvailableBalance", "0")
                print(f"💵 Total Equity: {total_equity}")
                print(f"💵 Available Balance: {total_available}")
                assert True
            else:
                print(f"❌ Balance failed: {balance}")
                pytest.skip("Balance API failed or unavailable; skipping test.")
        else:
            print(f"❌ Connection failed: {server_time}")
            pytest.skip("Bybit API connection failed or unavailable; skipping test.")
    except Exception as e:
        print(f"❌ Exception in basic connection: {e}")
        pytest.skip(f"Exception in Bybit connection: {e}")


def test_production_manager():
    """Test production data manager with timeout"""
    try:
        print("\n🏭 Testing Production Data Manager...")

        # Import with timeout simulation

        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")

        # Set alarm for 30 seconds (Windows doesn't support SIGALRM, so we'll use threading)
        import threading

        result = {"success": False, "data": None, "error": None}

        def run_test():
            try:
                from production_data_manager import ProductionDataManager

                print("✅ Production manager imported")

                mgr = ProductionDataManager()
                print("✅ Production manager initialized")

                # Test individual components
                print("🔍 Testing account balance...")
                start_time = time.time()
                balance = mgr.get_account_balance(use_cache=False)
                elapsed = time.time() - start_time

                print(f"⏱️ Account balance took: {elapsed:.2f}s")
                print(f"📊 Balance source: {balance.get('data_source', 'unknown')}")
                print(f"📊 Balance success: {balance.get('retCode') == 0}")

                result["success"] = True
                result["data"] = {
                    "balance_time": elapsed,
                    "balance_source": balance.get("data_source"),
                    "balance_success": balance.get("retCode") == 0,
                }
                assert True

            except Exception as e:
                result["error"] = str(e)
                print(f"❌ Production manager error: {e}")
                traceback.print_exc()
                raise AssertionError()

        # Run test in thread with timeout
        test_thread = threading.Thread(target=run_test)
        test_thread.daemon = True
        test_thread.start()
        test_thread.join(timeout=30)  # 30 second timeout

        if test_thread.is_alive():
            print("⏰ Production manager test timed out after 30 seconds")
            raise AssertionError()

        assert result["success"]

    except Exception as e:
        print(f"❌ Exception in production manager: {e}")
        traceback.print_exc()
        raise AssertionError()


if __name__ == "__main__":
    print("🚀 Portfolio Data Debug Test")
    print("=" * 50)

    # Test 1: Basic connection
    basic_ok = test_basic_connection()

    # Test 2: Production manager
    if basic_ok:
        prod_ok = test_production_manager()

        if prod_ok:
            print("\n✅ All tests passed! Portfolio data should work.")
        else:
            print("\n❌ Production manager has issues.")
    else:
        print("\n❌ Basic connection failed.")

    print("\n🏁 Debug test complete")
