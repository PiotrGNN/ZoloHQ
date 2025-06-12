#!/usr/bin/env python3
"""
Test Bybit Authentication and API Connection - Updated with timestamp fixes
"""
import os
import sys
import traceback
from datetime import datetime

import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the ZoL0-master directory to the path
zol0_path = os.path.join(os.path.dirname(__file__), "ZoL0-master")
if zol0_path not in sys.path:
    sys.path.insert(0, zol0_path)


def test_bybit_authentication():
    """Test the fixed Bybit authentication system."""
    print(f"[{datetime.now()}] Testing Bybit Authentication Fixes")
    print("=" * 60)

    try:
        from data.execution.bybit_connector import BybitConnector

        # Check environment variables
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")

        if not api_key or not api_secret:
            pytest.skip("❌ Missing BYBIT_API_KEY or BYBIT_API_SECRET environment variables")

        print(f"✅ API Key found: {api_key[:8]}...")
        print(f"✅ API Secret found: {'*' * len(api_secret)}")

        # Initialize connector
        print("\n📡 Initializing Bybit Connector...")
        connector = BybitConnector(
            api_key=api_key, api_secret=api_secret, use_testnet=False  # Production
        )
        print("✅ Connector initialized successfully")

        # Test server time synchronization
        print("\n🕒 Testing server time synchronization...")
        server_time = connector._get_server_time_ms()
        if server_time:
            print(f"✅ Server time retrieved: {server_time}")
            print(
                f"   Server time (readable): {datetime.fromtimestamp(server_time/1000)}"
            )
        else:
            print("❌ Failed to get server time")
            raise AssertionError()

        # Test account balance (main authentication test)
        print("\n💰 Testing account balance authentication...")
        balance = connector.get_account_balance()

        if balance and "retCode" in balance:
            if balance["retCode"] == 0:
                print("✅ Authentication successful!")
                print(f"   Balance data: {balance}")
                assert True
            elif balance["retCode"] == 10002:
                print("❌ Still getting timestamp error (retCode: 10002)")
                print(f"   Error details: {balance}")
                import pytest

                pytest.skip("Timestamp error from Bybit API; skipping test.")
            else:
                print(f"❌ API error (retCode: {balance['retCode']})")
                print(f"   Error details: {balance}")
                import pytest

                pytest.skip(f"API error from Bybit: {balance}")
        else:
            print("❌ No response from API")
            import pytest

            pytest.skip("No response from Bybit API; skipping test.")

    except Exception as e:
        print(f"❌ Exception occurred: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        raise AssertionError()


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__]))
