#!/usr/bin/env python3
"""
Production API Verification Script
Verify that all dashboards can access real Bybit production data
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import requests

# Add ZoL0-master to path
sys.path.append(str(Path(__file__).parent / "ZoL0-master"))


def test_bybit_api_connection():
    """Test direct connection to Bybit API"""
    print("🔍 Testing Bybit API Connection...")
    print("-" * 40)

    try:
        # Test server time endpoint (no auth required)
        response = requests.get("https://api.bybit.com/v5/market/time", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                server_time = datetime.fromtimestamp(int(data["result"]["timeSecond"]))
                print("✅ Bybit API accessible")
                print(f"   Server time: {server_time}")
                return True
        print(f"❌ API response error: {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_environment_variables():
    """Test that production environment variables are set"""
    print("\n🔍 Testing Environment Variables...")
    print("-" * 40)

    required_vars = {
        "BYBIT_API_KEY": os.getenv("BYBIT_API_KEY"),
        "BYBIT_API_SECRET": os.getenv("BYBIT_API_SECRET"),
        "BYBIT_PRODUCTION_ENABLED": os.getenv("BYBIT_PRODUCTION_ENABLED"),
        "BYBIT_PRODUCTION_CONFIRMED": os.getenv("BYBIT_PRODUCTION_CONFIRMED"),
    }

    all_set = True
    for var, value in required_vars.items():
        if value:
            if "SECRET" in var:
                print(f"✅ {var}: {'*' * len(value)}")
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: Not set")
            all_set = False

    return all_set


def test_bybit_connector():
    """Test BybitConnector import and initialization"""
    print("\n🔍 Testing BybitConnector...")
    print("-" * 40)

    try:
        from data.execution.bybit_connector import BybitConnector

        # Initialize connector in production mode
        connector = BybitConnector(
            api_key=os.getenv("BYBIT_API_KEY"),
            api_secret=os.getenv("BYBIT_API_SECRET"),
            use_testnet=False,  # Production mode
        )

        # Test server time call
        server_time = connector.get_server_time()
        if server_time.get("success"):
            print("✅ BybitConnector working correctly")
            print(f"   Production mode: {not connector.use_testnet}")
            return True
        else:
            print(f"❌ API call failed: {server_time}")
            return False

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Connector test failed: {e}")
        return False


def test_market_data_access():
    """Test real market data access"""
    print("\n🔍 Testing Market Data Access...")
    print("-" * 40)

    try:
        # Test basic ticker data
        response = requests.get(
            "https://api.bybit.com/v5/market/tickers?category=spot&symbol=BTCUSDT",
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                ticker = data["result"]["list"][0]
                print("✅ Real market data accessible")
                print(f"   BTCUSDT Price: ${float(ticker.get('lastPrice', 0)):,.2f}")
                print(f"   24h Volume: {float(ticker.get('volume24h', 0)):,.0f}")
                return True
        print(f"❌ Market data access failed: {response.status_code}")
        return False
    except Exception as e:
        print(f"❌ Market data test failed: {e}")
        return False


def main():
    """Run all verification tests"""
    print("=" * 60)
    print("🚀 ZoL0 Production API Verification")
    print("=" * 60)
    print(f"⏰ Test run: {datetime.now()}")
    print()

    tests = [
        ("Environment Variables", test_environment_variables),
        ("Bybit API Connection", test_bybit_api_connection),
        ("BybitConnector Module", test_bybit_connector),
        ("Market Data Access", test_market_data_access),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)

    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1

    print()
    print(f"Tests passed: {passed}/{len(results)}")

    if passed == len(results):
        print("🎉 ALL TESTS PASSED - Production API is working correctly!")
        print("✅ All dashboards should now show real Bybit production data")
    else:
        print("⚠️  Some tests failed - please check configuration")

    print("=" * 60)


if __name__ == "__main__":
    main()
