#!/usr/bin/env python3
"""
Test script to debug the specific timeout endpoints
"""

import time

import requests


def test_trading_statistics_endpoint():
    """Test the /api/trading/statistics endpoint that's timing out"""
    print("🔍 Testing /api/trading/statistics endpoint...")

    def make_request():
        try:
            start_time = time.time()
            response = requests.get(
                "http://localhost:5001/api/trading/statistics", timeout=25
            )
            elapsed = time.time() - start_time

            print(f"⏱️ Trading statistics response time: {elapsed:.2f}s")
            print(f"📊 Status code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: {data.get('success', False)}")
                print(f"📡 Data source: {data.get('data_source', 'unknown')}")
                return True
            else:
                print(f"❌ Failed with status {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            print(f"⏰ Request timed out after {elapsed:.2f}s")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Error after {elapsed:.2f}s: {e}")
            return False

    return make_request()


def test_cache_init_endpoint():
    """Test the /api/cache/init endpoint that's timing out"""
    print("\n🔍 Testing /api/cache/init endpoint...")

    def make_request():
        try:
            start_time = time.time()
            response = requests.get("http://localhost:5001/api/cache/init", timeout=25)
            elapsed = time.time() - start_time

            print(f"⏱️ Cache init response time: {elapsed:.2f}s")
            print(f"📊 Status code: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: {data.get('success', False)}")
                print(f"📡 Message: {data.get('message', 'unknown')}")
                return True
            else:
                print(f"❌ Failed with status {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            print(f"⏰ Request timed out after {elapsed:.2f}s")
            return False
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Error after {elapsed:.2f}s: {e}")
            return False

    return make_request()


def test_production_manager_methods():
    """Test the specific production manager methods causing issues"""
    print("\n🏭 Testing production manager methods directly...")

    try:
        from production_data_manager import ProductionDataManager

        manager = ProductionDataManager()

        # Test get_trading_stats (used by /api/trading/statistics)
        print("\n📊 Testing get_trading_stats()...")
        start_time = time.time()
        stats = manager.get_trading_stats(use_cache=False)
        elapsed = time.time() - start_time

        print(f"⏱️ get_trading_stats took: {elapsed:.2f}s")
        print(f"📡 Stats success: {stats.get('success', 'No success field')}")
        print(
            f"📡 Stats keys: {list(stats.keys()) if isinstance(stats, dict) else 'Not a dict'}"
        )

        # Test methods that get_trading_stats calls
        print("\n🔍 Testing sub-methods called by get_trading_stats...")

        # Test get_account_balance
        print("💰 Testing get_account_balance...")
        start_time = time.time()
        balance = manager.get_account_balance(use_cache=False)
        elapsed = time.time() - start_time
        print(f"⏱️ get_account_balance took: {elapsed:.2f}s")
        print(f"📡 Balance success: {balance.get('success', False)}")

        # Test get_positions
        print("📈 Testing get_positions...")
        start_time = time.time()
        positions = manager.get_positions(use_cache=False)
        elapsed = time.time() - start_time
        print(f"⏱️ get_positions took: {elapsed:.2f}s")
        print(f"📡 Positions success: {positions.get('success', False)}")

        # Test get_market_data
        print("📊 Testing get_market_data...")
        start_time = time.time()
        market = manager.get_market_data("BTCUSDT", use_cache=False)
        elapsed = time.time() - start_time
        print(f"⏱️ get_market_data took: {elapsed:.2f}s")
        print(f"📡 Market success: {market.get('success', False)}")

        return True

    except Exception as e:
        print(f"❌ Exception testing production manager: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("🚀 Detailed Timeout Endpoint Investigation")
    print("=" * 60)

    # Test 1: Enhanced API server is running
    try:
        response = requests.get("http://localhost:5001/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Enhanced API server is running")
        else:
            print(f"⚠️ Enhanced API server returned status {response.status_code}")
    except Exception:
        print("❌ Enhanced API server is not responding")
        return

    # Test 2: Working endpoints for comparison
    try:
        response = requests.get("http://localhost:5001/api/portfolio", timeout=10)
        if response.status_code == 200:
            print("✅ /api/portfolio endpoint is working")
        else:
            print(f"⚠️ /api/portfolio endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"❌ /api/portfolio endpoint failed: {e}")

    # Test 3: Production manager methods
    manager_ok = test_production_manager_methods()

    # Test 4: Timeout endpoints
    if manager_ok:
        print("\n" + "=" * 60)
        print("Testing timeout endpoints...")

        stats_ok = test_trading_statistics_endpoint()
        cache_ok = test_cache_init_endpoint()

        print("\n" + "=" * 60)
        print("SUMMARY:")
        print(
            f"📊 Trading statistics endpoint: {'✅ OK' if stats_ok else '❌ TIMEOUT'}"
        )
        print(f"🗄️ Cache init endpoint: {'✅ OK' if cache_ok else '❌ TIMEOUT'}")

        if not stats_ok or not cache_ok:
            print("\n🔧 ISSUE: Endpoints are still timing out!")
            print(
                "This suggests the issue may be in the Flask route handlers themselves,"
            )
            print("not in the underlying production manager methods.")
    else:
        print("❌ Production manager has issues, skipping endpoint tests")


if __name__ == "__main__":
    main()
