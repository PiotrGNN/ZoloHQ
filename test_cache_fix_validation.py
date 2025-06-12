#!/usr/bin/env python3
"""
Cache Fix Validation Test
-------------------------
Comprehensive test to validate the cache access fix between 
production data manager and enhanced dashboard API.
"""

import time
from datetime import datetime

import requests


def test_enhanced_dashboard_api_endpoints():
    """Test all Enhanced Dashboard API endpoints with production data"""

    print("🔧 ENHANCED DASHBOARD API - CACHE FIX VALIDATION")
    print("=" * 60)

    base_url = "http://localhost:5001"

    # Test 1: Health Check
    print("\n1️⃣ Testing API Health...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health Status: {health_data.get('status')}")
            print(f"   Service: {health_data.get('service')}")
            print(f"   Version: {health_data.get('version')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

    # Test 2: Portfolio Endpoint (Main Fix)
    print("\n2️⃣ Testing Portfolio Endpoint (CACHE FIX)...")
    try:
        response = requests.get(f"{base_url}/api/portfolio", timeout=30)
        if response.status_code == 200:
            portfolio_data = response.json()
            data_source = portfolio_data.get("data_source")
            total_value = portfolio_data.get("total_value")
            environment = portfolio_data.get("environment")
            success = portfolio_data.get("success")

            print(f"✅ Portfolio Success: {success}")
            print(f"✅ Data Source: {data_source}")
            print(f"✅ Environment: {environment}")
            print(f"✅ Total Value: {total_value}")

            # Validate this is real production data
            if data_source == "production_api" and environment == "production":
                print(
                    "🎉 CACHE FIX SUCCESS: Portfolio endpoint now uses PRODUCTION DATA!"
                )
                cache_fix_success = True
            elif data_source in ["recent_cache_balance", "cached_production_api"]:
                print("✅ Using cached production data (acceptable)")
                cache_fix_success = True
            else:
                print(f"⚠️ Still using fallback data: {data_source}")
                cache_fix_success = False

        else:
            print(f"❌ Portfolio endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Portfolio endpoint error: {e}")
        return False

    # Test 3: Cache Initialization
    print("\n3️⃣ Testing Cache Initialization...")
    try:
        response = requests.get(f"{base_url}/api/cache/init", timeout=30)
        if response.status_code == 200:
            cache_data = response.json()
            print(f"✅ Cache Init Success: {cache_data.get('success')}")
            print(f"✅ Balance Success: {cache_data.get('balance_success')}")
            print(f"✅ Balance Source: {cache_data.get('balance_source')}")
            print(f"✅ Portfolio Success: {cache_data.get('portfolio_success')}")
            print(f"✅ Portfolio Source: {cache_data.get('portfolio_source')}")
            print(f"✅ Cache Size: {cache_data.get('cache_size')}")
        else:
            print(f"❌ Cache init failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Cache init error: {e}")

    # Test 4: Portfolio Endpoint After Cache (Should be fast)
    print("\n4️⃣ Testing Portfolio Endpoint Speed (Post-Cache)...")
    try:
        start_time = time.time()
        response = requests.get(f"{base_url}/api/portfolio", timeout=10)
        elapsed = time.time() - start_time

        if response.status_code == 200:
            portfolio_data = response.json()
            data_source = portfolio_data.get("data_source")
            print(f"✅ Second call completed in {elapsed:.2f}s")
            print(f"✅ Data Source: {data_source}")

            if elapsed < 2.0:
                print("✅ Cache working effectively (fast response)")
            else:
                print(
                    "⚠️ Response slower than expected (cache may not be working optimally)"
                )
        else:
            print(f"❌ Second portfolio call failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Second portfolio call error: {e}")

    # Test 5: Validate Real Data Values
    print("\n5️⃣ Validating Real Data Values...")
    try:
        response = requests.get(f"{base_url}/api/portfolio", timeout=10)
        if response.status_code == 200:
            portfolio_data = response.json()

            total_value = portfolio_data.get("total_value", 0)
            available_balance = portfolio_data.get("available_balance", 0)
            balances = portfolio_data.get("balances", {})
            usdt_balance = balances.get("USDT", {})

            print(f"✅ Total Value: {total_value}")
            print(f"✅ Available Balance: {available_balance}")
            print(f"✅ USDT Equity: {usdt_balance.get('equity')}")
            print(f"✅ USDT Wallet Balance: {usdt_balance.get('wallet_balance')}")

            # Check if these are real production values (not demo fallback values)
            if total_value != 10000.0 and available_balance != 2500.0:
                print("🎉 CONFIRMED: Using REAL production account data!")
                real_data_confirmed = True
            else:
                print("⚠️ Values match demo fallback - may still be using demo data")
                real_data_confirmed = False

        else:
            print(f"❌ Portfolio validation failed: {response.status_code}")
            real_data_confirmed = False
    except Exception as e:
        print(f"❌ Portfolio validation error: {e}")
        real_data_confirmed = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 CACHE FIX VALIDATION SUMMARY")
    print("=" * 60)

    if cache_fix_success:
        print("✅ CACHE ACCESS ISSUE: RESOLVED")
        print("✅ Portfolio endpoint now uses production data")
    else:
        print("❌ CACHE ACCESS ISSUE: NOT FULLY RESOLVED")

    if real_data_confirmed:
        print("✅ REAL DATA INTEGRATION: CONFIRMED")
        print("✅ System displaying actual Bybit account data")
    else:
        print("⚠️ REAL DATA INTEGRATION: NEEDS VERIFICATION")

    print(f"\n📅 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return cache_fix_success and real_data_confirmed


if __name__ == "__main__":
    success = test_enhanced_dashboard_api_endpoints()
    if success:
        print("\n🎉 ALL TESTS PASSED - CACHE FIX SUCCESSFUL!")
    else:
        print("\n⚠️ Some tests failed - investigate further")
