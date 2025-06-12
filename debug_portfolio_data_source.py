#!/usr/bin/env python3
"""
Debug script to check why Enhanced API is returning fallback data
"""

import logging

from production_data_manager import get_production_data

logging.basicConfig(level=logging.INFO)


def test_data_sources():
    """Test production data manager methods to identify data source issues"""

    print("=== Testing Production Data Manager Data Sources ===")

    mgr = get_production_data()
    print(f"Manager initialized: {mgr is not None}")

    # Test 1: Account balance method (working in previous tests)
    print("\n1️⃣ Testing account balance...")
    try:
        balance = mgr.get_account_balance(use_cache=False)
        print(f"✅ Balance success: {balance.get('success')}")
        print(f"✅ Balance data source: {balance.get('data_source')}")
        print(f"✅ Balance environment: {balance.get('environment')}")
        if balance.get("balances"):
            usdt = balance["balances"].get("USDT", {})
            print(f"✅ USDT Balance: {usdt.get('wallet_balance')}")
    except Exception as e:
        print(f"❌ Balance error: {e}")

    # Test 2: Enhanced portfolio details method (called by portfolio_data)
    print("\n2️⃣ Testing enhanced portfolio details...")
    try:
        portfolio = mgr.get_enhanced_portfolio_details(use_cache=False)
        print(f"✅ Enhanced portfolio success: {portfolio.get('success')}")
        print(f"✅ Enhanced portfolio data source: {portfolio.get('data_source')}")
        print(f"✅ Enhanced portfolio environment: {portfolio.get('environment')}")
    except Exception as e:
        print(f"❌ Enhanced portfolio error: {e}")
        import traceback

        traceback.print_exc()

    # Test 3: Portfolio data method (called by Enhanced API)
    print("\n3️⃣ Testing portfolio data (Enhanced API method)...")
    try:
        portfolio_api = mgr.get_portfolio_data(use_cache=False)
        print(f"✅ Portfolio API success: {portfolio_api.get('success')}")
        print(f"✅ Portfolio API data source: {portfolio_api.get('data_source')}")
        print(f"✅ Portfolio API environment: {portfolio_api.get('environment')}")
        print(f"✅ Portfolio API total value: {portfolio_api.get('total_value')}")
    except Exception as e:
        print(f"❌ Portfolio API error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_data_sources()
