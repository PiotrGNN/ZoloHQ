#!/usr/bin/env python3
"""
Test production manager with fixed authentication
"""

import time

try:
    print("🚀 Testing Fixed Production Manager")
    print("=" * 50)

    from production_data_manager import ProductionDataManager

    print("✅ Production manager imported")

    mgr = ProductionDataManager()
    print("✅ Production manager initialized")
    print(f"🔧 Production mode: {mgr.is_production}")
    print(f"🔗 API key available: {'Yes' if mgr.api_key else 'No'}")

    # Test connection
    print("\n🌐 Testing connection...")
    connected = mgr.connection_status.get("bybit", {}).get("connected", False)
    print(f"Connection status: {connected}")

    # Test account balance (the main issue)
    print("\n💰 Testing account balance (direct call)...")
    start_time = time.time()
    balance = mgr.get_account_balance(use_cache=False)
    elapsed = time.time() - start_time

    print(f"⏱️ Balance call took: {elapsed:.2f}s")
    print(f"📊 Balance success: {balance.get('success', False)}")
    print(f"📊 Return code: {balance.get('retCode', 'N/A')}")
    print(f"📊 Data source: {balance.get('data_source', 'unknown')}")

    if balance.get("success") and balance.get("data_source") in [
        "production_api",
        "testnet_api",
    ]:
        print("🎉 SUCCESS! Real production data is working!")
        if "balances" in balance:
            print(f"💰 Balances: {balance['balances']}")
    elif balance.get("data_source") == "fallback":
        print("⚠️ Using fallback data - API may have issues")
    else:
        print("❌ Something is wrong with the balance call")
        print(f"Full response: {balance}")

    # Test the enhanced portfolio (which was causing timeouts)
    print("\n📊 Testing enhanced portfolio...")
    start_time = time.time()
    portfolio = mgr.get_enhanced_portfolio_details(use_cache=False)
    elapsed = time.time() - start_time

    print(f"⏱️ Enhanced portfolio took: {elapsed:.2f}s")
    print(f"📊 Portfolio success: {portfolio.get('success', False)}")
    print(f"📊 Portfolio data source: {portfolio.get('data_source', 'unknown')}")
    print(f"📊 Environment: {portfolio.get('environment', 'unknown')}")

    if portfolio.get("data_source") == "production_api":
        print("🎉 Enhanced portfolio is using real production data!")
    else:
        print(f"⚠️ Enhanced portfolio using: {portfolio.get('data_source')}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
