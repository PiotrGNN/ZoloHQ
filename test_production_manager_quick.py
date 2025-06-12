#!/usr/bin/env python3
"""
Quick test of production manager
"""


try:
    print("Starting test...")
    from production_data_manager import ProductionDataManager

    print("Import successful")

    mgr = ProductionDataManager()
    print("✅ Production Manager loaded")
    print(f"🔧 Production mode: {mgr.is_production}")
    print(f'🔗 API key available: {"Yes" if mgr.api_key else "No"}')
    print(f'🔗 API secret available: {"Yes" if mgr.api_secret else "No"}')

    # Test connection
    status = mgr.connection_status
    print(f'🌐 Connection status: {status.get("bybit", {}).get("connected", False)}')

    # Test real data
    print("Testing balance...")
    balance = mgr.get_account_balance(use_cache=False)
    print(f'💰 Balance test success: {balance.get("success", False)}')
    print(f'📊 Balance data source: {balance.get("data_source", "unknown")}')

    if balance.get("success"):
        print("🎉 PRODUCTION DATA MANAGER IS WORKING WITH REAL DATA!")
    else:
        print("⚠️ Production Data Manager fallback to demo data")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
