#!/usr/bin/env python3
"""
Detailed production manager test
"""

import json

try:
    print("Starting detailed test...")
    from production_data_manager import ProductionDataManager

    mgr = ProductionDataManager()
    print("✅ Production Manager loaded")
    print(f"🔧 Production mode: {mgr.is_production}")

    # Test real data with detailed output
    print("\n=== Testing Account Balance ===")
    balance = mgr.get_account_balance(use_cache=False)

    print("Raw balance response:")
    print(json.dumps(balance, indent=2, default=str))

    print(
        f"\nBalance keys: {list(balance.keys()) if isinstance(balance, dict) else 'Not a dict'}"
    )
    print(f"Success field: {balance.get('success')}")
    print(f"Data source: {balance.get('data_source')}")
    print(f"RetCode: {balance.get('retCode')}")

    # Test the connector directly
    print("\n=== Testing Bybit Connector Direct ===")
    if hasattr(mgr, "bybit_connector") and mgr.bybit_connector:
        try:
            direct_balance = mgr.bybit_connector.get_account_balance()
            print("Direct connector response:")
            print(json.dumps(direct_balance, indent=2, default=str))
        except Exception as e:
            print(f"Direct connector error: {e}")
    else:
        print("No bybit_connector available")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
