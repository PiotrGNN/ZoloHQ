#!/usr/bin/env python3
"""
Quick test of direct API call without timeout wrapper
"""


try:
    print("Testing direct API call...")
    from production_data_manager import ProductionDataManager

    print("Import successful")

    mgr = ProductionDataManager()
    print("✅ Production Manager loaded")

    # Test direct API call without timeout wrapper
    print("Testing direct balance call...")
    if mgr.bybit_connector:
        result = mgr.bybit_connector.get_account_balance()
        print(
            f'✅ Direct balance call result: {result.get("retCode")} - {result.get("retMsg")}'
        )
        if result.get("retCode") == 0:
            print("🎉 DIRECT API CALL WORKS!")

            # Transform the result like production manager does
            transformed = mgr._transform_bybit_balance_response(result)
            print(f'✅ Transformed result: {transformed.get("success")}')
            print(f'✅ Data source: {transformed.get("data_source")}')
        else:
            print(f"❌ API call failed: {result}")
    else:
        print("❌ No connector available")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
