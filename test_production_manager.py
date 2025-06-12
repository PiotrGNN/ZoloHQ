#!/usr/bin/env python3
"""Test the production data manager directly to verify authentication fixes."""

import sys
from datetime import datetime

# Add path
sys.path.append(".")


def test_production_data_manager():
    """Test production data manager with fixed authentication."""
    print(f"[{datetime.now()}] Testing Production Data Manager")
    print("=" * 60)

    try:
        from production_data_manager import ProductionDataManager

        print("📡 Initializing Production Data Manager...")
        manager = ProductionDataManager()

        print("✅ Production Data Manager initialized")

        # Test account balance
        print("\n💰 Testing account balance retrieval...")
        balance = manager.get_account_balance()

        if balance and balance.get("success"):
            print("✅ Account balance retrieved successfully!")
            print(f"   Data source: {balance.get('data_source', 'unknown')}")
            if "balances" in balance:
                print(f"   Balances: {balance['balances']}")
            assert True
        else:
            print("❌ Failed to get account balance")
            print(f"   Response: {balance}")
            raise AssertionError()

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback

        print(f"   Traceback: {traceback.format_exc()}")
        raise AssertionError()


if __name__ == "__main__":
    success = test_production_data_manager()
    if success:
        print("\n🎉 Production Data Manager test PASSED!")
    else:
        print("\n💥 Production Data Manager test FAILED!")
