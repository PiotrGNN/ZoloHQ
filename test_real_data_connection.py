#!/usr/bin/env python3
"""
Test script to verify real data connection after authentication fixes
"""

import logging
import os
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_environment_config():
    """Test environment configuration"""
    print("🔍 Testing Environment Configuration...")
    print("-" * 50)

    # Check critical environment variables
    config_items = [
        ("TRADING_MODE", os.getenv("TRADING_MODE")),
        ("BYBIT_PRODUCTION_ENABLED", os.getenv("BYBIT_PRODUCTION_ENABLED")),
        ("BYBIT_TESTNET", os.getenv("BYBIT_TESTNET")),
        ("BYBIT_API_KEY", "***" if os.getenv("BYBIT_API_KEY") else None),
        ("BYBIT_API_SECRET", "***" if os.getenv("BYBIT_API_SECRET") else None),
    ]

    all_good = True
    for key, value in config_items:
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")
        if not value:
            all_good = False

    print("\n" + "=" * 50)
    if all_good:
        print("✅ Environment configuration looks good for real data!")
    else:
        print("❌ Environment configuration has issues!")

    assert all_good


def test_bybit_connector():
    """Test Bybit connector with real data"""
    print("\n🔍 Testing Bybit Connector...")
    print("-" * 50)

    try:
        # Add path to find connector
        sys.path.append(str(Path(__file__).parent / "ZoL0-master"))
        from data.execution.bybit_connector import BybitConnector

        # Initialize connector
        api_key = os.getenv("BYBIT_API_KEY")
        api_secret = os.getenv("BYBIT_API_SECRET")
        use_testnet = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

        print(f"📡 Initializing connector (testnet: {use_testnet})...")
        connector = BybitConnector(
            api_key=api_key, api_secret=api_secret, use_testnet=use_testnet
        )

        # Test server time (public endpoint)
        print("🕒 Testing server time...")
        time_result = connector.get_server_time()
        if time_result.get("success"):
            print(f"✅ Server time: {time_result.get('server_time')}")
        else:
            print(f"❌ Server time failed: {time_result.get('error')}")

        # Test wallet balance (authenticated endpoint)
        print("💰 Testing wallet balance...")
        balance_result = connector.get_wallet_balance()
        if balance_result.get("success"):
            print("✅ Wallet balance retrieved successfully!")
            print(f"   Source: {balance_result.get('source', 'unknown')}")

            # Show some balance details
            balances = balance_result.get("balances", {})
            if isinstance(balances, dict) and "list" in balances:
                for account in balances["list"][:3]:  # Show first 3 accounts
                    account_type = account.get("accountType", "Unknown")
                    total_equity = account.get("totalEquity", "0")
                    print(f"   Account {account_type}: {total_equity} USD")
            elif isinstance(balances, list):
                for account in balances[:3]:  # Show first 3 accounts
                    print(f"   Balance entry: {account}")
            else:
                print(f"   Balance data: {balances}")
        else:
            print(f"❌ Wallet balance failed: {balance_result.get('error')}")

        assert True

    except Exception as e:
        print(f"❌ Connector test failed: {e}")
        logger.exception("Full error details:")
        raise AssertionError()


def test_production_data_manager():
    """Test production data manager"""
    print("\n🔍 Testing Production Data Manager...")
    print("-" * 50)

    try:
        from production_data_manager import ProductionDataManager

        manager = ProductionDataManager()

        # Check configuration
        print(f"📊 Production mode: {manager.is_production}")
        print(
            f"🔗 API credentials: {'✅' if manager.api_key and manager.api_secret else '❌'}"
        )
        # Test connection status
        connection_status = manager.connection_status
        print(f"🌐 Connection status: {connection_status}")

        # Test real data retrieval
        print("💹 Testing market data retrieval...")
        market_data = manager.get_market_data("BTCUSDT")
        if market_data:
            print("✅ Market data retrieved for BTCUSDT")
            print(f"   Source: {market_data.get('source', 'unknown')}")
            price = market_data.get("price", "N/A")
            print(f"   Current price: {price}")
        else:
            print("❌ Failed to retrieve market data")

        assert True

    except Exception as e:
        print(f"❌ Production data manager test failed: {e}")
        logger.exception("Full error details:")
        raise AssertionError()


def main():
    """Main test function"""
    print("🚀 TESTING REAL DATA CONNECTION")
    print("=" * 50)

    # Test environment
    env_ok = test_environment_config()

    # Test connector
    connector_ok = test_bybit_connector()

    # Test production manager
    manager_ok = test_production_data_manager()

    # Final summary
    print("\n" + "=" * 50)
    print("📋 FINAL TEST SUMMARY")
    print("=" * 50)

    tests = [
        ("Environment Config", env_ok),
        ("Bybit Connector", connector_ok),
        ("Production Data Manager", manager_ok),
    ]

    all_passed = True
    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED - REAL DATA IS WORKING!")
    else:
        print("⚠️ SOME TESTS FAILED - CHECK ISSUES ABOVE")

    assert all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
