#!/usr/bin/env python3
"""
Debug Bybit Balance API Issue
"""
import os
import sys
import traceback

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the ZoL0-master directory to the path
zol0_path = os.path.join(os.path.dirname(__file__), "ZoL0-master")
if zol0_path not in sys.path:
    sys.path.insert(0, zol0_path)

from data.execution.bybit_connector import BybitConnector


def debug_balance():
    print("🔍 DEBUGGING BALANCE API ISSUE")
    print("=" * 50)

    try:
        connector = BybitConnector()
        print("✅ BybitConnector initialized")

        print("\n🔎 Calling get_account_balance()...")
        balance = connector.get_account_balance()
        print(f"✅ Balance result: {balance}")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()


if __name__ == "__main__":
    debug_balance()
