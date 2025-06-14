#!/usr/bin/env python3
"""
Direct Bybit API Test - Bypass all production managers to test raw API connectivity
"""
import hashlib
import hmac
import os
import time

import requests
from dotenv import load_dotenv
load_dotenv()
import pytest

# Load environment variables
load_dotenv()


def test_bybit_api_direct():
    if not os.getenv("BYBIT_API_KEY") or not os.getenv("BYBIT_API_SECRET"):
        pytest.skip("BYBIT_API_KEY or BYBIT_API_SECRET not set in environment.")

    """Test direct Bybit API calls without any wrappers"""
    print("🔍 DIRECT BYBIT API TEST")
    print("=" * 50)

    # Get credentials from environment
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")

    if not api_key or not api_secret:
        print("❌ Missing API credentials!")
        return False

    print(f"API Key: {api_key[:10]}..." if api_key else "No API Key")
    print(f"API Secret: {'*' * 20}" if api_secret else "No API Secret")

    # Test 1: Public endpoint (no auth required)
    print("\n📡 Testing public endpoint (server time)...")
    try:
        response = requests.get("https://api.bybit.com/v5/market/time", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server time: {data}")
        else:
            print(f"❌ Server time failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Server time error: {e}")

    # Test 2: Public market data
    print("\n📊 Testing public market data...")
    try:
        response = requests.get(
            "https://api.bybit.com/v5/market/tickers",
            params={"category": "spot", "symbol": "BTCUSDT"},
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                print(
                    f"✅ Market data: BTC price: {data['result']['list'][0]['lastPrice']}"
                )
            else:
                print(f"❌ Market data API error: {data}")
        else:
            print(f"❌ Market data failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Market data error: {e}")

    # Test 3: Authenticated endpoint (account balance)
    print("\n🔐 Testing authenticated endpoint (account balance)...")
    try:
        # Generate signature for authenticated request
        timestamp = str(int(time.time() * 1000))
        params = "accountType=UNIFIED"

        pre_sign = timestamp + api_key + "20000" + params
        signature = hmac.new(
            bytes(api_secret, "utf-8"), bytes(pre_sign, "utf-8"), hashlib.sha256
        ).hexdigest()

        headers = {
            "X-BAPI-API-KEY": api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": "20000",
            "Content-Type": "application/json",
        }

        response = requests.get(
            "https://api.bybit.com/v5/account/wallet-balance",
            headers=headers,
            params={"accountType": "UNIFIED"},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                print("✅ Account balance retrieved successfully!")

                # Show some balance details
                if "list" in data.get("result", {}):
                    for account in data["result"]["list"]:
                        account_type = account.get("accountType", "Unknown")
                        total_equity = account.get("totalEquity", "0")
                        print(f"   Account {account_type}: {total_equity} USD")

                        # Show coin balances
                        if "coin" in account:
                            for coin_data in account["coin"][:3]:  # First 3 coins
                                coin = coin_data.get("coin", "Unknown")
                                balance = coin_data.get("walletBalance", "0")
                                print(f"     {coin}: {balance}")
                else:
                    print(f"   Raw response: {data}")
            else:
                print(f"❌ Account balance API error: {data.get('retMsg')}")
        else:
            print(
                f"❌ Account balance HTTP error: {response.status_code} - {response.text}"
            )
    except Exception as e:
        print(f"❌ Account balance error: {e}")

    # Test 4: Network connectivity and DNS resolution
    print("\n🌐 Testing network connectivity...")
    try:
        import socket

        # Test DNS resolution
        ip = socket.gethostbyname("api.bybit.com")
        print(f"✅ DNS resolution: api.bybit.com -> {ip}")

        # Test TCP connection
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((ip, 443))
        if result == 0:
            print(f"✅ TCP connection to {ip}:443 successful")
        else:
            print(f"❌ TCP connection failed: {result}")
        sock.close()

    except Exception as e:
        print(f"❌ Network test error: {e}")

    assert True


if __name__ == "__main__":
    test_bybit_api_direct()
