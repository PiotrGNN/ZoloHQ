#!/usr/bin/env python3
"""
Quick test of Enhanced API portfolio endpoint
"""

import requests

print("Testing Enhanced API Portfolio Endpoint...")

try:
    response = requests.get("http://localhost:5001/api/portfolio", timeout=30)
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Data Source: {data.get('data_source')}")
        print(f"Environment: {data.get('environment')}")
        print(f"Total Value: {data.get('total_value')}")
        print(f"Available Balance: {data.get('available_balance')}")

        if data.get("data_source") == "production_api":
            print("🎉 SUCCESS: Using production data!")
        else:
            print(f"⚠️ Using: {data.get('data_source')}")
    else:
        print(f"❌ HTTP Error: {response.status_code}")

except Exception as e:
    print(f"❌ Error: {e}")
