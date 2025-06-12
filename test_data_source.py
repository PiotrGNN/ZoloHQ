import time

import requests

print("Testing Enhanced Dashboard API data source...")
start = time.time()

try:
    response = requests.get("http://localhost:5001/api/portfolio", timeout=20)
    elapsed = time.time() - start
    print(f"Response time: {elapsed:.2f}s")
    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f'Data source: {data.get("data_source", "unknown")}')
        print(f'Environment: {data.get("environment", "unknown")}')
        print(f'Connection status: {data.get("connection_status", {})}')
        print(f'Available balance: {data.get("available_balance", "N/A")}')

        # Check balances
        balances = data.get("balances", {})
        if balances:
            print("Balance details:")
            for currency, details in balances.items():
                print(f"  {currency}: {details}")

    else:
        print(f"Error response: {response.text}")

except requests.exceptions.Timeout:
    elapsed = time.time() - start
    print(f"Request timed out after {elapsed:.2f}s")
except Exception as e:
    elapsed = time.time() - start
    print(f"Error after {elapsed:.2f}s: {e}")
