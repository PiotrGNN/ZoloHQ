import requests

try:
    print("Testing health endpoint...")
    response = requests.get("http://localhost:4001/health", timeout=5)
    print(f"Health Status: {response.status_code}")
    print(f"Health Data: {response.json()}")

    print("\nTesting portfolio endpoint (quick timeout)...")
    response = requests.get("http://localhost:4001/api/portfolio", timeout=5)
    print(f"Portfolio Status: {response.status_code}")
    data = response.json()
    print(f'Data source: {data.get("data_source", "unknown")}')

except requests.exceptions.Timeout:
    print("Request timed out")
except Exception as e:
    print(f"Error: {e}")
