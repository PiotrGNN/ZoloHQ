#!/usr/bin/env python3
"""
Test API connections for master control dashboard
"""
import requests


def test_api_endpoints():
    """Test all API endpoints used by master control dashboard"""

    endpoints = [
        "http://localhost:5000/api/portfolio",
        "http://localhost:5001/api/portfolio",
        "http://localhost:5001/api/trading/statistics",
        "http://localhost:5001/health",
    ]

    for endpoint in endpoints:
        try:
            print(f"\n🔍 Testing: {endpoint}")
            response = requests.get(endpoint, timeout=5)
            print(f"✅ Status: {response.status_code}")

            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"📊 Data type: {type(data)}")
                    if isinstance(data, dict):
                        print(f"🔑 Keys: {list(data.keys())[:5]}")
                    else:
                        print(f"📄 Content: {str(data)[:200]}")
                except Exception:
                    print(f"📄 Text: {response.text[:200]}")
            else:
                print(f"❌ Error: {response.text[:200]}")

        except Exception as e:
            print(f"❌ Connection failed: {e}")


if __name__ == "__main__":
    test_api_endpoints()
