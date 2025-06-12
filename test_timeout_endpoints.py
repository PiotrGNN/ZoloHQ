#!/usr/bin/env python3
"""
Test the remaining timeout endpoints in Enhanced API
"""

import logging
import time

import requests


def test_timeout_endpoints():
    """Test the remaining Enhanced API endpoints that are timing out"""

    print("🔍 Testing Enhanced API Timeout Endpoints")
    print("=" * 50)

    base_url = "http://localhost:5001"

    endpoints = [
        ("/api/trading/statistics", "Trading Statistics"),
        ("/api/cache/init", "Cache Initialization"),
    ]

    for endpoint, name in endpoints:
        print(f"\n📊 Testing {name}...")
        print(f"   URL: {base_url}{endpoint}")

        start_time = time.time()
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=20)
            elapsed = time.time() - start_time

            print(f"   ⏱️ Response time: {elapsed:.2f}s")
            print(f"   📡 HTTP Status: {response.status_code}")

            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success: {data.get('success')}")

                if "data_source" in data:
                    print(f"   🎯 Data Source: {data.get('data_source')}")

                if "statistics" in data:
                    stats = data["statistics"]
                    print(f"   📈 Sample stats: {dict(list(stats.items())[:3])}")

                if "cache_size" in data:
                    print(f"   🗄️ Cache size: {data.get('cache_size')}")

                print(f"   🎉 {name} endpoint working!")

            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   📋 Error: {error_data.get('error', 'Unknown')}")
                except Exception:
                    logging.exception(
                        "Exception occurred in test_timeout_endpoints at line 57"
                    )
                    print(f"   📋 Response: {response.text[:100]}")

        except requests.exceptions.Timeout:
            elapsed = time.time() - start_time
            print(f"   ❌ Timeout after {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ❌ Error after {elapsed:.2f}s: {e}")


if __name__ == "__main__":
    test_timeout_endpoints()
