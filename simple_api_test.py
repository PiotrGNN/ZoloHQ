#!/usr/bin/env python3
"""
Simple API test with CLI, retry, and SaaS-ready hooks
"""
import requests
import argparse
import time
import sys

DEFAULT_URL = "http://localhost:5001/health"


def test_simple(url, retries=3, delay=2):
    for attempt in range(1, retries + 1):
        try:
            print(f"Testing {url} (attempt {attempt})...")
            r = requests.get(url, timeout=3)
            print(f"Status: {r.status_code}")
            print(f"Text: {r.text}")
            # SaaS/monetization hook: send result to central monitoring/analytics
            return r.status_code == 200
        except Exception as e:
            print(f"Error: {e}")
            if attempt < retries:
                time.sleep(delay)
            else:
                # SaaS/monetization hook: log failure to central system
                return False


def main():
    parser = argparse.ArgumentParser(description="Simple API health check with retries and SaaS hooks.")
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="API endpoint to test")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries")
    parser.add_argument("--delay", type=int, default=2, help="Delay between retries (seconds)")
    args = parser.parse_args()
    success = test_simple(args.url, args.retries, args.delay)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
