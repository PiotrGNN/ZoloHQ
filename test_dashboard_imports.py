#!/usr/bin/env python3
"""
Test Dashboard Imports
"""

print("🔍 TESTING DASHBOARD IMPORTS")
print("=" * 50)

dashboards_to_test = [
    "unified_trading_dashboard",
    "enhanced_dashboard",
    "master_control_dashboard",
    "advanced_trading_analytics",
]

for dashboard in dashboards_to_test:
    try:
        print(f"\n📊 Testing {dashboard}...")
        exec(f"import {dashboard}")
        print(f"✅ {dashboard} imports successfully")
    except Exception as e:
        print(f"❌ {dashboard} import error: {e}")

print("\n🎯 Dashboard import tests completed!")
