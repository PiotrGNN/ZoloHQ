#!/usr/bin/env python3
import os
import sys

print("🔥 ZoL0 SYSTEM STARTUP - TESTING...")
print("=" * 50)

# Set production environment
os.environ["BYBIT_PRODUCTION_CONFIRMED"] = "true"
os.environ["BYBIT_PRODUCTION_ENABLED"] = "true"

print("✅ Environment variables set for production")
print(f"📂 Current directory: {os.getcwd()}")
print(f"🐍 Python version: {sys.version}")

# Test if files exist
files_to_check = [
    "master_control_dashboard.py",
    "enhanced_dashboard_api.py",
    "ZoL0-master/dashboard_api.py",
]

print("\n📁 Checking required files:")
for file in files_to_check:
    if os.path.exists(file):
        print(f"✅ {file} - EXISTS")
    else:
        print(f"❌ {file} - MISSING")

print("\n🚀 READY TO LAUNCH SYSTEM!")
print("Run manually: python EMERGENCY_LAUNCH.py")
print("Or double-click: LAUNCH_ZOL0_SYSTEM.bat")
print("=" * 50)
