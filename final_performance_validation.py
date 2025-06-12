#!/usr/bin/env python3
"""
ZoL0 Performance Monitoring System - Final Validation
======================================================

Ostateczna walidacja wszystkich komponentów systemu monitorowania wydajności.
"""

import sys
from datetime import datetime

import requests


def test_api_availability():
    """Test Enhanced Dashboard API availability"""
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Enhanced Dashboard API: OPERATIONAL")
            return True
        else:
            print(f"⚠️ Enhanced Dashboard API: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Enhanced Dashboard API: NOT AVAILABLE - {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring components"""
    try:
        from advanced_performance_monitor import get_performance_monitor

        monitor = get_performance_monitor()

        # Test recording functionality
        monitor.record_api_call("/test", "GET", 0.5, True)

        # Test summary generation
        summary = monitor.get_performance_summary(hours=1)

        if summary and "total_api_calls" in summary:
            print("✅ Performance Monitor: OPERATIONAL")
            return True
        else:
            print("⚠️ Performance Monitor: Limited functionality")
            return False

    except Exception as e:
        print(f"❌ Performance Monitor: ERROR - {e}")
        return False


def test_cache_system():
    """Test intelligent cache system"""
    try:
        from api_cache_system import get_cache_instance

        cache = get_cache_instance()

        # Test cache operations
        cache.set("test_key", {"test": "data"})
        data, cache_hit = cache.get("test_key")

        if data and data.get("test") == "data" and cache_hit:
            print("✅ Intelligent Cache: OPERATIONAL")
            return True
        else:
            print("⚠️ Intelligent Cache: Limited functionality")
            return False

    except Exception as e:
        print(f"❌ Intelligent Cache: ERROR - {e}")
        return False


def test_production_monitor():
    """Test production monitoring system"""
    try:
        from production_usage_monitor import get_production_monitor

        monitor = get_production_monitor()

        # Test monitoring functionality
        monitor.record_api_request("/test", "GET", 0.3, True, False)

        dashboard_data = monitor.get_production_dashboard_data()

        if dashboard_data and "realtime_metrics" in dashboard_data:
            print("✅ Production Monitor: OPERATIONAL")
            return True
        else:
            print("⚠️ Production Monitor: Limited functionality")
            return False

    except Exception as e:
        print(f"❌ Production Monitor: ERROR - {e}")
        return False


def test_rate_limiter():
    """Test adaptive rate limiter"""
    try:
        from advanced_rate_limit_optimizer import get_adaptive_rate_limiter

        limiter = get_adaptive_rate_limiter()

        # Test rate limiting
        allowed, wait_time = limiter.can_make_request("/test")

        if allowed is not None:
            print("✅ Adaptive Rate Limiter: OPERATIONAL")
            return True
        else:
            print("⚠️ Adaptive Rate Limiter: Limited functionality")
            return False

    except Exception as e:
        print(f"❌ Adaptive Rate Limiter: ERROR - {e}")
        return False


def test_dashboard_integration():
    """Test dashboard integration"""
    try:
        from enhanced_performance_dashboard import PerformanceDashboard

        dashboard = PerformanceDashboard()

        if dashboard:
            print("✅ Performance Dashboard: OPERATIONAL")
            return True
        else:
            print("⚠️ Performance Dashboard: Limited functionality")
            return False

    except Exception as e:
        print(f"❌ Performance Dashboard: ERROR - {e}")
        return False


def test_production_integration():
    """Test production system integration"""
    try:
        from production_performance_integration import get_production_system

        system = get_production_system()

        if system:
            print("✅ Production Integration: OPERATIONAL")
            return True
        else:
            print("⚠️ Production Integration: Limited functionality")
            return False

    except Exception as e:
        print(f"❌ Production Integration: ERROR - {e}")
        return False


def main():
    """Run final validation tests"""
    print("🎯 ZoL0 Performance Monitoring System - Final Validation")
    print("=" * 60)
    print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    tests = [
        ("Enhanced Dashboard API", test_api_availability),
        ("Performance Monitor", test_performance_monitoring),
        ("Intelligent Cache", test_cache_system),
        ("Production Monitor", test_production_monitor),
        ("Rate Limiter", test_rate_limiter),
        ("Dashboard Integration", test_dashboard_integration),
        ("Production Integration", test_production_integration),
    ]

    results = []
    passed = 0
    total = len(tests)

    print("🔍 Running Component Tests...")
    print("-" * 40)

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}: CRASH - {e}")
            results.append((test_name, False))

    print("")
    print("📊 FINAL VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"📈 Success Rate: {(passed/total)*100:.1f}%")

    if passed == total:
        print("🎉 STATUS: ALL SYSTEMS OPERATIONAL")
        overall_status = "OPERATIONAL"
    elif passed >= total * 0.8:
        print("✅ STATUS: MOSTLY OPERATIONAL")
        overall_status = "MOSTLY_OPERATIONAL"
    elif passed >= total * 0.5:
        print("⚠️ STATUS: PARTIALLY OPERATIONAL")
        overall_status = "PARTIALLY_OPERATIONAL"
    else:
        print("❌ STATUS: CRITICAL ISSUES")
        overall_status = "CRITICAL"

    print("")
    print("📋 Component Status:")
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")

    print("")
    print("🏆 DEPLOYMENT READINESS")
    print("-" * 30)

    if overall_status == "OPERATIONAL":
        print("🚀 Ready for Production Deployment")
        print("🎯 All monitoring systems active")
        print("📈 Performance optimization enabled")
    elif overall_status in ["MOSTLY_OPERATIONAL", "PARTIALLY_OPERATIONAL"]:
        print("⚠️ Ready for Testing Environment")
        print("🔧 Some components need attention")
        print("📊 Basic monitoring functional")
    else:
        print("🚨 System needs significant repairs")
        print("🛠️ Multiple components failing")
        print("❌ Not ready for deployment")

    print("")
    print("=" * 60)
    print("🕐 Validation completed successfully")

    return overall_status == "OPERATIONAL"


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n💥 Validation crashed: {e}")
        sys.exit(3)
