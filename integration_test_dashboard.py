#!/usr/bin/env python3
"""
Integration Test for Enhanced Dashboard and Core System
Test integracyjny dla rozszerzonego dashboard i systemu core
"""

import json
from datetime import datetime

import requests


class DashboardIntegrationTest:
    def __init__(self):
        self.api_base = "http://localhost:5001"
        self.dashboard_url = "http://localhost:8502"
        self.results = {}

    def test_api_endpoints(self):
        """Test wszystkich endpointów API"""
        print("🧪 Testing API Endpoints...")

        endpoints = [
            "/core/status",
            "/core/strategies",
            "/core/ai-models",
            "/core/system-metrics",
            "/core/health",
            "/api/trading-signals",
            "/api/portfolio",
        ]

        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.api_base}{endpoint}", timeout=5)
                if response.status_code == 200:
                    self.results[endpoint] = {
                        "status": "✅ PASS",
                        "response_time": response.elapsed.total_seconds(),
                        "data_size": len(response.content),
                    }
                    print(
                        f"  ✅ {endpoint} - OK ({response.elapsed.total_seconds():.3f}s)"
                    )
                else:
                    self.results[endpoint] = {
                        "status": "❌ FAIL",
                        "error": f"HTTP {response.status_code}",
                    }
                    print(f"  ❌ {endpoint} - HTTP {response.status_code}")
            except Exception as e:
                self.results[endpoint] = {"status": "❌ ERROR", "error": str(e)}
                print(f"  ❌ {endpoint} - {str(e)}")

    def test_core_system_health(self):
        """Test zdrowia systemu core"""
        print("\n🏥 Testing Core System Health...")

        try:
            health = requests.get(f"{self.api_base}/core/health").json()

            print(f"  📊 Overall Status: {health['overall_status']}")
            print(f"  📈 Health Percentage: {health['health_percentage']}%")
            print(
                f"  🔧 Components Healthy: {health['components_healthy']}/{health['total_components']}"
            )

            for component, status in health["details"].items():
                status_icon = "✅" if status else "❌"
                print(f"    {status_icon} {component}")

            self.results["health_check"] = health

        except Exception as e:
            print(f"  ❌ Health check failed: {e}")
            self.results["health_check"] = {"error": str(e)}

    def test_strategy_integration(self):
        """Test integracji strategii"""
        print("\n🎯 Testing Strategy Integration...")

        try:
            strategies = requests.get(f"{self.api_base}/core/strategies").json()

            print(f"  📊 Active Strategies: {strategies['count']}")
            print(
                f"  ✅ Load Status: {'Success' if strategies['loaded_successfully'] else 'Failed'}"
            )

            for strategy in strategies["strategies"]:
                print(f"    🔄 {strategy}")

            self.results["strategies"] = strategies

        except Exception as e:
            print(f"  ❌ Strategy test failed: {e}")
            self.results["strategies"] = {"error": str(e)}

    def test_ai_models_integration(self):
        """Test integracji modeli AI"""
        print("\n🤖 Testing AI Models Integration...")

        try:
            ai_models = requests.get(f"{self.api_base}/core/ai-models").json()

            print(f"  📊 Total Models: {ai_models['total_models']}")
            print(
                f"  🧠 RL Trader: {'Available' if ai_models['rl_trader_available'] else 'Not Available'}"
            )
            print(f"  ⚡ Status: {ai_models['status']}")

            # Test komponentów
            for component, status in ai_models["components"].items():
                status_icon = "✅" if status == "active" else "❌"
                print(f"    {status_icon} {component}: {status}")

            self.results["ai_models"] = ai_models

        except Exception as e:
            print(f"  ❌ AI Models test failed: {e}")
            self.results["ai_models"] = {"error": str(e)}

    def test_system_performance(self):
        """Test wydajności systemu"""
        print("\n⚡ Testing System Performance...")

        try:
            metrics = requests.get(f"{self.api_base}/core/system-metrics").json()

            print(f"  💻 CPU Usage: {metrics['cpu_percent']:.1f}%")
            print(f"  🧠 Memory Usage: {metrics['memory']['percent']:.1f}%")
            print(f"  💽 Disk Usage: {metrics['disk']['percent']:.1f}%")
            print(f"  🔄 Active Processes: {metrics['processes']}")

            # Performance warnings
            if metrics["cpu_percent"] > 80:
                print("  ⚠️  High CPU usage detected!")
            if metrics["memory"]["percent"] > 80:
                print("  ⚠️  High memory usage detected!")
            if metrics["disk"]["percent"] > 90:
                print("  ⚠️  Low disk space!")

            self.results["performance"] = metrics

        except Exception as e:
            print(f"  ❌ Performance test failed: {e}")
            self.results["performance"] = {"error": str(e)}

    def generate_report(self):
        """Generuj raport końcowy"""
        print("\n" + "=" * 60)
        print("📋 INTEGRATION TEST REPORT")
        print("=" * 60)

        passed_tests = 0
        total_tests = 0

        for _test_name, result in self.results.items():
            total_tests += 1
            if isinstance(result, dict) and result.get("status") == "✅ PASS":
                passed_tests += 1
            elif isinstance(result, dict) and "error" not in result:
                passed_tests += 1

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        print(f"✅ Tests Passed: {passed_tests}/{total_tests}")
        print(f"📊 Success Rate: {success_rate:.1f}%")
        print(f"🕒 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        if success_rate >= 90:
            print("🎉 EXCELLENT! Core system integration is working perfectly!")
        elif success_rate >= 70:
            print("✅ GOOD! Core system integration is mostly working.")
        else:
            print("⚠️  WARNING! Core system has integration issues.")

        # Save detailed report
        with open("integration_test_report.json", "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "success_rate": success_rate,
                    "passed_tests": passed_tests,
                    "total_tests": total_tests,
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print("\n📄 Detailed report saved to: integration_test_report.json")

    def run_all_tests(self):
        """Uruchom wszystkie testy"""
        print("🚀 Starting Core System Integration Tests...")
        print(f"🕒 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)

        self.test_api_endpoints()
        self.test_core_system_health()
        self.test_strategy_integration()
        self.test_ai_models_integration()
        self.test_system_performance()
        self.generate_report()


if __name__ == "__main__":
    tester = DashboardIntegrationTest()
    tester.run_all_tests()
