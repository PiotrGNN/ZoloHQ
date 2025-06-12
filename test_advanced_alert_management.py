import unittest

from advanced_alert_management import AdvancedAlertManager


class TestAdvancedAlertManagement(unittest.TestCase):
    def setUp(self):
        self.manager = AdvancedAlertManager()

    def test_generate_system_alerts(self):
        alerts = self.manager._generate_system_alerts()
        self.assertIsInstance(alerts, list)
        # At least one alert if system is under load, else empty
        for alert in alerts:
            self.assertIn(alert["level"], ["critical", "warning"])
            self.assertIn("message", alert)

    def test_generate_risk_alerts(self):
        risk_data = {
            "current_leverage": 3.5,
            "max_leverage": 3.0,
            "drawdown_current": -16,
            "var_95": -6,
        }
        alerts = self.manager._generate_risk_alerts(risk_data)
        self.assertTrue(any(a["level"] == "critical" for a in alerts))
        self.assertTrue(any(a["level"] == "warning" for a in alerts))

    def test_generate_performance_alerts(self):
        perf_data = {"win_rate": 35, "net_profit": -2000}
        alerts = self.manager._generate_performance_alerts(perf_data)
        self.assertTrue(any(a["level"] == "warning" for a in alerts))
        self.assertTrue(any(a["level"] == "critical" for a in alerts))

    def test_alert_statistics(self):
        alerts = [
            {"level": "critical", "category": "system"},
            {"level": "warning", "category": "risk"},
            {"level": "info", "category": "performance"},
        ]
        stats = self.manager.get_alert_statistics(alerts)
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["critical"], 1)
        self.assertEqual(stats["warning"], 1)
        self.assertEqual(stats["info"], 1)
        self.assertIn("system", stats["categories"])


if __name__ == "__main__":
    unittest.main()
