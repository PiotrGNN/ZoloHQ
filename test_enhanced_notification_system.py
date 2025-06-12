import unittest
from datetime import datetime

from enhanced_notification_system import EnhancedNotificationManager, NotificationConfig


class TestEnhancedNotificationSystem(unittest.TestCase):
    def setUp(self):
        self.config = NotificationConfig(
            email_enabled=False, sms_enabled=False, slack_enabled=False
        )
        self.manager = EnhancedNotificationManager(self.config)

    def test_should_send_notification(self):
        alert = {
            "level": "critical",
            "category": "system",
            "timestamp": datetime.now().isoformat(),
        }
        self.assertTrue(self.manager.should_send_notification(alert))
        # Cooldown logic
        self.manager.last_notifications = {"system_critical": datetime.now()}
        self.assertFalse(self.manager.should_send_notification(alert))

    def test_format_alert_message(self):
        alert = {
            "level": "warning",
            "category": "risk",
            "message": "Test alert",
            "timestamp": datetime.now().isoformat(),
        }
        text = self.manager.format_alert_message(alert, "text")
        html = self.manager.format_alert_message(alert, "html")
        self.assertIn("Test alert", text)
        self.assertIn("Test alert", html)

    def test_send_notification_disabled(self):
        alert = {
            "level": "critical",
            "category": "system",
            "message": "Test",
            "timestamp": datetime.now().isoformat(),
        }
        result = self.manager.send_notification(alert)
        self.assertEqual(result, {"email": False, "sms": False, "slack": False})


if __name__ == "__main__":
    unittest.main()
