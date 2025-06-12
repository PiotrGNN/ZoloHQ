import logging


def notify_system(message: str) -> None:
    """Send a notification to the system log."""
    logger = logging.getLogger("notification_system")
    try:
        logger.info(f"System notification: {message}")
    except Exception as e:
        logger.error(f"Notification system error: {e}")
