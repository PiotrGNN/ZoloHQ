"""
Enhanced Notification System for ZoL0 Advanced Monitoring Suite
Integrates email and SMS notifications with the existing alert management system.
"""

import json
import os
import smtplib
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List

import requests
import streamlit as st
import structlog
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
import sentry_sdk
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as aioredis
from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import RequestValidationError
from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse


# --- Sentry Initialization ---
sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN", ""),
    traces_sample_rate=1.0,
    environment=os.environ.get("SENTRY_ENV", "development"),
)

# --- Structlog Configuration ---
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger("enhanced_notification_system")

# --- OpenTelemetry Tracing ---
tracer_provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "zol0-enhanced-notification-system"}))
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
tracer_provider.add_span_processor(span_processor)

# --- FastAPI App with Security, CORS, GZip, HTTPS, Session, Rate Limiting ---
notification_api = FastAPI(
    title="Enhanced Notification System API",
    version="2.0-maximal",
    description="Comprehensive, observable, and secure enhanced notification and alerting API.",
    contact={"name": "ZoL0 Engineering", "email": "support@zol0.ai"},
    openapi_tags=[
        {"name": "notification", "description": "Notification endpoints"},
        {"name": "ci", "description": "CI/CD and test endpoints"},
        {"name": "info", "description": "Info endpoints"},
    ],
)

# --- Middleware ---
notification_api.add_middleware(GZipMiddleware, minimum_size=1000)
notification_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
notification_api.add_middleware(HTTPSRedirectMiddleware)
notification_api.add_middleware(TrustedHostMiddleware, allowed_hosts=["*", ".zol0.ai"])
notification_api.add_middleware(SessionMiddleware, secret_key=os.environ.get("SESSION_SECRET", "supersecret"))
notification_api.add_middleware(SentryAsgiMiddleware)

# --- Rate Limiting Initialization ---
@notification_api.on_event("startup")
async def startup_event() -> None:
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    redis = await aioredis.from_url(redis_url, encoding="utf8", decode_responses=True)
    await FastAPILimiter.init(redis)

# --- Instrumentation ---
FastAPIInstrumentor.instrument_app(notification_api)
LoggingInstrumentor().instrument(set_logging_format=True)

# --- Security Headers Middleware ---
from starlette.middleware.base import BaseHTTPMiddleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "no-referrer-when-downgrade"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
        return response
notification_api.add_middleware(SecurityHeadersMiddleware)

# --- Pydantic Models with OpenAPI Examples and Validators ---
class NotificationRequest(BaseModel):
    """Request model for sending notifications."""
    recipient: str = Field(..., example="user@example.com", description="Notification recipient.")
    message: str = Field(..., example="Critical system alert", description="Notification message.")
    channel: str = Field(..., example="email", description="Notification channel (email, sms, slack, etc.)")

class HealthResponse(BaseModel):
    status: str = Field(example="ok")
    ts: str = Field(example="2025-06-14T12:00:00Z")

# --- Robust Error Handling: Global Exception Handler with Logging, Tracing, Sentry ---
@notification_api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception", error=str(exc), path=str(request.url))
    sentry_sdk.capture_exception(exc)
    with tracer.start_as_current_span("global_exception_handler"):
        return JSONResponse(status_code=500, content={"error": str(exc)})

@notification_api.exception_handler(FastAPIRequestValidationError)
async def validation_exception_handler(request: Request, exc: FastAPIRequestValidationError) -> JSONResponse:
    logger.error("Validation error", error=str(exc), path=str(request.url))
    sentry_sdk.capture_exception(exc)
    with tracer.start_as_current_span("validation_exception_handler"):
        return JSONResponse(status_code=422, content={"error": str(exc)})

# --- CI/CD Test Endpoint ---
@notification_api.get("/api/ci/test", tags=["ci"])
async def api_ci_test() -> Dict[str, str]:
    """CI/CD pipeline test endpoint."""
    logger.info("CI/CD test endpoint hit")
    return {"ci": "ok"}


@dataclass
class NotificationConfig:
    """Configuration for notification services"""

    # Email settings
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_user: str = ""
    email_password: str = ""

    # SMS settings (using Twilio)
    twilio_sid: str = ""
    twilio_token: str = ""
    twilio_phone: str = ""

    # Slack settings
    slack_webhook_url: str = ""

    # Recipients
    email_recipients: List[str] = None
    sms_recipients: List[str] = None

    # Notification rules
    email_enabled: bool = False
    sms_enabled: bool = False
    slack_enabled: bool = False
    min_severity: str = "warning"  # Only send notifications for warning and above
    cooldown_minutes: int = 5  # Minimum time between notifications of same type

    def __post_init__(self):
        if self.email_recipients is None:
            self.email_recipients = []
        if self.sms_recipients is None:
            self.sms_recipients = []


class SlackNotifier:
    def __init__(self, url=os.getenv("SLACK_WEBHOOK_URL")):
        if not url:
            raise ValueError("SLACK_WEBHOOK_URL not set")
        self.url = url

    def send(self, text, severity="INFO"):
        payload = {"text": f"[{severity}] {text}"}
        r = requests.post(self.url, json=payload, timeout=5)
        r.raise_for_status()


class PagerDutyNotifier:
    def __init__(self, integration_key=os.getenv("PAGERDUTY_INTEGRATION_KEY")):
        if not integration_key:
            # Placeholder for PagerDuty integration
            raise ValueError("PAGERDUTY_INTEGRATION_KEY not set")
        self.integration_key = integration_key

    def send(self, text, severity="CRITICAL"):
        # Implement PagerDuty Events V2 API call
        url = "https://events.pagerduty.com/v2/enqueue"
        payload = {
            "routing_key": self.integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": text,
                "severity": severity.lower(),
                "source": "zol0-notification-system",
                "component": "alertmanager",
                "group": "zol0",
                "class": "infrastructure",
            },
        }
        headers = {"Content-Type": "application/json"}
        r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=5)
        r.raise_for_status()


# Example: trigger PagerDuty if severity==CRITICAL and no ACK after 5 min
class AlertManagerWithPagerDuty:
    def __init__(self, slack_notifier, pagerduty_notifier):
        self.slack_notifier = slack_notifier
        self.pagerduty_notifier = pagerduty_notifier
        self.acks = {}

    def send_alert(self, text, severity="INFO", alert_id=None):
        self.slack_notifier.send(text, severity)
        if severity == "CRITICAL":

            def escalate():
                time.sleep(300)  # 5 min
                if not self.acks.get(alert_id):
                    self.pagerduty_notifier.send(f"No ACK for alert: {text}", severity)

            t = threading.Thread(target=escalate)
            t.start()

    def ack_alert(self, alert_id):
        self.acks[alert_id] = True


class EnhancedNotificationManager:
    """Enhanced notification manager with email and SMS capabilities"""

    def __init__(self, config: NotificationConfig = None):
        self.config = config or NotificationConfig()
        self.last_notifications = {}  # Track last notification times for cooldown

    def should_send_notification(self, alert: Dict[str, Any]) -> bool:
        """Determine if notification should be sent based on rules"""
        # Check severity level
        severity_order = {"info": 0, "success": 1, "warning": 2, "critical": 3}
        alert_severity = severity_order.get(alert.get("level", "info"), 0)
        min_severity = severity_order.get(self.config.min_severity, 2)

        if alert_severity < min_severity:
            return False

        # Check cooldown
        alert_key = f"{alert.get('category', 'unknown')}_{alert.get('level', 'info')}"
        now = datetime.now()

        if alert_key in self.last_notifications:
            last_time = self.last_notifications[alert_key]
            if (now - last_time).total_seconds() < (self.config.cooldown_minutes * 60):
                return False

        self.last_notifications[alert_key] = now
        return True

    def format_alert_message(
        self, alert: Dict[str, Any], format_type: str = "text"
    ) -> str:
        """Format alert message for different notification types"""
        level = alert.get("level", "info").upper()
        message = alert.get("message", "Unknown alert")
        category = alert.get("category", "system").title()
        timestamp = alert.get("timestamp", datetime.now().isoformat())

        # Format timestamp
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            time_str = timestamp

        if format_type == "html":
            severity_colors = {
                "CRITICAL": "#ff4444",
                "WARNING": "#ffaa00",
                "INFO": "#4488ff",
                "SUCCESS": "#44ff44",
            }
            color = severity_colors.get(level, "#666666")

            return f"""
            <div style="border-left: 4px solid {color}; padding: 10px; margin: 10px 0;">
                <h3 style="color: {color}; margin: 0;">🚨 ZoL0 Trading Bot Alert</h3>
                <p><strong>Level:</strong> {level}</p>
                <p><strong>Category:</strong> {category}</p>
                <p><strong>Message:</strong> {message}</p>
                <p><strong>Time:</strong> {time_str}</p>
            </div>
            """
        else:
            return f"""
🚨 ZoL0 Trading Bot Alert

Level: {level}
Category: {category}
Message: {message}
Time: {time_str}

This is an automated notification from your ZoL0 trading bot monitoring system.
            """.strip()

    def send_email_notification(self, alert: Dict[str, Any]) -> bool:
        """Send email notification for alert"""
        if not self.config.email_enabled or not self.config.email_recipients:
            return False

        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"ZoL0 Alert: {alert.get('level', 'Unknown').title()}"
            msg["From"] = self.config.email_user
            msg["To"] = ", ".join(self.config.email_recipients)

            # Add text and HTML parts
            text_content = self.format_alert_message(alert, "text")
            html_content = self.format_alert_message(alert, "html")

            msg.attach(MIMEText(text_content, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.email_user, self.config.email_password)
                server.send_message(msg)

            return True

        except Exception as e:
            print(f"Email notification failed: {e}")
            return False

    def send_sms_notification(self, alert: Dict[str, Any]) -> bool:
        """Send SMS notification using Twilio"""
        if not self.config.sms_enabled or not self.config.sms_recipients:
            return False

        try:
            # Prepare SMS message (limited to 160 characters)
            level = alert.get("level", "info").upper()
            message = alert.get("message", "Unknown alert")
            sms_text = f"ZoL0 Alert [{level}]: {message}"

            if len(sms_text) > 155:
                sms_text = sms_text[:152] + "..."

            # Send SMS via Twilio API
            url = f"https://api.twilio.com/2010-04-01/Accounts/{self.config.twilio_sid}/Messages.json"

            for recipient in self.config.sms_recipients:
                data = {
                    "From": self.config.twilio_phone,
                    "To": recipient,
                    "Body": sms_text,
                }

                response = requests.post(
                    url,
                    data=data,
                    auth=(self.config.twilio_sid, self.config.twilio_token),
                )

                if response.status_code != 201:
                    print(f"SMS failed to {recipient}: {response.text}")
                    return False

            return True

        except Exception as e:
            print(f"SMS notification failed: {e}")
            return False

    def send_slack_notification(self, alert: Dict[str, Any]) -> bool:
        """Send alert to Slack via webhook"""
        if not self.config.slack_enabled or not self.config.slack_webhook_url:
            return False

        try:
            slack_message = {
                "text": f"*ZoL0 Alert*\n*Level:* {alert.get('level', '').upper()}\n*Category:* {alert.get('category', '')}\n*Message:* {alert.get('message', '')}\n*Time:* {alert.get('timestamp', '')}"
            }
            response = requests.post(
                self.config.slack_webhook_url, json=slack_message, timeout=10
            )

            if response.status_code != 200:
                print(f"Slack notification failed: {response.text}")
                return False

            return True

        except Exception as e:
            print(f"Slack notification error: {e}")
            return False

    def send_notification(self, alert: Dict[str, Any]) -> Dict[str, bool]:
        """Send notification via all enabled channels (email, sms, slack)"""
        results = {"email": False, "sms": False, "slack": False}

        if not self.should_send_notification(alert):
            return results

        # Send email notification
        if self.config.email_enabled:
            results["email"] = self.send_email_notification(alert)

        # Send SMS notification
        if self.config.sms_enabled:
            results["sms"] = self.send_sms_notification(alert)

        # Send Slack notification
        if self.config.slack_enabled:
            results["slack"] = self.send_slack_notification(alert)

        return results

    def test_notifications(self) -> Dict[str, bool]:
        """Send test notifications to verify configuration"""
        test_alert = {
            "level": "info",
            "message": "This is a test notification from ZoL0 monitoring system",
            "category": "system",
            "timestamp": datetime.now().isoformat(),
        }

        return self.send_notification(test_alert)


def create_notification_config_ui():
    """Create Streamlit UI for notification configuration"""
    st.header("📧 Notification Configuration")

    with st.expander("Email Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            email_enabled = st.checkbox("Enable Email Notifications", value=False)
            smtp_server = st.text_input("SMTP Server", value="smtp.gmail.com")
            smtp_port = st.number_input(
                "SMTP Port", value=587, min_value=1, max_value=65535
            )

        with col2:
            email_user = st.text_input(
                "Email Address", placeholder="your-email@gmail.com"
            )
            email_password = st.text_input(
                "Email Password",
                type="password",
                help="Use app-specific password for Gmail",
            )

        email_recipients = st.text_area(
            "Email Recipients",
            placeholder="recipient1@gmail.com, recipient2@gmail.com",
            help="Comma-separated list of email addresses",
        )

    with st.expander("SMS Settings", expanded=False):
        col3, col4 = st.columns(2)

        with col3:
            sms_enabled = st.checkbox("Enable SMS Notifications", value=False)
            twilio_sid = st.text_input("Twilio Account SID", type="password")
            twilio_token = st.text_input("Twilio Auth Token", type="password")

        with col4:
            twilio_phone = st.text_input(
                "Twilio Phone Number", placeholder="+1234567890"
            )
            sms_recipients = st.text_area(
                "SMS Recipients",
                placeholder="+1234567890, +0987654321",
                help="Comma-separated list of phone numbers with country codes",
            )

    with st.expander("Slack Settings", expanded=False):
        slack_enabled = st.checkbox("Enable Slack Notifications", value=False)
        slack_webhook_url = st.text_input(
            "Slack Webhook URL", placeholder="https://hooks.slack.com/services/..."
        )

    with st.expander("Notification Rules", expanded=True):
        col5, col6 = st.columns(2)

        with col5:
            min_severity = st.selectbox(
                "Minimum Severity Level",
                ["info", "warning", "critical"],
                index=1,
                help="Only send notifications for alerts at this level or higher",
            )

        with col6:
            cooldown_minutes = st.number_input(
                "Cooldown Period (minutes)",
                value=5,
                min_value=1,
                max_value=60,
                help="Minimum time between notifications of the same type",
            )

    # Create configuration object
    config = NotificationConfig(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        email_user=email_user,
        email_password=email_password,
        twilio_sid=twilio_sid,
        twilio_token=twilio_token,
        twilio_phone=twilio_phone,
        email_recipients=(
            [r.strip() for r in email_recipients.split(",") if r.strip()]
            if email_recipients
            else []
        ),
        sms_recipients=(
            [r.strip() for r in sms_recipients.split(",") if r.strip()]
            if sms_recipients
            else []
        ),
        slack_webhook_url=slack_webhook_url,
        email_enabled=email_enabled,
        sms_enabled=sms_enabled,
        slack_enabled=slack_enabled,
        min_severity=min_severity,
        cooldown_minutes=cooldown_minutes,
    )

    # Test buttons
    col7, col8, col9 = st.columns(3)

    with col7:
        if st.button("💾 Save Configuration"):
            # Save config to session state
            st.session_state.notification_config = config
            st.success("Configuration saved!")

    with col8:
        if st.button("📧 Test Email"):
            if email_enabled and email_user and email_recipients:
                manager = EnhancedNotificationManager(config)
                test_alert = {
                    "level": "info",
                    "message": "Email notification test successful!",
                    "category": "test",
                    "timestamp": datetime.now().isoformat(),
                }
                result = manager.send_email_notification(test_alert)
                if result:
                    st.success("Test email sent successfully!")
                else:
                    st.error("Failed to send test email. Check your configuration.")
            else:
                st.warning("Please configure email settings first.")

    with col9:
        if st.button("📱 Test SMS"):
            if sms_enabled and twilio_sid and sms_recipients:
                manager = EnhancedNotificationManager(config)
                test_alert = {
                    "level": "info",
                    "message": "SMS notification test successful!",
                    "category": "test",
                    "timestamp": datetime.now().isoformat(),
                }
                result = manager.send_sms_notification(test_alert)
                if result:
                    st.success("Test SMS sent successfully!")
                else:
                    st.error("Failed to send test SMS. Check your configuration.")
            else:
                st.warning("Please configure SMS settings first.")

    with col9:
        if st.button("💬 Test Slack"):
            if slack_enabled and slack_webhook_url:
                manager = EnhancedNotificationManager(config)
                test_alert = {
                    "level": "info",
                    "message": "Slack notification test successful!",
                    "category": "test",
                    "timestamp": datetime.now().isoformat(),
                }
                result = manager.send_slack_notification(test_alert)
                if result:
                    st.success("Test Slack notification sent successfully!")
                else:
                    st.error(
                        "Failed to send Slack notification. Check your configuration."
                    )
            else:
                st.warning("Please configure Slack settings first.")

    return config


# Global notification manager instance
_notification_manager = None


def get_notification_manager() -> EnhancedNotificationManager:
    """Get global notification manager instance"""
    global _notification_manager

    if _notification_manager is None:
        # Try to get config from session state
        config = getattr(st.session_state, "notification_config", NotificationConfig())
        _notification_manager = EnhancedNotificationManager(config)

    return _notification_manager


def update_notification_manager(config: NotificationConfig):
    """Update the global notification manager with new config"""
    global _notification_manager
    _notification_manager = EnhancedNotificationManager(config)
