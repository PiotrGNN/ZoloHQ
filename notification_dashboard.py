#!/usr/bin/env python3
"""
Notification Dashboard for ZoL0 Advanced Monitoring Suite
Comprehensive interface for managing and testing notifications
"""

import json
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from enhanced_notification_system import EnhancedNotificationManager, NotificationConfig, create_notification_config_ui

st.set_page_config(
    page_title="ZoL0 Notification Dashboard",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enhanced CSS for notification dashboard
st.markdown(
    """
<style>
    .notification-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        border-left: 5px solid #5a67d8;
    }
    .notification-success {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        border-left: 5px solid #38a169;
    }
    .notification-failed {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        border-left: 5px solid #e53e3e;
    }
    .notification-pending {
        background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
        border-left: 5px solid #dd6b20;
    }
    .metric-card {
        background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(66, 153, 225, 0.3);
    }
    .config-section {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(167, 139, 250, 0.3);
    }
    .test-section {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(52, 211, 153, 0.3);
    }
    .metric-number {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


class NotificationDashboard:
    """Comprehensive notification dashboard manager"""

    def __init__(self):
        self.config_file = "notification_config.json"
        self.history_file = "notification_history.json"
        self.load_config()
        self.load_history()

        # Initialize production data manager for real trading event notifications
        try:
            from production_data_manager import ProductionDataManager

            self.production_manager = ProductionDataManager()
            self.production_mode = True
            # Store the connection status for UI display
            self.real_data_available = True
        except ImportError:
            self.production_manager = None
            self.production_mode = False
            self.real_data_available = False

    def load_config(self):
        """Load notification configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r") as f:
                    config_data = json.load(f)
                    self.config = NotificationConfig(**config_data)
            else:
                self.config = NotificationConfig()
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            self.config = NotificationConfig()

    def save_config(self, config: NotificationConfig):
        """Save notification configuration to file"""
        try:
            config_dict = {
                "smtp_server": config.smtp_server,
                "smtp_port": config.smtp_port,
                "email_user": config.email_user,
                "email_password": config.email_password,
                "twilio_sid": config.twilio_sid,
                "twilio_token": config.twilio_token,
                "twilio_phone": config.twilio_phone,
                "email_recipients": config.email_recipients,
                "sms_recipients": config.sms_recipients,
                "email_enabled": config.email_enabled,
                "sms_enabled": config.sms_enabled,
                "min_severity": config.min_severity,
                "cooldown_minutes": config.cooldown_minutes,
            }

            with open(self.config_file, "w") as f:
                json.dump(config_dict, f, indent=2)

            self.config = config
            return True
        except Exception as e:
            st.error(f"Error saving configuration: {e}")
            return False

    def load_history(self):
        """Load notification history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, "r") as f:
                    self.history = json.load(f)
            else:
                self.history = []
        except Exception as e:
            st.error(f"Error loading history: {e}")
            self.history = []

    def save_history(self):
        """Save notification history to file"""
        try:
            # Keep only last 1000 entries
            if len(self.history) > 1000:
                self.history = self.history[-1000:]

            with open(self.history_file, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            st.error(f"Error saving history: {e}")

    def add_to_history(
        self, notification_type: str, recipient: str, status: str, message: str
    ):
        """Add notification to history"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": notification_type,
            "recipient": recipient,
            "status": status,
            "message": message,
        }
        self.history.insert(0, entry)  # Add to beginning
        self.save_history()

    def get_history_stats(self):
        """Get notification history statistics"""
        if not self.history:
            return {"total": 0, "email": 0, "sms": 0, "success": 0, "failed": 0}

        stats = {
            "total": len(self.history),
            "email": len([h for h in self.history if h["type"] == "email"]),
            "sms": len([h for h in self.history if h["type"] == "sms"]),
            "success": len([h for h in self.history if h["status"] == "success"]),
            "failed": len([h for h in self.history if h["status"] == "failed"]),
        }

        return stats

    def get_real_trading_events(self):
        """Get real trading events that might trigger notifications"""
        if not self.production_manager or not self.production_mode:
            return []

        try:
            # Get current account balance and positions
            balance_data = self.production_manager.get_account_balance()
            positions_data = self.production_manager.get_positions()
            self.production_manager.get_market_data("BTCUSDT")

            events = []

            if balance_data.get("success"):
                balance_info = balance_data.get("result", {})
                float(balance_info.get("totalWalletBalance", 0))
                unrealized_pnl = float(balance_info.get("totalUnrealisedPnl", 0))

                # Check for balance-based events
                if unrealized_pnl < -100:
                    events.append(
                        {
                            "type": "critical",
                            "message": f"High unrealized loss detected: ${unrealized_pnl:.2f}",
                            "category": "risk",
                            "timestamp": datetime.now(),
                        }
                    )
                elif unrealized_pnl > 200:
                    events.append(
                        {
                            "type": "info",
                            "message": f"Significant profit reached: ${unrealized_pnl:.2f}",
                            "category": "performance",
                            "timestamp": datetime.now(),
                        }
                    )

            if positions_data.get("success"):
                positions = positions_data.get("result", {}).get("list", [])
                if len(positions) > 5:
                    events.append(
                        {
                            "type": "warning",
                            "message": f"High number of open positions: {len(positions)}",
                            "category": "risk",
                            "timestamp": datetime.now(),
                        }
                    )

            return events

        except Exception as e:
            return [
                {
                    "type": "error",
                    "message": f"Error getting trading events: {e}",
                    "category": "system",
                    "timestamp": datetime.now(),
                }
            ]


def main():
    # Ensure NotificationDashboard is always in session state
    if "notification_dashboard" not in st.session_state:
        st.session_state.notification_dashboard = NotificationDashboard()
    dashboard = st.session_state.notification_dashboard

    # Header
    st.title("📧 ZoL0 Notification Dashboard")
    st.markdown("**Comprehensive notification management and testing interface**")

    # Data source indicator
    if dashboard.real_data_available:
        st.success("🟢 Real Data - Trading Events Available")
    else:
        st.warning("🟡 Simulated Data - Production Manager Not Available")

    # Sidebar navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["📊 Overview", "⚙️ Configuration", "🧪 Testing", "📋 History", "📈 Analytics"],
    )

    if page == "📊 Overview":
        show_overview_page(dashboard)
    elif page == "⚙️ Configuration":
        show_configuration_page(dashboard)
    elif page == "🧪 Testing":
        show_testing_page(dashboard)
    elif page == "📋 History":
        show_history_page(dashboard)
    elif page == "📈 Analytics":
        show_analytics_page(dashboard)


def show_overview_page(dashboard):
    """Show notification overview page"""
    st.header("📊 Notification Overview")

    # Real Trading Events Section
    if dashboard.real_data_available:
        st.subheader("⚡ Real Trading Events")
        trading_events = dashboard.get_real_trading_events()

        if trading_events:
            for event in trading_events:
                event_type = event.get("type", "info")
                message = event.get("message", "No message")
                category = event.get("category", "general")

                if event_type == "critical":
                    st.error(f"🔴 **{category.title()}**: {message}")
                elif event_type == "warning":
                    st.warning(f"🟡 **{category.title()}**: {message}")
                else:
                    st.info(f"🔵 **{category.title()}**: {message}")
        else:
            st.success("✅ No critical trading events detected")

        st.markdown("---")

    # Get current stats
    stats = dashboard.get_history_stats()

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>📧 Total Sent</h4>
            <div class="metric-number">{stats['total']}</div>
            <small>All notifications</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="metric-card notification-success">
            <h4>✅ Successful</h4>
            <div class="metric-number">{stats['success']}</div>
            <small>Delivered successfully</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>📧 Email</h4>
            <div class="metric-number">{stats['email']}</div>
            <small>Email notifications</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>📱 SMS</h4>
            <div class="metric-number">{stats['sms']}</div>
            <small>SMS notifications</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Configuration status
    st.header("⚙️ Configuration Status")

    col1, col2 = st.columns(2)

    with col1:
        email_status = "✅ Enabled" if dashboard.config.email_enabled else "❌ Disabled"
        st.markdown(
            f"""
        <div class="notification-card">
            <h4>📧 Email Notifications</h4>
            <p><strong>Status:</strong> {email_status}</p>
            <p><strong>Recipients:</strong> {len(dashboard.config.email_recipients)}</p>
            <p><strong>SMTP Server:</strong> {dashboard.config.smtp_server}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        sms_status = "✅ Enabled" if dashboard.config.sms_enabled else "❌ Disabled"
        st.markdown(
            f"""
        <div class="notification-card">
            <h4>📱 SMS Notifications</h4>
            <p><strong>Status:</strong> {sms_status}</p>
            <p><strong>Recipients:</strong> {len(dashboard.config.sms_recipients)}</p>
            <p><strong>Provider:</strong> Twilio</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Recent notifications
    st.header("📋 Recent Notifications")

    if dashboard.history:
        recent_notifications = dashboard.history[:10]

        for notification in recent_notifications:
            try:
                dt = datetime.fromisoformat(notification["timestamp"])
                time_ago = datetime.now() - dt
                time_str = f"{int(time_ago.total_seconds() // 60)}m ago"
            except Exception as e:
                logging.exception(
                    f"Exception calculating time ago for notification: {e}"
                )
                time_str = "Unknown time"

            status_class = (
                "notification-success"
                if notification["status"] == "success"
                else "notification-failed"
            )
            status_icon = "✅" if notification["status"] == "success" else "❌"
            type_icon = "📧" if notification["type"] == "email" else "📱"

            st.markdown(
                f"""
            <div class="notification-card {status_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{type_icon} {notification['type'].title()} to {notification['recipient']}</strong> {status_icon}
                        <br><small>{notification['message'][:100]}... | {time_str}</small>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.info(
            "No notifications sent yet. Configure your settings and test notifications!"
        )


def show_configuration_page(dashboard):
    """Show notification configuration page"""
    st.header("⚙️ Notification Configuration")

    # Create configuration UI
    config = create_notification_config_ui()

    if st.button("💾 Save Configuration", type="primary"):
        if dashboard.save_config(config):
            st.success("✅ Configuration saved successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to save configuration")


def show_testing_page(dashboard):
    """Show notification testing page"""
    st.header("🧪 Notification Testing")

    # Real Trading Data Notifications
    if dashboard.real_data_available:
        st.markdown(
            """
        <div class="test-section">
            <h3>📡 Real Trading Data Notifications</h3>
            <p>Test notifications using real trading events and data</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Test Portfolio Alert"):
                try:
                    manager = EnhancedNotificationManager(dashboard.config)
                    balance_data = dashboard.production_manager.get_account_balance()

                    if balance_data.get("success"):
                        balance_info = balance_data.get("result", {})
                        total_balance = float(balance_info.get("totalWalletBalance", 0))

                        test_alert = {
                            "level": "info",
                            "message": f"Portfolio Update: Current balance ${total_balance:.2f} USDT",
                            "category": "trading",
                            "timestamp": datetime.now().isoformat(),
                        }

                        result = manager.send_email_notification(test_alert)

                        for recipient in dashboard.config.email_recipients:
                            status = "success" if result else "failed"
                            dashboard.add_to_history(
                                "email", recipient, status, test_alert["message"]
                            )

                        if result:
                            st.success("✅ Real portfolio alert sent!")
                        else:
                            st.error("❌ Failed to send portfolio alert")
                    else:
                        st.error("❌ Failed to get real portfolio data")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with col2:
            if st.button("⚠️ Test Risk Alert"):
                try:
                    manager = EnhancedNotificationManager(dashboard.config)
                    positions_data = dashboard.production_manager.get_positions()

                    if positions_data.get("success"):
                        positions = positions_data.get("result", {}).get("list", [])
                        position_count = len(positions)

                        test_alert = {
                            "level": "warning" if position_count > 3 else "info",
                            "message": f"Risk Check: {position_count} active positions",
                            "category": "risk",
                            "timestamp": datetime.now().isoformat(),
                        }

                        result = manager.send_email_notification(test_alert)

                        for recipient in dashboard.config.email_recipients:
                            status = "success" if result else "failed"
                            dashboard.add_to_history(
                                "email", recipient, status, test_alert["message"]
                            )

                        if result:
                            st.success("✅ Real risk alert sent!")
                        else:
                            st.error("❌ Failed to send risk alert")
                    else:
                        st.error("❌ Failed to get real position data")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        st.markdown("---")

    # Test email notifications
    st.markdown(
        """
    <div class="test-section">
        <h3>📧 Test Email Notifications</h3>
        <p>Send test emails to verify your configuration</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if dashboard.config.email_enabled and dashboard.config.email_recipients:
        col1, col2 = st.columns(2)

        with col1:
            st.text_input("Test Email Subject", value="ZoL0 Test Notification")
            test_message = st.text_area(
                "Test Email Message",
                value="This is a test notification from ZoL0 monitoring system.",
            )

        with col2:
            severity_level = st.selectbox(
                "Test Severity Level", ["info", "warning", "critical"]
            )
            test_category = st.selectbox(
                "Test Category", ["system", "trading", "risk", "performance"]
            )

        if st.button("📧 Send Test Email"):
            try:
                manager = EnhancedNotificationManager(dashboard.config)
                test_alert = {
                    "level": severity_level,
                    "message": test_message,
                    "category": test_category,
                    "timestamp": datetime.now().isoformat(),
                }

                result = manager.send_email_notification(test_alert)

                for recipient in dashboard.config.email_recipients:
                    status = "success" if result else "failed"
                    dashboard.add_to_history("email", recipient, status, test_message)

                if result:
                    st.success("✅ Test email sent successfully!")
                else:
                    st.error("❌ Failed to send test email")

            except Exception as e:
                st.error(f"❌ Error sending test email: {e}")
    else:
        st.warning(
            "Email notifications not configured. Please configure email settings first."
        )

    st.markdown("---")

    # Test SMS notifications
    st.markdown(
        """
    <div class="test-section">
        <h3>📱 Test SMS Notifications</h3>
        <p>Send test SMS messages to verify your configuration</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if dashboard.config.sms_enabled and dashboard.config.sms_recipients:
        col1, col2 = st.columns(2)

        with col1:
            sms_message = st.text_area(
                "Test SMS Message", value="ZoL0 Test: This is a test SMS notification."
            )
            sms_severity = st.selectbox(
                "SMS Severity Level",
                ["info", "warning", "critical"],
                key="sms_severity",
            )

        with col2:
            sms_category = st.selectbox(
                "SMS Category",
                ["system", "trading", "risk", "performance"],
                key="sms_category",
            )

        if st.button("📱 Send Test SMS"):
            try:
                manager = EnhancedNotificationManager(dashboard.config)
                test_alert = {
                    "level": sms_severity,
                    "message": sms_message,
                    "category": sms_category,
                    "timestamp": datetime.now().isoformat(),
                }

                result = manager.send_sms_notification(test_alert)

                for recipient in dashboard.config.sms_recipients:
                    status = "success" if result else "failed"
                    dashboard.add_to_history("sms", recipient, status, sms_message)

                if result:
                    st.success("✅ Test SMS sent successfully!")
                else:
                    st.error("❌ Failed to send test SMS")

            except Exception as e:
                st.error(f"❌ Error sending test SMS: {e}")
    else:
        st.warning(
            "SMS notifications not configured. Please configure SMS settings first."
        )


def show_history_page(dashboard):
    """Show notification history page"""
    st.header("📋 Notification History")

    if not dashboard.history:
        st.info("No notification history available.")
        return

    # Filter options
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_type = st.selectbox("Filter by Type", ["All", "Email", "SMS"])

    with col2:
        filter_status = st.selectbox("Filter by Status", ["All", "Success", "Failed"])

    with col3:
        days_back = st.selectbox(
            "Show Last", [7, 30, 90, 365], format_func=lambda x: f"{x} days"
        )

    # Filter history
    filtered_history = []
    cutoff_date = datetime.now() - timedelta(days=days_back)

    for entry in dashboard.history:
        try:
            entry_date = datetime.fromisoformat(entry["timestamp"])
            if entry_date < cutoff_date:
                continue
        except Exception as e:
            logging.exception(
                f"Exception parsing timestamp in notification history: {e}"
            )
            continue

        if filter_type != "All" and entry["type"].title() != filter_type:
            continue

        if filter_status != "All" and entry["status"].title() != filter_status:
            continue

        filtered_history.append(entry)

    # Display history
    if filtered_history:
        st.subheader(f"Showing {len(filtered_history)} notifications")

        # Convert to DataFrame for better display
        df_data = []
        for entry in filtered_history:
            try:
                dt = datetime.fromisoformat(entry["timestamp"])
                time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logging.exception(
                    f"Exception parsing timestamp in notification history: {e}"
                )
                time_str = entry["timestamp"]

            df_data.append(
                {
                    "Time": time_str,
                    "Type": entry["type"].title(),
                    "Recipient": entry["recipient"],
                    "Status": entry["status"].title(),
                    "Message": (
                        entry["message"][:100] + "..."
                        if len(entry["message"]) > 100
                        else entry["message"]
                    ),
                }
            )

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

        # Export options
        if st.button("📥 Export History to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"notification_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
    else:
        st.info("No notifications match the selected filters.")


def show_analytics_page(dashboard):
    """Show notification analytics page"""
    st.header("📈 Notification Analytics")

    if not dashboard.history:
        st.info("No data available for analytics.")
        return

    # Prepare data
    df_data = []
    for entry in dashboard.history:
        try:
            dt = datetime.fromisoformat(entry["timestamp"])
            df_data.append(
                {
                    "timestamp": dt,
                    "type": entry["type"],
                    "status": entry["status"],
                    "recipient": entry["recipient"],
                }
            )
        except Exception:
            continue

    if not df_data:
        st.info("No valid data for analytics.")
        return

    df = pd.DataFrame(df_data)
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    # Success rate over time
    st.subheader("📊 Success Rate Over Time")

    daily_stats = df.groupby(["date", "status"]).size().unstack(fill_value=0)
    if "success" in daily_stats.columns and "failed" in daily_stats.columns:
        daily_stats["success_rate"] = (
            daily_stats["success"]
            / (daily_stats["success"] + daily_stats["failed"])
            * 100
        )

        fig = px.line(
            x=daily_stats.index,
            y=daily_stats["success_rate"],
            title="Daily Success Rate (%)",
            labels={"x": "Date", "y": "Success Rate (%)"},
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Notification type distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📧 Notification Types")
        type_counts = df["type"].value_counts()

        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Distribution by Type",
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("✅ Success/Failure Rate")
        status_counts = df["status"].value_counts()

        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Success vs Failure Rate",
            color_discrete_map={"success": "#48bb78", "failed": "#f56565"},
        )
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # Hourly distribution
    st.subheader("🕐 Notification Activity by Hour")

    hourly_counts = df.groupby("hour").size()

    fig = px.bar(
        x=hourly_counts.index,
        y=hourly_counts.values,
        title="Notifications by Hour of Day",
        labels={"x": "Hour", "y": "Number of Notifications"},
    )
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
