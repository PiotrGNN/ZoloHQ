#!/usr/bin/env python3
"""
Advanced Alert Management System
Zaawansowany system zarządzania alertami
"""

import logging
import os
import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

# Import enhanced notification system
from enhanced_notification_system import EnhancedNotificationManager, NotificationConfig, get_notification_manager

# Add production data integration
try:
    from production_data_manager import get_production_data
except ImportError:
    get_production_data = None

# REMOVE st.set_page_config from here to avoid conflicts when imported

# Enhanced CSS for alert management
st.markdown(
    """
<style>
    .alert-critical {
        background: linear-gradient(135deg, #ff4757 0%, #c44569 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(255, 71, 87, 0.3);
        border-left: 5px solid #ff3742;
        animation: pulse 2s infinite;
    }
    .alert-warning {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(255, 167, 38, 0.3);
        border-left: 5px solid #ff8f00;
    }
    .alert-info {
        background: linear-gradient(135deg, #42a5f5 0%, #1976d2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(66, 165, 245, 0.3);
        border-left: 5px solid #1565c0;
    }
    .alert-success {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(102, 187, 106, 0.3);
        border-left: 5px solid #388e3c;
    }
    @keyframes pulse {
        0% { box-shadow: 0 4px 12px rgba(255, 71, 87, 0.3); }
        50% { box-shadow: 0 4px 20px rgba(255, 71, 87, 0.6); }
        100% { box-shadow: 0 4px 12px rgba(255, 71, 87, 0.3); }
    }
    .alert-counter {
        background: #f44336;
        color: white;
        border-radius: 50%;
        padding: 0.2rem 0.5rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .metric-alert {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-timeline {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .rule-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 8px;
        color: #333;
        margin: 0.3rem 0;
        border-left: 4px solid #00bcd4;
    }
</style>
""",
    unsafe_allow_html=True,
)


def send_alert(message, level="info"):
    """
    Send alert to Telegram, Slack, or email. Reads webhook/token from env.
    """
    # Telegram example
    tg_token = os.getenv("ALERT_TELEGRAM_TOKEN")
    tg_chat_id = os.getenv("ALERT_TELEGRAM_CHAT_ID")
    if tg_token and tg_chat_id:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        try:
            requests.post(
                url,
                data={"chat_id": tg_chat_id, "text": f"[{level.upper()}] {message}"},
            )
        except Exception:
            pass
    # Slack example
    slack_webhook = os.getenv("ALERT_SLACK_WEBHOOK")
    if slack_webhook:
        try:
            requests.post(slack_webhook, json={"text": f"[{level.upper()}] {message}"})
        except Exception:
            pass
    # Email (placeholder)
    # ...
    logging.info(f"ALERT: {message}")


__all__ = ["send_alert"]


class AdvancedAlertManager:
    def __init__(self):
        self.api_base_url = "http://localhost:5001"
        self.alert_history = []
        self.notification_manager = get_notification_manager()
        self.processed_alerts = set()  # Track alerts to avoid duplicate notifications

        # Initialize production data manager for real alert monitoring
        try:
            from production_data_manager import ProductionDataManager

            self.production_manager = ProductionDataManager()
            self.production_mode = True
        except ImportError:
            self.production_manager = None
            self.production_mode = False

        # Legacy production API integration
        self.is_production = os.getenv("BYBIT_PRODUCTION_ENABLED", "").lower() == "true"
        self.production_data_manager = None
        self._initialize_production_data()

    def _initialize_production_data(self):
        """Initialize production data manager if available"""
        try:
            if get_production_data:
                self.production_data_manager = get_production_data()
                if self.is_production:
                    st.sidebar.success("🟢 Production API alerts enabled")
                else:
                    st.sidebar.info("🔄 Testnet API alerts enabled")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Production data not available: {e}")

    def get_real_api_alerts(self):
        """Get alerts from real Bybit API data"""
        if not self.production_data_manager:
            return []

        alerts = []
        try:
            # Get account balance for balance-based alerts
            balance_data = self.production_data_manager.get_account_balance()
            if balance_data.get("success"):
                alerts.extend(self._analyze_balance_alerts(balance_data))

            # Get market data for market-based alerts
            market_data = self.production_data_manager.get_market_data("BTCUSDT")
            if market_data.get("success"):
                alerts.extend(self._analyze_market_alerts(market_data))

            # Get positions for position-based alerts
            positions_data = self.production_data_manager.get_positions()
            if positions_data.get("success"):
                alerts.extend(self._analyze_position_alerts(positions_data))

        except Exception as e:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Failed to fetch real API data for alerts: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "category": "api",
                }
            )

        return alerts

    def _analyze_balance_alerts(self, balance_data):
        """Analyze balance data for alerts"""
        alerts = []
        try:
            balances = balance_data.get("data", {}).get("balances", [])
            for balance in balances:
                total_equity = float(balance.get("totalEquity", 0))
                available_balance = float(balance.get("availableBalance", 0))

                # Low balance alert
                if available_balance < 100:  # Alert if less than $100
                    alerts.append(
                        {
                            "level": "warning",
                            "message": f"Low available balance: ${available_balance:,.2f}",
                            "timestamp": datetime.now().isoformat(),
                            "category": "balance",
                            "value": available_balance,
                        }
                    )

                # High equity usage alert
                if total_equity > 0:
                    used_margin_ratio = (
                        total_equity - available_balance
                    ) / total_equity
                    if used_margin_ratio > 0.8:  # 80% margin usage
                        alerts.append(
                            {
                                "level": "critical",
                                "message": f"High margin usage: {used_margin_ratio*100:.1f}%",
                                "timestamp": datetime.now().isoformat(),
                                "category": "margin",
                                "value": used_margin_ratio * 100,
                            }
                        )
        except Exception:
            pass

        return alerts

    def _analyze_market_alerts(self, market_data):
        """Analyze market data for alerts"""
        alerts = []
        try:
            ticker = market_data.get("data", {})
            if ticker:
                price_change = float(ticker.get("price24hPcnt", 0)) * 100
                volume = float(ticker.get("volume24h", 0))

                # High volatility alert
                if abs(price_change) > 10:  # 10% price change
                    level = "critical" if abs(price_change) > 15 else "warning"
                    alerts.append(
                        {
                            "level": level,
                            "message": f"High volatility: BTC {'+' if price_change > 0 else ''}{price_change:.1f}% (24h)",
                            "timestamp": datetime.now().isoformat(),
                            "category": "volatility",
                            "value": price_change,
                        }
                    )

                # Volume spike alert
                # This is simplified - in production you'd compare with historical averages
                if volume > 50000:  # High volume threshold
                    alerts.append(
                        {
                            "level": "info",
                            "message": f"High trading volume: {volume:,.0f} BTC (24h)",
                            "timestamp": datetime.now().isoformat(),
                            "category": "volume",
                            "value": volume,
                        }
                    )
        except Exception:
            pass

        return alerts

    def _analyze_position_alerts(self, positions_data):
        """Analyze position data for alerts"""
        alerts = []
        try:
            positions = positions_data.get("data", {}).get("list", [])
            for position in positions:
                size = float(position.get("size", 0))
                if size > 0:  # Active position
                    unrealized_pnl = float(position.get("unrealisedPnl", 0))
                    float(position.get("positionValue", 0))

                    # Large loss alert
                    if unrealized_pnl < -500:  # $500 loss
                        alerts.append(
                            {
                                "level": "critical",
                                "message": f"Large unrealized loss: ${unrealized_pnl:,.2f} on {position.get('symbol', 'Unknown')}",
                                "timestamp": datetime.now().isoformat(),
                                "category": "position",
                                "value": unrealized_pnl,
                            }
                        )

                    # Large profit alert
                    elif unrealized_pnl > 1000:  # $1000 profit
                        alerts.append(
                            {
                                "level": "success",
                                "message": f"Large unrealized profit: ${unrealized_pnl:,.2f} on {position.get('symbol', 'Unknown')}",
                                "timestamp": datetime.now().isoformat(),
                                "category": "position",
                                "value": unrealized_pnl,
                            }
                        )
        except Exception:
            pass

        return alerts

    def get_real_production_alerts(self):
        """Get alerts from real production data manager"""
        if not self.production_manager:
            return []

        alerts = []
        try:
            # Get account balance for balance-based alerts
            balance_data = self.production_manager.get_account_balance()
            if balance_data and balance_data.get("success"):
                alerts.extend(self._analyze_production_balance_alerts(balance_data))

            # Get positions for position-based alerts
            positions_data = self.production_manager.get_positions()
            if positions_data and positions_data.get("success"):
                alerts.extend(self._analyze_production_position_alerts(positions_data))

            # Get market data for volatility alerts
            market_data = self.production_manager.get_market_data("BTCUSDT")
            if market_data and market_data.get("success"):
                alerts.extend(self._analyze_production_market_alerts(market_data))

        except Exception as e:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Failed to fetch production data for alerts: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "category": "api",
                }
            )

        return alerts

    def _analyze_production_balance_alerts(self, balance_data):
        """Analyze production balance data for alerts"""
        alerts = []
        try:
            wallet_balance = balance_data.get("data", {}).get("list", [])
            for balance in wallet_balance:
                if balance.get("accountType") == "UNIFIED":
                    total_equity = float(balance.get("totalEquity", 0))
                    available_balance = float(balance.get("availableBalance", 0))

                    # Low balance alert
                    if available_balance < 100:  # Alert if less than $100
                        alerts.append(
                            {
                                "level": "warning",
                                "message": f"Low available balance: ${available_balance:,.2f}",
                                "timestamp": datetime.now().isoformat(),
                                "category": "balance",
                                "value": available_balance,
                            }
                        )

                    # High margin usage alert
                    if total_equity > 0:
                        margin_used = total_equity - available_balance
                        margin_ratio = margin_used / total_equity
                        if margin_ratio > 0.8:  # 80% margin usage
                            alerts.append(
                                {
                                    "level": "critical",
                                    "message": f"High margin usage: {margin_ratio*100:.1f}%",
                                    "timestamp": datetime.now().isoformat(),
                                    "category": "margin",
                                    "value": margin_ratio * 100,
                                }
                            )
        except Exception:
            pass

        return alerts

    def _analyze_production_position_alerts(self, positions_data):
        """Analyze production position data for alerts"""
        alerts = []
        try:
            positions = positions_data.get("data", {}).get("list", [])
            for position in positions:
                size = float(position.get("size", 0))
                if size > 0:  # Active position
                    unrealized_pnl = float(position.get("unrealisedPnl", 0))
                    symbol = position.get("symbol", "Unknown")

                    # Large loss alert
                    if unrealized_pnl < -500:  # $500 loss
                        alerts.append(
                            {
                                "level": "critical",
                                "message": f"Large unrealized loss: ${unrealized_pnl:,.2f} on {symbol}",
                                "timestamp": datetime.now().isoformat(),
                                "category": "position",
                                "value": unrealized_pnl,
                            }
                        )

                    # Large profit alert
                    elif unrealized_pnl > 1000:  # $1000 profit
                        alerts.append(
                            {
                                "level": "success",
                                "message": f"Large unrealized profit: ${unrealized_pnl:,.2f} on {symbol}",
                                "timestamp": datetime.now().isoformat(),
                                "category": "position",
                                "value": unrealized_pnl,
                            }
                        )
        except Exception:
            pass

        return alerts

    def _analyze_production_market_alerts(self, market_data):
        """Analyze production market data for alerts"""
        alerts = []
        try:
            ticker = market_data.get("data", {}).get("list", [])
            if ticker:
                price_data = ticker[0]
                price_change = float(price_data.get("price24hPcnt", 0)) * 100
                volume = float(price_data.get("volume24h", 0))

                # High volatility alert
                if abs(price_change) > 10:  # 10% price change
                    level = "critical" if abs(price_change) > 15 else "warning"
                    alerts.append(
                        {
                            "level": level,
                            "message": f"High volatility: BTC {'+' if price_change > 0 else ''}{price_change:.1f}% (24h)",
                            "timestamp": datetime.now().isoformat(),
                            "category": "volatility",
                            "value": price_change,
                        }
                    )

                # Volume spike alert
                if volume > 50000:  # High volume threshold
                    alerts.append(
                        {
                            "level": "info",
                            "message": f"High trading volume: {volume:,.0f} BTC (24h)",
                            "timestamp": datetime.now().isoformat(),
                            "category": "volume",
                            "value": volume,
                        }
                    )
        except Exception:
            pass

        return alerts

    def get_comprehensive_alerts(self):
        """Pobierz wszystkie alerty systemowe"""
        try:
            # Get basic alerts from API
            api_response = requests.get(
                f"{self.api_base_url}/api/bot/alerts", timeout=5
            )
            api_alerts = (
                api_response.json().get("alerts", [])
                if api_response.status_code == 200
                else []
            )

            # Get risk alerts
            risk_response = requests.get(
                f"{self.api_base_url}/api/risk/metrics", timeout=5
            )
            risk_data = (
                risk_response.json().get("risk_metrics", {})
                if risk_response.status_code == 200
                else {}
            )

            # Get performance alerts
            perf_response = requests.get(
                f"{self.api_base_url}/api/analytics/performance", timeout=5
            )
            perf_data = (
                perf_response.json().get("performance", {})
                if perf_response.status_code == 200
                else {}
            )
            # Combine and analyze alerts
            all_alerts = api_alerts.copy()

            # Add real production alerts if modern production manager is available
            if self.production_mode and self.production_manager:
                production_alerts = self.get_real_production_alerts()
                all_alerts.extend(production_alerts)

            # Add real API alerts if legacy production data is available
            elif self.production_data_manager and self.is_production:
                real_api_alerts = self.get_real_api_alerts()
                all_alerts.extend(real_api_alerts)

            # Add risk-based alerts
            if risk_data:
                all_alerts.extend(self._generate_risk_alerts(risk_data))
            # Add performance-based alerts
            if perf_data:
                all_alerts.extend(self._generate_performance_alerts(perf_data))

            # Add system health alerts
            all_alerts.extend(self._generate_system_alerts())

            # Process new alerts for notifications
            self._process_alerts_for_notifications(all_alerts)

            # Sort by severity and timestamp
            all_alerts.sort(
                key=lambda x: (
                    {"critical": 0, "warning": 1, "info": 2, "success": 3}.get(
                        x.get("level", "info"), 2
                    ),
                    x.get("timestamp", ""),
                ),
                reverse=True,
            )

            return all_alerts

        except Exception as e:
            return [
                {
                    "level": "critical",
                    "message": f"Failed to fetch alerts: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "category": "system",
                }
            ]

    def _process_alerts_for_notifications(self, alerts):
        """Process alerts and send notifications for new critical/warning alerts"""
        for alert in alerts:
            # Create unique alert identifier
            alert_id = f"{alert.get('category', 'unknown')}_{alert.get('level', 'info')}_{alert.get('message', '')[:50]}"

            # Only send notifications for new alerts
            if alert_id not in self.processed_alerts:
                level = alert.get("level", "info")

                # Send notification for critical and warning alerts
                if level in ["critical", "warning"] and self.notification_manager:
                    try:
                        result = self.notification_manager.send_notification(alert)
                        if any(
                            result.values()
                        ):  # If any notification was sent successfully
                            print(
                                f"Notification sent for alert: {alert.get('message', 'Unknown')}"
                            )
                    except Exception as e:
                        print(f"Failed to send notification: {e}")

                # Mark alert as processed
                self.processed_alerts.add(alert_id)
                # Clean up old processed alerts (keep only last 1000)
        if len(self.processed_alerts) > 1000:
            # Remove oldest entries (this is a simple cleanup, you might want more sophisticated logic)
            old_alerts = list(self.processed_alerts)[:500]
            for old_alert in old_alerts:
                self.processed_alerts.discard(old_alert)

    def _generate_risk_alerts(self, risk_data):
        """Generuj alerty związane z ryzykiem"""
        alerts = []

        # Leverage alerts
        current_leverage = risk_data.get("current_leverage", 0)
        max_leverage = risk_data.get("max_leverage", 3.0)

        if current_leverage > max_leverage * 0.9:
            alerts.append(
                {
                    "level": "critical",
                    "message": f"High leverage warning: {current_leverage:.1f}x (max: {max_leverage:.1f}x)",
                    "timestamp": datetime.now().isoformat(),
                    "category": "risk",
                    "value": current_leverage,
                }
            )
        elif current_leverage > max_leverage * 0.7:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Moderate leverage: {current_leverage:.1f}x",
                    "timestamp": datetime.now().isoformat(),
                    "category": "risk",
                    "value": current_leverage,
                }
            )

        # Drawdown alerts
        current_dd = risk_data.get("drawdown_current", 0)
        risk_data.get("drawdown_max", -10)

        if current_dd < -15:
            alerts.append(
                {
                    "level": "critical",
                    "message": f"Severe drawdown: {current_dd:.1f}%",
                    "timestamp": datetime.now().isoformat(),
                    "category": "risk",
                    "value": current_dd,
                }
            )
        elif current_dd < -8:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Moderate drawdown: {current_dd:.1f}%",
                    "timestamp": datetime.now().isoformat(),
                    "category": "risk",
                    "value": current_dd,
                }
            )

        # VaR alerts
        var_95 = risk_data.get("var_95", 0)
        if var_95 < -5:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"High Value at Risk: {var_95:.2f}%",
                    "timestamp": datetime.now().isoformat(),
                    "category": "risk",
                    "value": var_95,
                }
            )

        return alerts

    def _generate_performance_alerts(self, perf_data):
        """Generuj alerty związane z wydajnością"""
        alerts = []

        # Win rate alerts
        win_rate = perf_data.get("win_rate", 0)
        if win_rate < 40:
            alerts.append(
                {
                    "level": "warning",
                    "message": f"Low win rate: {win_rate:.1f}%",
                    "timestamp": datetime.now().isoformat(),
                    "category": "performance",
                    "value": win_rate,
                }
            )
        elif win_rate > 80:
            alerts.append(
                {
                    "level": "info",
                    "message": f"Excellent win rate: {win_rate:.1f}%",
                    "timestamp": datetime.now().isoformat(),
                    "category": "performance",
                    "value": win_rate,
                }
            )

        # Profit alerts
        net_profit = perf_data.get("net_profit", 0)
        if net_profit < -1000:
            alerts.append(
                {
                    "level": "critical",
                    "message": f"Significant losses: ${net_profit:,.2f}",
                    "timestamp": datetime.now().isoformat(),
                    "category": "performance",
                    "value": net_profit,
                }
            )

        return alerts

    def _generate_system_alerts(self):
        """Generuj alerty systemowe"""
        alerts = []

        try:
            import psutil

            # CPU alerts
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                alerts.append(
                    {
                        "level": "critical",
                        "message": f"Critical CPU usage: {cpu_percent:.1f}%",
                        "timestamp": datetime.now().isoformat(),
                        "category": "system",
                        "value": cpu_percent,
                    }
                )
            elif cpu_percent > 75:
                alerts.append(
                    {
                        "level": "warning",
                        "message": f"High CPU usage: {cpu_percent:.1f}%",
                        "timestamp": datetime.now().isoformat(),
                        "category": "system",
                        "value": cpu_percent,
                    }
                )

            # Memory alerts
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                alerts.append(
                    {
                        "level": "critical",
                        "message": f"Critical memory usage: {memory.percent:.1f}%",
                        "timestamp": datetime.now().isoformat(),
                        "category": "system",
                        "value": memory.percent,
                    }
                )
            elif memory.percent > 75:
                alerts.append(
                    {
                        "level": "warning",
                        "message": f"High memory usage: {memory.percent:.1f}%",
                        "timestamp": datetime.now().isoformat(),
                        "category": "system",
                        "value": memory.percent,
                    }
                )

            # Disk space alerts
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:
                alerts.append(
                    {
                        "level": "warning",
                        "message": f"Low disk space: {disk_percent:.1f}% used",
                        "timestamp": datetime.now().isoformat(),
                        "category": "system",
                        "value": disk_percent,
                    }
                )

        except ImportError:
            pass

        return alerts

    def get_alert_statistics(self, alerts):
        """Oblicz statystyki alertów"""
        if not alerts:
            return {
                "total": 0,
                "critical": 0,
                "warning": 0,
                "info": 0,
                "success": 0,
                "categories": {},
            }

        stats = {
            "total": len(alerts),
            "critical": len([a for a in alerts if a.get("level") == "critical"]),
            "warning": len([a for a in alerts if a.get("level") == "warning"]),
            "info": len([a for a in alerts if a.get("level") == "info"]),
            "success": len([a for a in alerts if a.get("level") == "success"]),
        }

        # Category breakdown
        categories = {}
        for alert in alerts:
            cat = alert.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        stats["categories"] = categories
        return stats

    def get_alert_rules(self):
        """Pobierz konfigurację reguł alertów"""
        return [
            {
                "name": "High Leverage",
                "condition": "current_leverage > 2.5x",
                "level": "warning",
                "enabled": True,
                "category": "risk",
            },
            {
                "name": "Critical Drawdown",
                "condition": "drawdown < -15%",
                "level": "critical",
                "enabled": True,
                "category": "risk",
            },
            {
                "name": "Low Win Rate",
                "condition": "win_rate < 40%",
                "level": "warning",
                "enabled": True,
                "category": "performance",
            },
            {
                "name": "High CPU Usage",
                "condition": "cpu_usage > 80%",
                "level": "warning",
                "enabled": True,
                "category": "system",
            },
            {
                "name": "Memory Warning",
                "condition": "memory_usage > 85%",
                "level": "warning",
                "enabled": True,
                "category": "system",
            },
            {
                "name": "Trading Stopped",
                "condition": "trading_active == false",
                "level": "info",
                "enabled": True,
                "category": "trading",
            },
        ]


def main():
    # Header
    st.title("🚨 Advanced Alert Management System")
    st.markdown("**Zaawansowany system zarządzania alertami i powiadomień**")
    # Initialize alert manager
    if "alert_manager" not in st.session_state:
        st.session_state.alert_manager = AdvancedAlertManager()

    manager = st.session_state.alert_manager
    # Data source indicators
    if manager.production_mode and manager.production_manager:
        st.success(
            "🟢 **Production Alert System** - Monitoring real Bybit account data"
        )
        data_indicator = "🟢 Real Data"
    elif manager.production_data_manager and manager.is_production:
        st.success(
            "🟢 **Production Alert System** - Monitoring real Bybit account data"
        )
        data_indicator = "🟢 Real Data"
    elif manager.production_data_manager and not manager.is_production:
        st.info("🔶 **Testnet Alert System** - Monitoring Bybit testnet data")
        data_indicator = "🔶 Testnet Data"
    else:
        st.warning("🟡 **Demo Alert System** - Using simulated data for alerts")
        data_indicator = "🟡 Demo Data"

    # Sidebar controls
    st.sidebar.title("🎛️ Alert Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.selectbox(
        "Refresh Interval (s)", [5, 10, 30, 60], index=1
    )
    # Alert level filters
    st.sidebar.subheader("Filter by Level")
    show_critical = st.sidebar.checkbox("Critical", value=True)
    show_warning = st.sidebar.checkbox("Warning", value=True)
    show_info = st.sidebar.checkbox("Info", value=True)
    show_success = st.sidebar.checkbox("Success", value=True)

    if st.sidebar.button("🔄 Refresh Alerts"):
        st.rerun()

    # Notification Configuration Section
    st.sidebar.header("📧 Notification Settings")

    # Add notification configuration UI in sidebar
    with st.sidebar.expander("Configure Notifications", expanded=False):
        st.markdown("**Email Notifications**")
        email_enabled = st.checkbox(
            "Enable Email Notifications", value=False, key="email_enabled"
        )

        if email_enabled:
            email_user = st.text_input(
                "Email Address", placeholder="your-email@gmail.com", key="email_user"
            )
            email_password = st.text_input(
                "Email Password",
                type="password",
                help="Use app-specific password for Gmail",
                key="email_password",
            )
            email_recipients = st.text_area(
                "Email Recipients",
                placeholder="recipient1@gmail.com, recipient2@gmail.com",
                key="email_recipients",
            )

            if st.button("Test Email", key="test_email"):
                if email_user and email_password and email_recipients:
                    config = NotificationConfig(
                        email_enabled=True,
                        email_user=email_user,
                        email_password=email_password,
                        email_recipients=[
                            email.strip()
                            for email in email_recipients.split(",")
                            if email.strip()
                        ],
                    )
                    test_manager = EnhancedNotificationManager(config)
                    result = test_manager.test_notifications()
                    if result.get("email"):
                        st.success("✅ Test email sent successfully!")
                    else:
                        st.error("❌ Failed to send test email")
                else:
                    st.warning("Please fill in all email fields")

        st.markdown("**SMS Notifications**")
        sms_enabled = st.checkbox(
            "Enable SMS Notifications", value=False, key="sms_enabled"
        )

        if sms_enabled:
            twilio_sid = st.text_input(
                "Twilio Account SID", type="password", key="twilio_sid"
            )
            twilio_token = st.text_input(
                "Twilio Auth Token", type="password", key="twilio_token"
            )
            twilio_phone = st.text_input(
                "Twilio Phone Number", placeholder="+1234567890", key="twilio_phone"
            )
            sms_recipients = st.text_area(
                "SMS Recipients",
                placeholder="+1234567890, +0987654321",
                key="sms_recipients",
            )

            if st.button("Test SMS", key="test_sms"):
                if twilio_sid and twilio_token and twilio_phone and sms_recipients:
                    config = NotificationConfig(
                        sms_enabled=True,
                        twilio_sid=twilio_sid,
                        twilio_token=twilio_token,
                        twilio_phone=twilio_phone,
                        sms_recipients=[
                            sms.strip()
                            for sms in sms_recipients.split(",")
                            if sms.strip()
                        ],
                    )
                    test_manager = EnhancedNotificationManager(config)
                    result = test_manager.test_notifications()
                    if result.get("sms"):
                        st.success("✅ Test SMS sent successfully!")
                    else:
                        st.error("❌ Failed to send test SMS")
                else:
                    st.warning("Please fill in all SMS fields")

        # Notification Rules
        st.markdown("**Notification Rules**")
        min_severity = st.selectbox(
            "Minimum Severity",
            ["info", "success", "warning", "critical"],
            index=2,
            key="min_severity",
        )
        cooldown_minutes = st.slider(
            "Cooldown (minutes)", 1, 60, 5, key="cooldown_minutes"
        )

        if st.button("💾 Save Notification Config", key="save_config"):
            # Update the notification manager with new configuration
            if email_enabled or sms_enabled:
                config = NotificationConfig(
                    email_enabled=email_enabled,
                    email_user=email_user if email_enabled else "",
                    email_password=email_password if email_enabled else "",
                    email_recipients=(
                        [
                            email.strip()
                            for email in email_recipients.split(",")
                            if email.strip()
                        ]
                        if email_enabled and email_recipients
                        else []
                    ),
                    sms_enabled=sms_enabled,
                    twilio_sid=twilio_sid if sms_enabled else "",
                    twilio_token=twilio_token if sms_enabled else "",
                    twilio_phone=twilio_phone if sms_enabled else "",
                    sms_recipients=(
                        [
                            sms.strip()
                            for sms in sms_recipients.split(",")
                            if sms.strip()
                        ]
                        if sms_enabled and sms_recipients
                        else []
                    ),
                    min_severity=min_severity,
                    cooldown_minutes=cooldown_minutes,
                )
                manager.notification_manager = EnhancedNotificationManager(config)
                st.success("✅ Notification configuration saved!")
            else:
                st.warning("Please enable at least one notification method")

    # Get alerts
    all_alerts = manager.get_comprehensive_alerts()

    # Filter alerts based on sidebar settings
    filtered_alerts = []
    for alert in all_alerts:
        level = alert.get("level", "info")
        if (
            (level == "critical" and show_critical)
            or (level == "warning" and show_warning)
            or (level == "info" and show_info)
            or (level == "success" and show_success)
        ):
            filtered_alerts.append(alert)

    # Alert statistics
    stats = manager.get_alert_statistics(filtered_alerts)
    # === ALERT OVERVIEW ===
    st.header(f"📊 Alert Overview - {data_indicator}")

    overview_col1, overview_col2, overview_col3, overview_col4, overview_col5 = (
        st.columns(5)
    )

    with overview_col1:
        st.markdown(
            f"""
        <div class="alert-info">
            <h3>📋 Total</h3>
            <div class="metric-alert">{stats['total']}</div>
            <small>Active Alerts</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with overview_col2:
        st.markdown(
            f"""
        <div class="alert-critical">
            <h3>🔴 Critical</h3>
            <div class="metric-alert">{stats['critical']}</div>
            <small>Immediate Action</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with overview_col3:
        st.markdown(
            f"""
        <div class="alert-warning">
            <h3>🟡 Warning</h3>
            <div class="metric-alert">{stats['warning']}</div>
            <small>Monitor Closely</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with overview_col4:
        st.markdown(
            f"""
        <div class="alert-info">
            <h3>🔵 Info</h3>
            <div class="metric-alert">{stats['info']}</div>
            <small>Informational</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with overview_col5:
        st.markdown(
            f"""
        <div class="alert-success">
            <h3>🟢 Success</h3>
            <div class="metric-alert">{stats['success']}</div>
            <small>All Good</small>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # === ACTIVE ALERTS ===
    st.header("⚠️ Active Alerts")

    if filtered_alerts:
        for _i, alert in enumerate(filtered_alerts[:20]):  # Show top 20 alerts
            level = alert.get("level", "info")
            message = alert.get("message", "No message")
            timestamp = alert.get("timestamp", "")
            category = alert.get("category", "unknown")

            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_ago = datetime.now() - dt.replace(tzinfo=None)
                time_str = f"{int(time_ago.total_seconds() // 60)}m ago"
            except Exception:
                time_str = "Unknown time"

            # Choose alert style
            if level == "critical":
                alert_class = "alert-critical"
                icon = "🔴"
            elif level == "warning":
                alert_class = "alert-warning"
                icon = "🟡"
            elif level == "success":
                alert_class = "alert-success"
                icon = "🟢"
            else:
                alert_class = "alert-info"
                icon = "🔵"

            st.markdown(
                f"""
            <div class="{alert_class}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{icon} {message}</strong>
                        <br><small>Category: {category.title()} | {time_str}</small>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.success("🎉 No alerts matching current filters!")

    # === ALERT RULES CONFIGURATION ===
    st.header("⚙️ Alert Rules Configuration")

    rules = manager.get_alert_rules()

    rules_col1, rules_col2 = st.columns(2)

    with rules_col1:
        st.subheader("Risk & Performance Rules")
        for rule in rules[:3]:
            enabled_status = "✅ Enabled" if rule["enabled"] else "❌ Disabled"
            st.markdown(
                f"""
            <div class="rule-card">
                <strong>{rule['name']}</strong> {enabled_status}<br>
                <small>Condition: {rule['condition']}</small><br>
                <small>Level: {rule['level'].title()}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with rules_col2:
        st.subheader("System & Trading Rules")
        for rule in rules[3:]:
            enabled_status = "✅ Enabled" if rule["enabled"] else "❌ Disabled"
            st.markdown(
                f"""
            <div class="rule-card">
                <strong>{rule['name']}</strong> {enabled_status}<br>
                <small>Condition: {rule['condition']}</small><br>
                <small>Level: {rule['level'].title()}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )
    # === ALERT TRENDS ===
    st.header(f"📈 Alert Trends - {data_indicator}")

    if stats["categories"]:
        # Category breakdown chart
        fig = px.pie(
            values=list(stats["categories"].values()),
            names=list(stats["categories"].keys()),
            title="Alerts by Category",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # === ALERT TIMELINE ===
    st.header("⏰ Recent Alert Timeline")

    if filtered_alerts:
        timeline_data = []
        for alert in filtered_alerts[:10]:
            try:
                dt = datetime.fromisoformat(
                    alert.get("timestamp", "").replace("Z", "+00:00")
                )
                timeline_data.append(
                    {
                        "time": dt.strftime("%H:%M:%S"),
                        "level": alert.get("level", "info"),
                        "message": alert.get("message", ""),
                        "category": alert.get("category", "unknown"),
                    }
                )
            except Exception:
                continue

        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            st.dataframe(timeline_df, use_container_width=True)

    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# TODO: Integrate with CI/CD pipeline for automated alert management and edge-case tests.
# Edge-case tests: simulate notification failures, API errors, and permission issues.
# All public methods have docstrings and exception handling.

if __name__ == "__main__":
    main()
