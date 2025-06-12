"""
ZoL0 Trading Bot - Regulatory Compliance Dashboard
Advanced regulatory monitoring and compliance reporting system for professional trading operations.
Port: 8515
"""

import logging
import sqlite3
import threading
import time as time_module
import uuid
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    PENDING_REVIEW = "pending_review"


class RegulationType(Enum):
    POSITION_LIMITS = "position_limits"
    CONCENTRATION_LIMITS = "concentration_limits"
    RISK_LIMITS = "risk_limits"
    REPORTING_REQUIREMENTS = "reporting_requirements"
    TRADING_HOURS = "trading_hours"
    MARKET_MAKING = "market_making"
    ANTI_MONEY_LAUNDERING = "aml"
    KNOW_YOUR_CUSTOMER = "kyc"


class ReportType(Enum):
    DAILY_TRADE_REPORT = "daily_trade_report"
    POSITION_REPORT = "position_report"
    RISK_REPORT = "risk_report"
    AML_REPORT = "aml_report"
    REGULATORY_FILING = "regulatory_filing"


@dataclass
class ComplianceRule:
    rule_id: str
    name: str
    regulation_type: RegulationType
    description: str
    threshold_value: float
    threshold_type: str  # 'max', 'min', 'range'
    severity: str  # 'low', 'medium', 'high', 'critical'
    is_active: bool
    created_at: datetime
    last_updated: datetime


@dataclass
class ComplianceViolation:
    violation_id: str
    rule_id: str
    timestamp: datetime
    severity: str
    status: ComplianceStatus
    description: str
    current_value: float
    threshold_value: float
    affected_positions: List[str]
    remediation_actions: List[str]
    resolved_at: Optional[datetime] = None
    notes: str = ""


@dataclass
class RegulatoryReport:
    report_id: str
    report_type: ReportType
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    file_path: str
    status: str  # 'draft', 'submitted', 'approved', 'rejected'
    submission_deadline: datetime
    data_summary: Dict[str, Any]


@dataclass
class PositionMonitoring:
    symbol: str
    current_position: float
    position_limit: float
    concentration_percent: float
    concentration_limit: float
    risk_exposure: float
    last_updated: datetime


class ComplianceDatabase:
    def __init__(self, db_path: str = "compliance.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize compliance database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Compliance rules table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS compliance_rules (
                rule_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                regulation_type TEXT NOT NULL,
                description TEXT,
                threshold_value REAL,
                threshold_type TEXT,
                severity TEXT,
                is_active BOOLEAN,
                created_at TIMESTAMP,
                last_updated TIMESTAMP
            )
        """
        )

        # Compliance violations table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS compliance_violations (
                violation_id TEXT PRIMARY KEY,
                rule_id TEXT,
                timestamp TIMESTAMP,
                severity TEXT,
                status TEXT,
                description TEXT,
                current_value REAL,
                threshold_value REAL,
                affected_positions TEXT,
                remediation_actions TEXT,
                resolved_at TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (rule_id) REFERENCES compliance_rules (rule_id)
            )
        """
        )

        # Regulatory reports table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS regulatory_reports (
                report_id TEXT PRIMARY KEY,
                report_type TEXT NOT NULL,
                period_start TIMESTAMP,
                period_end TIMESTAMP,
                generated_at TIMESTAMP,
                file_path TEXT,
                status TEXT,
                submission_deadline TIMESTAMP,
                data_summary TEXT
            )
        """
        )

        # Position monitoring table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS position_monitoring (
                symbol TEXT PRIMARY KEY,
                current_position REAL,
                position_limit REAL,
                concentration_percent REAL,
                concentration_limit REAL,
                risk_exposure REAL,
                last_updated TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()


class RegulatoryComplianceSystem:
    def __init__(self):
        self.db = ComplianceDatabase()
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.violations: List[ComplianceViolation] = []
        self.reports: Dict[str, RegulatoryReport] = {}
        self.position_monitoring: Dict[str, PositionMonitoring] = {}
        self.monitoring_active = True
        self.load_default_rules()
        self.start_monitoring()

    def load_default_rules(self):
        """Load default compliance rules."""
        default_rules = [
            ComplianceRule(
                rule_id="POS_LIMIT_001",
                name="Maximum Position Size",
                regulation_type=RegulationType.POSITION_LIMITS,
                description="Maximum position size per symbol",
                threshold_value=1000000.0,
                threshold_type="max",
                severity="high",
                is_active=True,
                created_at=datetime.now(),
                last_updated=datetime.now(),
            ),
            ComplianceRule(
                rule_id="CONC_LIMIT_001",
                name="Portfolio Concentration Limit",
                regulation_type=RegulationType.CONCENTRATION_LIMITS,
                description="Maximum concentration in single asset",
                threshold_value=25.0,
                threshold_type="max",
                severity="medium",
                is_active=True,
                created_at=datetime.now(),
                last_updated=datetime.now(),
            ),
            ComplianceRule(
                rule_id="RISK_LIMIT_001",
                name="Daily Risk Limit",
                regulation_type=RegulationType.RISK_LIMITS,
                description="Maximum daily risk exposure",
                threshold_value=500000.0,
                threshold_type="max",
                severity="critical",
                is_active=True,
                created_at=datetime.now(),
                last_updated=datetime.now(),
            ),
            ComplianceRule(
                rule_id="TRADING_HOURS_001",
                name="Trading Hours Compliance",
                regulation_type=RegulationType.TRADING_HOURS,
                description="Trading only during market hours",
                threshold_value=0.0,
                threshold_type="range",
                severity="medium",
                is_active=True,
                created_at=datetime.now(),
                last_updated=datetime.now(),
            ),
        ]

        for rule in default_rules:
            self.compliance_rules[rule.rule_id] = rule

    def check_position_compliance(
        self, symbol: str, position_size: float, portfolio_value: float
    ) -> List[ComplianceViolation]:
        """Check position compliance against all relevant rules."""
        violations = []

        # Check position limits
        for rule_id, rule in self.compliance_rules.items():
            if not rule.is_active:
                continue

            if rule.regulation_type == RegulationType.POSITION_LIMITS:
                if abs(position_size) > rule.threshold_value:
                    violation = ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id=rule_id,
                        timestamp=datetime.now(),
                        severity=rule.severity,
                        status=ComplianceStatus.VIOLATION,
                        description=f"Position size {position_size:,.2f} exceeds limit {rule.threshold_value:,.2f}",
                        current_value=abs(position_size),
                        threshold_value=rule.threshold_value,
                        affected_positions=[symbol],
                        remediation_actions=[
                            "Reduce position size",
                            "Review position limits",
                        ],
                    )
                    violations.append(violation)

            elif rule.regulation_type == RegulationType.CONCENTRATION_LIMITS:
                concentration = (
                    (abs(position_size) / portfolio_value) * 100
                    if portfolio_value > 0
                    else 0
                )
                if concentration > rule.threshold_value:
                    violation = ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id=rule_id,
                        timestamp=datetime.now(),
                        severity=rule.severity,
                        status=ComplianceStatus.VIOLATION,
                        description=f"Concentration {concentration:.2f}% exceeds limit {rule.threshold_value:.2f}%",
                        current_value=concentration,
                        threshold_value=rule.threshold_value,
                        affected_positions=[symbol],
                        remediation_actions=[
                            "Diversify portfolio",
                            "Reduce position concentration",
                        ],
                    )
                    violations.append(violation)

        return violations

    def check_trading_hours_compliance(self) -> Optional[ComplianceViolation]:
        """Check if current time is within allowed trading hours."""
        current_time = datetime.now().time()
        market_open = time(9, 30)  # 9:30 AM
        market_close = time(16, 0)  # 4:00 PM

        if not (market_open <= current_time <= market_close):
            rule_id = "TRADING_HOURS_001"
            if rule_id in self.compliance_rules:
                violation = ComplianceViolation(
                    violation_id=str(uuid.uuid4()),
                    rule_id=rule_id,
                    timestamp=datetime.now(),
                    severity="medium",
                    status=ComplianceStatus.VIOLATION,
                    description=f"Trading outside market hours: {current_time}",
                    current_value=current_time.hour + current_time.minute / 60,
                    threshold_value=16.0,
                    affected_positions=[],
                    remediation_actions=["Stop trading", "Wait for market open"],
                )
                return violation
        return None

    def generate_daily_trade_report(self, date: datetime) -> RegulatoryReport:
        """Generate daily trade report."""
        report_id = f"DTR_{date.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"

        # Simulate trade data
        trades_data = {
            "total_trades": np.random.randint(50, 200),
            "total_volume": np.random.uniform(1000000, 5000000),
            "symbols_traded": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"],
            "avg_trade_size": np.random.uniform(10000, 50000),
            "largest_trade": np.random.uniform(100000, 500000),
        }

        report = RegulatoryReport(
            report_id=report_id,
            report_type=ReportType.DAILY_TRADE_REPORT,
            period_start=date.replace(hour=0, minute=0, second=0),
            period_end=date.replace(hour=23, minute=59, second=59),
            generated_at=datetime.now(),
            file_path=f"reports/{report_id}.json",
            status="draft",
            submission_deadline=date + timedelta(days=1),
            data_summary=trades_data,
        )

        self.reports[report_id] = report
        return report

    def generate_position_report(self) -> RegulatoryReport:
        """Generate current position report."""
        report_id = f"POS_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"

        # Simulate position data
        position_data = {
            "total_positions": len(self.position_monitoring),
            "total_exposure": sum(
                pos.risk_exposure for pos in self.position_monitoring.values()
            ),
            "largest_position": max(
                (pos.current_position for pos in self.position_monitoring.values()),
                default=0,
            ),
            "concentration_metrics": {
                symbol: pos.concentration_percent
                for symbol, pos in self.position_monitoring.items()
            },
        }

        report = RegulatoryReport(
            report_id=report_id,
            report_type=ReportType.POSITION_REPORT,
            period_start=datetime.now().replace(hour=0, minute=0, second=0),
            period_end=datetime.now(),
            generated_at=datetime.now(),
            file_path=f"reports/{report_id}.json",
            status="draft",
            submission_deadline=datetime.now() + timedelta(days=7),
            data_summary=position_data,
        )

        self.reports[report_id] = report
        return report

    def update_position_monitoring(
        self, symbol: str, position_size: float, portfolio_value: float
    ):
        """Update position monitoring data."""
        # Get or create position monitoring
        if symbol not in self.position_monitoring:
            self.position_monitoring[symbol] = PositionMonitoring(
                symbol=symbol,
                current_position=position_size,
                position_limit=1000000.0,  # Default limit
                concentration_percent=0.0,
                concentration_limit=25.0,  # Default 25%
                risk_exposure=abs(position_size),
                last_updated=datetime.now(),
            )

        pos_monitor = self.position_monitoring[symbol]
        pos_monitor.current_position = position_size
        pos_monitor.concentration_percent = (
            (abs(position_size) / portfolio_value * 100) if portfolio_value > 0 else 0
        )
        pos_monitor.risk_exposure = abs(position_size)
        pos_monitor.last_updated = datetime.now()

    def start_monitoring(self):
        """Start continuous compliance monitoring."""

        def monitor():
            while self.monitoring_active:
                try:
                    # Simulate position updates
                    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
                    portfolio_value = 10000000.0  # $10M portfolio

                    for symbol in symbols:
                        position_size = np.random.uniform(-500000, 500000)
                        self.update_position_monitoring(
                            symbol, position_size, portfolio_value
                        )

                        # Check compliance
                        violations = self.check_position_compliance(
                            symbol, position_size, portfolio_value
                        )
                        self.violations.extend(violations)

                    # Check trading hours
                    hours_violation = self.check_trading_hours_compliance()
                    if hours_violation:
                        self.violations.append(hours_violation)

                    # Limit violations list size
                    if len(self.violations) > 1000:
                        self.violations = self.violations[-500:]

                    time_module.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time_module.sleep(60)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()


# Initialize compliance system
@st.cache_resource
def get_compliance_system():
    return RegulatoryComplianceSystem()


def main():
    st.set_page_config(
        page_title="ZoL0 Regulatory Compliance Dashboard",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .violation-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }
    .compliant-card {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("⚖️ ZoL0 Regulatory Compliance Dashboard")
    st.markdown("**Advanced regulatory monitoring and compliance reporting system**")

    # Sidebar
    st.sidebar.title("🔧 Compliance Controls")

    # Get compliance system
    compliance_system = get_compliance_system()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Compliance Overview",
            "⚠️ Violations & Alerts",
            "📋 Position Monitoring",
            "📄 Regulatory Reports",
            "⚙️ Rules Management",
            "📈 Compliance Analytics",
        ]
    )

    with tab1:
        st.header("📊 Compliance Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            active_violations = len(
                [
                    v
                    for v in compliance_system.violations
                    if v.status != ComplianceStatus.COMPLIANT
                ]
            )
            st.markdown(
                f"""
            <div class="metric-card">
                <h3>Active Violations</h3>
                <h2>{active_violations}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            active_rules = len(
                [r for r in compliance_system.compliance_rules.values() if r.is_active]
            )
            st.markdown(
                f"""
            <div class="metric-card">
                <h3>Active Rules</h3>
                <h2>{active_rules}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            monitored_positions = len(compliance_system.position_monitoring)
            st.markdown(
                f"""
            <div class="metric-card">
                <h3>Monitored Positions</h3>
                <h2>{monitored_positions}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            pending_reports = len(
                [r for r in compliance_system.reports.values() if r.status == "draft"]
            )
            st.markdown(
                f"""
            <div class="metric-card">
                <h3>Pending Reports</h3>
                <h2>{pending_reports}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Compliance status chart
        st.subheader("📈 Compliance Status Overview")

        # Create compliance summary data
        regulation_types = [rt.value for rt in RegulationType]
        compliance_data = []

        for reg_type in regulation_types:
            rules = [
                r
                for r in compliance_system.compliance_rules.values()
                if r.regulation_type.value == reg_type
            ]
            violations = [
                v
                for v in compliance_system.violations
                if any(r.rule_id == v.rule_id for r in rules)
            ]

            compliance_data.append(
                {
                    "Regulation Type": reg_type.replace("_", " ").title(),
                    "Total Rules": len(rules),
                    "Active Violations": len(violations),
                    "Compliance Rate": (
                        ((len(rules) - len(violations)) / len(rules) * 100)
                        if rules
                        else 100
                    ),
                }
            )

        df_compliance = pd.DataFrame(compliance_data)

        col1, col2 = st.columns(2)

        with col1:
            fig_compliance = px.bar(
                df_compliance,
                x="Regulation Type",
                y=["Total Rules", "Active Violations"],
                title="Rules vs Violations by Regulation Type",
                barmode="group",
            )
            fig_compliance.update_layout(height=400)
            st.plotly_chart(fig_compliance, use_container_width=True)

        with col2:
            fig_rate = px.bar(
                df_compliance,
                x="Regulation Type",
                y="Compliance Rate",
                title="Compliance Rate by Regulation Type (%)",
                color="Compliance Rate",
                color_continuous_scale="RdYlGn",
            )
            fig_rate.update_layout(height=400)
            st.plotly_chart(fig_rate, use_container_width=True)

    with tab2:
        st.header("⚠️ Violations & Alerts")

        # Filter controls
        col1, col2, col3 = st.columns(3)

        with col1:
            severity_filter = st.selectbox(
                "Filter by Severity", ["All", "critical", "high", "medium", "low"]
            )

        with col2:
            status_filter = st.selectbox(
                "Filter by Status",
                ["All"] + [status.value for status in ComplianceStatus],
            )

        with col3:
            time_filter = st.selectbox(
                "Time Period",
                ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"],
            )

        # Filter violations
        filtered_violations = compliance_system.violations.copy()

        if severity_filter != "All":
            filtered_violations = [
                v for v in filtered_violations if v.severity == severity_filter
            ]

        if status_filter != "All":
            filtered_violations = [
                v for v in filtered_violations if v.status.value == status_filter
            ]

        # Time filtering
        if time_filter != "All time":
            time_delta_map = {
                "Last 24 hours": timedelta(hours=24),
                "Last 7 days": timedelta(days=7),
                "Last 30 days": timedelta(days=30),
            }
            cutoff_time = datetime.now() - time_delta_map[time_filter]
            filtered_violations = [
                v for v in filtered_violations if v.timestamp >= cutoff_time
            ]

        # Display violations
        st.subheader(f"📋 Active Violations ({len(filtered_violations)})")

        if filtered_violations:
            for i, violation in enumerate(filtered_violations[-20:]):  # Show last 20
                {
                    "critical": "#ff4444",
                    "high": "#ff8800",
                    "medium": "#ffaa00",
                    "low": "#ffcc00",
                }.get(violation.severity, "#cccccc")

                with st.expander(
                    f"🚨 {violation.description} - {violation.severity.upper()}",
                    expanded=(i < 5),
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Rule ID:** {violation.rule_id}")
                        st.write(
                            f"**Timestamp:** {violation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                        )
                        st.write(f"**Status:** {violation.status.value}")
                        st.write(f"**Current Value:** {violation.current_value:,.2f}")
                        st.write(f"**Threshold:** {violation.threshold_value:,.2f}")

                    with col2:
                        st.write(
                            f"**Affected Positions:** {', '.join(violation.affected_positions)}"
                        )
                        st.write("**Remediation Actions:**")
                        for action in violation.remediation_actions:
                            st.write(f"• {action}")

                        if violation.notes:
                            st.write(f"**Notes:** {violation.notes}")
        else:
            st.success("✅ No violations found for the selected criteria!")

        # Violations timeline
        if filtered_violations:
            st.subheader("📈 Violations Timeline")

            # Create timeline data
            timeline_data = []
            for violation in filtered_violations:
                timeline_data.append(
                    {
                        "timestamp": violation.timestamp,
                        "severity": violation.severity,
                        "description": (
                            violation.description[:50] + "..."
                            if len(violation.description) > 50
                            else violation.description
                        ),
                    }
                )

            df_timeline = pd.DataFrame(timeline_data)
            df_timeline["timestamp"] = pd.to_datetime(df_timeline["timestamp"])

            # Group by hour for timeline chart
            df_timeline["hour"] = df_timeline["timestamp"].dt.floor("h")
            timeline_counts = (
                df_timeline.groupby(["hour", "severity"])
                .size()
                .reset_index(name="count")
            )

            fig_timeline = px.line(
                timeline_counts,
                x="hour",
                y="count",
                color="severity",
                title="Violations Over Time",
                labels={"hour": "Time", "count": "Number of Violations"},
            )
            fig_timeline.update_layout(height=400)
            st.plotly_chart(fig_timeline, use_container_width=True)

    with tab3:
        st.header("📋 Position Monitoring")

        # Position overview
        st.subheader("💼 Current Positions")

        if compliance_system.position_monitoring:
            positions_data = []
            for symbol, pos in compliance_system.position_monitoring.items():
                positions_data.append(
                    {
                        "Symbol": symbol,
                        "Position Size": f"{pos.current_position:,.2f}",
                        "Position Limit": f"{pos.position_limit:,.2f}",
                        "Utilization %": f"{(abs(pos.current_position) / pos.position_limit * 100):.1f}%",
                        "Concentration %": f"{pos.concentration_percent:.1f}%",
                        "Risk Exposure": f"{pos.risk_exposure:,.2f}",
                        "Last Updated": pos.last_updated.strftime("%H:%M:%S"),
                    }
                )

            df_positions = pd.DataFrame(positions_data)
            st.dataframe(df_positions, use_container_width=True)

            # Position charts
            col1, col2 = st.columns(2)

            with col1:
                # Position utilization chart
                symbols = list(compliance_system.position_monitoring.keys())
                utilizations = [
                    abs(pos.current_position) / pos.position_limit * 100
                    for pos in compliance_system.position_monitoring.values()
                ]

                fig_util = go.Figure(
                    data=[
                        go.Bar(
                            x=symbols,
                            y=utilizations,
                            marker_color=[
                                "red" if u > 80 else "orange" if u > 60 else "green"
                                for u in utilizations
                            ],
                        )
                    ]
                )
                fig_util.update_layout(
                    title="Position Limit Utilization (%)",
                    xaxis_title="Symbol",
                    yaxis_title="Utilization %",
                    height=400,
                )
                fig_util.add_hline(
                    y=80,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Warning Level",
                )
                st.plotly_chart(fig_util, use_container_width=True)

            with col2:
                # Concentration chart
                concentrations = [
                    pos.concentration_percent
                    for pos in compliance_system.position_monitoring.values()
                ]

                fig_conc = go.Figure(
                    data=[
                        go.Bar(
                            x=symbols,
                            y=concentrations,
                            marker_color=[
                                "red" if c > 25 else "orange" if c > 15 else "green"
                                for c in concentrations
                            ],
                        )
                    ]
                )
                fig_conc.update_layout(
                    title="Portfolio Concentration (%)",
                    xaxis_title="Symbol",
                    yaxis_title="Concentration %",
                    height=400,
                )
                fig_conc.add_hline(
                    y=25, line_dash="dash", line_color="red", annotation_text="Limit"
                )
                st.plotly_chart(fig_conc, use_container_width=True)
        else:
            st.info(
                "📊 No position data available. Positions will appear here when trading begins."
            )

    with tab4:
        st.header("📄 Regulatory Reports")

        # Report generation controls
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📈 Generate Daily Trade Report", type="primary"):
                report = compliance_system.generate_daily_trade_report(datetime.now())
                st.success(f"✅ Generated report: {report.report_id}")

        with col2:
            if st.button("📊 Generate Position Report", type="primary"):
                report = compliance_system.generate_position_report()
                st.success(f"✅ Generated report: {report.report_id}")

        with col3:
            if st.button("🔄 Refresh Reports"):
                st.rerun()

        # Display reports
        st.subheader("📋 Report Status")

        if compliance_system.reports:
            reports_data = []
            for report_id, report in compliance_system.reports.items():
                reports_data.append(
                    {
                        "Report ID": report_id,
                        "Type": report.report_type.value.replace("_", " ").title(),
                        "Period": f"{report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}",
                        "Generated": report.generated_at.strftime("%Y-%m-%d %H:%M"),
                        "Status": report.status.title(),
                        "Deadline": report.submission_deadline.strftime("%Y-%m-%d"),
                        "Days Until Deadline": (
                            report.submission_deadline - datetime.now()
                        ).days,
                    }
                )

            df_reports = pd.DataFrame(reports_data)
            st.dataframe(df_reports, use_container_width=True)

            # Report details
            if (
                st.selectbox(
                    "Select Report for Details",
                    ["None"] + list(compliance_system.reports.keys()),
                )
                != "None"
            ):
                selected_report_id = st.selectbox(
                    "Select Report for Details",
                    ["None"] + list(compliance_system.reports.keys()),
                )
                if selected_report_id != "None":
                    report = compliance_system.reports[selected_report_id]

                    st.subheader(f"📄 Report Details: {selected_report_id}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**Type:** {report.report_type.value}")
                        st.write(f"**Status:** {report.status}")
                        st.write(f"**Generated:** {report.generated_at}")
                        st.write(f"**Deadline:** {report.submission_deadline}")

                    with col2:
                        st.write("**Data Summary:**")
                        st.json(report.data_summary)
        else:
            st.info(
                "📊 No reports generated yet. Use the buttons above to generate reports."
            )

    with tab5:
        st.header("⚙️ Rules Management")

        # Add new rule section
        with st.expander("➕ Add New Compliance Rule"):
            col1, col2 = st.columns(2)

            with col1:
                rule_name = st.text_input("Rule Name")
                regulation_type = st.selectbox(
                    "Regulation Type", [rt.value for rt in RegulationType]
                )
                threshold_value = st.number_input("Threshold Value", value=0.0)
                threshold_type = st.selectbox("Threshold Type", ["max", "min", "range"])

            with col2:
                severity = st.selectbox(
                    "Severity", ["low", "medium", "high", "critical"]
                )
                description = st.text_area("Description")
                is_active = st.checkbox("Active", value=True)

            if st.button("Add Rule") and rule_name:
                rule_id = f"{regulation_type.upper()}_{len(compliance_system.compliance_rules):03d}"
                new_rule = ComplianceRule(
                    rule_id=rule_id,
                    name=rule_name,
                    regulation_type=RegulationType(regulation_type),
                    description=description,
                    threshold_value=threshold_value,
                    threshold_type=threshold_type,
                    severity=severity,
                    is_active=is_active,
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                )
                compliance_system.compliance_rules[rule_id] = new_rule
                st.success(f"✅ Added rule: {rule_id}")

        # Existing rules
        st.subheader("📋 Current Compliance Rules")

        if compliance_system.compliance_rules:
            rules_data = []
            for rule_id, rule in compliance_system.compliance_rules.items():
                rules_data.append(
                    {
                        "Rule ID": rule_id,
                        "Name": rule.name,
                        "Type": rule.regulation_type.value.replace("_", " ").title(),
                        "Threshold": f"{rule.threshold_value:,.2f} ({rule.threshold_type})",
                        "Severity": rule.severity.title(),
                        "Active": "✅" if rule.is_active else "❌",
                        "Created": rule.created_at.strftime("%Y-%m-%d"),
                    }
                )

            df_rules = pd.DataFrame(rules_data)
            st.dataframe(df_rules, use_container_width=True)
        else:
            st.info("📋 No compliance rules configured.")

    with tab6:
        st.header("📈 Compliance Analytics")

        # Analytics time period selector
        analytics_period = st.selectbox(
            "Analytics Period", ["Last 24 hours", "Last 7 days", "Last 30 days"]
        )

        # Violation trends
        st.subheader("📊 Violation Trends")

        if compliance_system.violations:
            # Create violation trend data
            now = datetime.now()
            period_map = {
                "Last 24 hours": timedelta(hours=24),
                "Last 7 days": timedelta(days=7),
                "Last 30 days": timedelta(days=30),
            }

            start_time = now - period_map[analytics_period]
            period_violations = [
                v for v in compliance_system.violations if v.timestamp >= start_time
            ]

            if period_violations:
                # Violations by severity over time
                violation_data = []
                for violation in period_violations:
                    violation_data.append(
                        {
                            "timestamp": violation.timestamp,
                            "severity": violation.severity,
                            "rule_type": compliance_system.compliance_rules.get(
                                violation.rule_id,
                                ComplianceRule(
                                    "",
                                    "",
                                    RegulationType.POSITION_LIMITS,
                                    "",
                                    0,
                                    "",
                                    "",
                                    True,
                                    datetime.now(),
                                    datetime.now(),
                                ),
                            ).regulation_type.value,
                        }
                    )

                df_violations = pd.DataFrame(violation_data)

                col1, col2 = st.columns(2)

                with col1:
                    # Violations by severity
                    severity_counts = df_violations["severity"].value_counts()
                    fig_severity = px.pie(
                        values=severity_counts.values,
                        names=severity_counts.index,
                        title="Violations by Severity",
                    )
                    st.plotly_chart(fig_severity, use_container_width=True)

                with col2:
                    # Violations by rule type
                    rule_counts = df_violations["rule_type"].value_counts()
                    fig_rules = px.pie(
                        values=rule_counts.values,
                        names=[
                            name.replace("_", " ").title() for name in rule_counts.index
                        ],
                        title="Violations by Rule Type",
                    )
                    st.plotly_chart(fig_rules, use_container_width=True)

                # Compliance score calculation
                st.subheader("🎯 Compliance Score")

                total_checks = (
                    len(compliance_system.compliance_rules) * 24
                )  # Assume hourly checks
                violations_count = len(period_violations)
                compliance_score = (
                    max(0, 100 - (violations_count / total_checks * 100))
                    if total_checks > 0
                    else 100
                )

                # Display compliance score
                score_color = (
                    "green"
                    if compliance_score >= 90
                    else "orange" if compliance_score >= 70 else "red"
                )

                st.markdown(
                    f"""
                <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #{score_color}33, #{score_color}22); border-radius: 10px;">
                    <h2 style="color: {score_color}; margin: 0;">Compliance Score</h2>
                    <h1 style="color: {score_color}; margin: 10px 0;">{compliance_score:.1f}%</h1>
                    <p style="margin: 0;">Based on {analytics_period.lower()}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            else:
                st.success("✅ No violations in the selected period!")
        else:
            st.info("📊 No violation data available for analytics.")

    # Auto-refresh
    if st.sidebar.checkbox("🔄 Auto Refresh", value=True):
        time_module.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
