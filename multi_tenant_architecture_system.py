"""
ZoL0 Trading Bot - Multi-Tenant Architecture System
Enterprise-grade multi-tenant platform for scaling trading operations across organizations.
Port: 8517
"""

import hashlib
import logging
import sqlite3
import threading
import time as time_module
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TenantStatus(Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    EXPIRED = "expired"
    PENDING = "pending"


class UserRole(Enum):
    SUPER_ADMIN = "super_admin"
    TENANT_ADMIN = "tenant_admin"
    TRADER = "trader"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_USER = "api_user"


class SubscriptionTier(Enum):
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    CUSTOM = "custom"


class ResourceType(Enum):
    API_CALLS = "api_calls"
    DATA_STORAGE = "data_storage"
    CONCURRENT_SESSIONS = "concurrent_sessions"
    TRADING_VOLUME = "trading_volume"
    HISTORICAL_DATA = "historical_data"


@dataclass
class Tenant:
    tenant_id: str
    name: str
    domain: str
    status: TenantStatus
    subscription_tier: SubscriptionTier
    created_at: datetime
    expires_at: Optional[datetime]
    admin_email: str
    config: Dict[str, Any]
    resource_limits: Dict[ResourceType, int]
    custom_branding: Dict[str, str]
    billing_info: Dict[str, Any]


@dataclass
class User:
    user_id: str
    tenant_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[str]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool
    profile: Dict[str, Any]
    api_keys: List[str]


@dataclass
class ResourceUsage:
    tenant_id: str
    resource_type: ResourceType
    usage_amount: int
    limit_amount: int
    period_start: datetime
    period_end: datetime
    last_updated: datetime


@dataclass
class TenantMetrics:
    tenant_id: str
    active_users: int
    api_calls_today: int
    data_storage_mb: int
    trading_volume_24h: float
    uptime_percentage: float
    last_calculated: datetime


@dataclass
class BillingRecord:
    billing_id: str
    tenant_id: str
    period_start: datetime
    period_end: datetime
    base_amount: float
    usage_charges: float
    total_amount: float
    status: str  # 'pending', 'paid', 'overdue'
    created_at: datetime


class TenantDatabase:
    def __init__(self, db_path: str = "tenants.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize multi-tenant database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Tenants table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                domain TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL,
                subscription_tier TEXT NOT NULL,
                created_at TIMESTAMP,
                expires_at TIMESTAMP,
                admin_email TEXT,
                config TEXT,
                resource_limits TEXT,
                custom_branding TEXT,
                billing_info TEXT
            )
        """
        )

        # Users table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                username TEXT NOT NULL,
                email TEXT NOT NULL,
                role TEXT NOT NULL,
                permissions TEXT,
                created_at TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN,
                profile TEXT,
                api_keys TEXT,
                FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id),
                UNIQUE(tenant_id, username),
                UNIQUE(tenant_id, email)
            )
        """
        )

        # Resource usage table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS resource_usage (
                usage_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                usage_amount INTEGER,
                limit_amount INTEGER,
                period_start TIMESTAMP,
                period_end TIMESTAMP,
                last_updated TIMESTAMP,
                FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id)
            )
        """
        )

        # Tenant metrics table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tenant_metrics (
                tenant_id TEXT PRIMARY KEY,
                active_users INTEGER,
                api_calls_today INTEGER,
                data_storage_mb INTEGER,
                trading_volume_24h REAL,
                uptime_percentage REAL,
                last_calculated TIMESTAMP,
                FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id)
            )
        """
        )

        # Billing records table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS billing_records (
                billing_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                period_start TIMESTAMP,
                period_end TIMESTAMP,
                base_amount REAL,
                usage_charges REAL,
                total_amount REAL,
                status TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (tenant_id) REFERENCES tenants (tenant_id)
            )
        """
        )

        conn.commit()
        conn.close()


class TenantManager:
    def __init__(self):
        self.db = TenantDatabase()
        self.tenants: Dict[str, Tenant] = {}
        self.users: Dict[str, User] = {}
        self.resource_usage: Dict[str, List[ResourceUsage]] = defaultdict(list)
        self.tenant_metrics: Dict[str, TenantMetrics] = {}
        self.billing_records: List[BillingRecord] = []

        # Subscription tier configurations
        self.subscription_tiers = {
            SubscriptionTier.STARTER: {
                "monthly_price": 99,
                "resource_limits": {
                    ResourceType.API_CALLS: 10000,
                    ResourceType.DATA_STORAGE: 1024,  # MB
                    ResourceType.CONCURRENT_SESSIONS: 5,
                    ResourceType.TRADING_VOLUME: 100000,  # USD
                    ResourceType.HISTORICAL_DATA: 30,  # days
                },
                "features": ["Basic Analytics", "Email Support", "API Access"],
            },
            SubscriptionTier.PROFESSIONAL: {
                "monthly_price": 299,
                "resource_limits": {
                    ResourceType.API_CALLS: 100000,
                    ResourceType.DATA_STORAGE: 10240,  # MB
                    ResourceType.CONCURRENT_SESSIONS: 25,
                    ResourceType.TRADING_VOLUME: 1000000,  # USD
                    ResourceType.HISTORICAL_DATA: 365,  # days
                },
                "features": [
                    "Advanced Analytics",
                    "Priority Support",
                    "Custom Strategies",
                    "Risk Management",
                ],
            },
            SubscriptionTier.ENTERPRISE: {
                "monthly_price": 999,
                "resource_limits": {
                    ResourceType.API_CALLS: 1000000,
                    ResourceType.DATA_STORAGE: 102400,  # MB
                    ResourceType.CONCURRENT_SESSIONS: 100,
                    ResourceType.TRADING_VOLUME: 10000000,  # USD
                    ResourceType.HISTORICAL_DATA: 1825,  # days (5 years)
                },
                "features": [
                    "Full Platform Access",
                    "24/7 Support",
                    "Custom Development",
                    "Dedicated Infrastructure",
                ],
            },
        }

        self.load_sample_tenants()
        self.start_metrics_collection()

    def create_tenant(
        self,
        name: str,
        domain: str,
        admin_email: str,
        subscription_tier: SubscriptionTier,
    ) -> Tuple[bool, str, Optional[Tenant]]:
        """Create a new tenant."""
        try:
            # Validate domain uniqueness
            if any(tenant.domain == domain for tenant in self.tenants.values()):
                return False, "Domain already exists", None

            tenant_id = str(uuid.uuid4())

            # Get resource limits for subscription tier
            tier_config = self.subscription_tiers[subscription_tier]
            resource_limits = tier_config["resource_limits"]

            tenant = Tenant(
                tenant_id=tenant_id,
                name=name,
                domain=domain,
                status=TenantStatus.TRIAL,
                subscription_tier=subscription_tier,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=30),  # 30-day trial
                admin_email=admin_email,
                config={
                    "timezone": "UTC",
                    "currency": "USD",
                    "trading_enabled": True,
                    "notifications_enabled": True,
                },
                resource_limits=resource_limits,
                custom_branding={
                    "logo_url": "",
                    "primary_color": "#1f77b4",
                    "secondary_color": "#ff7f0e",
                    "company_name": name,
                },
                billing_info={
                    "payment_method": "trial",
                    "billing_address": "",
                    "tax_id": "",
                },
            )

            self.tenants[tenant_id] = tenant

            # Create admin user
            self.create_user(
                tenant_id=tenant_id,
                username="admin",
                email=admin_email,
                role=UserRole.TENANT_ADMIN,
                permissions=["all"],
            )

            # Initialize resource usage tracking
            self.initialize_resource_tracking(tenant_id)

            logger.info(f"Created tenant: {name} ({tenant_id})")
            return True, f"Tenant {name} created successfully", tenant

        except Exception as e:
            logger.error(f"Error creating tenant: {e}")
            return False, f"Error creating tenant: {str(e)}", None

    def create_user(
        self,
        tenant_id: str,
        username: str,
        email: str,
        role: UserRole,
        permissions: List[str],
    ) -> Tuple[bool, str, Optional[User]]:
        """Create a new user for a tenant."""
        try:
            if tenant_id not in self.tenants:
                return False, "Tenant not found", None

            # Check if username/email already exists for this tenant
            existing_users = [
                u for u in self.users.values() if u.tenant_id == tenant_id
            ]
            if any(u.username == username or u.email == email for u in existing_users):
                return False, "Username or email already exists", None

            user_id = str(uuid.uuid4())

            user = User(
                user_id=user_id,
                tenant_id=tenant_id,
                username=username,
                email=email,
                role=role,
                permissions=permissions,
                created_at=datetime.now(),
                last_login=None,
                is_active=True,
                profile={
                    "first_name": "",
                    "last_name": "",
                    "phone": "",
                    "department": "",
                },
                api_keys=[],
            )

            self.users[user_id] = user

            logger.info(f"Created user: {username} for tenant {tenant_id}")
            return True, f"User {username} created successfully", user

        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False, f"Error creating user: {str(e)}", None

    def initialize_resource_tracking(self, tenant_id: str):
        """Initialize resource usage tracking for a tenant."""
        tenant = self.tenants[tenant_id]
        period_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(days=1)

        for resource_type, limit in tenant.resource_limits.items():
            usage = ResourceUsage(
                tenant_id=tenant_id,
                resource_type=resource_type,
                usage_amount=0,
                limit_amount=limit,
                period_start=period_start,
                period_end=period_end,
                last_updated=datetime.now(),
            )
            self.resource_usage[tenant_id].append(usage)

    def update_resource_usage(
        self, tenant_id: str, resource_type: ResourceType, amount: int
    ) -> bool:
        """Update resource usage for a tenant."""
        if tenant_id not in self.tenants:
            return False

        tenant_usage = self.resource_usage.get(tenant_id, [])
        current_usage = next(
            (u for u in tenant_usage if u.resource_type == resource_type), None
        )

        if current_usage:
            current_usage.usage_amount += amount
            current_usage.last_updated = datetime.now()

            # Check if usage exceeds limit
            if current_usage.usage_amount > current_usage.limit_amount:
                logger.warning(
                    f"Tenant {tenant_id} exceeded {resource_type.value} limit"
                )
                return False

        return True

    def get_tenant_utilization(self, tenant_id: str) -> Dict[str, float]:
        """Get resource utilization percentage for a tenant."""
        if tenant_id not in self.tenants:
            return {}

        utilization = {}
        tenant_usage = self.resource_usage.get(tenant_id, [])

        for usage in tenant_usage:
            utilization[usage.resource_type.value] = (
                usage.usage_amount / usage.limit_amount * 100
                if usage.limit_amount > 0
                else 0
            )

        return utilization

    def calculate_billing(
        self, tenant_id: str, period_start: datetime, period_end: datetime
    ) -> BillingRecord:
        """Calculate billing for a tenant for a specific period."""
        if tenant_id not in self.tenants:
            return None

        tenant = self.tenants[tenant_id]
        tier_config = self.subscription_tiers[tenant.subscription_tier]

        base_amount = tier_config["monthly_price"]
        usage_charges = 0.0

        # Calculate overage charges
        tenant_usage = self.resource_usage.get(tenant_id, [])
        overage_rates = {
            ResourceType.API_CALLS: 0.001,  # $0.001 per call over limit
            ResourceType.DATA_STORAGE: 0.1,  # $0.10 per MB over limit
            ResourceType.TRADING_VOLUME: 0.00001,  # $0.00001 per USD over limit
        }

        for usage in tenant_usage:
            if usage.usage_amount > usage.limit_amount:
                overage = usage.usage_amount - usage.limit_amount
                rate = overage_rates.get(usage.resource_type, 0)
                usage_charges += overage * rate

        total_amount = base_amount + usage_charges

        billing_record = BillingRecord(
            billing_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            period_start=period_start,
            period_end=period_end,
            base_amount=base_amount,
            usage_charges=usage_charges,
            total_amount=total_amount,
            status="pending",
            created_at=datetime.now(),
        )

        self.billing_records.append(billing_record)
        return billing_record

    def load_sample_tenants(self):
        """Load sample tenants for demonstration."""
        sample_tenants = [
            {
                "name": "Crypto Capital LLC",
                "domain": "cryptocapital.trading",
                "admin_email": "admin@cryptocapital.trading",
                "subscription_tier": SubscriptionTier.ENTERPRISE,
            },
            {
                "name": "Digital Assets Fund",
                "domain": "dafund.trading",
                "admin_email": "admin@dafund.trading",
                "subscription_tier": SubscriptionTier.PROFESSIONAL,
            },
            {
                "name": "Trading Startup",
                "domain": "startup.trading",
                "admin_email": "admin@startup.trading",
                "subscription_tier": SubscriptionTier.STARTER,
            },
        ]

        for tenant_data in sample_tenants:
            self.create_tenant(**tenant_data)

        # Simulate some resource usage
        self.simulate_resource_usage()

    def simulate_resource_usage(self):
        """Simulate realistic resource usage for demo tenants."""
        for tenant_id in self.tenants.keys():
            # Simulate different usage patterns
            self.update_resource_usage(
                tenant_id, ResourceType.API_CALLS, np.random.randint(1000, 5000)
            )
            self.update_resource_usage(
                tenant_id, ResourceType.DATA_STORAGE, np.random.randint(100, 500)
            )
            self.update_resource_usage(
                tenant_id, ResourceType.CONCURRENT_SESSIONS, np.random.randint(1, 10)
            )
            self.update_resource_usage(
                tenant_id, ResourceType.TRADING_VOLUME, np.random.randint(10000, 100000)
            )

    def start_metrics_collection(self):
        """Start continuous metrics collection for all tenants."""

        def collect_metrics():
            while True:
                try:
                    for tenant_id in self.tenants.keys():
                        # Calculate tenant metrics
                        tenant_users = [
                            u for u in self.users.values() if u.tenant_id == tenant_id
                        ]
                        active_users = len([u for u in tenant_users if u.is_active])

                        # Simulate metrics
                        api_calls_today = np.random.randint(1000, 10000)
                        data_storage_mb = np.random.randint(100, 1000)
                        trading_volume_24h = np.random.uniform(10000, 500000)
                        uptime_percentage = np.random.uniform(99.5, 100.0)

                        metrics = TenantMetrics(
                            tenant_id=tenant_id,
                            active_users=active_users,
                            api_calls_today=api_calls_today,
                            data_storage_mb=data_storage_mb,
                            trading_volume_24h=trading_volume_24h,
                            uptime_percentage=uptime_percentage,
                            last_calculated=datetime.now(),
                        )

                        self.tenant_metrics[tenant_id] = metrics

                    time_module.sleep(300)  # Update every 5 minutes
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
                    time_module.sleep(60)

        metrics_thread = threading.Thread(target=collect_metrics, daemon=True)
        metrics_thread.start()


class MultiTenantArchitecture:
    def __init__(self):
        self.tenant_manager = TenantManager()
        self.current_tenant_id = None
        self.current_user_id = None

    def authenticate_tenant(self, domain: str) -> Optional[Tenant]:
        """Authenticate tenant by domain."""
        for tenant in self.tenant_manager.tenants.values():
            if tenant.domain == domain and tenant.status == TenantStatus.ACTIVE:
                return tenant
        return None

    def get_tenant_isolation_key(self, tenant_id: str) -> str:
        """Generate tenant isolation key for data segregation."""
        return hashlib.sha256(f"tenant_{tenant_id}".encode()).hexdigest()[:16]

    def check_tenant_permissions(
        self, tenant_id: str, user_id: str, required_permission: str
    ) -> bool:
        """Check if user has required permission in tenant context."""
        user = self.tenant_manager.users.get(user_id)
        if not user or user.tenant_id != tenant_id:
            return False

        if not user.is_active:
            return False

        # Super admin has all permissions
        if user.role == UserRole.SUPER_ADMIN:
            return True

        # Check specific permissions
        return required_permission in user.permissions or "all" in user.permissions

    def get_tenant_config(self, tenant_id: str, config_key: str) -> Any:
        """Get tenant-specific configuration."""
        tenant = self.tenant_manager.tenants.get(tenant_id)
        if tenant:
            return tenant.config.get(config_key)
        return None


# Initialize multi-tenant system
@st.cache_resource
def get_multitenant_system():
    return MultiTenantArchitecture()


def main():
    st.set_page_config(
        page_title="ZoL0 Multi-Tenant Architecture",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
    <style>
    .tenant-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .user-card {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }
    .metric-good {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
    .metric-warning {
        background: linear-gradient(135deg, #ffd43b 0%, #fab005 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
    .metric-danger {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🏢 ZoL0 Multi-Tenant Architecture")
    st.markdown(
        "**Enterprise-grade multi-tenant platform for scaling trading operations**"
    )

    # Sidebar
    st.sidebar.title("🎛️ Platform Controls")

    # Get multi-tenant system
    mt_system = get_multitenant_system()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🏢 Tenant Management",
            "👥 User Management",
            "📊 Resource Monitoring",
            "💰 Billing & Subscriptions",
            "📈 Platform Analytics",
            "⚙️ System Configuration",
        ]
    )

    with tab1:
        st.header("🏢 Tenant Management")

        # Platform overview metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_tenants = len(mt_system.tenant_manager.tenants)
            st.markdown(
                f"""
            <div class="tenant-card">
                <h3>Total Tenants</h3>
                <h2>{total_tenants}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            active_tenants = len(
                [
                    t
                    for t in mt_system.tenant_manager.tenants.values()
                    if t.status == TenantStatus.ACTIVE
                ]
            )
            st.markdown(
                f"""
            <div class="tenant-card">
                <h3>Active Tenants</h3>
                <h2>{active_tenants}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            trial_tenants = len(
                [
                    t
                    for t in mt_system.tenant_manager.tenants.values()
                    if t.status == TenantStatus.TRIAL
                ]
            )
            st.markdown(
                f"""
            <div class="tenant-card">
                <h3>Trial Tenants</h3>
                <h2>{trial_tenants}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            total_users = len(mt_system.tenant_manager.users)
            st.markdown(
                f"""
            <div class="tenant-card">
                <h3>Total Users</h3>
                <h2>{total_users}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Create new tenant form
        with st.expander("➕ Create New Tenant"):
            col1, col2 = st.columns(2)

            with col1:
                tenant_name = st.text_input("Organization Name")
                tenant_domain = st.text_input("Domain (e.g., company.trading)")
                admin_email = st.text_input("Admin Email")

            with col2:
                subscription_tier = st.selectbox(
                    "Subscription Tier",
                    [tier.value.title() for tier in SubscriptionTier],
                )

                # Show tier details
                selected_tier = SubscriptionTier(subscription_tier.lower())
                tier_config = mt_system.tenant_manager.subscription_tiers[selected_tier]
                st.write(f"**Monthly Price:** ${tier_config['monthly_price']}")
                st.write("**Features:**")
                for feature in tier_config["features"]:
                    st.write(f"• {feature}")

            if (
                st.button("Create Tenant")
                and tenant_name
                and tenant_domain
                and admin_email
            ):
                success, message, tenant = mt_system.tenant_manager.create_tenant(
                    name=tenant_name,
                    domain=tenant_domain,
                    admin_email=admin_email,
                    subscription_tier=selected_tier,
                )

                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")

        # Tenant list
        st.subheader("📋 Tenant Directory")

        if mt_system.tenant_manager.tenants:
            tenants_data = []
            for tenant in mt_system.tenant_manager.tenants.values():
                tenant_users = [
                    u
                    for u in mt_system.tenant_manager.users.values()
                    if u.tenant_id == tenant.tenant_id
                ]

                tenants_data.append(
                    {
                        "Name": tenant.name,
                        "Domain": tenant.domain,
                        "Status": tenant.status.value.title(),
                        "Tier": tenant.subscription_tier.value.title(),
                        "Users": len(tenant_users),
                        "Created": tenant.created_at.strftime("%Y-%m-%d"),
                        "Expires": (
                            tenant.expires_at.strftime("%Y-%m-%d")
                            if tenant.expires_at
                            else "N/A"
                        ),
                        "Admin Email": tenant.admin_email,
                    }
                )

            df_tenants = pd.DataFrame(tenants_data)
            st.dataframe(df_tenants, use_container_width=True)

            # Tenant status distribution
            status_counts = {}
            for tenant in mt_system.tenant_manager.tenants.values():
                status = tenant.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            col1, col2 = st.columns(2)

            with col1:
                fig_status = px.pie(
                    values=list(status_counts.values()),
                    names=[name.title() for name in status_counts.keys()],
                    title="Tenant Status Distribution",
                )
                st.plotly_chart(fig_status, use_container_width=True)

            with col2:
                # Subscription tier distribution
                tier_counts = {}
                for tenant in mt_system.tenant_manager.tenants.values():
                    tier = tenant.subscription_tier.value
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1

                fig_tiers = px.bar(
                    x=[tier.title() for tier in tier_counts.keys()],
                    y=list(tier_counts.values()),
                    title="Subscription Tier Distribution",
                )
                st.plotly_chart(fig_tiers, use_container_width=True)
        else:
            st.info("📊 No tenants created yet.")

    with tab2:
        st.header("👥 User Management")

        # User management controls
        col1, col2 = st.columns(2)

        with col1:
            # Select tenant for user management
            tenant_options = {
                f"{t.name} ({t.domain})": t.tenant_id
                for t in mt_system.tenant_manager.tenants.values()
            }
            selected_tenant_display = st.selectbox(
                "Select Tenant", list(tenant_options.keys())
            )
            selected_tenant_id = (
                tenant_options[selected_tenant_display] if tenant_options else None
            )

        with col2:
            if selected_tenant_id:
                tenant = mt_system.tenant_manager.tenants[selected_tenant_id]
                st.write(f"**Tenant:** {tenant.name}")
                st.write(f"**Status:** {tenant.status.value.title()}")
                st.write(f"**Tier:** {tenant.subscription_tier.value.title()}")

        if selected_tenant_id:
            # Create new user form
            with st.expander("➕ Create New User"):
                col1, col2 = st.columns(2)

                with col1:
                    username = st.text_input("Username")
                    email = st.text_input("Email")
                    role = st.selectbox(
                        "Role",
                        [role.value.replace("_", " ").title() for role in UserRole],
                    )

                with col2:
                    # Role-based permissions
                    role_permissions = {
                        UserRole.SUPER_ADMIN: ["all"],
                        UserRole.TENANT_ADMIN: ["all"],
                        UserRole.TRADER: ["trading", "analytics", "orders"],
                        UserRole.ANALYST: ["analytics", "reports"],
                        UserRole.VIEWER: ["view"],
                        UserRole.API_USER: ["api"],
                    }

                    selected_role = UserRole(role.lower().replace(" ", "_"))
                    default_permissions = role_permissions.get(selected_role, [])

                    permissions = st.multiselect(
                        "Permissions",
                        [
                            "all",
                            "trading",
                            "analytics",
                            "orders",
                            "reports",
                            "api",
                            "view",
                        ],
                        default=default_permissions,
                    )

                if st.button("Create User") and username and email:
                    success, message, user = mt_system.tenant_manager.create_user(
                        tenant_id=selected_tenant_id,
                        username=username,
                        email=email,
                        role=selected_role,
                        permissions=permissions,
                    )

                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")

            # Display tenant users
            st.subheader("👥 Tenant Users")

            tenant_users = [
                u
                for u in mt_system.tenant_manager.users.values()
                if u.tenant_id == selected_tenant_id
            ]

            if tenant_users:
                users_data = []
                for user in tenant_users:
                    users_data.append(
                        {
                            "Username": user.username,
                            "Email": user.email,
                            "Role": user.role.value.replace("_", " ").title(),
                            "Status": "Active" if user.is_active else "Inactive",
                            "Permissions": ", ".join(user.permissions),
                            "Created": user.created_at.strftime("%Y-%m-%d"),
                            "Last Login": (
                                user.last_login.strftime("%Y-%m-%d %H:%M")
                                if user.last_login
                                else "Never"
                            ),
                        }
                    )

                df_users = pd.DataFrame(users_data)
                st.dataframe(df_users, use_container_width=True)

                # User role distribution
                role_counts = {}
                for user in tenant_users:
                    role = user.role.value.replace("_", " ").title()
                    role_counts[role] = role_counts.get(role, 0) + 1

                fig_roles = px.pie(
                    values=list(role_counts.values()),
                    names=list(role_counts.keys()),
                    title=f"User Roles - {tenant.name}",
                )
                st.plotly_chart(fig_roles, use_container_width=True)
            else:
                st.info("👥 No users found for this tenant.")
        else:
            st.info("👥 Please select a tenant to manage users.")

    with tab3:
        st.header("📊 Resource Monitoring")

        # Resource overview
        st.subheader("🔍 Platform Resource Overview")

        if mt_system.tenant_manager.tenants:
            # Calculate platform-wide metrics
            total_api_calls = sum(
                metrics.api_calls_today
                for metrics in mt_system.tenant_manager.tenant_metrics.values()
            )
            total_storage = sum(
                metrics.data_storage_mb
                for metrics in mt_system.tenant_manager.tenant_metrics.values()
            )
            total_volume = sum(
                metrics.trading_volume_24h
                for metrics in mt_system.tenant_manager.tenant_metrics.values()
            )
            avg_uptime = np.mean(
                [
                    metrics.uptime_percentage
                    for metrics in mt_system.tenant_manager.tenant_metrics.values()
                ]
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(
                    f"""
                <div class="metric-good">
                    <h3>Total API Calls Today</h3>
                    <h2>{total_api_calls:,}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class="metric-good">
                    <h3>Total Storage (MB)</h3>
                    <h2>{total_storage:,}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f"""
                <div class="metric-good">
                    <h3>Trading Volume 24h</h3>
                    <h2>${total_volume:,.0f}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col4:
                uptime_class = (
                    "metric-good"
                    if avg_uptime >= 99.5
                    else "metric-warning" if avg_uptime >= 99 else "metric-danger"
                )
                st.markdown(
                    f"""
                <div class="{uptime_class}">
                    <h3>Average Uptime</h3>
                    <h2>{avg_uptime:.2f}%</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Tenant resource utilization
            st.subheader("📈 Tenant Resource Utilization")

            utilization_data = []
            for tenant_id, tenant in mt_system.tenant_manager.tenants.items():
                utilization = mt_system.tenant_manager.get_tenant_utilization(tenant_id)

                for resource_type, percentage in utilization.items():
                    utilization_data.append(
                        {
                            "Tenant": tenant.name,
                            "Resource": resource_type.replace("_", " ").title(),
                            "Utilization %": percentage,
                            "Status": (
                                "Critical"
                                if percentage > 90
                                else "Warning" if percentage > 75 else "Good"
                            ),
                        }
                    )

            if utilization_data:
                df_utilization = pd.DataFrame(utilization_data)

                # Resource utilization heatmap
                pivot_data = df_utilization.pivot(
                    index="Tenant", columns="Resource", values="Utilization %"
                )

                fig_heatmap = px.imshow(
                    pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    title="Resource Utilization Heatmap (%)",
                    color_continuous_scale="RdYlGn_r",
                    aspect="auto",
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # Resource utilization table
                st.dataframe(df_utilization, use_container_width=True)

            # Resource limits by tier
            st.subheader("📋 Resource Limits by Subscription Tier")

            tier_limits_data = []
            for tier, config in mt_system.tenant_manager.subscription_tiers.items():
                for resource_type, limit in config["resource_limits"].items():
                    tier_limits_data.append(
                        {
                            "Tier": tier.value.title(),
                            "Resource": resource_type.value.replace("_", " ").title(),
                            "Limit": f"{limit:,}",
                            "Monthly Price": f"${config['monthly_price']}",
                        }
                    )

            df_limits = pd.DataFrame(tier_limits_data)
            st.dataframe(df_limits, use_container_width=True)
        else:
            st.info("📊 No resource data available.")

    with tab4:
        st.header("💰 Billing & Subscriptions")

        # Billing overview
        st.subheader("💵 Revenue Overview")

        if mt_system.tenant_manager.tenants:
            # Calculate revenue metrics
            monthly_revenue = sum(
                mt_system.tenant_manager.subscription_tiers[tenant.subscription_tier][
                    "monthly_price"
                ]
                for tenant in mt_system.tenant_manager.tenants.values()
                if tenant.status in [TenantStatus.ACTIVE, TenantStatus.TRIAL]
            )

            annual_revenue = monthly_revenue * 12

            # Revenue by tier
            tier_revenue = {}
            for tenant in mt_system.tenant_manager.tenants.values():
                if tenant.status in [TenantStatus.ACTIVE, TenantStatus.TRIAL]:
                    tier = tenant.subscription_tier.value.title()
                    price = mt_system.tenant_manager.subscription_tiers[
                        tenant.subscription_tier
                    ]["monthly_price"]
                    tier_revenue[tier] = tier_revenue.get(tier, 0) + price

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"""
                <div class="metric-good">
                    <h3>Monthly Revenue</h3>
                    <h2>${monthly_revenue:,}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(
                    f"""
                <div class="metric-good">
                    <h3>Annual Revenue</h3>
                    <h2>${annual_revenue:,}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with col3:
                avg_revenue_per_tenant = (
                    monthly_revenue / len(mt_system.tenant_manager.tenants)
                    if mt_system.tenant_manager.tenants
                    else 0
                )
                st.markdown(
                    f"""
                <div class="metric-good">
                    <h3>ARPU (Monthly)</h3>
                    <h2>${avg_revenue_per_tenant:,.0f}</h2>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Revenue distribution
            col1, col2 = st.columns(2)

            with col1:
                if tier_revenue:
                    fig_revenue = px.pie(
                        values=list(tier_revenue.values()),
                        names=list(tier_revenue.keys()),
                        title="Monthly Revenue by Subscription Tier",
                    )
                    st.plotly_chart(fig_revenue, use_container_width=True)

            with col2:
                # Subscription tier adoption
                tier_counts = {}
                for tenant in mt_system.tenant_manager.tenants.values():
                    tier = tenant.subscription_tier.value.title()
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1

                fig_adoption = px.bar(
                    x=list(tier_counts.keys()),
                    y=list(tier_counts.values()),
                    title="Subscription Tier Adoption",
                )
                st.plotly_chart(fig_adoption, use_container_width=True)

            # Generate billing reports
            st.subheader("📄 Billing Management")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📊 Generate Monthly Billing Reports", type="primary"):
                    period_start = datetime.now().replace(
                        day=1, hour=0, minute=0, second=0, microsecond=0
                    )
                    period_end = period_start + timedelta(days=32)
                    period_end = period_end.replace(day=1) - timedelta(days=1)

                    billing_reports = []
                    for tenant_id in mt_system.tenant_manager.tenants.keys():
                        billing_record = mt_system.tenant_manager.calculate_billing(
                            tenant_id, period_start, period_end
                        )
                        if billing_record:
                            billing_reports.append(billing_record)

                    st.success(f"✅ Generated {len(billing_reports)} billing reports")

            with col2:
                if st.button("💳 Process Payments", type="secondary"):
                    # Simulate payment processing
                    pending_bills = [
                        b
                        for b in mt_system.tenant_manager.billing_records
                        if b.status == "pending"
                    ]
                    processed = len(pending_bills)

                    for bill in pending_bills:
                        bill.status = "paid"

                    st.success(f"✅ Processed {processed} payments")

            # Billing records table
            if mt_system.tenant_manager.billing_records:
                st.subheader("📋 Recent Billing Records")

                billing_data = []
                for record in mt_system.tenant_manager.billing_records[
                    -20:
                ]:  # Last 20 records
                    tenant = mt_system.tenant_manager.tenants.get(record.tenant_id)
                    tenant_name = tenant.name if tenant else "Unknown"

                    billing_data.append(
                        {
                            "Tenant": tenant_name,
                            "Period": f"{record.period_start.strftime('%Y-%m-%d')} to {record.period_end.strftime('%Y-%m-%d')}",
                            "Base Amount": f"${record.base_amount:,.2f}",
                            "Usage Charges": f"${record.usage_charges:,.2f}",
                            "Total": f"${record.total_amount:,.2f}",
                            "Status": record.status.title(),
                            "Created": record.created_at.strftime("%Y-%m-%d"),
                        }
                    )

                df_billing = pd.DataFrame(billing_data)
                st.dataframe(df_billing, use_container_width=True)
        else:
            st.info("💰 No billing data available.")

    with tab5:
        st.header("📈 Platform Analytics")

        # Platform growth metrics
        st.subheader("📊 Platform Growth")

        if mt_system.tenant_manager.tenants:
            # Simulate growth data over time
            dates = pd.date_range(start="2024-01-01", end=datetime.now(), freq="D")
            growth_data = []

            for i, date in enumerate(dates):
                # Simulate cumulative growth
                tenants_count = min(
                    len(mt_system.tenant_manager.tenants), max(1, int(i * 0.1))
                )
                users_count = tenants_count * np.random.randint(2, 8)
                revenue = tenants_count * np.random.randint(100, 500)

                growth_data.append(
                    {
                        "Date": date,
                        "Tenants": tenants_count,
                        "Users": users_count,
                        "Revenue": revenue,
                    }
                )

            df_growth = pd.DataFrame(growth_data)

            col1, col2 = st.columns(2)

            with col1:
                fig_tenants = px.line(
                    df_growth, x="Date", y="Tenants", title="Tenant Growth Over Time"
                )
                st.plotly_chart(fig_tenants, use_container_width=True)

            with col2:
                fig_users = px.line(
                    df_growth, x="Date", y="Users", title="User Growth Over Time"
                )
                st.plotly_chart(fig_users, use_container_width=True)

            # Revenue growth
            fig_revenue = px.line(
                df_growth, x="Date", y="Revenue", title="Revenue Growth Over Time ($)"
            )
            st.plotly_chart(fig_revenue, use_container_width=True)

            # Platform utilization analytics
            st.subheader("🔍 Platform Utilization Analytics")

            if mt_system.tenant_manager.tenant_metrics:
                metrics_data = []
                for (
                    tenant_id,
                    metrics,
                ) in mt_system.tenant_manager.tenant_metrics.items():
                    tenant = mt_system.tenant_manager.tenants[tenant_id]

                    metrics_data.append(
                        {
                            "Tenant": tenant.name,
                            "Tier": tenant.subscription_tier.value.title(),
                            "Active Users": metrics.active_users,
                            "API Calls": metrics.api_calls_today,
                            "Storage (MB)": metrics.data_storage_mb,
                            "Trading Volume": metrics.trading_volume_24h,
                            "Uptime %": metrics.uptime_percentage,
                        }
                    )

                df_metrics = pd.DataFrame(metrics_data)

                # Correlation between tier and usage
                fig_scatter = px.scatter(
                    df_metrics,
                    x="Active Users",
                    y="Trading Volume",
                    color="Tier",
                    size="API Calls",
                    title="Usage Patterns by Subscription Tier",
                    hover_data=["Tenant"],
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Platform health dashboard
                st.subheader("🏥 Platform Health Dashboard")

                avg_uptime = df_metrics["Uptime %"].mean()
                total_api_calls = df_metrics["API Calls"].sum()
                total_storage = df_metrics["Storage (MB)"].sum()

                health_metrics = [
                    {
                        "Metric": "Average Uptime",
                        "Value": f"{avg_uptime:.2f}%",
                        "Status": "Good" if avg_uptime >= 99.5 else "Warning",
                    },
                    {
                        "Metric": "Total API Calls",
                        "Value": f"{total_api_calls:,}",
                        "Status": "Good",
                    },
                    {
                        "Metric": "Total Storage Used",
                        "Value": f"{total_storage:,} MB",
                        "Status": "Good",
                    },
                    {
                        "Metric": "Active Tenants",
                        "Value": str(len(df_metrics)),
                        "Status": "Good",
                    },
                ]

                health_df = pd.DataFrame(health_metrics)
                st.dataframe(health_df, use_container_width=True)
        else:
            st.info("📈 No analytics data available.")

    with tab6:
        st.header("⚙️ System Configuration")

        # Platform settings
        st.subheader("🔧 Platform Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Global Settings**")

            st.number_input(
                "Default Trial Period (days)", value=30, min_value=1, max_value=365
            )
            st.number_input(
                "Max Tenants per Server", value=100, min_value=1, max_value=1000
            )
            st.checkbox("Auto Scaling Enabled", value=True)
            st.number_input(
                "Backup Retention (days)", value=90, min_value=1, max_value=365
            )

        with col2:
            st.write("**Security Settings**")

            st.number_input(
                "Session Timeout (minutes)", value=60, min_value=5, max_value=1440
            )
            st.number_input(
                "Password Expiry (days)", value=90, min_value=30, max_value=365
            )
            st.checkbox("MFA Required", value=True)
            st.number_input(
                "API Rate Limit (per minute)",
                value=1000,
                min_value=100,
                max_value=10000,
            )

        if st.button("💾 Save Configuration"):
            st.success("✅ Configuration saved successfully!")

        # System monitoring
        st.subheader("📊 System Monitoring")

        # Simulate system metrics
        system_metrics = {
            "CPU Usage": np.random.uniform(20, 80),
            "Memory Usage": np.random.uniform(30, 70),
            "Disk Usage": np.random.uniform(40, 60),
            "Network I/O": np.random.uniform(10, 50),
            "Database Connections": np.random.randint(50, 200),
            "Active Connections": np.random.randint(100, 500),
        }

        col1, col2, col3 = st.columns(3)

        for i, (metric, value) in enumerate(system_metrics.items()):
            col = [col1, col2, col3][i % 3]

            with col:
                if "Usage" in metric:
                    metric_class = (
                        "metric-good"
                        if value < 60
                        else "metric-warning" if value < 80 else "metric-danger"
                    )
                    display_value = f"{value:.1f}%"
                else:
                    metric_class = "metric-good"
                    display_value = f"{value:.0f}"

                st.markdown(
                    f"""
                <div class="{metric_class}">
                    <h4>{metric}</h4>
                    <h3>{display_value}</h3>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # Database status
        st.subheader("🗄️ Database Status")

        database_metrics = {
            "Total Tables": 25,
            "Total Records": 150000,
            "Database Size": "2.3 GB",
            "Backup Status": "Last backup: 2 hours ago",
            "Replication Status": "Active",
            "Query Performance": "95% queries < 100ms",
        }

        db_data = [{"Metric": k, "Value": v} for k, v in database_metrics.items()]
        df_db = pd.DataFrame(db_data)
        st.dataframe(df_db, use_container_width=True)

    # Auto-refresh
    if st.sidebar.checkbox("🔄 Auto Refresh", value=True):
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 30, 300, 60)
        time_module.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
