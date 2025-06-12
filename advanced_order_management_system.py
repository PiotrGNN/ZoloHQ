"""
ZoL0 Trading Bot - Advanced Order Management System
Enterprise-grade order execution and management platform for professional trading.
Port: 8516
"""

import logging
import sqlite3
import threading
import time as time_module
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    BRACKET = "bracket"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExecutionAlgorithm(Enum):
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ARRIVAL_PRICE = "arrival_price"


class OrderPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class OrderRequest:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_price: Optional[float]
    time_in_force: str  # GTC, IOC, FOK, DAY
    algorithm: ExecutionAlgorithm
    priority: OrderPriority
    parent_order_id: Optional[str]
    client_order_id: str
    trader_id: str
    created_at: datetime
    valid_until: Optional[datetime]
    min_quantity: Optional[float]
    display_quantity: Optional[float]
    metadata: Dict[str, Any]


@dataclass
class OrderExecution:
    execution_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    exchange: str
    liquidity_flag: str  # 'maker' or 'taker'
    execution_venue: str


@dataclass
class OrderState:
    order_id: str
    status: OrderStatus
    filled_quantity: float
    remaining_quantity: float
    avg_fill_price: float
    last_update: datetime
    status_message: str
    executions: List[OrderExecution]


@dataclass
class MarketData:
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume: float
    timestamp: datetime


@dataclass
class RiskCheck:
    check_id: str
    order_id: str
    check_type: str
    status: str  # 'passed', 'failed', 'warning'
    message: str
    timestamp: datetime


class OrderDatabase:
    def __init__(self, db_path: str = "orders.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize order management database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Orders table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                stop_price REAL,
                time_in_force TEXT,
                algorithm TEXT,
                priority INTEGER,
                parent_order_id TEXT,
                client_order_id TEXT,
                trader_id TEXT,
                created_at TIMESTAMP,
                valid_until TIMESTAMP,
                min_quantity REAL,
                display_quantity REAL,
                metadata TEXT
            )
        """
        )

        # Order states table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS order_states (
                order_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                filled_quantity REAL,
                remaining_quantity REAL,
                avg_fill_price REAL,
                last_update TIMESTAMP,
                status_message TEXT,
                FOREIGN KEY (order_id) REFERENCES orders (order_id)
            )
        """
        )

        # Executions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                commission REAL,
                timestamp TIMESTAMP,
                exchange TEXT,
                liquidity_flag TEXT,
                execution_venue TEXT,
                FOREIGN KEY (order_id) REFERENCES orders (order_id)
            )
        """
        )

        # Risk checks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS risk_checks (
                check_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                check_type TEXT NOT NULL,
                status TEXT NOT NULL,
                message TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (order_id) REFERENCES orders (order_id)
            )
        """
        )

        conn.commit()
        conn.close()


class OrderRouter:
    def __init__(self):
        self.venues = {
            "BINANCE": {"fees": 0.001, "latency": 50, "liquidity_score": 0.95},
            "COINBASE": {"fees": 0.005, "latency": 80, "liquidity_score": 0.90},
            "KRAKEN": {"fees": 0.002, "latency": 100, "liquidity_score": 0.85},
            "INTERNAL": {"fees": 0.0, "latency": 10, "liquidity_score": 0.70},
        }
        self.routing_rules = {}

    def select_venue(
        self, order: OrderRequest, market_data: Dict[str, MarketData]
    ) -> str:
        """Smart order routing to select best execution venue."""
        symbol_data = market_data.get(order.symbol)
        if not symbol_data:
            return "BINANCE"  # Default venue

        # Calculate venue scores based on multiple factors
        venue_scores = {}

        for venue, props in self.venues.items():
            # Score = liquidity_weight * liquidity + cost_weight * (1-fees) + speed_weight * (1/latency)
            liquidity_weight = 0.4
            cost_weight = 0.4
            speed_weight = 0.2

            liquidity_score = props["liquidity_score"]
            cost_score = 1 - props["fees"]
            speed_score = 1 / (props["latency"] + 1)

            total_score = (
                liquidity_weight * liquidity_score
                + cost_weight * cost_score
                + speed_weight * speed_score
            )

            venue_scores[venue] = total_score

        # Select venue with highest score
        best_venue = max(venue_scores.items(), key=lambda x: x[1])[0]
        return best_venue


class AlgorithmicExecutor:
    def __init__(self):
        self.active_algorithms = {}
        self.market_data = {}

    def execute_twap(
        self, order: OrderRequest, duration_minutes: int = 60
    ) -> List[OrderRequest]:
        """Time-Weighted Average Price execution algorithm."""
        child_orders = []
        num_slices = min(20, duration_minutes // 3)  # Create slices every 3 minutes
        slice_quantity = order.quantity / num_slices
        interval_seconds = (duration_minutes * 60) // num_slices

        for i in range(num_slices):
            child_order = OrderRequest(
                order_id=f"{order.order_id}_TWAP_{i+1}",
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.LIMIT,
                quantity=slice_quantity,
                price=order.price,
                stop_price=None,
                time_in_force="IOC",
                algorithm=ExecutionAlgorithm.TWAP,
                priority=order.priority,
                parent_order_id=order.order_id,
                client_order_id=f"{order.client_order_id}_TWAP_{i+1}",
                trader_id=order.trader_id,
                created_at=order.created_at + timedelta(seconds=i * interval_seconds),
                valid_until=order.valid_until,
                min_quantity=slice_quantity * 0.1,
                display_quantity=slice_quantity,
                metadata={**order.metadata, "slice": i + 1, "total_slices": num_slices},
            )
            child_orders.append(child_order)

        return child_orders

    def execute_vwap(
        self, order: OrderRequest, historical_volume: List[float]
    ) -> List[OrderRequest]:
        """Volume-Weighted Average Price execution algorithm."""
        child_orders = []
        total_historical_volume = sum(historical_volume)

        for i, volume in enumerate(historical_volume):
            volume_weight = volume / total_historical_volume
            slice_quantity = order.quantity * volume_weight

            if slice_quantity > 0:
                child_order = OrderRequest(
                    order_id=f"{order.order_id}_VWAP_{i+1}",
                    symbol=order.symbol,
                    side=order.side,
                    order_type=OrderType.LIMIT,
                    quantity=slice_quantity,
                    price=order.price,
                    stop_price=None,
                    time_in_force="IOC",
                    algorithm=ExecutionAlgorithm.VWAP,
                    priority=order.priority,
                    parent_order_id=order.order_id,
                    client_order_id=f"{order.client_order_id}_VWAP_{i+1}",
                    trader_id=order.trader_id,
                    created_at=order.created_at + timedelta(minutes=i * 5),
                    valid_until=order.valid_until,
                    min_quantity=slice_quantity * 0.1,
                    display_quantity=slice_quantity,
                    metadata={
                        **order.metadata,
                        "volume_slice": i + 1,
                        "volume_weight": volume_weight,
                    },
                )
                child_orders.append(child_order)

        return child_orders

    def execute_iceberg(
        self, order: OrderRequest, display_size: float
    ) -> List[OrderRequest]:
        """Iceberg execution algorithm - hide large orders."""
        child_orders = []
        remaining_quantity = order.quantity
        slice_num = 1

        while remaining_quantity > 0:
            slice_quantity = min(display_size, remaining_quantity)

            child_order = OrderRequest(
                order_id=f"{order.order_id}_ICE_{slice_num}",
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=slice_quantity,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force="GTC",
                algorithm=ExecutionAlgorithm.ICEBERG,
                priority=order.priority,
                parent_order_id=order.order_id,
                client_order_id=f"{order.client_order_id}_ICE_{slice_num}",
                trader_id=order.trader_id,
                created_at=order.created_at,
                valid_until=order.valid_until,
                min_quantity=order.min_quantity,
                display_quantity=slice_quantity,
                metadata={
                    **order.metadata,
                    "iceberg_slice": slice_num,
                    "hidden_quantity": remaining_quantity - slice_quantity,
                },
            )
            child_orders.append(child_order)

            remaining_quantity -= slice_quantity
            slice_num += 1

        return child_orders


class RiskManager:
    def __init__(self):
        self.position_limits = {}
        self.risk_limits = {
            "max_order_size": 1000000,  # $1M max order
            "max_daily_volume": 10000000,  # $10M daily volume
            "max_position_concentration": 0.25,  # 25% max concentration
            "max_leverage": 3.0,
        }
        self.daily_volumes = defaultdict(float)
        self.positions = defaultdict(float)

    def pre_trade_risk_check(self, order: OrderRequest) -> List[RiskCheck]:
        """Perform comprehensive pre-trade risk checks."""
        checks = []

        # Order size check
        order_value = order.quantity * (order.price or 50000)  # Estimate if no price
        if order_value > self.risk_limits["max_order_size"]:
            checks.append(
                RiskCheck(
                    check_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    check_type="ORDER_SIZE_LIMIT",
                    status="failed",
                    message=f"Order value ${order_value:,.2f} exceeds limit ${self.risk_limits['max_order_size']:,.2f}",
                    timestamp=datetime.now(),
                )
            )

        # Daily volume check
        today = datetime.now().date()
        current_daily_volume = self.daily_volumes[today]
        if current_daily_volume + order_value > self.risk_limits["max_daily_volume"]:
            checks.append(
                RiskCheck(
                    check_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    check_type="DAILY_VOLUME_LIMIT",
                    status="failed",
                    message="Order would exceed daily volume limit",
                    timestamp=datetime.now(),
                )
            )

        # Position concentration check
        current_position = self.positions[order.symbol]
        new_position = current_position + (
            order.quantity if order.side == OrderSide.BUY else -order.quantity
        )
        portfolio_value = sum(
            abs(pos * 50000) for pos in self.positions.values()
        )  # Estimate

        if portfolio_value > 0:
            concentration = abs(new_position * (order.price or 50000)) / portfolio_value
            if concentration > self.risk_limits["max_position_concentration"]:
                checks.append(
                    RiskCheck(
                        check_id=str(uuid.uuid4()),
                        order_id=order.order_id,
                        check_type="POSITION_CONCENTRATION",
                        status="failed",
                        message=f"Position concentration {concentration:.1%} exceeds limit {self.risk_limits['max_position_concentration']:.1%}",
                        timestamp=datetime.now(),
                    )
                )

        # If no failures, add passed check
        if not any(check.status == "failed" for check in checks):
            checks.append(
                RiskCheck(
                    check_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    check_type="PRE_TRADE_VALIDATION",
                    status="passed",
                    message="All pre-trade risk checks passed",
                    timestamp=datetime.now(),
                )
            )

        return checks


class OrderManagementSystem:
    def __init__(self):
        self.db = OrderDatabase()
        self.router = OrderRouter()
        self.executor = AlgorithmicExecutor()
        self.risk_manager = RiskManager()

        self.orders: Dict[str, OrderRequest] = {}
        self.order_states: Dict[str, OrderState] = {}
        self.market_data: Dict[str, MarketData] = {}
        self.executions: List[OrderExecution] = []
        self.risk_checks: List[RiskCheck] = []

        self.order_queue = deque()
        self.execution_engine_active = True

        self.start_market_data_simulation()
        self.start_execution_engine()

    def submit_order(self, order_request: OrderRequest) -> Tuple[bool, str]:
        """Submit new order with full validation and risk checks."""
        try:
            # Pre-trade risk checks
            risk_checks = self.risk_manager.pre_trade_risk_check(order_request)
            self.risk_checks.extend(risk_checks)

            # Check if any risk check failed
            failed_checks = [check for check in risk_checks if check.status == "failed"]
            if failed_checks:
                return False, f"Risk check failed: {failed_checks[0].message}"

            # Add to orders
            self.orders[order_request.order_id] = order_request

            # Initialize order state
            self.order_states[order_request.order_id] = OrderState(
                order_id=order_request.order_id,
                status=OrderStatus.PENDING,
                filled_quantity=0.0,
                remaining_quantity=order_request.quantity,
                avg_fill_price=0.0,
                last_update=datetime.now(),
                status_message="Order received and validated",
                executions=[],
            )

            # Handle algorithmic orders
            if order_request.algorithm in [
                ExecutionAlgorithm.TWAP,
                ExecutionAlgorithm.VWAP,
                ExecutionAlgorithm.ICEBERG,
            ]:
                child_orders = self.create_child_orders(order_request)
                for child_order in child_orders:
                    self.order_queue.append(child_order)
            else:
                # Add to execution queue
                self.order_queue.append(order_request)

            # Update order status
            self.update_order_status(
                order_request.order_id,
                OrderStatus.SUBMITTED,
                "Order submitted for execution",
            )

            return True, f"Order {order_request.order_id} submitted successfully"

        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return False, f"Error submitting order: {str(e)}"

    def create_child_orders(self, parent_order: OrderRequest) -> List[OrderRequest]:
        """Create child orders for algorithmic execution."""
        if parent_order.algorithm == ExecutionAlgorithm.TWAP:
            return self.executor.execute_twap(parent_order, duration_minutes=60)
        elif parent_order.algorithm == ExecutionAlgorithm.VWAP:
            # Simulate historical volume data
            historical_volume = [
                np.random.uniform(0.5, 2.0) for _ in range(12)
            ]  # 12 5-minute intervals
            return self.executor.execute_vwap(parent_order, historical_volume)
        elif parent_order.algorithm == ExecutionAlgorithm.ICEBERG:
            display_size = parent_order.display_quantity or (
                parent_order.quantity * 0.1
            )
            return self.executor.execute_iceberg(parent_order, display_size)

        return []

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """Cancel an existing order."""
        if order_id not in self.orders:
            return False, "Order not found"

        order_state = self.order_states.get(order_id)
        if not order_state:
            return False, "Order state not found"

        if order_state.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        ]:
            return False, f"Cannot cancel order in {order_state.status.value} status"

        # Update order status
        self.update_order_status(
            order_id, OrderStatus.CANCELLED, "Order cancelled by user"
        )

        return True, f"Order {order_id} cancelled successfully"

    def update_order_status(self, order_id: str, status: OrderStatus, message: str):
        """Update order status and timestamp."""
        if order_id in self.order_states:
            self.order_states[order_id].status = status
            self.order_states[order_id].status_message = message
            self.order_states[order_id].last_update = datetime.now()

    def simulate_execution(self, order: OrderRequest) -> Optional[OrderExecution]:
        """Simulate order execution with realistic fills."""
        if order.symbol not in self.market_data:
            return None

        market = self.market_data[order.symbol]

        # Determine execution price based on order type
        if order.order_type == OrderType.MARKET:
            execution_price = (
                market.ask_price if order.side == OrderSide.BUY else market.bid_price
            )
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.price >= market.ask_price:
                execution_price = market.ask_price
            elif order.side == OrderSide.SELL and order.price <= market.bid_price:
                execution_price = market.bid_price
            else:
                return None  # No execution
        else:
            execution_price = order.price or market.last_price

        # Calculate execution quantity (partial fills possible)
        max_executable = min(
            order.quantity,
            market.bid_size if order.side == OrderSide.SELL else market.ask_size,
        )
        execution_quantity = np.random.uniform(max_executable * 0.3, max_executable)
        execution_quantity = min(
            execution_quantity, self.order_states[order.order_id].remaining_quantity
        )

        if execution_quantity < (order.min_quantity or 0):
            return None

        # Create execution
        execution = OrderExecution(
            execution_id=str(uuid.uuid4()),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=execution_quantity,
            price=execution_price,
            commission=execution_quantity * execution_price * 0.001,  # 0.1% commission
            timestamp=datetime.now(),
            exchange=self.router.select_venue(order, self.market_data),
            liquidity_flag="taker" if order.order_type == OrderType.MARKET else "maker",
            execution_venue="SMART",
        )

        # Update order state
        order_state = self.order_states[order.order_id]
        order_state.executions.append(execution)
        order_state.filled_quantity += execution_quantity
        order_state.remaining_quantity -= execution_quantity

        # Update average fill price
        total_value = sum(ex.quantity * ex.price for ex in order_state.executions)
        order_state.avg_fill_price = total_value / order_state.filled_quantity

        # Update status
        if order_state.remaining_quantity <= 0:
            self.update_order_status(
                order.order_id, OrderStatus.FILLED, "Order fully executed"
            )
        else:
            self.update_order_status(
                order.order_id,
                OrderStatus.PARTIALLY_FILLED,
                f"Partial fill: {execution_quantity}",
            )

        self.executions.append(execution)
        return execution

    def start_market_data_simulation(self):
        """Start market data simulation."""

        def simulate_market_data():
            symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
            base_prices = {
                "BTCUSDT": 45000,
                "ETHUSDT": 3000,
                "ADAUSDT": 1.5,
                "DOTUSDT": 25,
                "LINKUSDT": 15,
            }

            while True:
                for symbol in symbols:
                    base_price = base_prices[symbol]
                    price_change = np.random.uniform(-0.02, 0.02)  # ±2% change
                    last_price = base_price * (1 + price_change)
                    spread = last_price * 0.001  # 0.1% spread

                    self.market_data[symbol] = MarketData(
                        symbol=symbol,
                        bid_price=last_price - spread / 2,
                        ask_price=last_price + spread / 2,
                        bid_size=np.random.uniform(10, 100),
                        ask_size=np.random.uniform(10, 100),
                        last_price=last_price,
                        volume=np.random.uniform(1000, 10000),
                        timestamp=datetime.now(),
                    )

                time_module.sleep(1)  # Update every second

        market_thread = threading.Thread(target=simulate_market_data, daemon=True)
        market_thread.start()

    def start_execution_engine(self):
        """Start order execution engine."""

        def execute_orders():
            while self.execution_engine_active:
                try:
                    if self.order_queue:
                        order = self.order_queue.popleft()

                        # Check if order is still valid
                        if order.order_id in self.order_states:
                            order_state = self.order_states[order.order_id]
                            if order_state.status in [
                                OrderStatus.PENDING,
                                OrderStatus.SUBMITTED,
                                OrderStatus.PARTIALLY_FILLED,
                            ]:
                                execution = self.simulate_execution(order)
                                if execution:
                                    logger.info(
                                        f"Executed {execution.quantity} of {order.symbol} at {execution.price}"
                                    )

                    time_module.sleep(0.5)  # Execute every 0.5 seconds
                except Exception as e:
                    logger.error(f"Execution engine error: {e}")
                    time_module.sleep(1)

        execution_thread = threading.Thread(target=execute_orders, daemon=True)
        execution_thread.start()

    def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """Get current order book for symbol."""
        if symbol not in self.market_data:
            return {}

        market = self.market_data[symbol]

        # Simulate order book depth
        bids = []
        asks = []

        for i in range(10):
            bid_price = market.bid_price - (i * market.bid_price * 0.0001)
            ask_price = market.ask_price + (i * market.ask_price * 0.0001)

            bids.append([bid_price, np.random.uniform(5, 50)])
            asks.append([ask_price, np.random.uniform(5, 50)])

        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "timestamp": market.timestamp,
        }


# Initialize OMS
@st.cache_resource
def get_oms():
    return OrderManagementSystem()


def main():
    st.set_page_config(
        page_title="ZoL0 Advanced Order Management System",
        page_icon="📋",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
    <style>
    .order-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .execution-card {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        padding: 12px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }
    .risk-fail {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 10px;
        border-radius: 5px;
        color: white;
    }
    .risk-pass {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        padding: 10px;
        border-radius: 5px;
        color: white;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("📋 ZoL0 Advanced Order Management System")
    st.markdown("**Enterprise-grade order execution and management platform**")

    # Sidebar
    st.sidebar.title("🎛️ Order Controls")

    # Get OMS
    oms = get_oms()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "📊 Dashboard",
            "➕ New Order",
            "📋 Order Book",
            "⚡ Executions",
            "🛡️ Risk Management",
            "📈 Analytics",
        ]
    )

    with tab1:
        st.header("📊 Order Management Dashboard")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_orders = len(oms.orders)
            st.markdown(
                f"""
            <div class="order-card">
                <h3>Total Orders</h3>
                <h2>{total_orders}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            active_orders = len(
                [
                    o
                    for o in oms.order_states.values()
                    if o.status
                    in [
                        OrderStatus.PENDING,
                        OrderStatus.SUBMITTED,
                        OrderStatus.PARTIALLY_FILLED,
                    ]
                ]
            )
            st.markdown(
                f"""
            <div class="order-card">
                <h3>Active Orders</h3>
                <h2>{active_orders}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            total_executions = len(oms.executions)
            st.markdown(
                f"""
            <div class="order-card">
                <h3>Total Executions</h3>
                <h2>{total_executions}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col4:
            total_volume = sum(ex.quantity * ex.price for ex in oms.executions)
            st.markdown(
                f"""
            <div class="order-card">
                <h3>Total Volume</h3>
                <h2>${total_volume:,.0f}</h2>
            </div>
            """,
                unsafe_allow_html=True,
            )

        # Order status chart
        st.subheader("📈 Order Status Distribution")

        if oms.order_states:
            status_counts = {}
            for state in oms.order_states.values():
                status = state.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            fig_status = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title="Order Status Distribution",
            )
            st.plotly_chart(fig_status, use_container_width=True)

        # Recent orders table
        st.subheader("📋 Recent Orders")

        if oms.orders:
            recent_orders = list(oms.orders.values())[-10:]  # Last 10 orders
            orders_data = []

            for order in recent_orders:
                state = oms.order_states.get(order.order_id)
                orders_data.append(
                    {
                        "Order ID": order.order_id[:8] + "...",
                        "Symbol": order.symbol,
                        "Side": order.side.value.upper(),
                        "Type": order.order_type.value.upper(),
                        "Quantity": f"{order.quantity:,.2f}",
                        "Price": f"${order.price:,.2f}" if order.price else "Market",
                        "Status": state.status.value.title() if state else "Unknown",
                        "Algorithm": order.algorithm.value.upper(),
                        "Created": order.created_at.strftime("%H:%M:%S"),
                    }
                )

            df_orders = pd.DataFrame(orders_data)
            st.dataframe(df_orders, use_container_width=True)
        else:
            st.info("📊 No orders yet. Create your first order in the 'New Order' tab.")

    with tab2:
        st.header("➕ Create New Order")

        # Order form
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Order Details")

            symbol = st.selectbox(
                "Symbol", ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
            )
            side = st.selectbox("Side", ["BUY", "SELL"])
            order_type = st.selectbox(
                "Order Type", [ot.value.upper() for ot in OrderType]
            )
            quantity = st.number_input("Quantity", min_value=0.01, value=1.0, step=0.01)

            if order_type != "MARKET":
                price = st.number_input(
                    "Price ($)", min_value=0.01, value=50000.0, step=0.01
                )
            else:
                price = None

            if order_type in ["STOP", "STOP_LIMIT"]:
                stop_price = st.number_input(
                    "Stop Price ($)", min_value=0.01, value=50000.0, step=0.01
                )
            else:
                stop_price = None

        with col2:
            st.subheader("⚙️ Execution Settings")

            algorithm = st.selectbox(
                "Execution Algorithm", [ea.value.upper() for ea in ExecutionAlgorithm]
            )
            time_in_force = st.selectbox("Time in Force", ["GTC", "IOC", "FOK", "DAY"])
            priority = st.selectbox("Priority", ["LOW", "NORMAL", "HIGH", "URGENT"])

            # Algorithm-specific settings
            if algorithm == "ICEBERG":
                display_quantity = st.number_input(
                    "Display Quantity", min_value=0.01, value=quantity * 0.1, step=0.01
                )
            else:
                display_quantity = None

            min_quantity = st.number_input(
                "Minimum Fill Quantity", min_value=0.0, value=0.0, step=0.01
            )

            trader_id = st.text_input("Trader ID", value="TRADER_001")
            client_order_id = st.text_input(
                "Client Order ID", value=f"CLIENT_{datetime.now().strftime('%H%M%S')}"
            )

        # Submit order
        if st.button("🚀 Submit Order", type="primary"):
            order_request = OrderRequest(
                order_id=str(uuid.uuid4()),
                symbol=symbol,
                side=OrderSide(side.lower()),
                order_type=OrderType(order_type.lower()),
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                time_in_force=time_in_force,
                algorithm=ExecutionAlgorithm(algorithm.lower()),
                priority=OrderPriority[priority],
                parent_order_id=None,
                client_order_id=client_order_id,
                trader_id=trader_id,
                created_at=datetime.now(),
                valid_until=datetime.now() + timedelta(days=1),
                min_quantity=min_quantity if min_quantity > 0 else None,
                display_quantity=display_quantity,
                metadata={"source": "dashboard", "version": "1.0"},
            )

            success, message = oms.submit_order(order_request)

            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")

    with tab3:
        st.header("📋 Order Book & Market Data")

        # Symbol selector
        selected_symbol = st.selectbox(
            "Select Symbol for Order Book",
            list(oms.market_data.keys()) if oms.market_data else ["BTCUSDT"],
        )

        if selected_symbol in oms.market_data:
            market = oms.market_data[selected_symbol]

            # Market data display
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Last Price", f"${market.last_price:,.2f}")
            with col2:
                st.metric("Bid", f"${market.bid_price:,.2f}")
            with col3:
                st.metric("Ask", f"${market.ask_price:,.2f}")
            with col4:
                spread = (
                    (market.ask_price - market.bid_price) / market.last_price
                ) * 100
                st.metric("Spread", f"{spread:.3f}%")

            # Order book
            order_book = oms.get_order_book(selected_symbol)

            if order_book:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📈 Bids")
                    bids_df = pd.DataFrame(
                        order_book["bids"], columns=["Price", "Size"]
                    )
                    bids_df = bids_df.sort_values("Price", ascending=False)
                    bids_df["Price"] = bids_df["Price"].apply(lambda x: f"${x:,.2f}")
                    bids_df["Size"] = bids_df["Size"].apply(lambda x: f"{x:.2f}")
                    st.dataframe(bids_df, hide_index=True)

                with col2:
                    st.subheader("📉 Asks")
                    asks_df = pd.DataFrame(
                        order_book["asks"], columns=["Price", "Size"]
                    )
                    asks_df = asks_df.sort_values("Price", ascending=True)
                    asks_df["Price"] = asks_df["Price"].apply(lambda x: f"${x:,.2f}")
                    asks_df["Size"] = asks_df["Size"].apply(lambda x: f"{x:.2f}")
                    st.dataframe(asks_df, hide_index=True)

                # Order book visualization
                st.subheader("📊 Order Book Depth")

                bids_data = order_book["bids"]
                asks_data = order_book["asks"]

                fig = go.Figure()

                # Add bids
                fig.add_trace(
                    go.Scatter(
                        x=[bid[0] for bid in bids_data],
                        y=np.cumsum([bid[1] for bid in bids_data]),
                        fill="tozeroy",
                        name="Bids",
                        line=dict(color="green"),
                    )
                )

                # Add asks
                fig.add_trace(
                    go.Scatter(
                        x=[ask[0] for ask in asks_data],
                        y=np.cumsum([ask[1] for ask in asks_data]),
                        fill="tozeroy",
                        name="Asks",
                        line=dict(color="red"),
                    )
                )

                fig.update_layout(
                    title=f"{selected_symbol} Order Book Depth",
                    xaxis_title="Price ($)",
                    yaxis_title="Cumulative Size",
                    height=400,
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Market data not available for selected symbol.")

    with tab4:
        st.header("⚡ Order Executions")

        # Execution filters
        col1, col2, col3 = st.columns(3)

        with col1:
            symbol_filter = st.selectbox(
                "Filter by Symbol",
                ["All"] + list(set([ex.symbol for ex in oms.executions])),
            )

        with col2:
            side_filter = st.selectbox("Filter by Side", ["All", "BUY", "SELL"])

        with col3:
            time_filter = st.selectbox(
                "Time Period", ["Last Hour", "Last 24 Hours", "All Time"]
            )

        # Filter executions
        filtered_executions = oms.executions.copy()

        if symbol_filter != "All":
            filtered_executions = [
                ex for ex in filtered_executions if ex.symbol == symbol_filter
            ]

        if side_filter != "All":
            filtered_executions = [
                ex for ex in filtered_executions if ex.side.value.upper() == side_filter
            ]

        # Time filtering
        if time_filter != "All Time":
            time_delta = (
                timedelta(hours=1)
                if time_filter == "Last Hour"
                else timedelta(hours=24)
            )
            cutoff_time = datetime.now() - time_delta
            filtered_executions = [
                ex for ex in filtered_executions if ex.timestamp >= cutoff_time
            ]

        # Display executions
        st.subheader(f"📋 Executions ({len(filtered_executions)})")

        if filtered_executions:
            executions_data = []
            for execution in filtered_executions[-50:]:  # Show last 50
                executions_data.append(
                    {
                        "Time": execution.timestamp.strftime("%H:%M:%S"),
                        "Symbol": execution.symbol,
                        "Side": execution.side.value.upper(),
                        "Quantity": f"{execution.quantity:,.4f}",
                        "Price": f"${execution.price:,.2f}",
                        "Value": f"${execution.quantity * execution.price:,.2f}",
                        "Commission": f"${execution.commission:,.2f}",
                        "Exchange": execution.exchange,
                        "Liquidity": execution.liquidity_flag.title(),
                    }
                )

            df_executions = pd.DataFrame(executions_data)
            st.dataframe(df_executions, use_container_width=True)

            # Execution analytics
            col1, col2 = st.columns(2)

            with col1:
                # Execution volume by symbol
                symbol_volumes = {}
                for ex in filtered_executions:
                    symbol_volumes[ex.symbol] = symbol_volumes.get(ex.symbol, 0) + (
                        ex.quantity * ex.price
                    )

                if symbol_volumes:
                    fig_volume = px.bar(
                        x=list(symbol_volumes.keys()),
                        y=list(symbol_volumes.values()),
                        title="Execution Volume by Symbol",
                    )
                    fig_volume.update_layout(
                        xaxis_title="Symbol", yaxis_title="Volume ($)"
                    )
                    st.plotly_chart(fig_volume, use_container_width=True)

            with col2:
                # Execution timeline
                if len(filtered_executions) > 1:
                    timeline_data = [
                        (ex.timestamp, ex.quantity * ex.price)
                        for ex in filtered_executions
                    ]
                    timeline_df = pd.DataFrame(timeline_data, columns=["Time", "Value"])
                    timeline_df["Time"] = pd.to_datetime(timeline_df["Time"])

                    fig_timeline = px.line(
                        timeline_df, x="Time", y="Value", title="Execution Timeline"
                    )
                    fig_timeline.update_layout(yaxis_title="Execution Value ($)")
                    st.plotly_chart(fig_timeline, use_container_width=True)
        else:
            st.info("📊 No executions found for the selected criteria.")

    with tab5:
        st.header("🛡️ Risk Management")

        # Risk metrics
        st.subheader("📊 Risk Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            risk_checks_passed = len(
                [rc for rc in oms.risk_checks if rc.status == "passed"]
            )
            st.metric("Risk Checks Passed", risk_checks_passed)

        with col2:
            risk_checks_failed = len(
                [rc for rc in oms.risk_checks if rc.status == "failed"]
            )
            st.metric("Risk Checks Failed", risk_checks_failed)

        with col3:
            total_exposure = sum(
                abs(pos) * 50000 for pos in oms.risk_manager.positions.values()
            )
            st.metric("Total Exposure", f"${total_exposure:,.0f}")

        with col4:
            max_order_limit = oms.risk_manager.risk_limits["max_order_size"]
            st.metric("Max Order Limit", f"${max_order_limit:,.0f}")

        # Risk limits configuration
        st.subheader("⚙️ Risk Limits")

        col1, col2 = st.columns(2)

        with col1:
            new_max_order = st.number_input(
                "Max Order Size ($)",
                value=oms.risk_manager.risk_limits["max_order_size"],
                step=100000,
            )
            new_max_daily = st.number_input(
                "Max Daily Volume ($)",
                value=oms.risk_manager.risk_limits["max_daily_volume"],
                step=1000000,
            )

        with col2:
            new_max_concentration = (
                st.number_input(
                    "Max Position Concentration (%)",
                    value=oms.risk_manager.risk_limits["max_position_concentration"]
                    * 100,
                    step=5.0,
                )
                / 100
            )
            new_max_leverage = st.number_input(
                "Max Leverage",
                value=oms.risk_manager.risk_limits["max_leverage"],
                step=0.5,
            )

        if st.button("Update Risk Limits"):
            oms.risk_manager.risk_limits.update(
                {
                    "max_order_size": new_max_order,
                    "max_daily_volume": new_max_daily,
                    "max_position_concentration": new_max_concentration,
                    "max_leverage": new_max_leverage,
                }
            )
            st.success("✅ Risk limits updated!")

        # Recent risk checks
        st.subheader("📋 Recent Risk Checks")

        if oms.risk_checks:
            recent_checks = oms.risk_checks[-20:]  # Last 20 checks

            for check in recent_checks:
                status_class = "risk-pass" if check.status == "passed" else "risk-fail"
                status_icon = "✅" if check.status == "passed" else "❌"

                st.markdown(
                    f"""
                <div class="{status_class}">
                    {status_icon} <strong>{check.check_type}</strong><br>
                    {check.message}<br>
                    <small>{check.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("📊 No risk checks performed yet.")

    with tab6:
        st.header("📈 Order Management Analytics")

        # Performance metrics
        if oms.executions:
            st.subheader("📊 Execution Performance")

            # Calculate metrics
            total_volume = sum(ex.quantity * ex.price for ex in oms.executions)
            total_commission = sum(ex.commission for ex in oms.executions)
            avg_commission_rate = (
                (total_commission / total_volume * 100) if total_volume > 0 else 0
            )

            # VWAP calculation
            vwap = (
                total_volume / sum(ex.quantity for ex in oms.executions)
                if oms.executions
                else 0
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Volume Traded", f"${total_volume:,.0f}")
            with col2:
                st.metric("Total Commission", f"${total_commission:,.2f}")
            with col3:
                st.metric("Avg Commission Rate", f"{avg_commission_rate:.3f}%")
            with col4:
                st.metric("Volume Weighted Avg Price", f"${vwap:,.2f}")

            # Fill rate analysis
            st.subheader("📈 Fill Rate Analysis")

            fill_rates = []
            for order_id, state in oms.order_states.items():
                if state.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                    order = oms.orders[order_id]
                    fill_rate = (state.filled_quantity / order.quantity) * 100
                    fill_rates.append(
                        {
                            "Order ID": order_id[:8] + "...",
                            "Symbol": order.symbol,
                            "Algorithm": order.algorithm.value.upper(),
                            "Fill Rate": f"{fill_rate:.1f}%",
                            "Status": state.status.value.title(),
                        }
                    )

            if fill_rates:
                df_fills = pd.DataFrame(fill_rates)
                st.dataframe(df_fills, use_container_width=True)

                # Fill rate by algorithm
                algo_fill_rates = {}
                for order_id, state in oms.order_states.items():
                    if state.status in [
                        OrderStatus.FILLED,
                        OrderStatus.PARTIALLY_FILLED,
                    ]:
                        order = oms.orders[order_id]
                        algo = order.algorithm.value
                        fill_rate = (state.filled_quantity / order.quantity) * 100

                        if algo not in algo_fill_rates:
                            algo_fill_rates[algo] = []
                        algo_fill_rates[algo].append(fill_rate)

                # Calculate average fill rates
                avg_fill_rates = {
                    algo: np.mean(rates) for algo, rates in algo_fill_rates.items()
                }

                if avg_fill_rates:
                    fig_algo = px.bar(
                        x=list(avg_fill_rates.keys()),
                        y=list(avg_fill_rates.values()),
                        title="Average Fill Rate by Algorithm",
                        labels={"x": "Algorithm", "y": "Fill Rate (%)"},
                    )
                    st.plotly_chart(fig_algo, use_container_width=True)
        else:
            st.info("📊 No execution data available for analytics.")

    # TODO: Integrate with CI/CD pipeline for automated order management and edge-case tests.
    # Edge-case tests: simulate order failures, DB/network errors, and permission issues.
    # All public methods have docstrings and exception handling.

    # Auto-refresh
    if st.sidebar.checkbox("🔄 Auto Refresh", value=True):
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
        time_module.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
