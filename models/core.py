# Minimal stub for models.core to unblock tests
from enum import Enum

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class Trade:
    def __init__(self, *args, **kwargs):
        pass

class BacktestResult:
    def __init__(self, *args, **kwargs):
        self.trades = []
        self.pnl = 0
        self.stats = {}
