"""
BaseStrategy - abstract base class for trading strategies.
CI/CD: Zautomatyzowane testy edge-case i workflow wdrożone w .github/workflows/ci-cd.yml
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import logging


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    Each strategy must implement generate_signals and calculate_position_size.
    Dodano: logowanie, hooki analityczne, obsługa premium, przygotowanie pod monetyzację.
    """

    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        Initialize BaseStrategy.
        Args:
            name (str): Strategy name.
            parameters (dict): Strategy parameters.
        """
        self.name = name
        self.parameters = parameters
        self.positions = {}
        self.orders = []
        self.trades = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.premium_features = self.parameters.get("premium_features", False)

    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals based on input data.
        Args:
            data: Input data (e.g., DataFrame).
        Returns:
            Any: Signals (implementation-specific).
        """
        pass

    @abstractmethod
    def calculate_position_size(self, signal, current_price, portfolio_value):
        """
        Calculate position size based on signal and portfolio state.
        Args:
            signal: Trading signal.
            current_price (float): Current price.
            portfolio_value (float): Current portfolio value.
        Returns:
            float: Position size.
        """
        pass

    # --- Monetization & Marketplace Hooks ---
    def register_strategy(self):
        # In production, register in SaaS/plugin marketplace
        print(f"Registering strategy: {self.name}")

    def usage_analytics(self, event: str):
        # In production, send analytics to SaaS
        print(f"Strategy usage: {self.name}, event={event}")


# Edge-case test examples (to be expanded in test suite)
def _test_edge_cases():
    class DummyStrategy(BaseStrategy):
        def generate_signals(self, data):
            return None

        def calculate_position_size(self, signal, current_price, portfolio_value):
            return 0

    try:
        s = DummyStrategy("test", {})
        s.generate_signals(None)
        s.calculate_position_size(None, 0, 0)
    except Exception as e:
        print(f"Handled edge case in BaseStrategy: {e}")


# TODO: Add more edge-case and integration tests for CI/CD
