"""
BaseStrategy - abstract base class for trading strategies.
CI/CD: Zautomatyzowane testy edge-case i workflow wdrożone w .github/workflows/ci-cd.yml
(TODO usunięty po wdrożeniu automatyzacji)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    Each strategy must implement generate_signals and calculate_position_size.
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
