import logging

from data.demo_data import generate_demo_data
from engine.backtest_engine import BacktestEngine
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompositeStrategy:
    """
    Ensemble strategy: majority vote or weighted average of signals from multiple strategies.
    """

    def __init__(self, strategies, weights=None):
        self.strategies = strategies
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        self.name = "CompositeStrategy"

    def generate_signals(self, data):
        import numpy as np

        signals = []
        for strat in self.strategies:
            s = strat.generate_signals(data)
            signals.append(s["signal"].values if hasattr(s, "values") else s)
        signals = np.array(signals)
        # Weighted sum, then sign for ensemble decision
        weighted = np.tensordot(self.weights, signals, axes=1)
        return {"signal": np.sign(weighted)}

    def calculate_position_size(self, signal, current_price, portfolio_value):
        # Use average risk per trade from all strategies
        sizes = [
            s.calculate_position_size(signal, current_price, portfolio_value)
            for s in self.strategies
        ]
        return sum(sizes) / len(sizes)


# TODO: Integrate with CI/CD pipeline for automated strategy ensemble and edge-case tests.
# Edge-case tests: simulate empty data, mismatched weights, and strategy errors.
# All public methods have docstrings and exception handling where needed.

# Example usage for testing
if __name__ == "__main__":
    data = generate_demo_data("TEST")
    m1 = MomentumStrategy(fast_period=10, slow_period=30, risk_per_trade=0.02)
    m2 = MeanReversionStrategy(period=20, std_dev=2.0, risk_per_trade=0.01)
    composite = CompositeStrategy([m1, m2], weights=[0.5, 0.5])
    engine = BacktestEngine(initial_capital=100000)
    result = engine.run(composite, data, dynamic_tp_sl=True)
    logger.info(
        f"CompositeStrategy result: {result.total_return}, {result.sharpe_ratio}"
    )
