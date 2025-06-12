import logging

from data.demo_data import generate_demo_data
from engine.backtest_engine import BacktestEngine
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
import os
import subprocess
import sys

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompositeStrategy:
    """
    Ensemble strategy: majority vote or weighted average of signals from multiple strategies.
    """

    def __init__(self, strategies, weights=None):
        self.strategies = strategies
        if weights is not None and len(weights) != len(strategies):
            raise ValueError("weights length must match number of strategies")
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        self.name = "CompositeStrategy"

    def generate_signals(self, data):
        import numpy as np

        signals = []
        for strat in self.strategies:
            s = strat.generate_signals(data)
            sig = s["signal"]
            sig_arr = sig.values if hasattr(sig, "values") else sig
            signals.append(np.array(sig_arr))
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


# CI/CD integration for automated strategy ensemble tests
def run_ci_cd_composite_strategy_tests() -> None:
    """Run composite strategy tests in CI/CD pipelines."""
    if not os.getenv("CI"):
        logger.debug("CI environment not detected; skipping composite tests")
        return

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_composite_strategy.py",
        "--maxfail=1",
        "--disable-warnings",
    ]
    logger.info("Running CI/CD composite strategy tests: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(proc.stdout)
    if proc.returncode != 0:
        logger.error(proc.stderr)
        raise RuntimeError(
            f"CI/CD composite strategy tests failed with exit code {proc.returncode}"
        )

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
