import pytest
from composite_strategy import CompositeStrategy

class DummyStrategy:
    def generate_signals(self, data):
        return {"signal": [1, -1, 0]}
    def calculate_position_size(self, signal, current_price, portfolio_value):
        return 1

def test_composite_strategy_basic():
    s1 = DummyStrategy()
    s2 = DummyStrategy()
    composite = CompositeStrategy([s1, s2], weights=[0.5, 0.5])
    signals = composite.generate_signals([1,2,3])
    assert "signal" in signals
    size = composite.calculate_position_size(1, 100, 1000)
    assert size == 1

def test_composite_strategy_mismatched_weights():
    s1 = DummyStrategy()
    s2 = DummyStrategy()
    with pytest.raises(Exception):
        CompositeStrategy([s1, s2], weights=[0.7])
