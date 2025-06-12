import pytest
from engine.base_strategy import BaseStrategy

class DummyStrategy(BaseStrategy):
    def generate_signals(self, data):
        return None
    def calculate_position_size(self, signal, current_price, portfolio_value):
        return 0

def test_base_strategy_instantiation():
    s = DummyStrategy("test", {})
    assert s.name == "test"
    assert s.parameters == {}
    assert hasattr(s, "generate_signals")
    assert hasattr(s, "calculate_position_size")

def test_generate_signals_and_position_size():
    s = DummyStrategy("test", {})
    assert s.generate_signals(None) is None
    assert s.calculate_position_size(None, 0, 0) == 0
