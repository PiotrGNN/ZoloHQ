import pytest
from engine.backtest_engine import BacktestEngine

class DummyStrategy:
    def generate_signals(self, data):
        # Always return a DataFrame-like object with 'signal' column
        import pandas as pd
        return pd.DataFrame({"signal": [1]})
    def calculate_position_size(self, signal, current_price, portfolio_value):
        return 1

def test_run_with_invalid_data():
    engine = BacktestEngine(1000)
    with pytest.raises(Exception):
        engine.run(DummyStrategy(), None)

def test_run_with_empty_data():
    import pandas as pd
    engine = BacktestEngine(1000)
    data = pd.DataFrame()
    with pytest.raises(Exception):
        engine.run(DummyStrategy(), data)

def test_load_production_dynamic_tp_sl_params_missing():
    engine = BacktestEngine(1000)
    params = engine.load_production_dynamic_tp_sl_params(path="nonexistent.json")
    assert params is None
