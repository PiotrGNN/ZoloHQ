import pytest
from engine.portfolio_manager import PortfolioManager

def test_update_and_get_positions():
    pm = PortfolioManager()
    pm.update("BTC", "buy", 1, 10000)
    pm.update("BTC", "sell", 1, 11000)
    positions = pm.get_positions()
    assert positions["BTC"] == 0

def test_get_advanced_analytics_empty():
    pm = PortfolioManager()
    result = pm.get_advanced_analytics({})
    assert result == {}

def test_dynamic_position_sizing_zero_stop():
    pm = PortfolioManager()
    size = pm.dynamic_position_sizing("BTC", 0.01, 0, 10000)
    assert size == 0

def test_kelly_position_sizing_negative():
    pm = PortfolioManager()
    size = pm.kelly_position_sizing(0.1, 0.5, 10000)
    assert size >= 0
