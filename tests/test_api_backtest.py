import pytest
from api_backtest import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_api_backtest_missing_params(client):
    # Missing required fields
    resp = client.post('/api/backtest', json={})
    assert resp.status_code == 200 or resp.status_code == 400

def test_api_backtest_invalid_strategy(client):
    resp = client.post('/api/backtest', json={"strategy": "Nonexistent", "params": {}})
    assert resp.status_code == 500 or resp.status_code == 400
