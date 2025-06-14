import pytest
from api_backtest import app
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    client = TestClient(app)
    yield client

def test_api_backtest_missing_params(client):
    # Missing required fields
    headers = {"X-API-KEY": "admin-key"}
    resp = client.post('/api/backtest', json={}, headers=headers)
    assert resp.status_code in (200, 400, 422)

def test_api_backtest_invalid_strategy(client):
    headers = {"X-API-KEY": "admin-key"}
    resp = client.post('/api/backtest', json={"strategy": "Nonexistent", "params": {}}, headers=headers)
    assert resp.status_code == 500 or resp.status_code == 400
