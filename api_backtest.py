"""
API Backtest - REST API do uruchamiania backtestów ZoL0
Autor: Twój zespół
Data: 2025-06-03
Opis: Umożliwia zdalne uruchamianie backtestów i pobieranie wyników przez API.
"""

from flask import Flask, jsonify, request

from data.demo_data import generate_demo_data
from engine.backtest_engine import BacktestEngine
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy

app = Flask(__name__)


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """
    Endpoint REST API do uruchamiania backtestu.
    Oczekuje JSON z polami: strategy, params, stop_loss_pct, take_profit_pct.
    Zwraca: metryki backtestu w formacie JSON.
    Obsługa błędów danych wejściowych.
    """
    try:
        req = request.json
        strategy_name = req.get("strategy", "Momentum")
        params = req.get("params", {})
        stop_loss_pct = req.get("stop_loss_pct", 0.02)
        take_profit_pct = req.get("take_profit_pct", 0.04)
        data = generate_demo_data("TEST")
        engine = BacktestEngine(initial_capital=100000)
        STRATEGY_MAP = {
            "Momentum": MomentumStrategy,
            "Mean Reversion": MeanReversionStrategy,
        }
        strategy = STRATEGY_MAP[strategy_name](**params)
        result = engine.run(
            strategy, data, stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct
        )
        return jsonify(
            {
                "final_capital": result.final_capital,
                "total_return": result.total_return,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_trades": result.total_trades,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Test edge-case: błąd danych wejściowych
if __name__ == "__main__":
    def test_invalid_input():
        """Testuje obsługę błędnych danych wejściowych do API."""
        import requests

        try:
            resp = requests.post(
                "http://localhost:8520/api/backtest", json={"strategy": "Nonexistent"}
            )
            if resp.status_code == 400:
                print("OK: Invalid input handled gracefully.")
            else:
                print(f"FAIL: Unexpected status code: {resp.status_code}")
        except Exception as e:
            print(f"FAIL: Unexpected exception: {e}")

    # Uruchom serwer Flask w osobnym procesie przed testem
    # test_invalid_input()
# CI/CD: Zautomatyzowane testy edge-case i workflow wdrożone w .github/workflows/ci-cd.yml
# (TODO usunięty po wdrożeniu automatyzacji)
