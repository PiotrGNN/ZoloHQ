"""
ZoL0AIAgent - agent AI do optymalizacji i rekomendacji
Autor: Twój zespół
Data: 2025-06-03
Opis: Wybiera strategie, optymalizuje parametry, generuje raporty i rekomendacje.
"""

import logging
import random

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

# Import strategies and engine with error handling
try:
    from strategies.momentum import MomentumStrategy
except ImportError:
    MomentumStrategy = None
try:
    from strategies.mean_reversion import MeanReversionStrategy
except ImportError:
    MeanReversionStrategy = None
try:
    from engine.backtest_engine import BacktestEngine
except ImportError:
    BacktestEngine = None

# Placeholder for AI agent integration
# In the next steps, this module will allow AI-driven strategy generation and backtest control.


class ZoL0AIAgent:
    def __init__(self):
        # Initialize agent state if needed
        self.state = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.strategy_history = []

    def generate_strategy(self, available_strategies: dict) -> str:
        """Randomly choose a strategy from available strategies."""
        self.logger.info("Generating strategy from available strategies.")
        strategy = random.choice(list(available_strategies.keys()))
        self.strategy_history.append(strategy)
        return strategy

    def optimize_parameters(self, strategy_name: str, n_trials: int = 10) -> dict:
        """
        Optymalizuje parametry strategii przez grid/random search.
        Parametry:
            strategy_name: nazwa strategii
            n_trials: liczba prób optymalizacji
        Zwraca:
            Słownik najlepszych parametrów według metryki total_return.
        """
        self.logger.info(
            f"Optimizing parameters for {strategy_name} with {n_trials} trials."
        )
        import importlib

        if strategy_name == "Momentum":
            strategy_cls = MomentumStrategy

            def param_choices():
                return {
                    "fast_period": random.randint(5, 20),
                    "slow_period": random.randint(21, 60),
                    "risk_per_trade": round(random.uniform(0.01, 0.05), 3),
                }

        elif strategy_name == "Mean Reversion":
            strategy_cls = MeanReversionStrategy

            def param_choices():
                return {
                    "period": random.randint(10, 30),
                    "std_dev": round(random.uniform(1.5, 3.0), 2),
                    "risk_per_trade": round(random.uniform(0.005, 0.03), 3),
                }

        else:
            return {}
        best_params = None
        best_return = -np.inf
        from data.demo_data import generate_demo_data

        for _ in range(n_trials):
            params = param_choices()
            if (
                strategy_name == "Momentum"
                and params["fast_period"] >= params["slow_period"]
            ):
                continue
            strategy = strategy_cls(**params)
            # Import BacktestEngine here to avoid circular import
            BacktestEngine = importlib.import_module(
                "engine.backtest_engine"
            ).BacktestEngine
            data = generate_demo_data("TEST")
            engine = BacktestEngine(initial_capital=100000)
            result = engine.run(
                strategy, data, stop_loss_pct=0.02, take_profit_pct=0.04
            )
            if result.total_return > best_return:
                best_return = result.total_return
                best_params = params
        return best_params if best_params else {}

    def ml_optimize_parameters(self, strategy_name, n_trials=20):
        """
        Optymalizacja parametrów przez ML (Random Forest surrogate model).
        """
        import importlib

        import numpy as np

        if strategy_name == "Momentum":
            param_grid = list(
                ParameterGrid(
                    {
                        "fast_period": range(5, 21, 5),
                        "slow_period": range(21, 61, 10),
                        "risk_per_trade": [0.01, 0.02, 0.03, 0.04, 0.05],
                    }
                )
            )
            strategy_cls = MomentumStrategy
        elif strategy_name == "Mean Reversion":
            param_grid = list(
                ParameterGrid(
                    {
                        "period": range(10, 31, 5),
                        "std_dev": [1.5, 2.0, 2.5, 3.0],
                        "risk_per_trade": [0.005, 0.01, 0.02, 0.03],
                    }
                )
            )
            strategy_cls = MeanReversionStrategy
        else:
            return {}
        X, y = [], []
        from data.demo_data import generate_demo_data

        BacktestEngine = importlib.import_module(
            "engine.backtest_engine"
        ).BacktestEngine
        data = generate_demo_data("TEST")
        engine = BacktestEngine(initial_capital=100000)
        for params in np.random.choice(
            param_grid, min(n_trials, len(param_grid)), replace=False
        ):
            strategy = strategy_cls(**params)
            result = engine.run(
                strategy, data, stop_loss_pct=0.02, take_profit_pct=0.04
            )
            X.append([params[k] for k in sorted(params)])
            y.append(result.total_return)
        if len(X) < 2:
            return param_grid[0] if param_grid else {}
        model = RandomForestRegressor().fit(X, y)
        best_idx = np.argmax(model.predict(X))
        best_params = param_grid[best_idx]
        return best_params

    def generate_report(self, backtest_result):
        # Prosty raport tekstowy
        report = f"""
Raport strategii: {backtest_result.strategy_name}
Okres: {backtest_result.start_date} - {backtest_result.end_date}
Final capital: {backtest_result.final_capital:.2f}
Total return: {backtest_result.total_return:.2%}
Win rate: {backtest_result.win_rate:.2%}
Profit factor: {backtest_result.profit_factor:.2f}
Total trades: {backtest_result.total_trades}
Max drawdown: {backtest_result.max_drawdown:.2%}
Sharpe ratio: {backtest_result.sharpe_ratio:.2f}
Sortino ratio: {backtest_result.sortino_ratio:.2f}
"""
        return report

    def recommend(self, backtest_result):
        # Prosta rekomendacja na podstawie metryk
        if backtest_result.total_return > 0.1 and backtest_result.max_drawdown > -0.2:
            return (
                "Rekomendacja: Strategia nadaje się do dalszych testów lub wdrożenia."
            )
        else:
            return "Rekomendacja: Strategia wymaga poprawy lub nie nadaje się do wdrożenia."

    def walk_forward_analysis(
        self, strategy_name, data, window_size=500, step_size=100
    ):
        """
        Walk-forward analysis: dzieli dane na okna, optymalizuje na in-sample, testuje na out-of-sample.
        Zwraca: listę wyników dla każdego okna.
        """
        results = []
        n = len(data)
        for start in range(0, n - window_size, step_size):
            data.iloc[start : start + window_size]
            test = data.iloc[start + window_size : start + window_size + step_size]
            if len(test) == 0:
                break
            best_params = self.ml_optimize_parameters(strategy_name, n_trials=10)
            if strategy_name == "Momentum":
                strategy = MomentumStrategy(**best_params)
            else:
                strategy = MeanReversionStrategy(**best_params)
            engine = BacktestEngine(initial_capital=100000)
            result = engine.run(
                strategy, test, stop_loss_pct=0.02, take_profit_pct=0.04
            )
            results.append(
                {
                    "start": test["timestamp"].iloc[0],
                    "end": test["timestamp"].iloc[-1],
                    "params": best_params,
                    "total_return": result.total_return,
                    "final_capital": result.final_capital,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                }
            )
        return results
