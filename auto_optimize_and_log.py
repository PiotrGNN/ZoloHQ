from datetime import datetime

from engine.backtest_engine import BacktestEngine


def run_and_log_optimization(
    strategy_name, dynamic_tp_sl=True, n_trials=20, log_path="logs/auto_optimize.log"
):
    engine = BacktestEngine(initial_capital=100000)
    result = engine.auto_optimize_strategy(
        strategy_name, n_trials=n_trials, dynamic_tp_sl=dynamic_tp_sl
    )
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy_name,
        "dynamic_tp_sl": dynamic_tp_sl,
        "best_params": result["best_params"],
        "metrics": {
            "total_return": getattr(result["metrics"], "total_return", None),
            "sharpe_ratio": getattr(result["metrics"], "sharpe_ratio", None),
            "win_rate": getattr(result["metrics"], "win_rate", None),
            "profit_factor": getattr(result["metrics"], "profit_factor", None),
        },
    }
    with open(log_path, "a") as f:
        f.write(str(log_entry) + "\n")
    print("Optimization complete. Log entry:", log_entry)


if __name__ == "__main__":
    # Example: run daily via cron/task scheduler
    run_and_log_optimization("Momentum", dynamic_tp_sl=True)
    run_and_log_optimization("Mean Reversion", dynamic_tp_sl=True)
