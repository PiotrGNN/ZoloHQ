import logging
import sys

from engine.backtest_engine import BacktestEngine

logger = logging.getLogger("auto_optimize")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.info(
            "Usage: python auto_optimize.py [Momentum|Mean Reversion] [--dynamic-tp-sl]"
        )
        sys.exit(1)
    strategy_name = sys.argv[1]
    dynamic_tp_sl = "--dynamic-tp-sl" in sys.argv
    engine = BacktestEngine(initial_capital=100000)
    result = engine.auto_optimize_strategy(
        strategy_name, n_trials=20, dynamic_tp_sl=dynamic_tp_sl
    )
    logger.info("Best parameters: %s", result["best_params"])
    logger.info("Profitability metrics:")
    metrics = result["metrics"]
    if metrics:
        logger.info("Total return: %.2f%%", metrics.total_return)
        logger.info("Sharpe ratio: %.2f", metrics.sharpe_ratio)
        logger.info("Win rate: %.2f%%", metrics.win_rate)
        logger.info("Profit factor: %.2f", metrics.profit_factor)
    else:
        logger.info("No result.")
