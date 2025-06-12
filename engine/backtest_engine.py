"""
BacktestEngine - backtesting engine for ZoL0
"""

import json
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ai.agent import ZoL0AIAgent
from models.core import BacktestResult, OrderSide, Trade


class BacktestEngine:
    """
    BacktestEngine executes backtests, manages trades, and computes metrics.
    Handles dynamic TP/SL, AI integration, and robust file operations.
    """

    def __init__(self, initial_capital: float):
        """
        Initialize BacktestEngine.
        Args:
            initial_capital (float): Starting capital for backtest.
        """
        self.initial_capital = initial_capital

    @staticmethod
    def load_production_dynamic_tp_sl_params(
        path="production_dynamic_tp_sl_params.json",
    ):
        """
        Load dynamic TP/SL parameters from production file (if exists).
        Args:
            path (str): Path to parameter file.
        Returns:
            dict or None: Parameters or None if not found/invalid.
        """
        from pathlib import Path
        import json

        p = Path(path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    params = json.load(f)
                    if isinstance(params, list):
                        params = max(params, key=lambda x: x.get("total_profit", 0))
                    return params
            except Exception as e:
                print(f"Error loading TP/SL params: {e}")
                return None
        return None

    def run(
        self,
        strategy,
        data,
        stop_loss_pct=None,
        take_profit_pct=None,
        trailing_stop_pct=None,
        dynamic_trailing=False,
        dynamic_tp_sl=False,
        tp_sl_window=None,
        tp_factor=None,
        sl_factor=None,
    ):
        """
        Run backtest on provided data and strategy.
        Args:
            strategy: Strategy object (inherits BaseStrategy).
            data: DataFrame with OHLCV data.
            stop_loss_pct: Stop loss as percent of entry (optional).
            take_profit_pct: Take profit as percent of entry (optional).
            dynamic_tp_sl: Use dynamic TP/SL (volatility-based).
            tp_sl_window: Window for volatility calculation.
            tp_factor: TP multiplier.
            sl_factor: SL multiplier.
        Returns:
            BacktestResult: Metrics, equity curve, trades.
        """
        # Zakładamy, że data ma kolumny: timestamp, close
        trades: List[Trade] = []
        equity_curve = []
        cash = self.initial_capital
        position = 0.0
        entry_price = 0.0
        entry_time = None
        commission_total = 0.0
        position_side = None  # BUY or SELL
        trailing_stop_price = None
        start_of_day_pnl = 0
        # Jeśli dynamiczne TP/SL i nie podano parametrów, spróbuj załadować z pliku produkcyjnego
        if dynamic_tp_sl and (
            tp_sl_window is None or tp_factor is None or sl_factor is None
        ):
            prod_params = self.load_production_dynamic_tp_sl_params()
            if prod_params:
                tp_sl_window = prod_params.get("tp_sl_window", 20)
                tp_factor = prod_params.get("tp_factor", 2.0)
                sl_factor = prod_params.get("sl_factor", 1.0)
            else:
                tp_sl_window = tp_sl_window or 20
                tp_factor = tp_factor or 2.0
                sl_factor = sl_factor or 1.0
        for i, row in data.iterrows():
            price = row["close"]
            timestamp = row["timestamp"]
            signal = 0
            if hasattr(strategy, "generate_signals"):
                signals = strategy.generate_signals(data.iloc[: i + 1])
                if "signal" in signals.columns:
                    signal = signals.iloc[-1]["signal"]
            # --- DYNAMIC TP/SL ---
            curr_take_profit = take_profit_pct
            curr_stop_loss = stop_loss_pct
            if dynamic_tp_sl and i >= tp_sl_window:
                prices_window = data["close"].iloc[i - tp_sl_window + 1 : i + 1].values
                volatility = np.std(prices_window)
                curr_take_profit = tp_factor * volatility / price  # as pct
                curr_stop_loss = sl_factor * volatility / price  # as pct
            # Otwórz pozycję long
            if signal == 1 and position == 0:
                qty = strategy.calculate_position_size(signal, price, cash)
                entry_price = price
                entry_time = timestamp
                position = qty
                position_side = OrderSide.BUY
                commission = qty * price * 0.001
                cash -= qty * price + commission
                commission_total += commission
                # Zapamiętaj dynamiczne TP/SL dla tej pozycji
                entry_take_profit = curr_take_profit
                entry_stop_loss = curr_stop_loss
            # Otwórz pozycję short
            elif signal == -1 and position == 0:
                qty = strategy.calculate_position_size(signal, price, cash)
                entry_price = price
                entry_time = timestamp
                position = qty
                position_side = OrderSide.SELL
                commission = qty * price * 0.001
                cash += qty * price - commission  # short: cash increases
                commission_total += commission
                entry_take_profit = curr_take_profit
                entry_stop_loss = curr_stop_loss
            # Zamknij pozycję long
            elif signal == -1 and position > 0 and position_side == OrderSide.BUY:
                exit_price = price
                exit_time = timestamp
                pnl = (exit_price - entry_price) * position
                commission = position * exit_price * 0.001
                cash += position * exit_price - commission
                commission_total += commission
                trades.append(
                    Trade(
                        id=str(uuid.uuid4()),
                        symbol="TEST",
                        entry_timestamp=entry_time,
                        exit_timestamp=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=position,
                        side=OrderSide.BUY,
                        pnl=pnl,
                        commission=commission,
                        duration=exit_time - entry_time,
                    )
                )
                position = 0
                entry_price = 0
                entry_time = None
                position_side = None
            # Zamknij pozycję short
            elif signal == 1 and position > 0 and position_side == OrderSide.SELL:
                exit_price = price
                exit_time = timestamp
                pnl = (entry_price - exit_price) * position
                commission = position * exit_price * 0.001
                cash -= position * exit_price + commission  # short: cash decreases
                commission_total += commission
                trades.append(
                    Trade(
                        id=str(uuid.uuid4()),
                        symbol="TEST",
                        entry_timestamp=entry_time,
                        exit_timestamp=exit_time,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        quantity=position,
                        side=OrderSide.SELL,
                        pnl=pnl,
                        commission=commission,
                        duration=exit_time - entry_time,
                    )
                )
                position = 0
                entry_price = 0
                entry_time = None
                position_side = None
            # Stop loss / take profit
            if position > 0 and entry_price > 0:
                if position_side == OrderSide.BUY:
                    if entry_stop_loss and price <= entry_price * (1 - entry_stop_loss):
                        # SL long
                        exit_price = price
                        exit_time = timestamp
                        pnl = (exit_price - entry_price) * position
                        commission = position * exit_price * 0.001
                        cash += position * exit_price - commission
                        commission_total += commission
                        trades.append(
                            Trade(
                                id=str(uuid.uuid4()),
                                symbol="TEST",
                                entry_timestamp=entry_time,
                                exit_timestamp=exit_time,
                                entry_price=entry_price,
                                exit_price=exit_price,
                                quantity=position,
                                side=OrderSide.BUY,
                                pnl=pnl,
                                commission=commission,
                                duration=exit_time - entry_time,
                            )
                        )
                        position = 0
                        entry_price = 0
                        entry_time = None
                        position_side = None
                    elif entry_take_profit and price >= entry_price * (
                        1 + entry_take_profit
                    ):
                        # TP long
                        exit_price = price
                        exit_time = timestamp
                        pnl = (exit_price - entry_price) * position
                        commission = position * exit_price * 0.001
                        cash += position * exit_price - commission
                        commission_total += commission
                        trades.append(
                            Trade(
                                id=str(uuid.uuid4()),
                                symbol="TEST",
                                entry_timestamp=entry_time,
                                exit_timestamp=exit_time,
                                entry_price=entry_price,
                                exit_price=exit_price,
                                quantity=position,
                                side=OrderSide.BUY,
                                pnl=pnl,
                                commission=commission,
                                duration=exit_time - entry_time,
                            )
                        )
                        position = 0
                        entry_price = 0
                        entry_time = None
                        position_side = None
                elif position_side == OrderSide.SELL:
                    if entry_stop_loss and price >= entry_price * (1 + entry_stop_loss):
                        # SL short
                        exit_price = price
                        exit_time = timestamp
                        pnl = (entry_price - exit_price) * position
                        commission = position * exit_price * 0.001
                        cash -= position * exit_price + commission
                        commission_total += commission
                        trades.append(
                            Trade(
                                id=str(uuid.uuid4()),
                                symbol="TEST",
                                entry_timestamp=entry_time,
                                exit_timestamp=exit_time,
                                entry_price=entry_price,
                                exit_price=exit_price,
                                quantity=position,
                                side=OrderSide.SELL,
                                pnl=pnl,
                                commission=commission,
                                duration=exit_time - entry_time,
                            )
                        )
                        position = 0
                        entry_price = 0
                        entry_time = None
                        position_side = None
                    elif entry_take_profit and price <= entry_price * (
                        1 - entry_take_profit
                    ):
                        # TP short
                        exit_price = price
                        exit_time = timestamp
                        pnl = (entry_price - exit_price) * position
                        commission = position * exit_price * 0.001
                        cash -= position * exit_price + commission
                        commission_total += commission
                        trades.append(
                            Trade(
                                id=str(uuid.uuid4()),
                                symbol="TEST",
                                entry_timestamp=entry_time,
                                exit_timestamp=exit_time,
                                entry_price=entry_price,
                                exit_price=exit_price,
                                quantity=position,
                                side=OrderSide.SELL,
                                pnl=pnl,
                                commission=commission,
                                duration=exit_time - entry_time,
                            )
                        )
                        position = 0
                        entry_price = 0
                        entry_time = None
                        position_side = None
            # Trailing stop logic (long)
            if position > 0 and position_side == OrderSide.BUY and trailing_stop_pct:
                if trailing_stop_price is None:
                    trailing_stop_price = price * (1 - trailing_stop_pct)
                else:
                    new_trailing = price * (1 - trailing_stop_pct)
                    if new_trailing > trailing_stop_price:
                        trailing_stop_price = new_trailing
                if price <= trailing_stop_price:
                    exit_price = price
                    exit_time = timestamp
                    pnl = (exit_price - entry_price) * position
                    commission = position * exit_price * 0.001
                    cash += position * exit_price - commission
                    commission_total += commission
                    trades.append(
                        Trade(
                            id=str(uuid.uuid4()),
                            symbol="TEST",
                            entry_timestamp=entry_time,
                            exit_timestamp=exit_time,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            quantity=position,
                            side=OrderSide.BUY,
                            pnl=pnl,
                            commission=commission,
                            duration=exit_time - entry_time,
                        )
                    )
                    position = 0
                    entry_price = 0
                    entry_time = None
                    position_side = None
                    trailing_stop_price = None
            # Trailing stop logic (short)
            if position > 0 and position_side == OrderSide.SELL and trailing_stop_pct:
                if trailing_stop_price is None:
                    trailing_stop_price = price * (1 + trailing_stop_pct)
                else:
                    new_trailing = price * (1 + trailing_stop_pct)
                    if new_trailing < trailing_stop_price:
                        trailing_stop_price = new_trailing
                if price >= trailing_stop_price:
                    exit_price = price
                    exit_time = timestamp
                    pnl = (entry_price - exit_price) * position
                    commission = position * exit_price * 0.001
                    cash -= position * exit_price + commission
                    commission_total += commission
                    trades.append(
                        Trade(
                            id=str(uuid.uuid4()),
                            symbol="TEST",
                            entry_timestamp=entry_time,
                            exit_timestamp=exit_time,
                            entry_price=entry_price,
                            exit_price=exit_price,
                            quantity=position,
                            side=OrderSide.SELL,
                            pnl=pnl,
                            commission=commission,
                            duration=exit_time - entry_time,
                        )
                    )
                    position = 0
                    entry_price = 0
                    entry_time = None
                    position_side = None
                    trailing_stop_price = None
            # Dynamic trailing stop logic
            if dynamic_trailing and position > 0:
                # Advanced ATR-based dynamic trailing stop
                window = 14
                if i >= window:
                    high = data["high"].iloc[i - window + 1 : i + 1]
                    low = data["low"].iloc[i - window + 1 : i + 1]
                    close = data["close"].iloc[i - window + 1 : i + 1]
                    tr = pd.concat(
                        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
                        axis=1,
                    ).max(axis=1)
                    atr = tr.mean()
                    dynamic_pct = atr / price
                    if position_side == OrderSide.BUY:
                        new_trailing = price * (1 - dynamic_pct)
                        if (
                            trailing_stop_price is None
                            or new_trailing > trailing_stop_price
                        ):
                            trailing_stop_price = new_trailing
                        if price <= trailing_stop_price:
                            # ...zamknięcie pozycji jak w trailing stop...
                            exit_price = price
                            exit_time = timestamp
                            pnl = (exit_price - entry_price) * position
                            commission = position * exit_price * 0.001
                            cash += position * exit_price - commission
                            commission_total += commission
                            trades.append(
                                Trade(
                                    id=str(uuid.uuid4()),
                                    symbol="TEST",
                                    entry_timestamp=entry_time,
                                    exit_timestamp=exit_time,
                                    entry_price=entry_price,
                                    exit_price=exit_price,
                                    quantity=position,
                                    side=OrderSide.BUY,
                                    pnl=pnl,
                                    commission=commission,
                                    duration=exit_time - entry_time,
                                )
                            )
                            position = 0
                            entry_price = 0
                            entry_time = None
                            position_side = None
                            trailing_stop_price = None
                    elif position_side == OrderSide.SELL:
                        new_trailing = price * (1 + dynamic_pct)
                        if (
                            trailing_stop_price is None
                            or new_trailing < trailing_stop_price
                        ):
                            trailing_stop_price = new_trailing
                        if price >= trailing_stop_price:
                            # ...zamknięcie pozycji jak w trailing stop...
                            exit_price = price
                            exit_time = timestamp
                            pnl = (entry_price - exit_price) * position
                            commission = position * exit_price * 0.001
                            cash -= position * exit_price + commission
                            commission_total += commission
                            trades.append(
                                Trade(
                                    id=str(uuid.uuid4()),
                                    symbol="TEST",
                                    entry_timestamp=entry_time,
                                    exit_timestamp=exit_time,
                                    entry_price=entry_price,
                                    exit_price=exit_price,
                                    quantity=position,
                                    side=OrderSide.SELL,
                                    pnl=pnl,
                                    commission=commission,
                                    duration=exit_time - entry_time,
                                )
                            )
                            position = 0
                            entry_price = 0
                            entry_time = None
                            position_side = None
                            trailing_stop_price = None
            portfolio_value = cash + (
                position * price
                if position_side == OrderSide.BUY
                else -position * price if position_side == OrderSide.SELL else 0
            )
            equity_curve.append(
                {
                    "timestamp": timestamp,
                    "portfolio_value": portfolio_value,
                    "cash": cash,
                    "position_value": (
                        position * price
                        if position_side == OrderSide.BUY
                        else -position * price if position_side == OrderSide.SELL else 0
                    ),
                }
            )
            # Sprawdź limit dziennej straty
            if self.enforce_daily_loss_limit(
                cash + position * price, start_of_day_pnl, daily_loss_limit_pct=0.05
            ):
                print(
                    f"Limit dziennej straty osiągnięty. Zatrzymanie handlu na dzień: {timestamp}"
                )
                break
        # Metryki
        equity_df = pd.DataFrame(equity_curve)
        returns = equity_df["portfolio_value"].pct_change().dropna()
        total_return = (
            equity_df["portfolio_value"].iloc[-1] - self.initial_capital
        ) / self.initial_capital
        days = (data["timestamp"].iloc[-1] - data["timestamp"].iloc[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        rolling_max = equity_df["portfolio_value"].cummax()
        drawdown = (equity_df["portfolio_value"] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(365 * 24)
            if returns.std() > 0
            else 0
        )
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 0 else 0
        sortino_ratio = (
            returns.mean() / downside_std * np.sqrt(365 * 24) if downside_std > 0 else 0
        )
        # Win rate, profit factor
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        win_rate = len(wins) / len(trades) if trades else 0
        profit_factor = (
            sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses))
            if losses
            else float("inf")
        )
        result = BacktestResult(
            strategy_name=strategy.name,
            start_date=data["timestamp"].iloc[0],
            end_date=data["timestamp"].iloc[-1],
            initial_capital=self.initial_capital,
            final_capital=equity_df["portfolio_value"].iloc[-1],
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            trades=trades,
            equity_curve=equity_df,
            metrics={
                "volatility": returns.std() * np.sqrt(365 * 24),
                "best_day": returns.max(),
                "worst_day": returns.min(),
                "total_days": days,
            },
        )
        # Advanced analytics
        rolling_sharpe = (
            returns.rolling(window=24).mean()
            / returns.rolling(window=24).std()
            * np.sqrt(24)
            if returns.std() > 0
            else 0
        )
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        for t in trades:
            if t.pnl > 0:
                current_wins += 1
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
                current_losses = 0
            elif t.pnl < 0:
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
                current_wins = 0
        result.metrics.update(
            {
                "rolling_sharpe_24h": (
                    rolling_sharpe.iloc[-1]
                    if hasattr(rolling_sharpe, "iloc")
                    else rolling_sharpe
                ),
                "max_consecutive_wins": max_consecutive_wins,
                "max_consecutive_losses": max_consecutive_losses,
            }
        )
        # --- AI scoring and recommendation ---
        try:
            agent = ZoL0AIAgent()
            result.ai_report = agent.generate_report(result)
            result.ai_recommendation = agent.recommend(result)
        except Exception as e:
            result.ai_report = f"AI report error: {e}"
            result.ai_recommendation = "AI recommendation unavailable."
        # ---
        return result

    def auto_optimize_strategy(self, strategy_name, n_trials=20, dynamic_tp_sl=False):
        """
        Automatyczna optymalizacja parametrów strategii (w tym dynamiczne TP/SL) na danych demo.
        Zwraca najlepsze parametry i metryki zyskowności.
        """
        best_params = None
        best_result = None
        best_return = -np.inf
        agent = ZoL0AIAgent()
        for _ in range(n_trials):
            params = agent.optimize_parameters(strategy_name, n_trials=1)
            if not params:
                continue
            if strategy_name == "Momentum":
                from strategies.momentum import MomentumStrategy

                strategy = MomentumStrategy(**params)
            elif strategy_name == "Mean Reversion":
                from strategies.mean_reversion import MeanReversionStrategy

                strategy = MeanReversionStrategy(**params)
            else:
                continue
            from data.demo_data import generate_demo_data

            data = generate_demo_data("TEST")
            # Testuj z dynamicznym TP/SL jeśli włączone
            result = self.run(strategy, data, dynamic_tp_sl=dynamic_tp_sl)
            if result.total_return > best_return:
                best_return = result.total_return
                best_params = params
                best_result = result
        return {"best_params": best_params, "metrics": best_result}

    def enforce_daily_loss_limit(
        self, current_pnl, start_of_day_pnl, daily_loss_limit_pct=0.05
    ):
        """
        Enforce daily loss limit. Returns True if trading should be stopped.
        """
        loss = start_of_day_pnl - current_pnl
        if loss / max(abs(start_of_day_pnl), 1) >= daily_loss_limit_pct:
            return True
        return False


# Edge-case test examples (to be expanded in test suite)
def _test_edge_cases():
    try:
        be = BacktestEngine(1000)
        be.run(None, None)  # invalid strategy/data
    except Exception as e:
        print(f"Handled edge case in BacktestEngine: {e}")


# CI/CD: Zautomatyzowane testy edge-case i workflow wdrożone w .github/workflows/ci-cd.yml
