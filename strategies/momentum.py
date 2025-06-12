from engine.base_strategy import BaseStrategy


class MomentumStrategy(BaseStrategy):
    """
    Prosta strategia momentum oparta na przecięciu średnich kroczących.
    Parametry: fast_period, slow_period, risk_per_trade.
    """

    def __init__(self, fast_period=20, slow_period=50, **kwargs):
        super().__init__(
            "Momentum",
            {"fast_period": fast_period, "slow_period": slow_period, **kwargs},
        )

    def generate_signals(self, data):
        df = data.copy()
        df["ma_fast"] = (
            df["close"].rolling(window=self.parameters["fast_period"]).mean()
        )
        df["ma_slow"] = (
            df["close"].rolling(window=self.parameters["slow_period"]).mean()
        )
        df["signal"] = 0
        df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1
        df.loc[df["ma_fast"] < df["ma_slow"], "signal"] = -1
        return df[["signal"]]

    def calculate_position_size(self, signal, current_price, portfolio_value):
        risk_per_trade = self.parameters.get("risk_per_trade", 0.02)
        return (portfolio_value * risk_per_trade) / current_price
