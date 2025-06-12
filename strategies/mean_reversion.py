from engine.base_strategy import BaseStrategy


class MeanReversionStrategy(BaseStrategy):
    """
    Prosta strategia mean reversion oparta na Bollinger Bands.
    Parametry: period, std_dev, risk_per_trade.
    """

    def __init__(self, period=20, std_dev=2.0, **kwargs):
        super().__init__(
            "Mean Reversion", {"period": period, "std_dev": std_dev, **kwargs}
        )

    def generate_signals(self, data):
        df = data.copy()
        df["ma"] = df["close"].rolling(window=self.parameters["period"]).mean()
        df["std"] = df["close"].rolling(window=self.parameters["period"]).std()
        df["upper_band"] = df["ma"] + (df["std"] * self.parameters["std_dev"])
        df["lower_band"] = df["ma"] - (df["std"] * self.parameters["std_dev"])
        df["signal"] = 0
        df.loc[df["close"] < df["lower_band"], "signal"] = 1
        df.loc[df["close"] > df["upper_band"], "signal"] = -1
        return df[["signal"]]

    def calculate_position_size(self, signal, current_price, portfolio_value):
        risk_per_trade = self.parameters.get("risk_per_trade", 0.01)
        return (portfolio_value * risk_per_trade) / current_price
