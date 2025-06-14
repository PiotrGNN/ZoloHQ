from engine.base_strategy import BaseStrategy
import logging


class MeanReversionStrategy(BaseStrategy):
    """
    Zaawansowana strategia mean reversion oparta na Bollinger Bands.
    Dodano: logowanie, hooki analityczne, obsługa premium, walidacja parametrów.
    Parametry: period, std_dev, risk_per_trade, premium_features.
    """

    def __init__(self, period=20, std_dev=2.0, premium_features=False, **kwargs):
        super().__init__(
            "Mean Reversion", {"period": period, "std_dev": std_dev, "premium_features": premium_features, **kwargs}
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.premium_features = premium_features
        self._validate_parameters()

    def _validate_parameters(self):
        if self.parameters["period"] < 5 or self.parameters["std_dev"] <= 0:
            raise ValueError("Invalid parameters for MeanReversionStrategy")

    def generate_signals(self, data):
        try:
            df = data.copy()
            df["ma"] = df["close"].rolling(window=self.parameters["period"]).mean()
            df["std"] = df["close"].rolling(window=self.parameters["period"]).std()
            df["upper_band"] = df["ma"] + (df["std"] * self.parameters["std_dev"])
            df["lower_band"] = df["ma"] - (df["std"] * self.parameters["std_dev"])
            df["signal"] = 0
            df.loc[df["close"] < df["lower_band"], "signal"] = 1
            df.loc[df["close"] > df["upper_band"], "signal"] = -1
            # Analytics hook
            self.logger.info(f"Signals generated: {df['signal'].value_counts().to_dict()}")
            if self.premium_features:
                df["premium_alert"] = (df["signal"] != 0)
            return df[["signal"]]
        except Exception as e:
            self.logger.error(f"Error in generate_signals: {e}")
            raise

    def calculate_position_size(self, signal, current_price, portfolio_value):
        try:
            risk_per_trade = self.parameters.get("risk_per_trade", 0.01)
            size = (portfolio_value * risk_per_trade) / current_price
            # Monetization hook: premium users get dynamic sizing
            if self.premium_features:
                size *= 1.1  # Example: premium users get 10% more exposure
            return size
        except Exception as e:
            self.logger.error(f"Error in calculate_position_size: {e}")
            raise

    def register_strategy(self):
        # Hook do rejestracji strategii w marketplace/monitoringu
        self.logger.info(f"Rejestracja strategii: {self.name}")
        # Możliwość rozbudowy o integrację z API marketplace

    def usage_analytics(self, event: str):
        # Hook do analityki użycia strategii (np. do monetyzacji)
        self.logger.info(f"Użycie strategii {self.name}: {event}")
        # Możliwość rozbudowy o wysyłkę do centralnego systemu analityki
