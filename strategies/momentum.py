from engine.base_strategy import BaseStrategy
import logging


class MomentumStrategy(BaseStrategy):
    """
    Zaawansowana strategia momentum oparta na przecięciu średnich kroczących.
    Dodano: logowanie, hooki analityczne, obsługa premium, walidacja parametrów.
    Parametry: fast_period, slow_period, risk_per_trade, premium_features.
    """

    def __init__(
        self, fast_period=20, slow_period=50, premium_features=False, **kwargs
    ):
        super().__init__(
            "Momentum",
            {
                "fast_period": fast_period,
                "slow_period": slow_period,
                "premium_features": premium_features,
                **kwargs,
            },
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.premium_features = premium_features
        self._validate_parameters()

    def _validate_parameters(self):
        if (
            self.parameters["fast_period"] < 1
            or self.parameters["slow_period"] <= self.parameters["fast_period"]
        ):
            raise ValueError("Invalid parameters for MomentumStrategy")

    def generate_signals(self, data):
        try:
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
            # Analytics hook
            self.logger.info(
                f"Signals generated: {df['signal'].value_counts().to_dict()}"
            )
            if self.premium_features:
                df["premium_alert"] = df["signal"] != 0
            return df[["signal"]]
        except Exception as e:
            self.logger.error(f"Error in generate_signals: {e}")
            raise

    def calculate_position_size(self, signal, current_price, portfolio_value):
        try:
            risk_per_trade = self.parameters.get("risk_per_trade", 0.02)
            size = (portfolio_value * risk_per_trade) / current_price
            # Monetization hook: premium users get dynamic sizing
            if self.premium_features:
                size *= 1.1
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
