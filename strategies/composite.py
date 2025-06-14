from engine.base_strategy import BaseStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy
import logging

class CompositeStrategy(BaseStrategy):
    """
    Composite strategy that combines signals from multiple strategies (mean reversion and momentum).
    Generates a consensus signal and dynamically allocates capital.
    Dodano: logowanie, hooki analityczne, obsługa premium, walidacja parametrów.
    Parametry: weights, risk_per_trade, premium_features.
    """
    def __init__(self, weights=None, premium_features=False, **kwargs):
        if weights is None:
            weights = {"mean_reversion": 0.5, "momentum": 0.5}
        super().__init__("Composite", {"weights": weights, "premium_features": premium_features, **kwargs})
        self.mean_reversion = MeanReversionStrategy(premium_features=premium_features, **kwargs)
        self.momentum = MomentumStrategy(premium_features=premium_features, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.premium_features = premium_features
        self._validate_parameters()

    def _validate_parameters(self):
        w = self.parameters["weights"]
        if not (0 <= w["mean_reversion"] <= 1 and 0 <= w["momentum"] <= 1):
            raise ValueError("Invalid weights for CompositeStrategy")
        if abs(w["mean_reversion"] + w["momentum"] - 1) > 1e-6:
            raise ValueError("Weights must sum to 1")

    def generate_signals(self, data):
        try:
            signals_mr = self.mean_reversion.generate_signals(data)
            signals_mom = self.momentum.generate_signals(data)
            consensus = (
                self.parameters["weights"]["mean_reversion"] * signals_mr["signal"] +
                self.parameters["weights"]["momentum"] * signals_mom["signal"]
            )
            result = signals_mr.copy()
            result["signal"] = consensus.apply(lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0))
            # Analytics hook
            self.logger.info(f"Composite signals: {result['signal'].value_counts().to_dict()}")
            if self.premium_features:
                result["premium_alert"] = (result["signal"] != 0)
            return result[["signal"]]
        except Exception as e:
            self.logger.error(f"Error in generate_signals: {e}")
            raise

    def calculate_position_size(self, signal, current_price, portfolio_value):
        try:
            risk_per_trade = self.parameters.get("risk_per_trade", 0.015)
            size = (portfolio_value * risk_per_trade) / current_price
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
