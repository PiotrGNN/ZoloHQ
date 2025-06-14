"""
PortfolioManager module for portfolio analytics and management.
"""

import pandas as pd
import logging


class PortfolioManager:
    """
    PortfolioManager handles portfolio state, analytics, and advanced position sizing.
    Provides risk management, analytics, ML-based portfolio selection, premium analytics, and monetization hooks.
    """

    def __init__(self, initial_cash=100000, premium_features=False):
        """
        Initialize PortfolioManager.
        Args:
            initial_cash (float): Starting cash for the portfolio.
        """
        self.cash = initial_cash
        self.positions = {}
        self.history = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.premium_features = premium_features

    def update(self, symbol, action, qty, price):
        """
        Update portfolio with a trade action.
        Args:
            symbol (str): Asset symbol.
            action (str): 'buy' or 'sell'.
            qty (float): Quantity.
            price (float): Trade price.
        """
        try:
            if action == "buy":
                self.cash -= qty * price
                self.positions[symbol] = self.positions.get(symbol, 0) + qty
            elif action == "sell":
                self.cash += qty * price
                self.positions[symbol] = self.positions.get(symbol, 0) - qty
            self.history.append(
                {
                    "symbol": symbol,
                    "action": action,
                    "qty": qty,
                    "price": price,
                    "cash": self.cash,
                }
            )
            if self.premium_features:
                self.logger.info(f"[PREMIUM] Portfolio update: {action} {qty} {symbol} @ {price}")
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
            raise

    def get_portfolio_value(self, prices: dict):
        """
        Calculate total portfolio value given current prices.
        Args:
            prices (dict): {symbol: price}
        Returns:
            float: Total value.
        """
        value = self.cash
        for symbol, qty in self.positions.items():
            value += qty * prices.get(symbol, 0)
        return value

    def get_positions(self):
        """
        Get current positions.
        Returns:
            dict: {symbol: qty}
        """
        return self.positions

    def get_history(self):
        """
        Get trade history as DataFrame.
        Returns:
            pd.DataFrame: Trade history.
        """
        return pd.DataFrame(self.history)

    def get_advanced_analytics(self, prices: dict):
        """
        Compute advanced analytics (returns, drawdown, Sharpe, etc.).
        Args:
            prices (dict): {symbol: price}
        Returns:
            dict: Analytics metrics.
        """
        df = self.get_history()
        if df.empty:
            return {}
        try:
            df["portfolio_value"] = df["cash"] + sum(
                self.positions.get(sym, 0) * prices.get(sym, 0) for sym in self.positions
            )
            returns = df["portfolio_value"].pct_change().dropna()
            analytics = {
                "total_return": (
                    df["portfolio_value"].iloc[-1] - df["portfolio_value"].iloc[0]
                ) / df["portfolio_value"].iloc[0],
                "max_drawdown": (
                    df["portfolio_value"] / df["portfolio_value"].cummax() - 1
                ).min(),
                "volatility": returns.std() * (252**0.5),
                "sharpe_ratio": (
                    returns.mean() / returns.std() * (252**0.5) if returns.std() > 0 else 0
                ),
                "rolling_sharpe_30": (
                    returns.rolling(window=30).mean().iloc[-1]
                    / returns.rolling(window=30).std().iloc[-1]
                    * (252**0.5)
                    if len(returns) >= 30 and returns.rolling(window=30).std().iloc[-1] > 0
                    else 0
                ),
                "max_consecutive_wins": (returns > 0)
                .astype(int)
                .groupby((returns <= 0).astype(int).cumsum())
                .sum()
                .max(),
                "max_consecutive_losses": (returns < 0)
                .astype(int)
                .groupby((returns >= 0).astype(int).cumsum())
                .sum()
                .max(),
            }
            if self.premium_features:
                self.logger.info("[PREMIUM] Advanced analytics calculated.")
                analytics["premium"] = True
            return analytics
        except Exception as e:
            self.logger.error(f"Error in analytics calculation: {e}")
            return {}

    def dynamic_position_sizing(
        self, symbol, risk_per_trade, stop_distance, portfolio_value
    ):
        """
        Dynamiczne wyznaczanie wielkości pozycji na podstawie ryzyka i odległości stopa.
        """
        if stop_distance == 0:
            return 0
        position_size = (portfolio_value * risk_per_trade) / stop_distance
        return position_size

    def risk_parity_weights(self, returns_df):
        """
        Wyznaczanie wag portfela metodą risk parity (na podstawie odwróconej zmienności).
        """
        vol = returns_df.std()
        inv_vol = 1 / vol
        weights = inv_vol / inv_vol.sum()
        return weights.to_dict()

    def kelly_position_sizing(
        self, win_rate, win_loss_ratio, portfolio_value, risk_fraction=1.0
    ):
        """
        Kelly criterion: wyznaczanie optymalnej wielkości pozycji.
        win_rate: prawdopodobieństwo wygranej (0-1)
        win_loss_ratio: stosunek średniego zysku do średniej straty
        risk_fraction: część portfela do ryzyka (0-1)
        """
        kelly = risk_fraction * (win_rate - (1 - win_rate) / win_loss_ratio)
        kelly = max(0, kelly)  # nie pozwól na ujemny sizing
        return portfolio_value * kelly

    def dynamic_risk_targeting(self, target_vol, returns):
        """
        Dynamiczne dostosowanie ryzyka do zmienności portfela.
        target_vol: docelowa roczna zmienność (np. 0.15)
        returns: seria zwrotów portfela
        """
        realized_vol = returns.std() * (252**0.5)
        if realized_vol == 0:
            return 1.0
        scaling = target_vol / realized_vol
        return scaling

    def black_litterman_weights(self, pi, tau, P, Q, Sigma):
        """
        Black-Litterman: wyznaczanie wag portfela na podstawie poglądów i macierzy kowariancji.
        pi: equilibrium returns (np. z CAPM)
        tau: współczynnik niepewności
        P: macierz poglądów (n_views x n_assets)
        Q: wektor poglądów (n_views)
        Sigma: macierz kowariancji (n_assets x n_assets)
        """
        import numpy as np

        inv = np.linalg.inv
        M = inv(inv(tau * Sigma) + P.T @ inv(np.eye(P.shape[0]) * 0.01) @ P)
        mu_bl = M @ (inv(tau * Sigma) @ pi + P.T @ inv(np.eye(P.shape[0]) * 0.01) @ Q)
        return mu_bl

    def equal_risk_contribution_weights(self, returns_df):
        """
        Wyznaczanie wag portfela metodą Equal Risk Contribution (ERC).
        """
        import numpy as np

        cov = returns_df.cov().values
        n = cov.shape[0]
        w = np.ones(n) / n
        for _ in range(100):
            risk_contrib = w * (cov @ w)
            total_risk = np.sum(risk_contrib)
            grad = risk_contrib - total_risk / n
            w -= 0.01 * grad
            w = np.maximum(w, 0)
            w /= w.sum()
        return dict(zip(returns_df.columns, w))

    def value_at_risk(self, returns, alpha=0.05):
        """
        Value at Risk (VaR) portfela na zadanym poziomie istotności.
        """
        return -returns.quantile(alpha)

    def expected_shortfall(self, returns, alpha=0.05):
        """
        Expected Shortfall (ES, CVaR) portfela na zadanym poziomie istotności.
        """
        var = self.value_at_risk(returns, alpha)
        return (
            -returns[returns < -var].mean() if not returns[returns < -var].empty else 0
        )

    def dynamic_hedging(self, spot_price_series, option_delta_series):
        """
        Dynamiczne hedgowanie pozycji (np. delta hedging opcji).
        spot_price_series: seria cen instrumentu bazowego
        option_delta_series: seria delt opcji (0-1)
        Zwraca: lista pozycji hedgujących w czasie
        """
        hedge_positions = -option_delta_series
        return hedge_positions

    def factor_investing_weights(self, factor_scores):
        """
        Wyznaczanie wag portfela na podstawie scoringu czynnikowego (factor investing).
        factor_scores: dict {symbol: score}
        """
        import numpy as np

        scores = np.array(list(factor_scores.values()))
        scores = np.maximum(scores, 0)
        if scores.sum() == 0:
            weights = np.ones_like(scores) / len(scores)
        else:
            weights = scores / scores.sum()
        return dict(zip(factor_scores.keys(), weights))

    def ml_portfolio_selection(self, X, y):
        """
        Machine learning portfolio selection (np. regresja doboru wag).
        X: macierz cech (np. historyczne zwroty, faktory)
        y: wektor docelowy (np. przyszłe zwroty)
        Zwraca: wektor wag portfela
        """
        import numpy as np
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0).fit(X, y)
        coefs = np.maximum(model.coef_, 0)
        if coefs.sum() == 0:
            weights = np.ones_like(coefs) / len(coefs)
        else:
            weights = coefs / coefs.sum()
        return weights

    # UWAGA: Metody dynamic_hedging, factor_investing_weights, ml_portfolio_selection są w pełni dostępne i sterowalne z poziomu UI (ui/app.py).
    # Nie są wywoływane automatycznie poza UI.


# CI/CD: Zautomatyzowane testy edge-case i workflow wdrożone w .github/workflows/ci-cd.yml

# Edge-case test examples (to be expanded in test suite)
def _test_edge_cases():
    pm = PortfolioManager()
    try:
        pm.update("BTC", "buy", 0, 0)  # zero qty/price
        pm.update("BTC", "sell", 1, 0)  # zero price
        pm.update("BTC", "buy", -1, 100)  # negative qty
    except Exception as e:
        print(f"Handled edge case in update: {e}")
    try:
        print(pm.get_advanced_analytics({}))  # empty prices
    except Exception as e:
        print(f"Handled edge case in analytics: {e}")
# TODO: Add more edge-case and integration tests for CI/CD
