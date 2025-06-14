import asyncio
import logging
import os
import subprocess
import sys
from typing import Callable, List, Optional

import numpy as np
from data.demo_data import generate_demo_data
from engine.backtest_engine import BacktestEngine
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from starlette_exporter import PrometheusMiddleware, handle_metrics

from strategies.mean_reversion import MeanReversionStrategy
from strategies.momentum import MomentumStrategy

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_KEYS = {
    "admin-key": "admin",
    "strategy-key": "strategy",
    "partner-key": "partner",
    "premium-key": "premium",
    "saas-key": "saas",
}
API_KEY_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)


def get_api_key(api_key: str = Depends(API_KEY_HEADER)):
    if api_key in API_KEYS:
        return API_KEYS[api_key]
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


class CompositeStrategy:
    """
    Advanced ensemble strategy: supports dynamic weight optimization, stacking, meta-models, and analytics.
    """

    def __init__(
        self,
        strategies: List,
        weights: Optional[List[float]] = None,
        ensemble_method: str = "weighted",
        meta_model: Optional[Callable] = None,
        window: int = 50,
    ):
        self.strategies = strategies
        if weights is not None and len(weights) != len(strategies):
            raise ValueError("weights length must match number of strategies")
        self.weights = weights or [1.0 / len(strategies)] * len(strategies)
        self.ensemble_method = ensemble_method  # 'weighted', 'majority', 'stacking'
        self.meta_model = meta_model  # for stacking/meta-ensemble
        self.window = window  # rolling window for dynamic weights
        self.performance_log = [[] for _ in strategies]  # track rolling performance
        self.name = "CompositeStrategy"

    def generate_signals(self, data):
        signals = []
        for i, strat in enumerate(self.strategies):
            try:
                s = strat.generate_signals(data)
                sig = s["signal"]
                sig_arr = sig.values if hasattr(sig, "values") else sig
                signals.append(np.array(sig_arr))
            except Exception as e:
                logger.error(f"Sub-strategy {i} failed: {e}")
                signals.append(np.zeros(len(data)))
        signals = np.array(signals)
        # Dynamic weight optimization
        self._update_weights()
        if self.ensemble_method == "weighted":
            weighted = np.tensordot(self.weights, signals, axes=1)
            return {"signal": np.sign(weighted)}
        elif self.ensemble_method == "majority":
            majority = np.sign(np.sum(signals, axis=0))
            return {"signal": majority}
        elif self.ensemble_method == "stacking" and self.meta_model:
            meta_input = signals.T  # shape: (n_samples, n_strategies)
            meta_pred = self.meta_model(meta_input)
            return {"signal": np.sign(meta_pred)}
        else:
            raise ValueError(f"Unknown ensemble_method: {self.ensemble_method}")

    def _update_weights(self):
        # Example: update weights based on rolling Sharpe ratio or profit
        for i, strat in enumerate(self.strategies):
            perf = self._get_recent_performance(strat)
            self.performance_log[i].append(perf)
        if len(self.performance_log[0]) >= self.window:
            scores = [np.mean(log[-self.window:]) for log in self.performance_log]
            total = sum(scores) or 1.0
            self.weights = [s / total for s in scores]
            logger.info(f"Dynamic weights updated: {self.weights}")

    def _get_recent_performance(self, strat):
        # Placeholder: use last trade PnL or Sharpe
        try:
            if hasattr(strat, "trades") and strat.trades:
                return np.mean([t.pnl for t in strat.trades[-self.window:]])
            return 0.0
        except Exception:
            return 0.0

    def calculate_position_size(self, signal, current_price, portfolio_value):
        sizes = [
            s.calculate_position_size(signal, current_price, portfolio_value)
            for s in self.strategies
        ]
        return sum(sizes) / len(sizes)

    def auto_select_ensemble_method(self, data):
        # Test all methods and pick the best by backtest result
        methods = ["weighted", "majority"]
        best_method = None
        best_score = float("-inf")
        for method in methods:
            self.ensemble_method = method
            signals = self.generate_signals(data)["signal"]
            score = np.sum(signals)  # Placeholder: use real backtest
            if score > best_score:
                best_score = score
                best_method = method
        self.ensemble_method = best_method
        logger.info(f"Auto-selected ensemble method: {best_method}")
        return best_method

    def stream_signals(self, data, callback: Callable):
        # Hook for real-time/live trading integration
        signals = self.generate_signals(data)["signal"]
        for sig in signals:
            callback(sig)


# CI/CD integration for automated strategy ensemble tests
def run_ci_cd_composite_strategy_tests() -> None:
    """Run composite strategy tests in CI/CD pipelines."""
    if not os.getenv("CI"):
        logger.debug("CI environment not detected; skipping composite tests")
        return

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_composite_strategy.py",
        "--maxfail=1",
        "--disable-warnings",
    ]
    logger.info("Running CI/CD composite strategy tests: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    logger.info(proc.stdout)
    if proc.returncode != 0:
        logger.error(proc.stderr)
        raise RuntimeError(
            f"CI/CD composite strategy tests failed with exit code {proc.returncode}"
        )


strategy_api = FastAPI(title="ZoL0 Composite Strategy API", version="2.0")
strategy_api.add_middleware(PrometheusMiddleware)
strategy_api.add_route("/metrics", handle_metrics)


class SignalQuery(BaseModel):
    data: dict
    weights: list[float] = None
    ensemble_method: str = "weighted"
    window: int = 50


class UpdateWeightsQuery(BaseModel):
    performances: list[float]
    window: int = 50


class AutoSelectQuery(BaseModel):
    data: dict


@strategy_api.get("/")
async def root():
    return {
        "status": "ok",
        "service": "ZoL0 Composite Strategy API",
        "version": "2.0",
    }


@strategy_api.get("/api/health")
async def api_health():
    return {
        "status": "ok",
        "service": "ZoL0 Composite Strategy API",
        "version": "2.0",
    }


@strategy_api.post(
    "/api/generate_signals", dependencies=[Depends(get_api_key)]
)
async def api_generate_signals(
    req: SignalQuery, role: str = Depends(get_api_key)
):
    # For demo, use two default strategies
    m1 = MomentumStrategy(fast_period=10, slow_period=30, risk_per_trade=0.02)
    m2 = MeanReversionStrategy(period=20, std_dev=2.0, risk_per_trade=0.01)
    composite = CompositeStrategy(
        [m1, m2],
        weights=req.weights,
        ensemble_method=req.ensemble_method,
        window=req.window,
    )
    signals = composite.generate_signals(req.data)
    return {"signals": signals["signal"].tolist()}


@strategy_api.post(
    "/api/update_weights", dependencies=[Depends(get_api_key)]
)
async def api_update_weights(
    req: UpdateWeightsQuery, role: str = Depends(get_api_key)
):
    # Simulate updating weights
    scores = req.performances[-req.window:]
    total = sum(scores) or 1.0
    weights = [s / total for s in scores]
    return {"weights": weights}


@strategy_api.post(
    "/api/auto_select", dependencies=[Depends(get_api_key)]
)
async def api_auto_select(
    req: AutoSelectQuery, role: str = Depends(get_api_key)
):
    m1 = MomentumStrategy(fast_period=10, slow_period=30, risk_per_trade=0.02)
    m2 = MeanReversionStrategy(period=20, std_dev=2.0, risk_per_trade=0.01)
    composite = CompositeStrategy([m1, m2])
    method = composite.auto_select_ensemble_method(req.data)
    return {"best_method": method}


@strategy_api.get(
    "/api/analytics", dependencies=[Depends(get_api_key)]
)
async def api_analytics(role: str = Depends(get_api_key)):
    # Placeholder for analytics
    return {"status": "analytics stub"}


@strategy_api.get(
    "/api/export/prometheus", dependencies=[Depends(get_api_key)]
)
async def api_export_prometheus(role: str = Depends(get_api_key)):
    # Placeholder for Prometheus export
    return PlainTextResponse(
        "# HELP composite_strategy_requests Number of requests\ncomposite_strategy_requests 1",
        media_type="text/plain",
    )


@strategy_api.get(
    "/api/report", dependencies=[Depends(get_api_key)]
)
async def api_report(role: str = Depends(get_api_key)):
    # Placeholder for PDF/CSV/email integration
    return {"status": "report generated (stub)"}


@strategy_api.get(
    "/api/recommendations", dependencies=[Depends(get_api_key)]
)
async def api_recommendations(role: str = Depends(get_api_key)):
    # Placeholder for recommendations
    return {"recommendations": ["Optimize ensemble weights for best performance."]}


@strategy_api.get(
    "/api/premium/score", dependencies=[Depends(get_api_key)]
)
async def api_premium_score(role: str = Depends(get_api_key)):
    # Placeholder for premium scoring
    return {"score": 100}


@strategy_api.get(
    "/api/saas/tenant/{tenant_id}/report", dependencies=[Depends(get_api_key)]
)
async def api_saas_tenant_report(tenant_id: str, role: str = Depends(get_api_key)):
    # Multi-tenant stub
    return {"tenant_id": tenant_id, "report": "stub"}


@strategy_api.get(
    "/api/partner/webhook", dependencies=[Depends(get_api_key)]
)
async def api_partner_webhook(payload: dict, role: str = Depends(get_api_key)):
    return {"status": "received", "payload": payload}


@strategy_api.get("/api/test/edge-case")
async def api_edge_case():
    try:
        raise RuntimeError("Simulated composite strategy edge-case error")
    except Exception as e:
        return {"edge_case": str(e)}


@strategy_api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": str(exc)})


# --- CI/CD test suite ---
import unittest


class TestCompositeStrategyAPI(unittest.TestCase):
    def test_generate_signals(self):
        m1 = MomentumStrategy(fast_period=10, slow_period=30, risk_per_trade=0.02)
        m2 = MeanReversionStrategy(period=20, std_dev=2.0, risk_per_trade=0.01)
        composite = CompositeStrategy([m1, m2])
        data = generate_demo_data("TEST")
        signals = composite.generate_signals(data)
        assert "signal" in signals


if __name__ == "__main__":
    import sys

    if "test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        uvicorn.run("composite_strategy:strategy_api", host="0.0.0.0", port=8511, reload=True)
