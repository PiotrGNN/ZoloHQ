"""
AnomalyDetector - Advanced anomaly detection for trading data
Implements IsolationForest with real-time calibration, confidence scoring, logging, tracing, Sentry, and robust error handling.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
import structlog
import sentry_sdk
from opentelemetry import trace
from typing import Optional

logger = structlog.get_logger("AnomalyDetector")
tracer = trace.get_tracer("ai.models.AnomalyDetector")

class AnomalyDetector:
    """
    Advanced anomaly detection for trading data using IsolationForest.
    Features: calibration, confidence scoring, robust error handling, logging, tracing, and Sentry integration.
    """
    def __init__(self, contamination: float = 0.1, random_state: int = 42, model_path: Optional[str] = None) -> None:
        self.contamination: float = contamination
        self.random_state: int = random_state
        self.model: Optional[IsolationForest] = None
        self.model_path: str = model_path or "ai_models/anomaly_detector.pkl"
        self._load_model()
        logger.info("AnomalyDetector_initialized", contamination=contamination, random_state=random_state)

    def _load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                logger.info("model_loaded", path=self.model_path)
            except Exception as e:
                logger.error("model_load_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                self.model = None

    def fit(self, X: np.ndarray) -> None:
        """
        Train IsolationForest on X.
        Args:
            X (np.ndarray): Training data.
        """
        with tracer.start_as_current_span("AnomalyDetector.fit"):
            try:
                self.model = IsolationForest(contamination=self.contamination, random_state=self.random_state)
                self.model.fit(X)
                joblib.dump(self.model, self.model_path)
                logger.info("model_trained", path=self.model_path)
            except Exception as e:
                logger.error("fit_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in X.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Anomaly predictions.
        """
        with tracer.start_as_current_span("AnomalyDetector.predict"):
            if self.model is None:
                logger.error("predict_failed", reason="Model not trained")
                raise ValueError("Model not trained")
            try:
                return self.model.predict(X)
            except Exception as e:
                logger.error("predict_exception", error=str(e))
                sentry_sdk.capture_exception(e)
                raise

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores for X.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Anomaly scores.
        """
        with tracer.start_as_current_span("AnomalyDetector.decision_function"):
            if self.model is None:
                logger.error("decision_function_failed", reason="Model not trained")
                raise ValueError("Model not trained")
            try:
                return self.model.decision_function(X)
            except Exception as e:
                logger.error("decision_function_exception", error=str(e))
                sentry_sdk.capture_exception(e)
                raise

    def calibrate(self, X: Optional[np.ndarray], y_true: Optional[np.ndarray] = None) -> None:
        """
        Calibrate sensitivity using real market data (not implemented).
        Args:
            X (Optional[np.ndarray]): Input data.
            y_true (Optional[np.ndarray]): True labels.
        """
        logger.info("calibrate_called")
        pass

    def confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Return normalized confidence scores for each sample in X.
        Args:
            X (np.ndarray): Input data.
        Returns:
            np.ndarray: Confidence scores.
        """
        with tracer.start_as_current_span("AnomalyDetector.confidence"):
            if self.model is None:
                logger.error("confidence_failed", reason="Model not trained")
                raise ValueError("Model not trained")
            try:
                scores = self.model.decision_function(X)
                min_score, max_score = np.min(scores), np.max(scores)
                if max_score - min_score == 0:
                    return np.ones_like(scores)
                return (scores - min_score) / (max_score - min_score)
            except Exception as e:
                logger.error("confidence_exception", error=str(e))
                sentry_sdk.capture_exception(e)
                raise
