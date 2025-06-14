"""
FeatureEngineer - Feature engineering utilities for trading AI
Supports feature selection, scaling, transformation, logging, tracing, and robust error handling.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import structlog
import sentry_sdk
from opentelemetry import trace
from typing import Any, List

logger = structlog.get_logger("FeatureEngineer")
tracer = trace.get_tracer("ai.models.FeatureEngineer")

class FeatureEngineer:
    """
    Feature engineering utilities for trading AI.
    Supports feature selection, scaling, transformation, logging, tracing, and robust error handling.
    """
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        logger.info("FeatureEngineer_initialized")

    def fit_transform(self, X: Any) -> Any:
        """
        Fit scaler and transform data.
        Args:
            X (Any): Input data.
        Returns:
            Any: Scaled data.
        """
        with tracer.start_as_current_span("FeatureEngineer.fit_transform"):
            try:
                return self.scaler.fit_transform(X)
            except Exception as e:
                logger.error("fit_transform_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                raise

    def transform(self, X: Any) -> Any:
        """
        Transform data using fitted scaler.
        Args:
            X (Any): Input data.
        Returns:
            Any: Scaled data.
        """
        with tracer.start_as_current_span("FeatureEngineer.transform"):
            try:
                return self.scaler.transform(X)
            except Exception as e:
                logger.error("transform_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                raise

    def select_features(self, X: pd.DataFrame, y: Any, threshold: float = 0.05) -> List[str]:
        """
        Select features by correlation threshold.
        Args:
            X (pd.DataFrame): Feature matrix.
            y (Any): Target vector.
            threshold (float): Correlation threshold.
        Returns:
            List[str]: Selected feature names.
        """
        with tracer.start_as_current_span("FeatureEngineer.select_features"):
            try:
                corr = pd.Series([abs(np.corrcoef(X.iloc[:, i], y)[0, 1]) for i in range(X.shape[1])], index=X.columns)
                selected = list(corr[corr > threshold].index)
                logger.info("features_selected", count=len(selected))
                return selected
            except Exception as e:
                logger.error("select_features_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                return []
