"""
DataScaler - Data scaling utility for AI/ML models
Supports standard and min-max scaling for arrays, with logging, tracing, and robust error handling.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import structlog
import sentry_sdk
from opentelemetry import trace
from typing import Any

logger = structlog.get_logger("DataScaler")
tracer = trace.get_tracer("ai.models.DataScaler")

class DataScaler:
    """
    Data scaling utility for arrays.
    Supports standard and min-max scaling, logging, tracing, and robust error handling.
    """
    def __init__(self, method: str = "standard") -> None:
        try:
            if method == "minmax":
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()
            logger.info("DataScaler_initialized", method=method)
        except Exception as e:
            logger.error("init_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            raise

    def fit_transform(self, X: Any) -> Any:
        """
        Fit scaler and transform data.
        Args:
            X (Any): Input data.
        Returns:
            Any: Scaled data.
        """
        with tracer.start_as_current_span("DataScaler.fit_transform"):
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
        with tracer.start_as_current_span("DataScaler.transform"):
            try:
                return self.scaler.transform(X)
            except Exception as e:
                logger.error("transform_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                raise

    def inverse_transform(self, X: Any) -> Any:
        """
        Inverse transform scaled data.
        Args:
            X (Any): Scaled data.
        Returns:
            Any: Original data.
        """
        with tracer.start_as_current_span("DataScaler.inverse_transform"):
            try:
                return self.scaler.inverse_transform(X)
            except Exception as e:
                logger.error("inverse_transform_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                raise
