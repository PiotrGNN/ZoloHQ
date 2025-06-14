"""
FeatureConfig - Configuration for feature engineering and selection
Supports dynamic feature sets, thresholds, logging, tracing, and robust error handling.
"""

import structlog
import sentry_sdk
from opentelemetry import trace
from typing import List, Any, Dict

logger = structlog.get_logger("FeatureConfig")
tracer = trace.get_tracer("ai.models.FeatureConfig")

class FeatureConfig:
    """
    Configuration for feature engineering and selection.
    Supports dynamic feature sets, thresholds, logging, tracing, and robust error handling.
    """
    def __init__(self, features: List[Any] = None, threshold: float = 0.05) -> None:
        self.features: List[Any] = features or []
        self.threshold: float = threshold
        logger.info("FeatureConfig_initialized", features=self.features, threshold=threshold)

    def set_features(self, features: List[Any]) -> None:
        """
        Set the list of features.
        Args:
            features (List[Any]): List of features.
        """
        with tracer.start_as_current_span("FeatureConfig.set_features"):
            try:
                self.features = features
                logger.info("features_set", count=len(features))
            except Exception as e:
                logger.error("set_features_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                raise

    def set_threshold(self, threshold: float) -> None:
        """
        Set the feature selection threshold.
        Args:
            threshold (float): Correlation threshold.
        """
        with tracer.start_as_current_span("FeatureConfig.set_threshold"):
            try:
                self.threshold = threshold
                logger.info("threshold_set", threshold=threshold)
            except Exception as e:
                logger.error("set_threshold_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                raise

    def get_config(self) -> Dict[str, Any]:
        """
        Get the current feature configuration.
        Returns:
            Dict[str, Any]: Feature configuration.
        """
        return {"features": self.features, "threshold": self.threshold}
