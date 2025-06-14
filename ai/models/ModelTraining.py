"""
ModelTraining - Model training orchestration for trading AI
Supports batch training, evaluation, logging, tracing, Sentry, and robust error handling.
"""

from typing import Any, Dict, List
import structlog
import sentry_sdk
from opentelemetry import trace

logger = structlog.get_logger("ModelTraining")
tracer = trace.get_tracer("ai.models.ModelTraining")

class ModelTraining:
    """
    Model training orchestration for trading AI.
    Supports batch training, evaluation, logging, tracing, Sentry, and robust error handling.
    """
    def __init__(self, trainer: Any) -> None:
        self.trainer: Any = trainer
        self.history: List[Any] = []
        logger.info("ModelTraining_initialized")

    def train_batch(self, models: Dict[str, Any], X: Any, y: Any) -> Dict[str, Any]:
        """
        Train a batch of models and log results.
        Args:
            models (Dict[str, Any]): Dictionary of model name to model instance.
            X (Any): Training features.
            y (Any): Training labels.
        Returns:
            Dict[str, Any]: Dictionary of trained models.
        """
        with tracer.start_as_current_span("ModelTraining.train_batch"):
            results: Dict[str, Any] = {}
            for name, model in models.items():
                try:
                    trained = self.trainer.train(model, X, y)
                    results[name] = trained
                    self.history.append((name, "trained"))
                    logger.info("model_trained", name=name)
                except Exception as e:
                    logger.error("train_batch_failed", name=name, error=str(e))
                    sentry_sdk.capture_exception(e)
            return results

    def get_history(self) -> List[Any]:
        """
        Get training history.
        Returns:
            List[Any]: Training history records.
        """
        return self.history
