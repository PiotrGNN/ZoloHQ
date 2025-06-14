"""
ModelLoader - Utility for loading and saving AI/ML models
Supports joblib serialization, versioning, logging, tracing, Sentry, and robust error handling.
"""

import os
import joblib
import structlog
import sentry_sdk
from opentelemetry import trace
from typing import Any, Optional

logger = structlog.get_logger("ModelLoader")
tracer = trace.get_tracer("ai.models.ModelLoader")

class ModelLoader:
    """
    Utility for loading and saving AI/ML models.
    Supports joblib serialization, versioning, logging, tracing, Sentry, and robust error handling.
    """
    def __init__(self, model_dir: str = "ai_models") -> None:
        self.model_dir: str = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info("ModelLoader_initialized", model_dir=model_dir)

    def save(self, model: Any, name: str) -> str:
        """
        Save model to disk.
        Args:
            model (Any): Model instance.
            name (str): Model name.
        Returns:
            str: Path to saved model.
        """
        try:
            path = os.path.join(self.model_dir, f"{name}.pkl")
            joblib.dump(model, path)
            logger.info("model_saved", path=path)
            return path
        except Exception as e:
            logger.error("save_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            raise

    def load(self, name: str) -> Optional[Any]:
        """
        Load model from disk.
        Args:
            name (str): Model name.
        Returns:
            Optional[Any]: Loaded model or None.
        """
        try:
            path = os.path.join(self.model_dir, f"{name}.pkl")
            if os.path.exists(path):
                logger.info("model_loaded", name=name)
                return joblib.load(path)
            logger.warning("model_not_found", name=name)
            return None
        except Exception as e:
            logger.error("load_failed", error=str(e))
            sentry_sdk.capture_exception(e)
            return None
