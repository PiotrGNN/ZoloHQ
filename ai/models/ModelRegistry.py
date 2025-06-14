"""
ModelRegistry - Registry for all AI/ML models in the system
Supports registration, lookup, versioning, logging, tracing, and robust error handling.
"""

from typing import Any, Dict, List, Optional
import structlog
import sentry_sdk
from opentelemetry import trace

logger = structlog.get_logger("ModelRegistry")
tracer = trace.get_tracer("ai.models.ModelRegistry")

class ModelRegistry:
    """
    Registry for all AI/ML models in the system.
    Supports registration, lookup, versioning, logging, tracing, and robust error handling.
    """
    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        logger.info("ModelRegistry_initialized")

    def register(self, name: str, model: Any) -> None:
        """
        Register a model in the registry.
        Args:
            name (str): Model name.
            model (Any): Model instance.
        """
        with tracer.start_as_current_span("ModelRegistry.register"):
            try:
                self.models[name] = model
                logger.info("model_registered", name=name)
            except Exception as e:
                logger.error("register_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                raise

    def get(self, name: str) -> Optional[Any]:
        """
        Retrieve a model by name.
        Args:
            name (str): Model name.
        Returns:
            Optional[Any]: Model instance or None.
        """
        with tracer.start_as_current_span("ModelRegistry.get"):
            try:
                model = self.models.get(name)
                if model is not None:
                    logger.info("model_found", name=name)
                else:
                    logger.warning("model_not_found", name=name)
                return model
            except Exception as e:
                logger.error("get_failed", error=str(e))
                sentry_sdk.capture_exception(e)
                return None

    def list(self) -> List[str]:
        """
        List all registered model names.
        Returns:
            List[str]: List of model names.
        """
        return list(self.models.keys())
