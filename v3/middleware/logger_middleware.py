"""
Logger middleware plugin for ACDEK v3.
"""
import logging

from registry.plugin_registry import register_plugin
from registry.registry_types import PluginType
from interfaces.middleware_interface import MiddlewareInterface

logger = logging.getLogger(__name__)

@register_plugin(
    plugin_type=PluginType.OBSERVABILITY,
    name="logger_middleware",
    version="1.0.0",
    description="Middleware that logs data payloads and passes them through",
    interfaces=["MiddlewareInterface"],
    is_default=True
)
class LoggerMiddleware(MiddlewareInterface):
    """
    A middleware plugin that logs incoming data.
    """
    def process(self, data: dict) -> dict:
        """
        Log the payload and return it unchanged.
        """
        logger.info(f"LoggerMiddleware processing data: {data}")
        return data