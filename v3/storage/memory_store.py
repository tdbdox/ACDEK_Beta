"""
In-memory storage plugin for ACDEK v3.
"""
import logging

from registry.plugin_registry import register_plugin
from registry.registry_types import PluginType
from interfaces.storage_interface import StorageInterface

logger = logging.getLogger(__name__)

@register_plugin(
    plugin_type=PluginType.STORAGE,
    name="in_memory_storage",
    version="1.0.0",
    description="In-memory key-value storage for testing",
    interfaces=["StorageInterface"],
    is_default=True
)
class InMemoryStorage(StorageInterface):
    """
    A simple in-memory storage implementation.
    """
    def __init__(self):
        self._store = {}
        logger.info("InMemoryStorage initialized")

    def store_data(self, key: str, value: Any) -> None:
        """
        Store value under key in memory.
        """
        self._store[key] = value
        logger.debug(f"Stored data under key '{key}'")