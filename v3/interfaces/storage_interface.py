"""
Interface for storage plugins in ACDEK v3.
"""
from abc import ABC, abstractmethod
from typing import Any


class StorageInterface(ABC):
    """
    Base interface for storage plugins.
    """
    @abstractmethod
    def store_data(self, key: str, value: Any) -> None:
        """
        Store a value by key in the storage backend.

        Args:
            key: Identifier for the data.
            value: Data to store.
        """
        pass