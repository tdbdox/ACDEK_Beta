"""
Interface for middleware plugins in ACDEK v3.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class MiddlewareInterface(ABC):
    """
    Base interface for middleware plugins.
    """
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and possibly modify data passing through middleware.

        Args:
            data: Arbitrary payload dictionary.
        Returns:
            Transformed payload dictionary.
        """
        pass