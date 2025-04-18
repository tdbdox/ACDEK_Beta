"""
Interface for agent plugins in ACDEK v3.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class AgentInterface(ABC):
    """
    Base interface for agent plugins.
    """
    @abstractmethod
    def run_agent(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent logic on input_data.

        Args:
            input_data: Dictionary of input parameters.
        Returns:
            Dictionary of agent outputs.
        """
        pass