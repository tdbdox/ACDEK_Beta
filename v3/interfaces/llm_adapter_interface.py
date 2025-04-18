"""
Interface for LLM adapter plugins in ACDEK v3.
"""
from abc import ABC, abstractmethod


class LLMAdapterInterface(ABC):
    """
    Base interface for LLM adapter plugins.
    """
    @abstractmethod
    def call_llm(self, prompt: str) -> str:
        """
        Send prompt to LLM and return the response text.

        Args:
            prompt: Text prompt for the LLM.
        Returns:
            LLM response as a string.
        """
        pass