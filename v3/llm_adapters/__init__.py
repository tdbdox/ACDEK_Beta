"""
ACDEK v3 LLM Adapters Package

This package provides standardized interfaces for interacting with various
LLM providers, ensuring consistent behavior and seamless fallback capabilities.
"""

# Import and expose key components
from .openai_adapter import OpenAIAdapter

__all__ = [
    'OpenAIAdapter',
]