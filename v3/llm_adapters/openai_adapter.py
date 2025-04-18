"""
Mock OpenAI adapter plugin for ACDEK v3.
"""
import logging

from registry.plugin_registry import register_plugin
from registry.registry_types import PluginType
from interfaces.llm_adapter_interface import LLMAdapterInterface

logger = logging.getLogger(__name__)

@register_plugin(
    plugin_type=PluginType.LLM_ADAPTER,
    name="openai_adapter",
    version="1.0.0",
    description="Mock OpenAI adapter returning a static response",
    interfaces=["LLMAdapterInterface"],
    is_default=True
)
class OpenAIAdapter(LLMAdapterInterface):
    """
    A mock implementation of an OpenAI LLM adapter.
    """
    def call_llm(self, prompt: str) -> str:
        """
        Returns a mock response for the given prompt.
        """
        logger.info(f"OpenAIAdapter.call_llm invoked with prompt: {prompt}")
        # Return a dummy completion
        return f"Mock response to: {prompt}"