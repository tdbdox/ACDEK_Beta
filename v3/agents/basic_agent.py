"""
Basic agent plugin implementation for ACDEK v3.
"""
import logging

from registry.plugin_registry import register_plugin
from registry.registry_types import PluginType
from interfaces.agent_interface import AgentInterface

logger = logging.getLogger(__name__)

@register_plugin(
    plugin_type=PluginType.DECISION_GATE,
    name="basic_agent",
    version="1.0.0",
    description="Basic agent plugin that returns a dummy response",
    interfaces=["AgentInterface"],
    is_default=True
)
class BasicAgent(AgentInterface):
    """
    A simple agent plugin that echoes the input data.
    """
    def run_agent(self, input_data: dict) -> dict:
        """
        Returns a dummy response containing the input.
        """
        logger.info("BasicAgent.run_agent invoked")
        return {"agent": "basic_agent", "received": input_data}