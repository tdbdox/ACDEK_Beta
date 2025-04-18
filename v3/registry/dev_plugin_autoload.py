"""
Development autoload for custom plugins.
Import plugin modules so they register at startup.
"""
# Agents
import agents.basic_agent  # noqa: F401
# LLM adapters
import llm_adapters.openai_adapter  # noqa: F401
# Storage
import storage.memory_store  # noqa: F401
# Middleware
import middleware.logger_middleware  # noqa: F401