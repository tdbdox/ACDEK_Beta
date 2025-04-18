"""
Unit tests for plugin interface compliance and basic plugin behavior in ACDEK v3.
"""
from agents.basic_agent import BasicAgent
from llm_adapters.openai_adapter import OpenAIAdapter
from middleware.logger_middleware import LoggerMiddleware
from storage.memory_store import InMemoryStorage

def test_basic_agent_plugin():
    agent = BasicAgent()
    assert hasattr(agent, 'run_agent')
    result = agent.run_agent({'x': 42})
    assert isinstance(result, dict)
    assert result.get('agent') == 'basic_agent'
    assert result.get('received') == {'x': 42}

def test_llm_adapter_plugin():
    adapter = OpenAIAdapter()
    response = adapter.call_llm('hello')
    assert isinstance(response, str)
    assert 'Mock response to: hello' in response

def test_middleware_plugin():
    mw = LoggerMiddleware()
    data = {'key': 'value'}
    out = mw.process(data)
    assert out == data

def test_storage_plugin():
    store = InMemoryStorage()
    # Test storing and internal state
    store.store_data('k', 123)
    assert hasattr(store, '_store')
    assert store._store.get('k') == 123