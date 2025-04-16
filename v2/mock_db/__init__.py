# Import common modules for easier access
from mock_db.agent_traits import agent_traits_dataset
from mock_db.environment_factors import environment_profiles
from mock_db.performance_metrics import performance_metrics, sales_data, trend_indicators
from mock_db.query_requests import query_requests, multi_intent_query_request, fused_query_request, memory_conditioned_query
from mock_db.modulation_profiles import modulation_profiles

# Memory persistence will be initialized when imported
try:
    from mock_db.memory_store import get_memory_trace, save_memory_trace, update_session_history, update_learned_traits
except ImportError:
    print("Warning: Memory store module not available")