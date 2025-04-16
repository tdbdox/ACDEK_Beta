from enum import Enum, auto
from typing import Dict, Any, Type, TypeVar, Generic, Callable, List

T = TypeVar('T')  # Generic type for plugin instance


class PluginType(str, Enum):
    """Enumeration of plugin types in the ACDEK system."""
    
    # Core pipeline components
    VECTOR_ALIGNMENT = "vector_alignment"          # Vector embedding and similarity components
    ENRICHMENT = "enrichment"                      # Data enrichment components  
    CONSTRAINT_VERIFIER = "constraint_verifier"    # Constraint validation components
    DECISION_GATE = "decision_gate"                # Decision making components
    MODULATION = "modulation"                      # Response modulation components
    
    # Infrastructure components
    STORAGE = "storage"                            # Data storage components
    CACHE = "cache"                                # Caching components
    OBSERVABILITY = "observability"                # Logging and monitoring components
    SECURITY = "security"                          # Security and access control components
    
    # Integration components
    LLM_ADAPTER = "llm_adapter"                    # Adapters for LLM providers
    CONNECTOR = "connector"                        # Connectors for external systems
    SERIALIZER = "serializer"                      # Data serialization components


class PluginMetadata:
    """Metadata for a registered plugin."""
    
    def __init__(
        self,
        name: str,
        plugin_type: PluginType,
        version: str,
        description: str,
        plugin_class: Type[T],
        interfaces: List[str],
        config_schema: Dict[str, Any] = None,
        dependencies: List[str] = None,
    ):
        self.name = name
        self.plugin_type = plugin_type
        self.version = version
        self.description = description
        self.plugin_class = plugin_class
        self.interfaces = interfaces
        self.config_schema = config_schema or {}
        self.dependencies = dependencies or []
        
    def __repr__(self) -> str:
        return f"<Plugin: {self.name} ({self.plugin_type.value}) v{self.version}>"

