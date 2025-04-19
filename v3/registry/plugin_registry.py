import logging
from typing import Dict, List, Type, TypeVar, Any, Optional, Callable, Set
import inspect
import functools

from .registry_types import PluginType, PluginMetadata

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Generic type for plugin instance


class PluginRegistry:
    """
    Central registry for ACDEK plugins.
    
    This registry manages plugin registration, discovery, and instantiation,
    providing a unified interface for accessing all available plugins.
    """
    
    def __init__(self):
        # Dictionary mapping plugin types to available plugins
        self._plugins: Dict[PluginType, Dict[str, PluginMetadata]] = {
            plugin_type: {} for plugin_type in PluginType
        }
        
        # Dictionary for default plugins by type
        self._default_plugins: Dict[PluginType, str] = {}
        
        # Set of registered interfaces
        self._interfaces: Set[str] = set()
        
        # Plugin dependency graph
        self._dependencies: Dict[str, List[str]] = {}
        
        logger.info("Plugin registry initialized")
    
    def register(
        self,
        name: str,
        plugin_type: PluginType,
        version: str,
        description: str,
        plugin_class: Type[T],
        interfaces: List[str],
        config_schema: Dict[str, Any] = None,
        dependencies: List[str] = None,
        is_default: bool = False
    ) -> None:
        """
        Register a plugin with the registry.
        
        Args:
            name: Unique name for the plugin
            plugin_type: Type of plugin from PluginType enum
            version: Version string (semver recommended)
            description: Short description of plugin functionality
            plugin_class: Class implementing the plugin
            interfaces: List of interface names this plugin implements
            config_schema: Optional JSON schema for plugin configuration
            dependencies: Optional list of plugin names this plugin depends on
            is_default: Whether this should be the default for its type
        """
        if name in self._plugins[plugin_type]:
            logger.warning(f"Plugin {name} already registered, updating registration")
        
        # Create metadata
        metadata = PluginMetadata(
            name=name,
            plugin_type=plugin_type,
            version=version,
            description=description,
            plugin_class=plugin_class,
            interfaces=interfaces,
            config_schema=config_schema or {},
            dependencies=dependencies or []
        )
        
        # Register plugin
        self._plugins[plugin_type][name] = metadata
        
        # Register interfaces
        self._interfaces.update(interfaces)
        
        # Register dependencies
        if dependencies:
            self._dependencies[name] = dependencies
        
        # Set as default if specified
        if is_default:
            self._default_plugins[plugin_type] = name
        
        logger.info(f"Registered plugin: {metadata}")
    
    def get_plugin(self, plugin_type: PluginType, name: Optional[str] = None) -> Optional[PluginMetadata]:
        """
        Get plugin metadata by type and name.
        
        Args:
            plugin_type: Type of plugin to retrieve
            name: Name of plugin to retrieve, or None to get the default
            
        Returns:
            Plugin metadata if found, None otherwise
        """
        if name is None:
            # Get default plugin
            if plugin_type not in self._default_plugins:
                logger.warning(f"No default plugin set for type {plugin_type}")
                return None
            name = self._default_plugins[plugin_type]
        
        if name not in self._plugins[plugin_type]:
            logger.warning(f"Plugin {name} of type {plugin_type} not found")
            return None
        
        return self._plugins[plugin_type][name]
    
    def create_instance(self, plugin_type: PluginType, name: Optional[str] = None, config: Dict[str, Any] = None) -> Optional[Any]:
        """
        Create an instance of a plugin.
        
        Args:
            plugin_type: Type of plugin to instantiate
            name: Name of plugin to instantiate, or None to use the default
            config: Configuration to pass to the plugin constructor
            
        Returns:
            Plugin instance if successful, None otherwise
        """
        metadata = self.get_plugin(plugin_type, name)
        if not metadata:
            return None
        
        try:
            # Create instance
            instance = metadata.plugin_class(**(config or {}))
            logger.info(f"Created instance of plugin {metadata.name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create instance of plugin {metadata.name}: {e}")
            return None
    
    def list_plugins(self, plugin_type: Optional[PluginType] = None) -> List[PluginMetadata]:
        """
        List available plugins, optionally filtered by type.
        
        Args:
            plugin_type: Optional type to filter by
            
        Returns:
            List of plugin metadata
        """
        if plugin_type:
            return list(self._plugins[plugin_type].values())
        
        # Return all plugins
        result = []
        for plugin_dict in self._plugins.values():
            result.extend(plugin_dict.values())
        return result
    
    def get_default_plugin(self, plugin_type: PluginType) -> Optional[PluginMetadata]:
        """
        Get the default plugin for a given type.
        
        Args:
            plugin_type: Type of plugin
            
        Returns:
            Plugin metadata if a default exists, None otherwise
        """
        if plugin_type not in self._default_plugins:
            return None
        
        name = self._default_plugins[plugin_type]
        return self._plugins[plugin_type].get(name)
    
    def set_default_plugin(self, plugin_type: PluginType, name: str) -> bool:
        """
        Set the default plugin for a given type.
        
        Args:
            plugin_type: Type of plugin
            name: Name of plugin to set as default
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self._plugins[plugin_type]:
            logger.warning(f"Cannot set unknown plugin {name} as default for {plugin_type}")
            return False
        
        self._default_plugins[plugin_type] = name
        logger.info(f"Set {name} as default plugin for {plugin_type}")
        return True


# Global registry instance
registry = PluginRegistry()


def register_plugin(
    plugin_type: PluginType,
    name: str,
    version: str,
    description: str,
    interfaces: List[str],
    config_schema: Dict[str, Any] = None,
    dependencies: List[str] = None,
    is_default: bool = False
) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator for registering plugins.
    
    Example:
        @register_plugin(
            plugin_type=PluginType.VECTOR_ALIGNMENT,
            name="my_vector_aligner",
            version="1.0.0",
            description="Custom vector alignment implementation",
            interfaces=["VectorAlignmentInterface"],
        )
        class MyVectorAligner(VectorAlignmentInterface):
            ...
    
    Args:
        plugin_type: Type of plugin from PluginType enum
        name: Unique name for the plugin
        version: Version string (semver recommended)
        description: Short description of plugin functionality
        interfaces: List of interface names this plugin implements
        config_schema: Optional JSON schema for plugin configuration
        dependencies: Optional list of plugin names this plugin depends on
        is_default: Whether this should be the default for its type
        
    Returns:
        Decorator function that registers the decorated class
    """
    def decorator(cls: Type[T]) -> Type[T]:
        registry.register(
            name=name,
            plugin_type=plugin_type,
            version=version,
            description=description,
            plugin_class=cls,
            interfaces=interfaces,
            config_schema=config_schema,
            dependencies=dependencies,
            is_default=is_default
        )
        return cls
    
    return decorator


def get_plugin(plugin_type: PluginType, name: Optional[str] = None) -> Optional[PluginMetadata]:
    """
    Get plugin metadata by type and name from the global registry.
    
    Args:
        plugin_type: Type of plugin to retrieve
        name: Name of plugin to retrieve, or None to get the default
        
    Returns:
        Plugin metadata if found, None otherwise
    """
    return registry.get_plugin(plugin_type, name)


def create_instance(plugin_type: PluginType, name: Optional[str] = None, config: Dict[str, Any] = None) -> Optional[Any]:
    """
    Create an instance of a plugin from the global registry.
    
    Args:
        plugin_type: Type of plugin to instantiate
        name: Name of plugin to instantiate, or None to use the default
        config: Configuration to pass to the plugin constructor
        
    Returns:
        Plugin instance if successful, None otherwise
    """
    return registry.create_instance(plugin_type, name, config)


def get_plugin_info(plugin_type: PluginType, name: Optional[str] = None) -> Optional[PluginMetadata]:
    """
    Get plugin metadata by type and name from the global registry.
    
    Args:
        plugin_type: Type of plugin to retrieve
        name: Name of plugin to retrieve, or None to get the default
        
    Returns:
        Plugin metadata if found, None otherwise
    """
    return get_plugin(plugin_type, name)