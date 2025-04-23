"""
Enhanced Plugin Registry with Improved Error Handling and Fallbacks.

This module extends the base plugin registry with better error handling,
automatic fallbacks, and dependency management to improve reliability.
"""
import logging
import importlib
import inspect
from typing import Dict, List, Any, Optional, Set, Type, Callable, Union, Tuple
from dataclasses import dataclass, field

from registry.registry_types import PluginType, PluginMetadata
from registry.plugin_registry import PluginRegistry
from observability.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)
metrics = MetricsCollector()

@dataclass
class FallbackStrategy:
    """Strategy for handling plugin loading failures."""
    use_default: bool = True  # Use default plugin if specified one fails
    allow_missing: bool = False  # Continue even if plugin is missing (optional)
    create_stub: bool = False  # Create a stub implementation if plugin is missing
    log_level: str = "WARNING"  # Log level for reporting issues

@dataclass
class PluginLoadResult:
    """Result of a plugin load operation."""
    success: bool
    instance: Optional[Any] = None
    error_message: Optional[str] = None
    used_fallback: bool = False
    fallback_plugin: Optional[str] = None
    
    def __bool__(self):
        """Allow using the result in boolean contexts to check success."""
        return self.success

class EnhancedPluginRegistry:
    """
    Enhanced plugin registry with improved error handling and fallbacks.
    
    This registry wraps the base PluginRegistry and adds:
    - Better error handling for missing plugins
    - Automatic fallback to default plugins
    - Clearer error messages
    - Dependency checking
    - Metrics tracking
    """
    
    def __init__(self, base_registry: Optional[PluginRegistry] = None):
        """
        Initialize the enhanced registry.
        
        Args:
            base_registry: Optional base registry to wrap. If not provided,
                           a new registry will be created.
        """
        self.registry = base_registry or PluginRegistry()
        self.stub_factories: Dict[PluginType, Callable] = {}
        
        # Cache of validated dependencies
        self._validated_deps: Dict[str, bool] = {}
    
    def register_stub_factory(self, plugin_type: PluginType, factory: Callable):
        """
        Register a factory function that creates stub implementations.
        
        Args:
            plugin_type: Type of plugin this factory creates stubs for
            factory: Function that creates stub implementations
        """
        self.stub_factories[plugin_type] = factory
    
    def load_plugin(
        self, 
        plugin_type: PluginType, 
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        fallback_strategy: Optional[FallbackStrategy] = None
    ) -> PluginLoadResult:
        """
        Load a plugin with enhanced error handling and fallback support.
        
        Args:
            plugin_type: Type of plugin to load
            name: Name of plugin, or None for default
            config: Configuration to pass to plugin constructor
            fallback_strategy: Strategy for handling errors
            
        Returns:
            PluginLoadResult with success status and instance if successful
        """
        strategy = fallback_strategy or FallbackStrategy()
        log_fn = getattr(logger, strategy.log_level.lower())
        
        # Track metrics if available
        if hasattr(metrics, 'increment_counter'):
            metrics.increment_counter(f"plugin_load_attempt.{plugin_type.value}")
        
        # Try to load the specified plugin
        instance = self.registry.create_instance(plugin_type, name, config)
        
        if instance is not None:
            if hasattr(metrics, 'increment_counter'):
                metrics.increment_counter(f"plugin_load_success.{plugin_type.value}")
            return PluginLoadResult(success=True, instance=instance)
        
        # Initial load failed - apply fallback strategy
        error_msg = f"Plugin {name} of type {plugin_type} not found or failed to load"
        log_fn(error_msg)
        if hasattr(metrics, 'increment_counter'):
            metrics.increment_counter(f"plugin_load_failure.{plugin_type.value}")
        
        # Try fallback to default if allowed
        if strategy.use_default and name is not None:
            default_plugin = self.registry.get_default_plugin(plugin_type)
            if default_plugin:
                log_fn(f"Falling back to default plugin {default_plugin.name}")
                fallback_instance = self.registry.create_instance(
                    plugin_type, default_plugin.name, config
                )
                if fallback_instance:
                    if hasattr(metrics, 'increment_counter'):
                        metrics.increment_counter(f"plugin_fallback_success.{plugin_type.value}")
                    return PluginLoadResult(
                        success=True, 
                        instance=fallback_instance,
                        used_fallback=True,
                        fallback_plugin=default_plugin.name
                    )
        
        # Try creating a stub if allowed
        if strategy.create_stub and plugin_type in self.stub_factories:
            log_fn(f"Creating stub implementation for {plugin_type}")
            stub_instance = self.stub_factories[plugin_type](config)
            if hasattr(metrics, 'increment_counter'):
                metrics.increment_counter(f"plugin_stub_created.{plugin_type.value}")
            return PluginLoadResult(
                success=True,
                instance=stub_instance,
                used_fallback=True,
                error_message=f"Using stub implementation for {plugin_type}"
            )
        
        # All fallback strategies failed
        if strategy.allow_missing:
            log_fn(f"Continuing without plugin {name} of type {plugin_type}")
            return PluginLoadResult(
                success=False,
                error_message=error_msg
            )
        
        # Critical failure - cannot continue
        logger.error(f"Critical plugin {name} of type {plugin_type} missing and no viable fallbacks")
        return PluginLoadResult(
            success=False,
            error_message=f"Critical plugin {name} missing and no viable fallbacks"
        )
    
    def check_dependencies(
        self, 
        plugin_name: str,
        recursive: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Check if all dependencies for a plugin are available.
        
        Args:
            plugin_name: Name of the plugin to check
            recursive: Whether to check dependencies of dependencies
            
        Returns:
            Tuple of (dependencies_satisfied, missing_dependencies)
        """
        # Check cache first
        if plugin_name in self._validated_deps and recursive:
            return (self._validated_deps[plugin_name], [])
        
        # Get direct dependencies
        dependencies = self.registry._dependencies.get(plugin_name, set())
        if not dependencies:
            self._validated_deps[plugin_name] = True
            return (True, [])
        
        # Check each dependency
        missing = []
        all_available = True
        
        for dep_name in dependencies:
            # Find this dependency in any plugin type
            found = False
            for plugin_type in PluginType:
                if dep_name in self.registry._plugins[plugin_type]:
                    found = True
                    
                    # Recursively check dependencies if needed
                    if recursive:
                        deps_satisfied, missing_deps = self.check_dependencies(dep_name, True)
                        if not deps_satisfied:
                            all_available = False
                            missing.extend(missing_deps)
                    
                    break
            
            if not found:
                all_available = False
                missing.append(dep_name)
        
        # Cache the result for this plugin
        self._validated_deps[plugin_name] = all_available
        
        return (all_available, missing)
    
    def list_plugins_with_status(self, plugin_type: Optional[PluginType] = None) -> List[Dict[str, Any]]:
        """
        List plugins with their availability status.
        
        Args:
            plugin_type: Optional type to filter by
            
        Returns:
            List of dicts with plugin info and status
        """
        plugins = self.registry.list_plugins(plugin_type)
        result = []
        
        for plugin in plugins:
            # Check dependencies
            deps_satisfied, missing_deps = self.check_dependencies(plugin.name)
            
            # Try instantiating to verify it works
            can_instantiate = False
            error_message = None
            try:
                instance = self.registry.create_instance(plugin.plugin_type, plugin.name)
                can_instantiate = instance is not None
            except Exception as e:
                error_message = str(e)
            
            result.append({
                "name": plugin.name,
                "type": plugin.plugin_type.value,
                "version": plugin.version,
                "dependencies_satisfied": deps_satisfied,
                "missing_dependencies": missing_deps,
                "can_instantiate": can_instantiate,
                "error_message": error_message,
                "is_default": self.registry._default_plugins.get(plugin.plugin_type) == plugin.name
            })
        
        return result

# Create a global instance
enhanced_registry = EnhancedPluginRegistry()

# Utility functions for easier access
def load_plugin(
    plugin_type: PluginType, 
    name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    fallback_strategy: Optional[FallbackStrategy] = None
) -> PluginLoadResult:
    """
    Load a plugin with enhanced error handling.
    
    Args:
        plugin_type: Type of plugin to load
        name: Name of plugin, or None for default
        config: Configuration to pass to plugin constructor
        fallback_strategy: Strategy for handling errors
        
    Returns:
        PluginLoadResult with success status and instance if successful
    """
    return enhanced_registry.load_plugin(plugin_type, name, config, fallback_strategy)

def check_plugin_dependencies(plugin_name: str) -> Tuple[bool, List[str]]:
    """
    Check if all dependencies for a plugin are available.
    
    Args:
        plugin_name: Name of the plugin to check
        
    Returns:
        Tuple of (dependencies_satisfied, missing_dependencies)
    """
    return enhanced_registry.check_dependencies(plugin_name)

def get_plugin_status(plugin_type: Optional[PluginType] = None) -> List[Dict[str, Any]]:
    """
    Get status information about available plugins.
    
    Args:
        plugin_type: Optional type to filter by
        
    Returns:
        List of dicts with plugin info and status
    """
    return enhanced_registry.list_plugins_with_status(plugin_type)