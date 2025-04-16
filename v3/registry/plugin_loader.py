import logging
import importlib.util
import inspect
import os
import sys
from typing import Dict, List, Any, Type, Optional, Set, Callable

from .registry_types import PluginType, PluginMetadata
from .plugin_registry import registry
from .interface_validator import is_valid_plugin

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Loads plugins from specified directories or modules.
    
    This class scans directories for plugin implementations and dynamically
    loads them, validating their interface compliance and registering them
    with the plugin registry.
    """
    
    def __init__(self):
        # Set of loaded plugin modules to prevent duplicates
        self._loaded_modules: Set[str] = set()
        
        # Directory paths to search for plugins
        self._plugin_dirs: List[str] = []
    
    def add_plugin_directory(self, directory: str) -> None:
        """
        Add a directory to search for plugins.
        
        Args:
            directory: Absolute path to directory containing plugin modules
        """
        if not os.path.isdir(directory):
            logger.warning(f"Plugin directory does not exist: {directory}")
            return
        
        if directory not in self._plugin_dirs:
            self._plugin_dirs.append(directory)
            logger.info(f"Added plugin directory: {directory}")
    
    def scan_plugins(self) -> int:
        """
        Scan all registered directories for plugins and load them.
        
        Returns:
            Number of plugins loaded
        """
        total_loaded = 0
        
        for directory in self._plugin_dirs:
            loaded = self._scan_directory(directory)
            total_loaded += loaded
        
        logger.info(f"Total plugins loaded: {total_loaded}")
        return total_loaded
    
    def _scan_directory(self, directory: str) -> int:
        """
        Scan a directory for Python modules containing plugins.
        
        Args:
            directory: Directory path to scan
            
        Returns:
            Number of plugins loaded from this directory
        """
        loaded_count = 0
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    module_path = os.path.join(root, file)
                    relative_path = os.path.relpath(module_path, directory)
                    module_name = os.path.splitext(relative_path)[0].replace(os.path.sep, '.')
                    
                    # Load plugins from this module
                    if self._load_module(module_name, module_path):
                        loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} plugins from directory: {directory}")
        return loaded_count
    
    def load_plugin_module(self, module_name: str) -> bool:
        """
        Load plugins from a specific module by name.
        
        Args:
            module_name: Fully qualified module name (e.g., 'acdek.plugins.my_plugin')
            
        Returns:
            True if successful, False otherwise
        """
        if module_name in self._loaded_modules:
            logger.info(f"Module already loaded: {module_name}")
            return True
        
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Scan module for plugins
            loaded = self._scan_module_for_plugins(module)
            
            if loaded:
                self._loaded_modules.add(module_name)
                logger.info(f"Loaded plugins from module: {module_name}")
                return True
            else:
                logger.warning(f"No plugins found in module: {module_name}")
                return False
                
        except ImportError as e:
            logger.error(f"Error importing module {module_name}: {e}")
            return False
    
    def _load_module(self, module_name: str, module_path: str) -> bool:
        """
        Load a module from file path and scan for plugins.
        
        Args:
            module_name: Name to assign to the module
            module_path: File path to the module
            
        Returns:
            True if successful, False otherwise
        """
        if module_name in self._loaded_modules:
            return True
        
        try:
            # Load module from file path
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                logger.error(f"Failed to load module spec: {module_path}")
                return False
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Scan module for plugins
            loaded = self._scan_module_for_plugins(module)
            
            if loaded:
                self._loaded_modules.add(module_name)
                logger.info(f"Loaded plugins from {module_path}")
                return True
            else:
                logger.warning(f"No plugins found in {module_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading module {module_path}: {e}")
            return False
    
    def _scan_module_for_plugins(self, module) -> bool:
        """
        Scan a module for plugin classes that use the @register_plugin decorator.
        
        Args:
            module: Module to scan
            
        Returns:
            True if any plugins were found, False otherwise
        """
        found_plugins = False
        
        # Check if module defines __plugin_classes__ list (convention for explicit declaration)
        if hasattr(module, '__plugin_classes__'):
            for plugin_class in module.__plugin_classes__:
                if hasattr(plugin_class, '__plugin_metadata__'):
                    # Already processed by the decorator
                    found_plugins = True
        
        # Look for classes that have the __plugin_metadata__ attribute (set by decorator)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if hasattr(obj, '__plugin_metadata__'):
                found_plugins = True
        
        return found_plugins


# Global loader instance
loader = PluginLoader()


def add_plugin_directory(directory: str) -> None:
    """
    Add a directory to search for plugins.
    
    Args:
        directory: Absolute path to directory containing plugin modules
    """
    loader.add_plugin_directory(directory)


def scan_plugins() -> int:
    """
    Scan all registered directories for plugins and load them.
    
    Returns:
        Number of plugins loaded
    """
    return loader.scan_plugins()


def load_plugin_module(module_name: str) -> bool:
    """
    Load plugins from a specific module by name.
    
    Args:
        module_name: Fully qualified module name (e.g., 'acdek.plugins.my_plugin')
        
    Returns:
        True if successful, False otherwise
    """
    return loader.load_plugin_module(module_name)