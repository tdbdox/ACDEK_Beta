import inspect
import logging
from typing import Any, Dict, List, Type, Set, Callable, Optional
import importlib

logger = logging.getLogger(__name__)


class InterfaceValidator:
    """
    Validates that plugin implementations correctly implement their interfaces.
    
    This class provides tools to verify that plugin classes fully implement
    all methods defined in their interfaces and that the method signatures
    are compatible.
    """
    
    def __init__(self):
        # Cache of loaded interface classes
        self._interface_cache: Dict[str, Type] = {}
        
        # Cache of interface method signatures
        self._method_signatures: Dict[str, Dict[str, inspect.Signature]] = {}
    
    def load_interface(self, interface_name: str) -> Optional[Type]:
        """
        Dynamically load an interface class.
        
        Args:
            interface_name: Name of the interface to load
            
        Returns:
            The interface class if found, None otherwise
        """
        if interface_name in self._interface_cache:
            return self._interface_cache[interface_name]
        
        # Try to load from known interface modules
        interface_modules = [
            "interfaces.vector_alignment_interface",
            "interfaces.enrichment_interface",
            "interfaces.constraint_verifier_interface",
            "interfaces.decision_gate_interface",
            "interfaces.modulation_interface"
        ]
        
        for module_name in interface_modules:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, interface_name):
                    interface_class = getattr(module, interface_name)
                    self._interface_cache[interface_name] = interface_class
                    # Cache method signatures
                    self._cache_method_signatures(interface_name, interface_class)
                    return interface_class
            except ImportError:
                # Try next module
                continue
        
        logger.warning(f"Interface {interface_name} not found in known modules")
        return None
    
    def _cache_method_signatures(self, interface_name: str, interface_class: Type) -> None:
        """Cache method signatures for an interface."""
        signatures = {}
        
        # Get all abstract methods from the interface
        for name, method in inspect.getmembers(interface_class, inspect.isfunction):
            if getattr(method, "__isabstractmethod__", False):
                signatures[name] = inspect.signature(method)
        
        self._method_signatures[interface_name] = signatures
    
    def get_required_methods(self, interface_name: str) -> Set[str]:
        """
        Get the set of required method names for an interface.
        
        Args:
            interface_name: Name of the interface
            
        Returns:
            Set of method names that must be implemented
        """
        interface_class = self.load_interface(interface_name)
        if not interface_class:
            return set()
        
        if interface_name in self._method_signatures:
            return set(self._method_signatures[interface_name].keys())
        
        # Manually find abstract methods if not in cache
        required = set()
        for name, method in inspect.getmembers(interface_class, inspect.isfunction):
            if getattr(method, "__isabstractmethod__", False):
                required.add(name)
        
        return required
    
    def validate_implementation(self, plugin_class: Type, interface_names: List[str]) -> Dict[str, List[str]]:
        """
        Validate that a plugin implementation properly implements its interfaces.
        
        Args:
            plugin_class: The plugin class to validate
            interface_names: List of interface names the plugin claims to implement
            
        Returns:
            Dictionary mapping interface names to lists of validation errors
        """
        results: Dict[str, List[str]] = {}
        
        for interface_name in interface_names:
            errors = []
            interface_class = self.load_interface(interface_name)
            
            if not interface_class:
                errors.append(f"Interface {interface_name} could not be loaded")
                results[interface_name] = errors
                continue
            
            # Get required methods
            required_methods = self.get_required_methods(interface_name)
            
            # Check for missing methods
            for method_name in required_methods:
                if not hasattr(plugin_class, method_name):
                    errors.append(f"Missing required method: {method_name}")
                    continue
                
                plugin_method = getattr(plugin_class, method_name)
                if not callable(plugin_method):
                    errors.append(f"Attribute {method_name} is not callable")
                    continue
                
                # Check signature compatibility
                if interface_name in self._method_signatures:
                    interface_sig = self._method_signatures[interface_name].get(method_name)
                    if interface_sig:
                        plugin_sig = inspect.signature(plugin_method)
                        # Verify parameter compatibility
                        self._check_signature_compatibility(method_name, interface_sig, plugin_sig, errors)
            
            results[interface_name] = errors
        
        return results
    
    def _check_signature_compatibility(
        self, 
        method_name: str, 
        interface_sig: inspect.Signature, 
        plugin_sig: inspect.Signature, 
        errors: List[str]
    ) -> None:
        """Check that plugin method signature is compatible with interface."""
        # Get required parameters from interface signature (excluding 'self')
        interface_params = list(interface_sig.parameters.values())[1:]  # Skip 'self'
        plugin_params = list(plugin_sig.parameters.values())[1:]  # Skip 'self'
        
        # Check if there are missing required parameters
        interface_required = [p for p in interface_params if p.default == inspect.Parameter.empty and p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        plugin_required = [p for p in plugin_params if p.default == inspect.Parameter.empty and p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        
        if len(plugin_required) > len(interface_required):
            errors.append(f"Method {method_name} has more required parameters than interface definition")
        
        # Check each interface parameter
        for i_param in interface_params:
            found = False
            for p_param in plugin_params:
                if i_param.name == p_param.name:
                    found = True
                    # Optional parameters in interface can't be required in implementation
                    if (i_param.default != inspect.Parameter.empty and 
                        p_param.default == inspect.Parameter.empty and 
                        p_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD):
                        errors.append(f"Method {method_name}: parameter {i_param.name} is optional in interface but required in implementation")
                    break
            
            if not found and i_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                errors.append(f"Method {method_name} missing parameter {i_param.name}")
        
        # Check return type annotation compatibility if present
        if interface_sig.return_annotation != inspect.Signature.empty:
            if plugin_sig.return_annotation == inspect.Signature.empty:
                errors.append(f"Method {method_name} is missing return type annotation")


# Global validator instance
validator = InterfaceValidator()


def validate_plugin(plugin_class: Type, interface_names: List[str]) -> Dict[str, List[str]]:
    """
    Validate a plugin implementation against its claimed interfaces.
    
    Args:
        plugin_class: The plugin class to validate
        interface_names: List of interface names the plugin claims to implement
        
    Returns:
        Dictionary mapping interface names to lists of validation errors
    """
    return validator.validate_implementation(plugin_class, interface_names)


def is_valid_plugin(plugin_class: Type, interface_names: List[str]) -> bool:
    """
    Check if a plugin validly implements all its claimed interfaces.
    
    Args:
        plugin_class: The plugin class to validate
        interface_names: List of interface names the plugin claims to implement
        
    Returns:
        True if valid, False otherwise
    """
    validation_results = validate_plugin(plugin_class, interface_names)
    
    # Check for any errors
    for errors in validation_results.values():
        if errors:
            return False
    
    return True