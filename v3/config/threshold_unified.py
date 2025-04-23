"""
ACDEK v3 Unified Threshold System

This module centralizes all threshold configurations across ACDEK v3 components,
ensuring consistency between benchmarks, production code, and domain-specific settings.
It eliminates inconsistencies by providing a single source of truth for threshold values.

Key features:
1. Common threshold access for all components (benchmarks, grounding, constraints)
2. Domain-specific threshold definitions
3. Model-specific threshold adjustments
4. Contextual thresholds based on deployment environment
5. Import/export capability for configuration management
"""
import os
import json
import yaml
import logging
import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Import Domain enum from domain_config to ensure consistency
from config.domain_config import Domain, get_threshold as domain_get_threshold


class ThresholdCategory(str, Enum):
    """Categories of thresholds used across the system."""
    FACTUALITY = "factuality"
    HALLUCINATION = "hallucination"
    CONFIDENCE = "confidence"
    CONSTRAINT = "constraint"
    LATENCY = "latency"
    RETRY = "retry"
    FALLBACK = "fallback"
    COMBINED = "combined"


class ExecutionMode(str, Enum):
    """Execution modes for the system, which may use different thresholds."""
    PRODUCTION = "production"
    BENCHMARK = "benchmark"
    DEVELOPMENT = "development"
    TESTING = "testing"


# Default thresholds by domain and category
# This consolidates all previously scattered threshold definitions
DEFAULT_THRESHOLDS = {
    Domain.MEDICAL: {
        ThresholdCategory.FACTUALITY: 0.85,
        ThresholdCategory.HALLUCINATION: 0.85,
        ThresholdCategory.CONFIDENCE: 0.80,
        ThresholdCategory.CONSTRAINT: 0.90,
        ThresholdCategory.LATENCY: 500,  # ms
        ThresholdCategory.RETRY: 0.75,
        ThresholdCategory.FALLBACK: 0.75,
        ThresholdCategory.COMBINED: 0.80
    },
    Domain.LEGAL: {
        ThresholdCategory.FACTUALITY: 0.85,
        ThresholdCategory.HALLUCINATION: 0.85,
        ThresholdCategory.CONFIDENCE: 0.80,
        ThresholdCategory.CONSTRAINT: 0.90,
        ThresholdCategory.LATENCY: 500,  # ms
        ThresholdCategory.RETRY: 0.75,
        ThresholdCategory.FALLBACK: 0.75,
        ThresholdCategory.COMBINED: 0.80
    },
    Domain.FINANCIAL: {
        ThresholdCategory.FACTUALITY: 0.80,
        ThresholdCategory.HALLUCINATION: 0.85,
        ThresholdCategory.CONFIDENCE: 0.75,
        ThresholdCategory.CONSTRAINT: 0.85,
        ThresholdCategory.LATENCY: 450,  # ms
        ThresholdCategory.RETRY: 0.70,
        ThresholdCategory.FALLBACK: 0.70,
        ThresholdCategory.COMBINED: 0.75
    },
    Domain.SCIENTIFIC: {
        ThresholdCategory.FACTUALITY: 0.80,
        ThresholdCategory.HALLUCINATION: 0.80,
        ThresholdCategory.CONFIDENCE: 0.75,
        ThresholdCategory.CONSTRAINT: 0.85,
        ThresholdCategory.LATENCY: 500,  # ms
        ThresholdCategory.RETRY: 0.70,
        ThresholdCategory.FALLBACK: 0.70,
        ThresholdCategory.COMBINED: 0.75
    },
    Domain.TECHNICAL: {
        ThresholdCategory.FACTUALITY: 0.75,
        ThresholdCategory.HALLUCINATION: 0.75,
        ThresholdCategory.CONFIDENCE: 0.70,
        ThresholdCategory.CONSTRAINT: 0.80,
        ThresholdCategory.LATENCY: 400,  # ms
        ThresholdCategory.RETRY: 0.65,
        ThresholdCategory.FALLBACK: 0.65,
        ThresholdCategory.COMBINED: 0.70
    },
    Domain.HISTORICAL: {
        ThresholdCategory.FACTUALITY: 0.80,
        ThresholdCategory.HALLUCINATION: 0.80,
        ThresholdCategory.CONFIDENCE: 0.75,
        ThresholdCategory.CONSTRAINT: 0.85,
        ThresholdCategory.LATENCY: 450,  # ms
        ThresholdCategory.RETRY: 0.65,
        ThresholdCategory.FALLBACK: 0.65,
        ThresholdCategory.COMBINED: 0.75
    },
    Domain.ACADEMIC: {
        ThresholdCategory.FACTUALITY: 0.80,
        ThresholdCategory.HALLUCINATION: 0.80,
        ThresholdCategory.CONFIDENCE: 0.75,
        ThresholdCategory.CONSTRAINT: 0.85,
        ThresholdCategory.LATENCY: 450,  # ms
        ThresholdCategory.RETRY: 0.65,
        ThresholdCategory.FALLBACK: 0.65,
        ThresholdCategory.COMBINED: 0.75
    },
    Domain.GENERAL: {
        ThresholdCategory.FACTUALITY: 0.70,
        ThresholdCategory.HALLUCINATION: 0.70,
        ThresholdCategory.CONFIDENCE: 0.65,
        ThresholdCategory.CONSTRAINT: 0.75,
        ThresholdCategory.LATENCY: 350,  # ms
        ThresholdCategory.RETRY: 0.60,
        ThresholdCategory.FALLBACK: 0.60,
        ThresholdCategory.COMBINED: 0.65
    }
}

# Mode-specific threshold adjustments
# These are multipliers applied to the default thresholds based on execution mode
MODE_ADJUSTMENTS = {
    ExecutionMode.PRODUCTION: {
        ThresholdCategory.FACTUALITY: 1.0,     # No adjustment
        ThresholdCategory.HALLUCINATION: 1.0,  # No adjustment
        ThresholdCategory.CONFIDENCE: 1.0,     # No adjustment
        ThresholdCategory.CONSTRAINT: 1.0,     # No adjustment
        ThresholdCategory.LATENCY: 1.0,        # No adjustment
        ThresholdCategory.RETRY: 1.0,          # No adjustment
        ThresholdCategory.FALLBACK: 1.0,       # No adjustment
        ThresholdCategory.COMBINED: 1.0        # No adjustment
    },
    ExecutionMode.BENCHMARK: {
        ThresholdCategory.FACTUALITY: 0.95,    # Slightly more lenient for benchmarks
        ThresholdCategory.HALLUCINATION: 0.95, # Slightly more lenient
        ThresholdCategory.CONFIDENCE: 0.95,    # Slightly more lenient
        ThresholdCategory.CONSTRAINT: 0.95,    # Slightly more lenient
        ThresholdCategory.LATENCY: 1.2,        # Allow more time in benchmarks
        ThresholdCategory.RETRY: 1.0,          # No adjustment
        ThresholdCategory.FALLBACK: 1.0,       # No adjustment
        ThresholdCategory.COMBINED: 0.95       # Slightly more lenient
    },
    ExecutionMode.DEVELOPMENT: {
        ThresholdCategory.FACTUALITY: 0.9,     # More lenient for development
        ThresholdCategory.HALLUCINATION: 0.9,  # More lenient
        ThresholdCategory.CONFIDENCE: 0.9,     # More lenient
        ThresholdCategory.CONSTRAINT: 0.9,     # More lenient
        ThresholdCategory.LATENCY: 1.5,        # Much more time allowed
        ThresholdCategory.RETRY: 0.8,          # More leniency in retries
        ThresholdCategory.FALLBACK: 0.8,       # More leniency in fallbacks
        ThresholdCategory.COMBINED: 0.9        # More lenient
    },
    ExecutionMode.TESTING: {
        ThresholdCategory.FACTUALITY: 0.8,     # Most lenient for testing
        ThresholdCategory.HALLUCINATION: 0.8,  # Most lenient
        ThresholdCategory.CONFIDENCE: 0.8,     # Most lenient
        ThresholdCategory.CONSTRAINT: 0.8,     # Most lenient
        ThresholdCategory.LATENCY: 2.0,        # Double time allowed
        ThresholdCategory.RETRY: 0.7,          # Much more leniency
        ThresholdCategory.FALLBACK: 0.7,       # Much more leniency
        ThresholdCategory.COMBINED: 0.8        # Most lenient
    }
}

# Container for runtime overrides
_runtime_overrides: Dict[Domain, Dict[ThresholdCategory, Dict[ExecutionMode, float]]] = {}

# Current execution mode - defaults to production
_current_mode = ExecutionMode.PRODUCTION

# Configuration metadata
_config_metadata = {
    "last_updated": datetime.datetime.now().isoformat(),
    "version": "1.0.0",
    "description": "ACDEK v3 unified threshold configuration"
}


def set_execution_mode(mode: Union[str, ExecutionMode]) -> None:
    """
    Set the current execution mode for threshold calculations.
    
    Args:
        mode: The execution mode to set (production, benchmark, development, testing)
    """
    global _current_mode
    
    if isinstance(mode, str):
        try:
            mode = ExecutionMode(mode.lower())
        except ValueError:
            logger.warning(f"Invalid execution mode '{mode}', defaulting to PRODUCTION")
            mode = ExecutionMode.PRODUCTION
    
    _current_mode = mode
    logger.info(f"Execution mode set to {_current_mode.value}")


def get_execution_mode() -> ExecutionMode:
    """
    Get the current execution mode.
    
    Returns:
        Current execution mode
    """
    return _current_mode


def get_threshold(
    domain: Union[str, Domain],
    category: Union[str, ThresholdCategory],
    mode: Optional[Union[str, ExecutionMode]] = None
) -> float:
    """
    Get threshold value for the specified domain, category, and mode.
    
    Args:
        domain: Domain for the threshold
        category: Threshold category
        mode: Execution mode (defaults to current mode)
        
    Returns:
        Threshold value
    """
    # Convert string domain to enum if needed
    if isinstance(domain, str):
        try:
            domain = Domain(domain.lower())
        except ValueError:
            logger.warning(f"Domain '{domain}' not recognized, using Domain.GENERAL")
            domain = Domain.GENERAL
    
    # Convert string category to enum if needed
    if isinstance(category, str):
        try:
            category = ThresholdCategory(category.lower())
        except ValueError:
            logger.warning(f"Category '{category}' not recognized, using ThresholdCategory.FACTUALITY")
            category = ThresholdCategory.FACTUALITY
    
    # Use provided mode or current mode
    if mode is None:
        mode = _current_mode
    elif isinstance(mode, str):
        try:
            mode = ExecutionMode(mode.lower())
        except ValueError:
            logger.warning(f"Mode '{mode}' not recognized, using current mode {_current_mode.value}")
            mode = _current_mode
    
    # Check runtime overrides first
    if domain in _runtime_overrides and category in _runtime_overrides[domain]:
        if mode in _runtime_overrides[domain][category]:
            return _runtime_overrides[domain][category][mode]
    
    # Get base threshold from default thresholds
    if domain in DEFAULT_THRESHOLDS and category in DEFAULT_THRESHOLDS[domain]:
        base_threshold = DEFAULT_THRESHOLDS[domain][category]
    else:
        # Fallback to general domain if specific domain not found
        base_threshold = DEFAULT_THRESHOLDS[Domain.GENERAL][category]
    
    # Apply mode-specific adjustment
    adjustment = MODE_ADJUSTMENTS[mode][category]
    
    # For latency, higher values are worse, so we multiply
    # For other metrics, lower values are worse, so we divide (making threshold lower)
    if category == ThresholdCategory.LATENCY:
        return base_threshold * adjustment
    else:
        return base_threshold * adjustment
    

def set_threshold_override(
    domain: Union[str, Domain],
    category: Union[str, ThresholdCategory],
    value: float,
    mode: Optional[Union[str, ExecutionMode]] = None
) -> None:
    """
    Set a runtime override for a threshold value.
    
    Args:
        domain: Domain for the threshold
        category: Threshold category
        value: Override value
        mode: Execution mode to override (defaults to all modes)
    """
    global _runtime_overrides
    
    # Convert string domain to enum if needed
    if isinstance(domain, str):
        try:
            domain = Domain(domain.lower())
        except ValueError:
            logger.warning(f"Domain '{domain}' not recognized, using Domain.GENERAL")
            domain = Domain.GENERAL
    
    # Convert string category to enum if needed
    if isinstance(category, str):
        try:
            category = ThresholdCategory(category.lower())
        except ValueError:
            logger.warning(f"Category '{category}' not recognized, using ThresholdCategory.FACTUALITY")
            category = ThresholdCategory.FACTUALITY
    
    # Use provided mode or apply to all modes
    if mode is None:
        # Apply to all modes
        if domain not in _runtime_overrides:
            _runtime_overrides[domain] = {}
        if category not in _runtime_overrides[domain]:
            _runtime_overrides[domain][category] = {}
        
        for m in ExecutionMode:
            _runtime_overrides[domain][category][m] = value
    else:
        # Apply to specific mode
        if isinstance(mode, str):
            try:
                mode = ExecutionMode(mode.lower())
            except ValueError:
                logger.warning(f"Mode '{mode}' not recognized, using current mode {_current_mode.value}")
                mode = _current_mode
        
        if domain not in _runtime_overrides:
            _runtime_overrides[domain] = {}
        if category not in _runtime_overrides[domain]:
            _runtime_overrides[domain][category] = {}
        
        _runtime_overrides[domain][category][mode] = value
    
    logger.info(f"Set threshold override for {domain.value}/{category.value}: {value}")


def clear_threshold_overrides(
    domain: Optional[Union[str, Domain]] = None,
    category: Optional[Union[str, ThresholdCategory]] = None,
    mode: Optional[Union[str, ExecutionMode]] = None
) -> None:
    """
    Clear runtime threshold overrides.
    
    Args:
        domain: Specific domain to clear (or all if None)
        category: Specific category to clear (or all if None)
        mode: Specific mode to clear (or all if None)
    """
    global _runtime_overrides
    
    # Convert string inputs to enums if provided
    domain_enum = None
    if domain is not None:
        if isinstance(domain, str):
            try:
                domain_enum = Domain(domain.lower())
            except ValueError:
                logger.warning(f"Domain '{domain}' not recognized, using Domain.GENERAL")
                domain_enum = Domain.GENERAL
        else:
            domain_enum = domain
    
    category_enum = None
    if category is not None:
        if isinstance(category, str):
            try:
                category_enum = ThresholdCategory(category.lower())
            except ValueError:
                logger.warning(f"Category '{category}' not recognized, using ThresholdCategory.FACTUALITY")
                category_enum = ThresholdCategory.FACTUALITY
        else:
            category_enum = category
    
    mode_enum = None
    if mode is not None:
        if isinstance(mode, str):
            try:
                mode_enum = ExecutionMode(mode.lower())
            except ValueError:
                logger.warning(f"Mode '{mode}' not recognized, using current mode {_current_mode.value}")
                mode_enum = _current_mode
        else:
            mode_enum = mode
    
    # Clear overrides based on provided parameters
    if domain_enum is None:
        # Clear all overrides
        _runtime_overrides = {}
        logger.info("Cleared all threshold overrides")
    elif category_enum is None:
        # Clear all categories for the domain
        if domain_enum in _runtime_overrides:
            del _runtime_overrides[domain_enum]
        logger.info(f"Cleared all threshold overrides for domain {domain_enum.value}")
    elif mode_enum is None:
        # Clear all modes for the domain and category
        if domain_enum in _runtime_overrides and category_enum in _runtime_overrides[domain_enum]:
            del _runtime_overrides[domain_enum][category_enum]
        logger.info(f"Cleared all threshold overrides for domain {domain_enum.value} and category {category_enum.value}")
    else:
        # Clear specific mode for the domain and category
        if (domain_enum in _runtime_overrides and 
            category_enum in _runtime_overrides[domain_enum] and
            mode_enum in _runtime_overrides[domain_enum][category_enum]):
            del _runtime_overrides[domain_enum][category_enum][mode_enum]
        logger.info(f"Cleared threshold override for domain {domain_enum.value}, category {category_enum.value}, and mode {mode_enum.value}")


def export_configuration(filepath: Union[str, Path]) -> bool:
    """
    Export the current threshold configuration to a file.
    
    Args:
        filepath: Path to save the configuration
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare configuration for export
        config = {
            "metadata": _config_metadata,
            "current_mode": _current_mode.value,
            "defaults": {
                domain.value: {
                    category.value: value
                    for category, value in thresholds.items()
                }
                for domain, thresholds in DEFAULT_THRESHOLDS.items()
            },
            "mode_adjustments": {
                mode.value: {
                    category.value: value
                    for category, value in adjustments.items()
                }
                for mode, adjustments in MODE_ADJUSTMENTS.items()
            },
            "overrides": {
                domain.value: {
                    category.value: {
                        mode.value: value
                        for mode, value in modes.items()
                    }
                    for category, modes in categories.items()
                }
                for domain, categories in _runtime_overrides.items()
            }
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            if filepath.suffix in ('.yaml', '.yml'):
                yaml.dump(config, f, sort_keys=False)
            else:
                json.dump(config, f, indent=2)
        
        logger.info(f"Threshold configuration exported to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error exporting threshold configuration: {e}")
        return False


def import_configuration(filepath: Union[str, Path]) -> bool:
    """
    Import threshold configuration from a file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    global _runtime_overrides, _current_mode, _config_metadata
    
    try:
        filepath = Path(filepath)
        
        # Read configuration file
        with open(filepath, 'r') as f:
            if filepath.suffix in ('.yaml', '.yml'):
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        # Import metadata
        if "metadata" in config:
            _config_metadata.update(config["metadata"])
        
        # Import current mode
        if "current_mode" in config:
            try:
                _current_mode = ExecutionMode(config["current_mode"].lower())
            except ValueError:
                logger.warning(f"Invalid execution mode '{config['current_mode']}', defaulting to PRODUCTION")
                _current_mode = ExecutionMode.PRODUCTION
        
        # Import overrides
        _runtime_overrides = {}
        if "overrides" in config:
            for domain_str, categories in config["overrides"].items():
                try:
                    domain = Domain(domain_str.lower())
                    _runtime_overrides[domain] = {}
                    
                    for category_str, modes in categories.items():
                        try:
                            category = ThresholdCategory(category_str.lower())
                            _runtime_overrides[domain][category] = {}
                            
                            for mode_str, value in modes.items():
                                try:
                                    mode = ExecutionMode(mode_str.lower())
                                    _runtime_overrides[domain][category][mode] = float(value)
                                except ValueError:
                                    logger.warning(f"Skipping invalid mode '{mode_str}' in configuration")
                        except ValueError:
                            logger.warning(f"Skipping invalid category '{category_str}' in configuration")
                except ValueError:
                    logger.warning(f"Skipping invalid domain '{domain_str}' in configuration")
        
        logger.info(f"Threshold configuration imported from {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error importing threshold configuration: {e}")
        return False


def synchronize_from_grounding_engine():
    """
    Synchronize thresholds from the grounding engine hard-coded values.
    This is for backward compatibility with existing code.
    """
    try:
        from v3.grounding.grounding_engine import DOMAIN_THRESHOLDS as grounding_thresholds
        from v3.grounding.grounding_engine import FACTUALITY_THRESHOLD as global_factuality
        
        # Iterate through domains in grounding engine
        for domain_str, factuality_threshold in grounding_thresholds.items():
            try:
                # Map from domain string to Domain enum
                domain = Domain.from_string(domain_str)
                
                # Set override for factuality in production mode
                set_threshold_override(
                    domain=domain,
                    category=ThresholdCategory.FACTUALITY,
                    value=factuality_threshold,
                    mode=ExecutionMode.PRODUCTION
                )
                
                logger.info(f"Synchronized factuality threshold for {domain.value} from grounding engine: {factuality_threshold}")
            except ValueError:
                logger.warning(f"Could not map grounding domain '{domain_str}' to Domain enum")
        
        # Set global factuality threshold for GENERAL domain
        set_threshold_override(
            domain=Domain.GENERAL,
            category=ThresholdCategory.FACTUALITY,
            value=global_factuality,
            mode=ExecutionMode.PRODUCTION
        )
        
        logger.info(f"Synchronized global factuality threshold from grounding engine: {global_factuality}")
        return True
    
    except ImportError:
        logger.warning("Could not import grounding engine thresholds")
        return False
    except Exception as e:
        logger.error(f"Error synchronizing from grounding engine: {e}")
        return False


def synchronize_from_benchmarks():
    """
    Synchronize thresholds from the benchmark threshold configuration.
    This is for backward compatibility with existing code.
    """
    try:
        from benchmarks.threshold_config import ThresholdConfig
        
        # Iterate through domains in benchmark config
        for domain_str in ThresholdConfig._default_domain_thresholds.keys():
            try:
                # Map from domain string to Domain enum
                domain = Domain.from_string(domain_str)
                
                # Get benchmark domain thresholds
                thresholds = ThresholdConfig.get_domain_thresholds(domain_str)
                
                # Set overrides for benchmark mode
                set_threshold_override(
                    domain=domain,
                    category=ThresholdCategory.FACTUALITY,
                    value=thresholds.factuality,
                    mode=ExecutionMode.BENCHMARK
                )
                
                set_threshold_override(
                    domain=domain,
                    category=ThresholdCategory.LATENCY,
                    value=thresholds.latency,
                    mode=ExecutionMode.BENCHMARK
                )
                
                set_threshold_override(
                    domain=domain,
                    category=ThresholdCategory.CONFIDENCE,
                    value=thresholds.confidence,
                    mode=ExecutionMode.BENCHMARK
                )
                
                logger.info(f"Synchronized benchmark thresholds for {domain.value}")
            except ValueError:
                logger.warning(f"Could not map benchmark domain '{domain_str}' to Domain enum")
        
        logger.info("Synchronized thresholds from benchmark configuration")
        return True
    
    except ImportError:
        logger.warning("Could not import benchmark threshold configuration")
        return False
    except Exception as e:
        logger.error(f"Error synchronizing from benchmarks: {e}")
        return False


def get_legacy_domain_threshold(domain: str, threshold_type: str) -> float:
    """
    Compatibility layer for code using the original domain_config.py get_threshold function.
    
    Args:
        domain: Domain name as string
        threshold_type: Threshold type as string
        
    Returns:
        Threshold value using unified threshold system
    """
    # Map threshold_type to ThresholdCategory
    category_map = {
        "factuality": ThresholdCategory.FACTUALITY,
        "hallucination": ThresholdCategory.HALLUCINATION,
        "confidence": ThresholdCategory.CONFIDENCE,
        "constraint": ThresholdCategory.CONSTRAINT,
        "latency_ms": ThresholdCategory.LATENCY,
        "latency": ThresholdCategory.LATENCY,
        "retry": ThresholdCategory.RETRY,
        "fallback": ThresholdCategory.FALLBACK,
        "combined": ThresholdCategory.COMBINED
    }
    
    category = category_map.get(threshold_type.lower(), ThresholdCategory.FACTUALITY)
    
    return get_threshold(domain, category)


# Initialize the module
def initialize():
    """Initialize the unified threshold system."""
    # Try to detect environment and set mode appropriately
    if os.environ.get("ACDEK_ENV") == "prod":
        set_execution_mode(ExecutionMode.PRODUCTION)
    elif os.environ.get("ACDEK_ENV") == "dev":
        set_execution_mode(ExecutionMode.DEVELOPMENT)
    elif os.environ.get("ACDEK_ENV") == "test":
        set_execution_mode(ExecutionMode.TESTING)
    elif os.environ.get("ACDEK_BENCHMARK") == "1":
        set_execution_mode(ExecutionMode.BENCHMARK)
    else:
        # Default to production mode
        set_execution_mode(ExecutionMode.PRODUCTION)
    
    # Try to synchronize with existing threshold systems
    # This ensures backward compatibility
    synchronize_from_grounding_engine()
    synchronize_from_benchmarks()
    
    logger.info(f"Unified threshold system initialized in {_current_mode.value} mode")


# Initialize on module import
initialize()