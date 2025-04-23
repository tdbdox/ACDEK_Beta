"""
ACDEK v3 Configuration Package

This package provides centralized configuration for all ACDEK v3 components,
ensuring consistent behavior across benchmarks, production systems, and tests.
"""

from config.domain_config import Domain, get_threshold, get_domain_thresholds, DEFAULT_THRESHOLDS

__all__ = ['Domain', 'get_threshold', 'get_domain_thresholds', 'DEFAULT_THRESHOLDS']