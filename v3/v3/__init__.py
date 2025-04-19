"""
ACDEK v3 core extension modules.

This package contains specialized modules that extend ACDEK v3's
core functionality with advanced features like grounding, domain-specific
validation, and hallucination prevention capabilities.
"""

__version__ = "3.0.0"

# Import grounding modules for direct access
try:
    from .grounding import (
        grounded_validation_pipeline,
        retrieve_reference,
        score_factuality,
        build_grounding_context,
        get_available_domains
    )
except ImportError:
    # Log error but don't crash on import
    import logging
    logging.getLogger(__name__).warning("Failed to import grounding module. Some functionality will be unavailable.")