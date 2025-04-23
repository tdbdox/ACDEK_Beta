"""
Grounding engine package for ACDEK v3.

This package provides modules for factuality checking and hallucination prevention
using Retrieval-Augmented Generation (RAG) techniques combined with constraint validation.
"""

from .grounding_engine import (
    grounded_validation_pipeline,
    retrieve_reference,
    score_factuality,
    build_grounding_context,
    get_available_domains,
    clear_reference_cache,
    FACTUALITY_THRESHOLD,
    DOMAIN_THRESHOLDS
)

__all__ = [
    'grounded_validation_pipeline',
    'retrieve_reference', 
    'score_factuality',
    'build_grounding_context',
    'get_available_domains',
    'clear_reference_cache',
    'FACTUALITY_THRESHOLD',
    'DOMAIN_THRESHOLDS'
]