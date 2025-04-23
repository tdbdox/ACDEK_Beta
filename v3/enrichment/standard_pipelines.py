"""
Standard pipeline configurations for the ACDEK system.

This module provides factory functions to create various pre-configured
pipeline configurations that work out of the box with proper error handling
and fallback strategies.
"""
import logging
from typing import Dict, Any, List, Optional

from enrichment.enrichment_pipeline import (
    EnrichmentPipeline, 
    EnrichmentStage, 
    ExecutionMode,
    EnrichmentResult
)
from registry.registry_types import PluginType
from registry.enhanced_plugin_registry import load_plugin, FallbackStrategy

logger = logging.getLogger(__name__)

def create_minimal_pipeline() -> EnrichmentPipeline:
    """
    Create a minimal pipeline with only essential components.
    
    This pipeline includes only confidence scoring and is designed to work
    even when other components are unavailable. Perfect for testing and
    environments with limited requirements.
    
    Returns:
        Configured minimal enrichment pipeline
    """
    # Create a pipeline with just confidence scoring
    try:
        # Try with enable_caching parameter
        pipeline = EnrichmentPipeline(
            stages=[
                EnrichmentStage(
                    name="confidence_scoring",
                    plugin_type=PluginType.ENRICHMENT,
                    plugin_name="vector_confidence_scorer",
                    required=True,
                    fallback_strategy="default",
                    cache_result=True
                )
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            enable_caching=True
        )
    except TypeError:
        # Fall back to version without enable_caching if not supported
        pipeline = EnrichmentPipeline(
            stages=[
                EnrichmentStage(
                    name="confidence_scoring",
                    plugin_type=PluginType.ENRICHMENT,
                    plugin_name="vector_confidence_scorer",
                    required=True,
                    fallback_strategy="default",
                    cache_result=True
                )
            ],
            execution_mode=ExecutionMode.SEQUENTIAL
        )
    
    return pipeline

def create_standard_pipeline() -> EnrichmentPipeline:
    """
    Create a standard enrichment pipeline with common stages and fallbacks.
    
    This pipeline includes confidence scoring with proper fallback strategies
    and is designed to work out of the box in most environments.
    
    Returns:
        Configured enrichment pipeline
    """
    # Check if vector_alignment is available
    vector_alignment_result = load_plugin(
        PluginType.VECTOR_ALIGNMENT, 
        fallback_strategy=FallbackStrategy(
            use_default=True,
            allow_missing=True,
            log_level="INFO"
        )
    )
    
    # Check if ar_vector_enricher is available
    ar_enricher_result = load_plugin(
        PluginType.ENRICHMENT,
        name="ar_vector_enricher",
        fallback_strategy=FallbackStrategy(
            use_default=False,
            allow_missing=True,
            log_level="INFO"
        )
    )
    
    # Configure stages based on available plugins
    stages = []
    
    # Only add vector_alignment if available
    if vector_alignment_result.success:
        stages.append(
            EnrichmentStage(
                name="vector_alignment",
                plugin_type=PluginType.VECTOR_ALIGNMENT,
                required=True,
                fallback_strategy="default",
                cache_result=True
            )
        )
    
    # Only add ar_enrichment if both it and vector_alignment are available
    if ar_enricher_result.success and vector_alignment_result.success:
        stages.append(
            EnrichmentStage(
                name="ar_enrichment",
                plugin_type=PluginType.ENRICHMENT,
                plugin_name="ar_vector_enricher",
                required=False,  # Make optional to avoid pipeline failure
                fallback_strategy="skip",
                dependencies=["vector_alignment"],
                cache_result=True
            )
        )
    
    # Always include confidence scoring
    stages.append(
        EnrichmentStage(
            name="confidence_scoring",
            plugin_type=PluginType.ENRICHMENT,
            plugin_name="vector_confidence_scorer",
            required=True,
            fallback_strategy="default",
            cache_result=True,
            # Only depend on vector_alignment if it's included
            dependencies=["vector_alignment"] if vector_alignment_result.success else []
        )
    )
    
    # Create the pipeline with the available stages
    try:
        # Try with enable_caching parameter
        pipeline = EnrichmentPipeline(
            stages=stages,
            execution_mode=ExecutionMode.ADAPTIVE,
            enable_caching=True
        )
    except TypeError:
        # Fall back to version without enable_caching if not supported
        pipeline = EnrichmentPipeline(
            stages=stages,
            execution_mode=ExecutionMode.ADAPTIVE
        )
    
    logger.info(f"Created standard pipeline with {len(stages)} stages")
    return pipeline

def create_advanced_pipeline() -> EnrichmentPipeline:
    """
    Create an advanced enrichment pipeline with all available components.
    
    This pipeline includes all available enrichment stages with proper
    dependency management and parallel execution where possible.
    
    Returns:
        Configured advanced enrichment pipeline
    """
    # Basic stages that are always included
    stages = [
        # Core stages
        EnrichmentStage(
            name="vector_alignment",
            plugin_type=PluginType.VECTOR_ALIGNMENT,
            required=False,  # Not required to allow pipeline to work without it
            fallback_strategy="skip",
            cache_result=True
        ),
        EnrichmentStage(
            name="confidence_scoring",
            plugin_type=PluginType.ENRICHMENT,
            plugin_name="vector_confidence_scorer",
            required=True,
            fallback_strategy="default",
            dependencies=["vector_alignment"] if load_plugin(PluginType.VECTOR_ALIGNMENT).success else [],
            cache_result=True
        )
    ]
    
    # Optional stages - only add if available
    optional_stages = [
        # Drift detection
        {
            "type": PluginType.ENRICHMENT,
            "name": "drift_detector",
            "stage_name": "drift_detection",
            "dependencies": ["vector_alignment"],
            "required": False
        },
        # AR enrichment
        {
            "type": PluginType.ENRICHMENT,
            "name": "ar_vector_enricher",
            "stage_name": "ar_enrichment",
            "dependencies": ["vector_alignment"],
            "required": False
        },
        # Constraint validation
        {
            "type": PluginType.CONSTRAINT,
            "name": None,  # Use default
            "stage_name": "constraint_validation",
            "dependencies": ["ar_enrichment"] if load_plugin(PluginType.ENRICHMENT, "ar_vector_enricher").success else [],
            "required": False
        }
    ]
    
    # Add optional stages if available
    for stage_info in optional_stages:
        plugin_result = load_plugin(
            stage_info["type"],
            stage_info["name"],
            fallback_strategy=FallbackStrategy(
                use_default=True,
                allow_missing=True,
                log_level="INFO"
            )
        )
        
        if plugin_result.success:
            # Verify all dependencies are included
            missing_deps = []
            for dep in stage_info["dependencies"]:
                if not any(s.name == dep for s in stages):
                    missing_deps.append(dep)
            
            # Only add if all dependencies are satisfied
            if not missing_deps:
                stages.append(
                    EnrichmentStage(
                        name=stage_info["stage_name"],
                        plugin_type=stage_info["type"],
                        plugin_name=stage_info["name"],
                        required=stage_info["required"],
                        fallback_strategy="skip",
                        dependencies=stage_info["dependencies"],
                        cache_result=True
                    )
                )
                logger.info(f"Added optional stage {stage_info['stage_name']} to advanced pipeline")
            else:
                logger.info(f"Skipped stage {stage_info['stage_name']} due to missing dependencies: {missing_deps}")
    
    # Create the pipeline with all available stages
    try:
        # Try with enable_caching parameter
        pipeline = EnrichmentPipeline(
            stages=stages,
            execution_mode=ExecutionMode.ADAPTIVE,
            enable_caching=True,
            max_workers=4  # Allow parallel execution where possible
        )
    except TypeError:
        # Fall back to version without enable_caching if not supported
        try:
            pipeline = EnrichmentPipeline(
                stages=stages,
                execution_mode=ExecutionMode.ADAPTIVE,
                max_workers=4  # Allow parallel execution where possible
            )
        except TypeError:
            # Fall back to bare minimum if max_workers is also not supported
            pipeline = EnrichmentPipeline(
                stages=stages,
                execution_mode=ExecutionMode.ADAPTIVE
            )
    
    logger.info(f"Created advanced pipeline with {len(stages)} stages")
    return pipeline

def dummy_execute_pipeline(
    input_data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> EnrichmentResult:
    """
    Dummy pipeline executor that always succeeds with minimal output.
    
    This function is useful for testing and as a fallback when the
    real pipeline cannot be initialized.
    
    Args:
        input_data: Input data to process
        context: Optional context information
        
    Returns:
        Basic EnrichmentResult with success status
    """
    logger.warning("Using dummy pipeline executor - this should only be used for testing")
    
    # Create basic confidence score for the output
    confidence_score = 0.7
    
    return EnrichmentResult(
        success=True,
        data=input_data,
        stage_results={
            "dummy": {
                "success": True,
                "confidence": confidence_score
            }
        },
        metrics={
            "duration": 0.001,
            "confidence": confidence_score
        }
    )