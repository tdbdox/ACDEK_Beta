import logging
from typing import Dict, List, Any, Optional, Union, Type, Callable
import time
from dataclasses import dataclass, field

from registry.registry_types import PluginType
from registry.plugin_registry import register_plugin, get_plugin, create_instance
from interfaces.enrichment_interface import EnrichmentInterface
from interfaces.vector_alignment_interface import VectorAlignmentInterface

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentStage:
    """Configuration for a stage in the enrichment pipeline."""
    name: str
    plugin_type: PluginType
    plugin_name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    fallback_strategy: str = "skip"  # skip, default, error
    

@dataclass
class EnrichmentResult:
    """Result of running the enrichment pipeline."""
    success: bool
    data: Dict[str, Any]
    errors: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, Any] = field(default_factory=dict)


class EnrichmentPipeline:
    """
    Pipeline for enriching data through multiple stages.
    
    This pipeline orchestrates the flow of data through multiple enrichment
    components, handling errors, fallbacks, and metrics collection.
    """
    
    def __init__(self, stages: List[EnrichmentStage] = None):
        """
        Initialize the enrichment pipeline.
        
        Args:
            stages: List of enrichment stages to execute
        """
        self.stages = stages or []
        self.stage_instances: Dict[str, Any] = {}
        self._initialize_stages()
        
        logger.info(f"Initialized enrichment pipeline with {len(self.stages)} stages")
    
    def _initialize_stages(self) -> None:
        """Initialize plugin instances for each stage."""
        for stage in self.stages:
            try:
                # Create plugin instance
                instance = create_instance(
                    plugin_type=stage.plugin_type,
                    name=stage.plugin_name,
                    config=stage.config
                )
                
                if instance:
                    self.stage_instances[stage.name] = instance
                    logger.info(f"Initialized stage '{stage.name}' with plugin {stage.plugin_name or 'default'}")
                else:
                    logger.error(f"Failed to initialize stage '{stage.name}'")
            except Exception as e:
                logger.error(f"Error initializing stage '{stage.name}': {str(e)}")
    
    def add_stage(self, stage: EnrichmentStage) -> bool:
        """
        Add a new stage to the pipeline.
        
        Args:
            stage: Stage configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if stage with same name already exists
            if any(s.name == stage.name for s in self.stages):
                logger.warning(f"Stage with name '{stage.name}' already exists")
                return False
            
            # Create plugin instance
            instance = create_instance(
                plugin_type=stage.plugin_type,
                name=stage.plugin_name,
                config=stage.config
            )
            
            if instance:
                self.stages.append(stage)
                self.stage_instances[stage.name] = instance
                logger.info(f"Added stage '{stage.name}' to pipeline")
                return True
            else:
                logger.error(f"Failed to create instance for stage '{stage.name}'")
                return False
                
        except Exception as e:
            logger.error(f"Error adding stage '{stage.name}': {str(e)}")
            return False
    
    def execute(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> EnrichmentResult:
        """
        Execute the enrichment pipeline.
        
        Args:
            input_data: Input data to enrich
            context: Optional context information
            
        Returns:
            Result of enrichment pipeline execution
        """
        start_time = time.time()
        result_data = dict(input_data)  # Start with input data
        errors = {}
        metrics = {"timing": {}}
        stage_results = {}
        
        context = context or {}
        
        for stage in self.stages:
            stage_start = time.time()
            
            try:
                # Get stage instance
                instance = self.stage_instances.get(stage.name)
                
                if not instance:
                    if stage.required:
                        if stage.fallback_strategy == "error":
                            logger.error(f"Required stage '{stage.name}' not initialized")
                            return EnrichmentResult(
                                success=False,
                                data=result_data,
                                errors={stage.name: "Stage not initialized"},
                                metrics=metrics,
                                stage_results=stage_results
                            )
                        elif stage.fallback_strategy == "default":
                            # Try to create a default instance
                            instance = create_instance(plugin_type=stage.plugin_type)
                            if not instance:
                                logger.error(f"Failed to create default instance for stage '{stage.name}'")
                                errors[stage.name] = "Failed to create default instance"
                                continue
                        else:  # skip
                            logger.warning(f"Skipping uninitialized stage '{stage.name}'")
                            errors[stage.name] = "Stage not initialized, skipped"
                            continue
                    else:
                        logger.info(f"Skipping optional stage '{stage.name}'")
                        continue
                
                # Execute stage
                if isinstance(instance, EnrichmentInterface):
                    stage_result = instance.enrich(result_data, context)
                    stage_results[stage.name] = stage_result
                    
                    # Update result data
                    if stage_result:
                        # For enrichment interfaces with a to_dict method
                        if hasattr(stage_result, 'to_dict'):
                            result_data[stage.name] = stage_result.to_dict()
                        else:
                            result_data[stage.name] = stage_result
                else:
                    # For custom stage implementations
                    stage_method = getattr(instance, "process", None)
                    if callable(stage_method):
                        stage_result = stage_method(result_data, context)
                        stage_results[stage.name] = stage_result
                        
                        # Update result data
                        if stage_result:
                            if hasattr(stage_result, 'to_dict'):
                                result_data[stage.name] = stage_result.to_dict()
                            else:
                                result_data[stage.name] = stage_result
                    else:
                        logger.error(f"Stage '{stage.name}' does not have a valid processing method")
                        errors[stage.name] = "Invalid processing method"
                        
                        if stage.required and stage.fallback_strategy == "error":
                            return EnrichmentResult(
                                success=False,
                                data=result_data,
                                errors=errors,
                                metrics=metrics,
                                stage_results=stage_results
                            )
                
            except Exception as e:
                logger.error(f"Error executing stage '{stage.name}': {str(e)}")
                errors[stage.name] = str(e)
                
                if stage.required and stage.fallback_strategy == "error":
                    return EnrichmentResult(
                        success=False,
                        data=result_data,
                        errors=errors,
                        metrics=metrics,
                        stage_results=stage_results
                    )
            
            finally:
                # Record timing
                stage_time = time.time() - stage_start
                metrics["timing"][stage.name] = stage_time
        
        # Calculate total time
        total_time = time.time() - start_time
        metrics["timing"]["total"] = total_time
        
        return EnrichmentResult(
            success=len(errors) == 0 or all(not self.stages[i].required for i, stage_name in enumerate(errors.keys())),
            data=result_data,
            errors=errors,
            metrics=metrics,
            stage_results=stage_results
        )
    
    def reset(self) -> None:
        """Reset the pipeline, clearing all stages."""
        self.stages = []
        self.stage_instances = {}
        logger.info("Enrichment pipeline reset")


# Factory function to create common pipeline configurations
def create_standard_pipeline() -> EnrichmentPipeline:
    """
    Create a standard enrichment pipeline with common stages.
    
    Returns:
        Configured enrichment pipeline
    """
    pipeline = EnrichmentPipeline([
        EnrichmentStage(
            name="vector_alignment",
            plugin_type=PluginType.VECTOR_ALIGNMENT,
            required=True,
            fallback_strategy="default"
        ),
        EnrichmentStage(
            name="ar_enrichment",
            plugin_type=PluginType.ENRICHMENT,
            plugin_name="ar_vector_enricher",
            required=True,
            fallback_strategy="error"
        )
    ])
    
    return pipeline
