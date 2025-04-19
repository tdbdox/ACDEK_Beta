import logging
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional, Union, Type, Callable, Tuple, Set
import time
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
import json

from registry.registry_types import PluginType
from registry.plugin_registry import register_plugin, get_plugin, create_instance
from interfaces.enrichment_interface import EnrichmentInterface
from interfaces.vector_alignment_interface import VectorAlignmentInterface
from observability.metrics_collector import MetricsCollector, TimingContext
from observability.telemetry_server import get_telemetry_client

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode for pipeline stages."""
    SEQUENTIAL = "sequential"  # Execute stages in sequence
    PARALLEL = "parallel"      # Execute independent stages in parallel
    ADAPTIVE = "adaptive"      # Automatically determine parallelization based on dependencies


@dataclass
class EnrichmentStage:
    """
    Configuration for a stage in the enrichment pipeline.
    
    Defines a processing stage with metadata about dependencies, execution requirements,
    and fallback strategies for resilient processing.
    """
    name: str
    plugin_type: PluginType
    plugin_name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    fallback_strategy: str = "skip"  # skip, default, error
    execution_timeout: Optional[float] = None  # Timeout in seconds
    cache_result: bool = True  # Whether to cache this stage's results
    dependencies: List[str] = field(default_factory=list)  # Names of stages this stage depends on
    concurrency_key: Optional[str] = None  # Key for concurrency control
    

@dataclass
class EnrichmentResult:
    """
    Result of running the enrichment pipeline.
    
    Contains the enrichment results, success status, errors, timing metrics,
    and other metadata from pipeline execution.
    """
    success: bool
    data: Dict[str, Any]
    errors: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    stage_results: Dict[str, Any] = field(default_factory=dict)
    fallback_depth: int = 0
    fallback_penalty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)
    
    @classmethod
    def create_error_result(cls, error: str, data: Optional[Dict[str, Any]] = None) -> 'EnrichmentResult':
        """Create a result representing an error condition."""
        return cls(
            success=False,
            data=data or {},
            errors={"pipeline": error},
            metrics={"timing": {"total": 0.0}}
        )


class StageDependencyGraph:
    """
    Dependency graph for pipeline stages to determine execution order.
    
    This class builds and manages a directed graph of stage dependencies,
    providing methods to determine execution order and parallelization opportunities.
    """
    def __init__(self, stages: List[EnrichmentStage]):
        """Initialize with a list of stages."""
        self.stages = {stage.name: stage for stage in stages}
        self.dependencies: Dict[str, Set[str]] = {}
        self.dependents: Dict[str, Set[str]] = {}
        self._build_graph()
        
    def _build_graph(self) -> None:
        """Build the dependency graph from stage definitions."""
        # Initialize empty dependency sets
        for name in self.stages:
            self.dependencies[name] = set()
            self.dependents[name] = set()
            
        # Populate dependencies and dependents
        for name, stage in self.stages.items():
            for dep in stage.dependencies:
                if dep in self.stages:
                    self.dependencies[name].add(dep)
                    self.dependents[dep].add(name)
        
        # Validate no circular dependencies
        self._check_cycles()
    
    def _check_cycles(self) -> None:
        """Check for cycles in the dependency graph."""
        visited = set()
        temp_visit = set()
        
        def dfs(node):
            if node in temp_visit:
                raise ValueError(f"Circular dependency detected involving stage '{node}'")
            if node in visited:
                return
                
            temp_visit.add(node)
            for dep in self.dependencies[node]:
                dfs(dep)
            temp_visit.remove(node)
            visited.add(node)
        
        for node in self.stages:
            if node not in visited:
                dfs(node)
    
    def get_execution_levels(self) -> List[List[str]]:
        """
        Group stages into execution levels based on dependencies.
        
        Returns a list of lists, where each inner list contains stage names
        that can be executed in parallel.
        """
        remaining_deps = {name: set(deps) for name, deps in self.dependencies.items()}
        levels = []
        
        while remaining_deps:
            # Find nodes with no dependencies
            level = [name for name, deps in remaining_deps.items() if not deps]
            
            if not level:
                # If there are no nodes without dependencies, we have a cycle
                cycle_stages = ", ".join(remaining_deps.keys())
                raise ValueError(f"Unable to determine execution order. Possible cycle in: {cycle_stages}")
            
            # Add this level to our execution plan
            levels.append(level)
            
            # Remove nodes we've processed
            for name in level:
                del remaining_deps[name]
                
            # Update dependencies
            for deps in remaining_deps.values():
                deps.difference_update(level)
        
        return levels
    
    def get_independent_stages(self) -> List[List[str]]:
        """
        Group stages that can be executed independently of each other.
        
        This is more fine-grained than execution levels and allows for
        maximum parallelization opportunities.
        """
        # Start with execution levels as the base
        levels = self.get_execution_levels()
        independent_groups = []
        
        for level in levels:
            # Further divide each level based on shared dependents
            groups: List[Set[str]] = []
            for stage in level:
                # Get all direct and indirect dependents
                dependent_closure = self._get_dependent_closure(stage)
                
                # Try to find an existing group that doesn't conflict
                assigned = False
                for group in groups:
                    # Check if stage conflicts with any stage in this group
                    conflict = False
                    for existing_stage in group:
                        existing_dependents = self._get_dependent_closure(existing_stage)
                        if dependent_closure.intersection(existing_dependents):
                            conflict = True
                            break
                    
                    # If no conflict, add to this group
                    if not conflict:
                        group.add(stage)
                        assigned = True
                        break
                
                # If couldn't assign to any group, create a new one
                if not assigned:
                    groups.append({stage})
            
            # Add all groups from this level
            independent_groups.extend([list(group) for group in groups])
        
        return independent_groups
    
    def _get_dependent_closure(self, stage: str) -> Set[str]:
        """Get the transitive closure of all dependents for a stage."""
        closure = set()
        queue = list(self.dependents[stage])
        
        while queue:
            current = queue.pop(0)
            if current not in closure:
                closure.add(current)
                queue.extend(self.dependents[current])
        
        return closure


class EnrichmentPipeline:
    """
    Pipeline for enriching data through multiple stages with parallel execution support.
    
    This pipeline orchestrates the flow of data through multiple enrichment
    components, handling errors, fallbacks, and metrics collection. It supports
    parallel execution of independent stages and provides comprehensive
    observability and caching capabilities.
    """
    
    def __init__(
        self,
        stages: List[EnrichmentStage] = None,
        execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE,
        max_workers: Optional[int] = None,
        cache_config: Optional[Dict[str, Any]] = None,
        telemetry_enabled: bool = True
    ):
        """
        Initialize the enrichment pipeline.
        
        Args:
            stages: List of enrichment stages to execute
            execution_mode: Strategy for executing stages (sequential, parallel, adaptive)
            max_workers: Maximum number of concurrent stage executions
            cache_config: Configuration for result caching
            telemetry_enabled: Whether to collect and report telemetry
        """
        self.stages = stages or []
        self.stage_instances: Dict[str, Any] = {}
        self.execution_mode = execution_mode
        
        # Set up thread pool for parallel execution
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers or min(32, (len(self.stages) * 2))
        )
        
        # Concurrency control
        self.concurrency_semaphores: Dict[str, threading.Semaphore] = {}
        
        # Telemetry and metrics
        self.telemetry_enabled = telemetry_enabled
        if telemetry_enabled:
            self.metrics = MetricsCollector(service_name="enrichment_pipeline")
            self.telemetry = get_telemetry_client()
            self._register_metrics()
        
        # Set up caching
        self.cache_config = cache_config or {}
        self.enable_caching = self.cache_config.get("enabled", True)
        if self.enable_caching:
            self.result_cache: Dict[str, Dict[str, Any]] = {}
            self.cache_lock = threading.RLock()
        
        # Initialize components
        self._initialize_stages()
        self._initialize_concurrency_control()
        
        # Build dependency graph if needed
        if self.execution_mode != ExecutionMode.SEQUENTIAL:
            try:
                self.dependency_graph = StageDependencyGraph(self.stages)
                logger.info("Stage dependency graph built successfully")
            except ValueError as e:
                logger.error(f"Error building dependency graph: {str(e)}, falling back to sequential execution")
                self.execution_mode = ExecutionMode.SEQUENTIAL
        
        logger.info(f"Initialized enrichment pipeline with {len(self.stages)} stages in {self.execution_mode.value} mode")
    
    def _register_metrics(self) -> None:
        """Register metrics for observability."""
        self.metrics.register_counter("pipeline.stages_total", "Total number of enrichment stages executed")
        self.metrics.register_counter("pipeline.errors_total", "Total number of stage errors")
        self.metrics.register_counter("pipeline.cache_hits", "Cache hit count")
        self.metrics.register_counter("pipeline.cache_misses", "Cache miss count")
        
        self.metrics.register_histogram(
            "pipeline.stage_duration_seconds", 
            "Stage execution time in seconds",
            buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        self.metrics.register_histogram(
            "pipeline.pipeline_duration_seconds", 
            "Pipeline execution time in seconds",
            buckets=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        )
        
        self.metrics.register_gauge(
            "pipeline.concurrent_stages", 
            "Current number of concurrently executing stages"
        )
    
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
    
    def _initialize_concurrency_control(self) -> None:
        """Initialize concurrency control mechanisms."""
        # Group stages by concurrency key and create semaphores
        concurrency_groups: Dict[str, List[str]] = {}
        
        for stage in self.stages:
            if stage.concurrency_key:
                if stage.concurrency_key not in concurrency_groups:
                    concurrency_groups[stage.concurrency_key] = []
                concurrency_groups[stage.concurrency_key].append(stage.name)
        
        # Create semaphores for each concurrency group
        for key, group in concurrency_groups.items():
            # Default to allowing half of max_workers for a given key
            limit = min(len(group), max(1, (self.max_workers or 4) // 2))
            self.concurrency_semaphores[key] = threading.Semaphore(limit)
            logger.info(f"Created concurrency control for key '{key}' with limit {limit} for stages: {', '.join(group)}")
    
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
                
                # Update dependency graph if needed
                if self.execution_mode != ExecutionMode.SEQUENTIAL:
                    try:
                        self.dependency_graph = StageDependencyGraph(self.stages)
                    except ValueError as e:
                        logger.error(f"Error updating dependency graph: {str(e)}")
                        return False
                
                # Set up concurrency control if needed
                if stage.concurrency_key and stage.concurrency_key not in self.concurrency_semaphores:
                    self.concurrency_semaphores[stage.concurrency_key] = threading.Semaphore(
                        max(1, (self.max_workers or 4) // 2)
                    )
                
                logger.info(f"Added stage '{stage.name}' to pipeline")
                return True
            else:
                logger.error(f"Failed to create instance for stage '{stage.name}'")
                return False
                
        except Exception as e:
            logger.error(f"Error adding stage '{stage.name}': {str(e)}")
            return False
    
    def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> EnrichmentResult:
        """
        Execute the enrichment pipeline.
        
        Args:
            input_data: Input data to enrich
            context: Optional context information
            
        Returns:
            Result of enrichment pipeline execution
        """
        # Ensure we have a proper context dictionary and request ID
        context = context or {}
        request_id = context.get("request_id", f"req_{int(time.time() * 1000)}")
        if "request_id" not in context:
            context["request_id"] = request_id
        
        # Start timing the pipeline execution
        with TimingContext() as pipeline_timer:
            # Track metrics
            if self.telemetry_enabled:
                self.metrics.counter_inc("pipeline.stages_total", len(self.stages))
            
            # Select execution mode
            if self.execution_mode == ExecutionMode.SEQUENTIAL:
                result = self._execute_sequential(input_data, context, request_id)
            else:  # PARALLEL or ADAPTIVE
                result = self._execute_parallel(input_data, context, request_id)
            
            # Record execution time
            if self.telemetry_enabled:
                self.metrics.histogram_observe(
                    "pipeline.pipeline_duration_seconds", 
                    pipeline_timer.elapsed
                )
            
            # Log and return result
            logger.info(f"Pipeline execution completed in {pipeline_timer.elapsed:.4f}s for request {request_id}, " + 
                       f"success={result.success}, error_count={len(result.errors)}")
            return result
    
    def _execute_sequential(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        request_id: str
    ) -> EnrichmentResult:
        """Execute pipeline stages sequentially."""
        # Create span for the whole pipeline execution
        if self.telemetry_enabled:
            pipeline_span = self.telemetry.create_span(
                name="enrichment_pipeline.execute",
                attributes={"request_id": request_id}
            )
            pipeline_span.__enter__()
        else:
            pipeline_span = None
        
        try:
            start_time = time.time()
            result_data = dict(input_data)  # Start with input data
            errors = {}
            metrics = {"timing": {}}
            stage_results = {}
            
            # Process stages in order
            for stage in self.stages:
                stage_result = self._execute_stage(
                    stage, result_data, context, request_id, errors, metrics, stage_results
                )
                
                # If we got a proper result (not None) update the result data
                if stage_result:
                    self._update_result_data(stage.name, stage_result, result_data)
                
                # If a required stage failed with error strategy, exit early
                if (stage.required and 
                    stage.name in errors and 
                    stage.fallback_strategy == "error"):
                    logger.error(f"Required stage '{stage.name}' failed with error strategy, stopping pipeline")
                    return EnrichmentResult(
                        success=False,
                        data=result_data,
                        errors=errors,
                        metrics=metrics,
                        stage_results=stage_results
                    )
            
            # Calculate total time
            total_time = time.time() - start_time
            metrics["timing"]["total"] = total_time
            
            return EnrichmentResult(
                success=len(errors) == 0 or all(
                    not self._find_stage_by_name(stage_name).required 
                    for stage_name in errors.keys()
                ),
                data=result_data,
                errors=errors,
                metrics=metrics,
                stage_results=stage_results
            )
        finally:
            if pipeline_span:
                pipeline_span.__exit__(None, None, None)
    
    def _execute_parallel(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any],
        request_id: str
    ) -> EnrichmentResult:
        """Execute pipeline stages with parallel processing based on dependencies."""
        # Create span for the whole pipeline execution
        if self.telemetry_enabled:
            pipeline_span = self.telemetry.create_span(
                name="enrichment_pipeline.execute_parallel",
                attributes={"request_id": request_id}
            )
            pipeline_span.__enter__()
        else:
            pipeline_span = None
        
        try:
            start_time = time.time()
            result_data = dict(input_data)
            errors = {}
            metrics = {"timing": {}}
            stage_results = {}
            
            # Get execution levels from dependency graph
            try:
                execution_levels = self.dependency_graph.get_execution_levels()
            except ValueError as e:
                logger.error(f"Error determining execution order: {str(e)}, falling back to sequential execution")
                return self._execute_sequential(input_data, context, request_id)
            
            # Process stages level by level (stages within a level can be parallelized)
            for level_idx, level in enumerate(execution_levels):
                logger.info(f"Executing pipeline level {level_idx+1}/{len(execution_levels)} with {len(level)} stages")
                
                # Create futures for all stages in this level
                futures = {}
                for stage_name in level:
                    stage = self._find_stage_by_name(stage_name)
                    if not stage:
                        logger.error(f"Stage '{stage_name}' not found in pipeline configuration")
                        continue
                    
                    # Check if any dependencies failed
                    dependency_failed = False
                    for dep_name in stage.dependencies:
                        if dep_name in errors and self._find_stage_by_name(dep_name).required:
                            logger.warning(f"Skipping stage '{stage_name}' due to failed dependency '{dep_name}'")
                            errors[stage_name] = f"Dependency '{dep_name}' failed"
                            dependency_failed = True
                            break
                    
                    if dependency_failed:
                        continue
                    
                    # Submit stage execution to thread pool
                    future = self.executor.submit(
                        self._execute_stage,
                        stage, 
                        result_data.copy(),  # Create a copy of current data
                        context,
                        request_id,
                        errors,
                        metrics,
                        stage_results
                    )
                    futures[stage_name] = future
                
                # Wait for all futures to complete
                for stage_name, future in futures.items():
                    try:
                        stage_result = future.result()
                        
                        # Update result data with stage result if we got one
                        if stage_result:
                            self._update_result_data(stage_name, stage_result, result_data)
                            
                    except Exception as e:
                        logger.error(f"Unhandled error in stage '{stage_name}': {str(e)}")
                        errors[stage_name] = f"Unhandled execution error: {str(e)}"
                        
                        # If this is a required stage with error strategy, fail the pipeline
                        stage = self._find_stage_by_name(stage_name)
                        if stage and stage.required and stage.fallback_strategy == "error":
                            logger.error(f"Required stage '{stage_name}' failed with error strategy, stopping pipeline")
                            
                            return EnrichmentResult(
                                success=False,
                                data=result_data,
                                errors=errors,
                                metrics=metrics,
                                stage_results=stage_results
                            )
            
            # Calculate total time
            total_time = time.time() - start_time
            metrics["timing"]["total"] = total_time
            
            return EnrichmentResult(
                success=len(errors) == 0 or all(
                    not self._find_stage_by_name(stage_name).required 
                    for stage_name in errors.keys()
                ),
                data=result_data,
                errors=errors,
                metrics=metrics,
                stage_results=stage_results
            )
        finally:
            if pipeline_span:
                pipeline_span.__exit__(None, None, None)
    
    def _execute_stage(
        self,
        stage: EnrichmentStage, 
        result_data: Dict[str, Any],
        context: Dict[str, Any],
        request_id: str,
        errors: Dict[str, str],
        metrics: Dict[str, Any],
        stage_results: Dict[str, Any]
    ) -> Any:
        """
        Execute a single pipeline stage with instrumentation and error handling.
        
        Args:
            stage: Stage configuration
            result_data: Current result data
            context: Processing context
            request_id: Request identifier
            errors: Dictionary to collect errors
            metrics: Dictionary to collect metrics
            stage_results: Dictionary to collect stage results
            
        Returns:
            Stage execution result or None if execution failed
        """
        # Track concurrency
        if self.telemetry_enabled:
            self.metrics.gauge_inc("pipeline.concurrent_stages")
        
        # Create a span for this stage
        if self.telemetry_enabled:
            stage_span = self.telemetry.create_span(
                name=f"enrichment_pipeline.stage.{stage.name}",
                attributes={"request_id": request_id}
            )
            stage_span.__enter__()
        else:
            stage_span = None
            
        # Acquire concurrency semaphore if specified
        semaphore = self.concurrency_semaphores.get(stage.concurrency_key)
        if semaphore:
            semaphore_acquired = semaphore.acquire(timeout=10)  # 10 second timeout
            if not semaphore_acquired:
                logger.warning(f"Failed to acquire semaphore for stage '{stage.name}', executing anyway")
        else:
            semaphore_acquired = False
            
        stage_start = time.time()
        stage_result = None
        
        try:
            # Check cache first if enabled
            if self.enable_caching and stage.cache_result:
                cache_key = self._generate_cache_key(stage.name, result_data, context)
                cached_result = self._get_from_cache(cache_key)
                
                if cached_result:
                    if self.telemetry_enabled:
                        self.metrics.counter_inc("pipeline.cache_hits")
                    logger.info(f"Cache hit for stage '{stage.name}' with key {cache_key[:8]}...")
                    stage_result = cached_result
                    return stage_result
                
                if self.telemetry_enabled:
                    self.metrics.counter_inc("pipeline.cache_misses")
            
            # Get stage instance
            instance = self.stage_instances.get(stage.name)
            
            if not instance:
                if stage.required:
                    if stage.fallback_strategy == "error":
                        logger.error(f"Required stage '{stage.name}' not initialized")
                        errors[stage.name] = "Stage not initialized"
                        return None
                    elif stage.fallback_strategy == "default":
                        # Try to create a default instance
                        instance = create_instance(plugin_type=stage.plugin_type)
                        if not instance:
                            logger.error(f"Failed to create default instance for stage '{stage.name}'")
                            errors[stage.name] = "Failed to create default instance"
                            return None
                    else:  # skip
                        logger.warning(f"Skipping uninitialized stage '{stage.name}'")
                        errors[stage.name] = "Stage not initialized, skipped"
                        return None
                else:
                    logger.info(f"Skipping optional stage '{stage.name}'")
                    return None
            
            # Execute stage with timeout if specified
            if stage.execution_timeout:
                # Use executor with timeout
                future = self.executor.submit(self._execute_stage_instance, instance, stage.name, result_data, context)
                try:
                    stage_result = future.result(timeout=stage.execution_timeout)
                    stage_results[stage.name] = stage_result
                except concurrent.futures.TimeoutError:
                    logger.error(f"Stage '{stage.name}' timed out after {stage.execution_timeout} seconds")
                    errors[stage.name] = f"Execution timed out after {stage.execution_timeout}s"
                    return None
            else:
                # Execute directly
                stage_result = self._execute_stage_instance(instance, stage.name, result_data, context)
                stage_results[stage.name] = stage_result
            
            # Cache result if enabled and execution succeeded
            if self.enable_caching and stage.cache_result and stage_result:
                cache_key = self._generate_cache_key(stage.name, result_data, context)
                self._store_in_cache(cache_key, stage_result)
                
            return stage_result
                
        except Exception as e:
            logger.error(f"Error executing stage '{stage.name}': {str(e)}")
            errors[stage.name] = str(e)
            
            if self.telemetry_enabled:
                self.metrics.counter_inc("pipeline.errors_total")
                
            return None
            
        finally:
            # Record timing
            stage_time = time.time() - stage_start
            metrics["timing"][stage.name] = stage_time
            
            if self.telemetry_enabled:
                self.metrics.histogram_observe("pipeline.stage_duration_seconds", stage_time)
                self.metrics.gauge_dec("pipeline.concurrent_stages")
                
            # Release semaphore if acquired
            if semaphore_acquired and semaphore:
                semaphore.release()
                
            # End span
            if stage_span:
                stage_span.__exit__(None, None, None)
    
    def _execute_stage_instance(
        self,
        instance: Any,
        stage_name: str,
        result_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Any:
        """Execute the actual stage instance with proper interface handling."""
        if isinstance(instance, EnrichmentInterface):
            # Use the standard EnrichmentInterface
            return instance.enrich(result_data, context)
        else:
            # For custom stage implementations
            stage_method = getattr(instance, "process", None)
            if callable(stage_method):
                return stage_method(result_data, context)
            else:
                # Try to find any method that matches our signature
                for method_name in ["execute", "run", "process_data", "enrich_data", "__call__"]:
                    method = getattr(instance, method_name, None)
                    if callable(method):
                        return method(result_data, context)
                        
                raise ValueError(f"Stage '{stage_name}' does not have a valid processing method")
    
    def _update_result_data(
        self,
        stage_name: str,
        stage_result: Any,
        result_data: Dict[str, Any]
    ) -> None:
        """Update the result data with a stage result."""
        if stage_result:
            # For results with a to_dict method
            if hasattr(stage_result, 'to_dict') and callable(getattr(stage_result, 'to_dict')):
                result_data[stage_name] = stage_result.to_dict()
            # For dataclass instances
            elif hasattr(stage_result, '__dataclass_fields__'):
                result_data[stage_name] = asdict(stage_result)
            # For other types
            else:
                result_data[stage_name] = stage_result
    
    def _find_stage_by_name(self, name: str) -> Optional[EnrichmentStage]:
        """Find a stage configuration by name."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None
    
    def _generate_cache_key(
        self,
        stage_name: str,
        data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate a cache key for a stage execution."""
        # Import here to avoid circular imports
        import hashlib
        
        # Extract cacheable parts of the context
        cacheable_context = {}
        for field in ["user_id", "tenant_id", "parameters"]:
            if field in context:
                cacheable_context[field] = context[field]
        
        # Create a deterministic representation of the inputs
        try:
            # Use JSON for most types
            key_data = {
                "stage": stage_name,
                "data": data,
                "context": cacheable_context
            }
            json_str = json.dumps(key_data, sort_keys=True)
            return hashlib.md5(json_str.encode()).hexdigest()
        except (TypeError, ValueError):
            # Fallback for non-serializable content
            return f"{stage_name}_{id(data)}_{id(context)}"
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache."""
        if not self.enable_caching:
            return None
            
        with self.cache_lock:
            entry = self.result_cache.get(key)
            if entry:
                # Check expiration
                if time.time() - entry.get("timestamp", 0) < entry.get("ttl", 300):
                    return entry.get("value")
                # Expired, remove it
                del self.result_cache[key]
                
        return None
    
    def _store_in_cache(self, key: str, value: Any, ttl: float = 300) -> None:
        """Store a value in the cache."""
        if not self.enable_caching:
            return
            
        with self.cache_lock:
            # Implement simple LRU if we're at capacity
            if len(self.result_cache) >= 1000:  # Hard limit of 1000 entries
                # Find and remove oldest entry
                oldest_key = min(
                    self.result_cache.items(),
                    key=lambda x: x[1].get("timestamp", 0)
                )[0]
                del self.result_cache[oldest_key]
                
            self.result_cache[key] = {
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl
            }
    
    def reset(self) -> None:
        """Reset the pipeline, clearing all stages and cache."""
        self.stages = []
        self.stage_instances = {}
        
        if self.enable_caching:
            with self.cache_lock:
                self.result_cache.clear()
                
        logger.info("Enrichment pipeline reset")
    
    async def execute_async(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> EnrichmentResult:
        """
        Execute the enrichment pipeline asynchronously.
        
        Args:
            input_data: Input data to enrich
            context: Optional context information
            
        Returns:
            Result of enrichment pipeline execution
        """
        # Get current event loop
        loop = asyncio.get_running_loop()
        
        # Run in executor to avoid blocking
        return await loop.run_in_executor(
            self.executor,
            self.execute,
            input_data,
            context
        )


# Factory functions to create common pipeline configurations
def create_standard_pipeline() -> EnrichmentPipeline:
    """
    Create a standard enrichment pipeline with common stages.
    
    Returns:
        Configured enrichment pipeline
    """
    pipeline = EnrichmentPipeline(
        stages=[
            EnrichmentStage(
                name="vector_alignment",
                plugin_type=PluginType.VECTOR_ALIGNMENT,
                required=True,
                fallback_strategy="default",
                cache_result=True
            ),
            EnrichmentStage(
                name="ar_enrichment",
                plugin_type=PluginType.ENRICHMENT,
                plugin_name="ar_vector_enricher",
                required=True,
                fallback_strategy="error",
                dependencies=["vector_alignment"],
                cache_result=True
            )
        ],
        execution_mode=ExecutionMode.ADAPTIVE
    )
    
    return pipeline

def create_advanced_pipeline() -> EnrichmentPipeline:
    """
    Create an advanced enrichment pipeline with parallel processing support.
    
    Returns:
        Configured enrichment pipeline with advanced stages
    """
    pipeline = EnrichmentPipeline(
        stages=[
            # Base Vector Stage
            EnrichmentStage(
                name="vector_alignment",
                plugin_type=PluginType.VECTOR_ALIGNMENT,
                required=True,
                fallback_strategy="default",
                cache_result=True
            ),
            # Core Enrichment Stages
            EnrichmentStage(
                name="ar_enrichment",
                plugin_type=PluginType.ENRICHMENT,
                plugin_name="ar_vector_enricher",
                required=True,
                fallback_strategy="error",
                dependencies=["vector_alignment"],
                cache_result=True
            ),
            EnrichmentStage(
                name="drift_detection",
                plugin_type=PluginType.ENRICHMENT,
                plugin_name="drift_detector",
                required=False,
                dependencies=["vector_alignment"],
                cache_result=True
            ),
            # Advanced Processing Stages
            EnrichmentStage(
                name="constraint_validation",
                plugin_type=PluginType.CONSTRAINT,
                required=False,
                dependencies=["ar_enrichment"],
                execution_timeout=2.0  # 2 second timeout
            ),
            EnrichmentStage(
                name="confidence_scoring",
                plugin_type=PluginType.ENRICHMENT,
                plugin_name="confidence_scorer",
                required=True,
                dependencies=["ar_enrichment", "constraint_validation"],
                fallback_strategy="default"
            )
        ],
        execution_mode=ExecutionMode.ADAPTIVE,
        telemetry_enabled=True,
        cache_config={"enabled": True}
    )
    
    return pipeline
