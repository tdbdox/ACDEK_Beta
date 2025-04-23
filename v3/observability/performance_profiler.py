"""
PerformanceProfiler for ACDEK v3: Comprehensive performance measurement system.

This module provides a distributed profiling system for measuring execution time
across all ACDEK v3 pipeline stages, identifying bottlenecks, and tracking
scaling efficiency metrics.
"""
import time
import threading
import statistics
import json
import os
import logging
import sys
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
import psutil
import uuid
from enum import Enum

# Configure comprehensive logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add handler if not already added to ensure logs are properly captured
if not logger.handlers:
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info("PerformanceProfiler logger initialized")

# Default thresholds - can be overridden via config
DEFAULT_THRESHOLDS = {
    "critical": 200.0,   # milliseconds
    "warning": 100.0,    # milliseconds
    "info": 50.0,        # milliseconds
    "trace": 10.0        # milliseconds
}

class LatencyCategory(Enum):
    """Categories for classifying latency measurements."""
    CRITICAL = "critical"     # Unacceptable latency
    WARNING = "warning"       # Concerning latency that needs attention
    INFO = "info"             # Normal but notable latency
    TRACE = "trace"           # Expected and acceptable latency
    EXCELLENT = "excellent"   # Better than expected latency

@dataclass
class ResourceUsage:
    """Resource usage information for a profiled operation."""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    thread_count: int = 0
    io_read_mb: float = 0.0
    io_write_mb: float = 0.0

@dataclass
class StageProfile:
    """Detailed profile information for a pipeline stage."""
    name: str
    count: int = 0
    total_time: float = 0.0
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    avg_time: float = 0.0
    median_time: Optional[float] = None
    p95_time: Optional[float] = None
    p99_time: Optional[float] = None
    std_dev: Optional[float] = None
    category: LatencyCategory = LatencyCategory.TRACE
    resources: ResourceUsage = field(default_factory=ResourceUsage)
    execution_times: List[float] = field(default_factory=list)
    parent_stages: Set[str] = field(default_factory=set)
    child_stages: Set[str] = field(default_factory=set)
    errors: int = 0
    last_execution_time: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_execution(self, time_ms: float, resources: Optional[ResourceUsage] = None):
        """Add an execution time measurement to this stage profile."""
        self.count += 1
        self.total_time += time_ms
        self.last_execution_time = time_ms
        self.execution_times.append(time_ms)
        
        # Update min/max
        if self.min_time is None or time_ms < self.min_time:
            self.min_time = time_ms
        if self.max_time is None or time_ms > self.max_time:
            self.max_time = time_ms
            
        # Update avg
        self.avg_time = self.total_time / self.count
        
        # Update resource usage if provided
        if resources:
            self.resources = resources
    
    def compute_statistics(self):
        """Compute detailed statistics for this stage profile."""
        if not self.execution_times:
            return
            
        # Compute percentiles and standard deviation
        if len(self.execution_times) >= 2:
            try:
                self.median_time = statistics.median(self.execution_times)
                self.std_dev = statistics.stdev(self.execution_times)
                
                # Only compute percentiles if we have enough data
                if len(self.execution_times) >= 10:
                    sorted_times = sorted(self.execution_times)
                    idx_95 = int(len(sorted_times) * 0.95)
                    idx_99 = int(len(sorted_times) * 0.99)
                    self.p95_time = sorted_times[idx_95]
                    self.p99_time = sorted_times[idx_99]
            except Exception as e:
                logger.warning(f"Error computing statistics for stage {self.name}: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding the raw execution times."""
        result = asdict(self)
        # Don't include the full list of execution times in the result
        result.pop('execution_times', None)
        # Convert enum to string
        result['category'] = self.category.value
        # Convert sets to lists
        result['parent_stages'] = list(self.parent_stages)
        result['child_stages'] = list(self.child_stages)
        return result

@dataclass
class ProfileSnapshot:
    """A snapshot of the entire performance profile at a point in time."""
    timestamp: float
    request_id: str
    stage_profiles: Dict[str, Dict[str, Any]]
    system_load: Dict[str, float]
    thread_count: int
    total_time: float
    total_cpu_time: float
    bottlenecks: List[Dict[str, Any]]
    thresholds: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "stage_profiles": self.stage_profiles,
            "system_load": self.system_load,
            "thread_count": self.thread_count,
            "total_time": self.total_time,
            "total_cpu_time": self.total_cpu_time,
            "bottlenecks": self.bottlenecks,
            "thresholds": self.thresholds
        }
    
    def to_json(self) -> str:
        """Convert to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

@dataclass
class ScalingMetrics:
    """Metrics for tracking scaling efficiency."""
    concurrent_requests: int = 0
    avg_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    resource_utilization: float = 0.0
    efficiency_score: float = 0.0
    saturation_point: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

class PerformanceProfiler:
    """
    Distributed performance profiling system for ACDEK v3.
    
    This class provides comprehensive performance measurement capabilities:
    - Execution time tracking across all pipeline stages
    - Resource utilization monitoring (CPU, memory, I/O)
    - Bottleneck detection with recommendations
    - Scaling efficiency analysis
    - Performance visualization data generation
    """
    def __init__(
        self,
        instance_id: Optional[str] = None,
        enable_resource_monitoring: bool = True,
        max_history_size: int = 1000,
        thresholds: Optional[Dict[str, float]] = None,
        export_directory: Optional[str] = None
    ):
        """
        Initialize the performance profiler.
        
        Args:
            instance_id: Unique identifier for this profiler instance
            enable_resource_monitoring: Whether to track CPU, memory usage
            max_history_size: Maximum number of stage executions to retain
            thresholds: Latency thresholds for categorization in milliseconds
            export_directory: Directory for exporting performance data
        """
        logger.info("Initializing PerformanceProfiler...")
        
        # Generate ID if not provided
        self.instance_id = instance_id or str(uuid.uuid4())
        logger.info(f"Performance profiler instance ID: {self.instance_id}")
        
        # Configure settings
        self.enable_resource_monitoring = enable_resource_monitoring
        logger.debug(f"Resource monitoring: {'enabled' if enable_resource_monitoring else 'disabled'}")
        
        self.max_history_size = max_history_size
        logger.debug(f"Max history size: {max_history_size} executions")
        
        # Set up thresholds with validation
        self.thresholds = DEFAULT_THRESHOLDS.copy()
        if thresholds:
            logger.debug(f"Custom thresholds provided: {thresholds}")
            # Validate thresholds
            valid_threshold_keys = set(DEFAULT_THRESHOLDS.keys())
            for key, value in thresholds.items():
                if key not in valid_threshold_keys:
                    logger.warning(f"Unknown threshold key '{key}' - will be ignored")
                elif not isinstance(value, (int, float)) or value < 0:
                    logger.warning(f"Invalid threshold value for '{key}': {value} - using default")
                else:
                    self.thresholds[key] = value
        
        logger.debug(f"Active thresholds: {self.thresholds}")
        
        # Set up export directory with validation
        self.export_directory = export_directory
        if export_directory:
            # Validate export directory
            if not os.path.exists(export_directory):
                try:
                    os.makedirs(export_directory, exist_ok=True)
                    logger.info(f"Created export directory: {export_directory}")
                except Exception as e:
                    logger.error(f"Failed to create export directory {export_directory}: {e}")
                    self.export_directory = None  # Default to None if creation fails
            
            logger.debug(f"Export directory: {self.export_directory}")
        
        # Initialize threading and storage structures
        logger.debug("Initializing internal data structures")
        self._lock = threading.RLock()
        self._stage_profiles: Dict[str, StageProfile] = {}
        self._request_profiles: Dict[str, Dict[str, Any]] = {}
        self._stage_dependencies: Dict[str, Set[str]] = {}
        self._execution_stack: Dict[int, List[str]] = {}
        
        # Initialize scaling metrics
        logger.debug("Initializing scaling metrics")
        self._scaling_metrics = ScalingMetrics()
        self._concurrent_requests = 0
        self._request_start_times: Dict[str, float] = {}
        self._request_timings: List[float] = []
        
        # Initialize system monitoring
        if self.enable_resource_monitoring:
            logger.debug("Initializing system resource monitoring")
            try:
                self._init_process = psutil.Process()
                logger.debug(f"Process ID: {self._init_process.pid}")
                
                if hasattr(self._init_process, 'io_counters'):
                    self._io_counters_last = self._init_process.io_counters()
                    logger.debug("I/O counters initialized")
                else:
                    logger.warning("I/O counters not available for this platform - I/O monitoring disabled")
                    self._io_counters_last = None
                
                self._io_last_time = time.time()
            except Exception as e:
                logger.error(f"Failed to initialize psutil process monitoring: {e}")
                logger.warning("Resource monitoring will be disabled")
                self.enable_resource_monitoring = False
                self._init_process = None
                self._io_counters_last = None
        
        logger.info(f"PerformanceProfiler initialization complete - ID: {self.instance_id}, Monitoring: {'enabled' if self.enable_resource_monitoring else 'disabled'}")
    
    @contextmanager
    def profile_stage(self, stage_name: str, request_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling a pipeline stage execution.
        
        Args:
            stage_name: Name of the stage being profiled
            request_id: Optional request identifier for correlation
            metadata: Optional metadata about the stage execution
            
        Yields:
            None
        """
        thread_id = threading.get_ident()
        logger.debug(f"Starting profile for stage '{stage_name}' on thread {thread_id} [request: {request_id or 'none'}]")
        
        # Get start resource usage if enabled
        start_resources = None
        if self.enable_resource_monitoring:
            logger.debug(f"Collecting start resource usage for stage '{stage_name}'")
            start_resources = self._get_resource_usage()
            if start_resources:
                logger.debug(f"Stage '{stage_name}' starting resources - CPU: {start_resources.cpu_percent:.1f}%, Memory: {start_resources.memory_mb:.1f}MB")
        
        # Record entry in execution stack to track parent-child relationships
        parent_stage = None
        with self._lock:
            # Initialize execution stack for this thread if not present
            if thread_id not in self._execution_stack:
                logger.debug(f"Initializing execution stack for thread {thread_id}")
                self._execution_stack[thread_id] = []
            
            # Find parent stage (if any)
            if self._execution_stack[thread_id]:
                parent_stage = self._execution_stack[thread_id][-1]
                logger.debug(f"Stage '{stage_name}' has parent stage '{parent_stage}'")
            else:
                logger.debug(f"Stage '{stage_name}' is a root stage (no parent)")
                
            # Register dependency if we have a parent
            if parent_stage:
                logger.debug(f"Registering dependency: '{parent_stage}' -> '{stage_name}'")
                self._register_dependency(parent_stage, stage_name)
                
            # Add current stage to execution stack
            self._execution_stack[thread_id].append(stage_name)
            logger.debug(f"Added '{stage_name}' to execution stack for thread {thread_id}")
        
        # Start timing
        start_time = time.time()
        logger.debug(f"Stage '{stage_name}' execution started at {start_time:.6f}")
        
        try:
            # Execute the profiled code
            logger.debug(f"Executing profiled code for stage '{stage_name}'")
            yield
            logger.debug(f"Stage '{stage_name}' executed successfully")
        except Exception as e:
            # Record error but re-raise
            logger.warning(f"Error in stage '{stage_name}': {type(e).__name__}: {str(e)}")
            with self._lock:
                if stage_name in self._stage_profiles:
                    self._stage_profiles[stage_name].errors += 1
                    logger.debug(f"Incremented error count for stage '{stage_name}' to {self._stage_profiles[stage_name].errors}")
            raise
        finally:
            # End timing and calculate duration in milliseconds
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000.0
            logger.debug(f"Stage '{stage_name}' completed in {duration_ms:.3f}ms")
            
            # Get end resource usage if enabled
            end_resources = None
            if self.enable_resource_monitoring:
                logger.debug(f"Collecting end resource usage for stage '{stage_name}'")
                end_resources = self._get_resource_usage()
                if end_resources:
                    logger.debug(f"Stage '{stage_name}' ending resources - CPU: {end_resources.cpu_percent:.1f}%, Memory: {end_resources.memory_mb:.1f}MB")
                
            # Calculate resource delta
            resource_delta = None
            if start_resources and end_resources:
                logger.debug(f"Calculating resource usage delta for stage '{stage_name}'")
                resource_delta = ResourceUsage(
                    cpu_percent=end_resources.cpu_percent,
                    memory_mb=end_resources.memory_mb,
                    thread_count=end_resources.thread_count,
                    io_read_mb=end_resources.io_read_mb - start_resources.io_read_mb,
                    io_write_mb=end_resources.io_write_mb - start_resources.io_write_mb
                )
                logger.debug(f"Stage '{stage_name}' resource delta - CPU: {resource_delta.cpu_percent:.1f}%, " +
                           f"Memory: {resource_delta.memory_mb:.1f}MB, I/O read: {resource_delta.io_read_mb:.1f}MB, " +
                           f"I/O write: {resource_delta.io_write_mb:.1f}MB")
            
            # Record the profiling data
            with self._lock:
                # Create stage profile if first execution
                if stage_name not in self._stage_profiles:
                    logger.debug(f"Creating new stage profile for '{stage_name}'")
                    self._stage_profiles[stage_name] = StageProfile(name=stage_name)
                
                # Add this execution to the profile
                prev_count = self._stage_profiles[stage_name].count
                self._stage_profiles[stage_name].add_execution(duration_ms, resource_delta)
                logger.debug(f"Added execution #{prev_count + 1} to stage '{stage_name}' - duration: {duration_ms:.3f}ms")
                
                # Categorize the latency
                prev_category = self._stage_profiles[stage_name].category
                self._categorize_latency(stage_name, duration_ms)
                current_category = self._stage_profiles[stage_name].category
                
                if prev_category != current_category:
                    logger.info(f"Stage '{stage_name}' latency category changed: {prev_category.value} -> {current_category.value}")
                    if current_category in [LatencyCategory.WARNING, LatencyCategory.CRITICAL]:
                        logger.warning(f"Stage '{stage_name}' latency ({duration_ms:.2f}ms) exceeds threshold for {current_category.value}")
                
                # Record in request profile if we have a request ID
                if request_id:
                    # Create request profile if first stage for this request
                    if request_id not in self._request_profiles:
                        logger.debug(f"Creating new request profile for request '{request_id}'")
                        self._request_profiles[request_id] = {
                            "stages": {},
                            "start_time": start_time,
                            "metadata": metadata or {}
                        }
                    
                    # Add stage execution to request profile
                    self._request_profiles[request_id]["stages"][stage_name] = {
                        "duration_ms": duration_ms,
                        "start_time": start_time,
                        "end_time": end_time
                    }
                    logger.debug(f"Added stage '{stage_name}' to request profile '{request_id}'")
                
                # Remove stage from execution stack
                if thread_id in self._execution_stack and self._execution_stack[thread_id]:
                    removed_stage = self._execution_stack[thread_id].pop()
                    logger.debug(f"Removed '{removed_stage}' from execution stack for thread {thread_id}")
                    # Validate the popped stage matches expected
                    if removed_stage != stage_name:
                        logger.warning(f"Execution stack inconsistency: expected '{stage_name}' but popped '{removed_stage}'")
            
            # Log completion based on performance category
            if current_category == LatencyCategory.EXCELLENT:
                logger.debug(f"Stage '{stage_name}' completed with excellent performance: {duration_ms:.2f}ms")
            elif current_category == LatencyCategory.TRACE:
                logger.debug(f"Stage '{stage_name}' completed with acceptable performance: {duration_ms:.2f}ms")
            elif current_category == LatencyCategory.INFO:
                logger.info(f"Stage '{stage_name}' completed with notable latency: {duration_ms:.2f}ms")
            elif current_category == LatencyCategory.WARNING:
                logger.warning(f"Stage '{stage_name}' completed with concerning latency: {duration_ms:.2f}ms")
            elif current_category == LatencyCategory.CRITICAL:
                logger.warning(f"Stage '{stage_name}' completed with critical latency: {duration_ms:.2f}ms")
    
    @contextmanager
    def profile_request(self, request_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Context manager for profiling a complete request execution.
        
        Args:
            request_id: Optional request identifier
            metadata: Optional metadata about the request
            
        Yields:
            str: The request ID (generated if not provided)
        """
        # Generate request ID if not provided
        req_id = request_id or f"req_{uuid.uuid4()}"
        logger.info(f"Starting request profiling for request ID: {req_id}")
        
        if metadata:
            logger.debug(f"Request metadata: {metadata}")
        
        # Increment concurrent request counter
        with self._lock:
            self._concurrent_requests += 1
            prev_concurrent = self._concurrent_requests - 1
            self._scaling_metrics.concurrent_requests = self._concurrent_requests
            logger.debug(f"Incremented concurrent requests: {prev_concurrent} -> {self._concurrent_requests}")
            
            self._request_start_times[req_id] = time.time()
            logger.debug(f"Recorded start time for request {req_id}: {self._request_start_times[req_id]:.6f}")
            
            # Log high concurrency as potential scaling concern
            if self._concurrent_requests > 10:
                logger.warning(f"High concurrency detected: {self._concurrent_requests} concurrent requests")
        
        success = False
        try:
            # Start request profiling using "request_total" as the outermost stage
            logger.debug(f"Starting request_total stage for request {req_id}")
            with self.profile_stage("request_total", req_id, metadata):
                yield req_id
            
            # If we got here without an exception, the request was successful
            success = True
            logger.info(f"Request {req_id} completed successfully")
            
        except Exception as e:
            # Record error in scaling metrics
            logger.warning(f"Error processing request {req_id}: {type(e).__name__}: {str(e)}")
            with self._lock:
                # Update error rate with weighted average
                prev_error_rate = self._scaling_metrics.error_rate
                self._scaling_metrics.error_rate = (
                    (prev_error_rate * (self._concurrent_requests - 1) + 1.0) / 
                    self._concurrent_requests
                )
                logger.debug(f"Updated error rate: {prev_error_rate:.4f} -> {self._scaling_metrics.error_rate:.4f}")
                
                # Log high error rates as a concern
                if self._scaling_metrics.error_rate > 0.1:
                    logger.warning(f"High error rate detected: {self._scaling_metrics.error_rate:.2%}")
            raise
        finally:
            # Decrement concurrent request counter and update metrics
            with self._lock:
                logger.debug(f"Finalizing request metrics for {req_id}")
                
                # Calculate request duration
                if req_id in self._request_start_times:
                    start_time = self._request_start_times[req_id]
                    current_time = time.time()
                    duration = current_time - start_time
                    # Convert to milliseconds
                    duration_ms = duration * 1000.0
                    
                    logger.debug(f"Request {req_id} duration: {duration_ms:.2f}ms (start: {start_time:.6f}, end: {current_time:.6f})")
                    
                    # Add to request timings for throughput calculation
                    self._request_timings.append(duration_ms)
                    logger.debug(f"Added duration to request timings (count: {len(self._request_timings)})")
                    
                    # Keep only the last 100 request timings
                    if len(self._request_timings) > 100:
                        self._request_timings = self._request_timings[-100:]
                        logger.debug("Trimmed request timings to last 100 entries")
                    
                    # Update average latency in scaling metrics
                    if self._request_timings:
                        prev_avg = self._scaling_metrics.avg_latency_ms
                        self._scaling_metrics.avg_latency_ms = sum(self._request_timings) / len(self._request_timings)
                        logger.debug(f"Updated average latency: {prev_avg:.2f}ms -> {self._scaling_metrics.avg_latency_ms:.2f}ms")
                        
                        # Log significant latency changes
                        if prev_avg > 0 and abs(self._scaling_metrics.avg_latency_ms - prev_avg) / prev_avg > 0.2:
                            change_pct = (self._scaling_metrics.avg_latency_ms - prev_avg) / prev_avg * 100
                            change_direction = "increased" if change_pct > 0 else "decreased"
                            logger.info(f"Average latency {change_direction} by {abs(change_pct):.1f}%")
                    
                    # Calculate throughput (requests per second)
                    if self._scaling_metrics.avg_latency_ms > 0:
                        prev_throughput = self._scaling_metrics.throughput_rps
                        self._scaling_metrics.throughput_rps = 1000.0 / self._scaling_metrics.avg_latency_ms
                        logger.debug(f"Updated throughput: {prev_throughput:.2f} -> {self._scaling_metrics.throughput_rps:.2f} req/sec")
                    
                    # Remove from start times
                    del self._request_start_times[req_id]
                    logger.debug(f"Removed request {req_id} from start times")
                    
                    # Log long-running requests
                    if duration_ms > 5000:  # 5 seconds
                        logger.warning(f"Long-running request detected - Request {req_id} took {duration_ms:.2f}ms")
                else:
                    logger.warning(f"Request {req_id} not found in start times - unable to calculate duration")
                
                # Update concurrent requests count
                prev_concurrent = self._concurrent_requests
                self._concurrent_requests = max(0, self._concurrent_requests - 1)
                self._scaling_metrics.concurrent_requests = self._concurrent_requests
                logger.debug(f"Decremented concurrent requests: {prev_concurrent} -> {self._concurrent_requests}")
                
                # Update resource utilization if enabled
                if self.enable_resource_monitoring:
                    try:
                        prev_utilization = self._scaling_metrics.resource_utilization
                        self._scaling_metrics.resource_utilization = psutil.cpu_percent(interval=None) / 100.0
                        logger.debug(f"Updated resource utilization: {prev_utilization:.2f} -> {self._scaling_metrics.resource_utilization:.2f}")
                        
                        # Log high resource utilization
                        if self._scaling_metrics.resource_utilization > 0.8:  # 80%
                            logger.warning(f"High resource utilization: {self._scaling_metrics.resource_utilization:.1%}")
                    except Exception as e:
                        logger.warning(f"Failed to get CPU percent: {str(e)}")
                
                # Calculate efficiency score (throughput / resource utilization)
                if self._scaling_metrics.resource_utilization > 0:
                    prev_efficiency = self._scaling_metrics.efficiency_score
                    self._scaling_metrics.efficiency_score = (
                        self._scaling_metrics.throughput_rps / 
                        self._scaling_metrics.resource_utilization
                    )
                    logger.debug(f"Updated efficiency score: {prev_efficiency:.2f} -> {self._scaling_metrics.efficiency_score:.2f}")
                    
                    # Log efficiency degradation
                    if prev_efficiency > 0 and self._scaling_metrics.efficiency_score < prev_efficiency * 0.7:
                        logger.warning(f"Efficiency degradation detected: {prev_efficiency:.2f} -> {self._scaling_metrics.efficiency_score:.2f}")
            
            # Log request completion
            status = "succeeded" if success else "failed"
            logger.info(f"Request profiling for {req_id} {status} - metrics updated")
    
    def _register_dependency(self, parent_stage: str, child_stage: str):
        """Register a dependency between two stages."""
        # Create the stage profiles if they don't exist
        if parent_stage not in self._stage_profiles:
            self._stage_profiles[parent_stage] = StageProfile(name=parent_stage)
        if child_stage not in self._stage_profiles:
            self._stage_profiles[child_stage] = StageProfile(name=child_stage)
        
        # Register the dependency
        self._stage_profiles[parent_stage].child_stages.add(child_stage)
        self._stage_profiles[child_stage].parent_stages.add(parent_stage)
    
    def _categorize_latency(self, stage_name: str, duration_ms: float):
        """Categorize a latency measurement based on thresholds."""
        if duration_ms >= self.thresholds.get("critical", DEFAULT_THRESHOLDS["critical"]):
            category = LatencyCategory.CRITICAL
        elif duration_ms >= self.thresholds.get("warning", DEFAULT_THRESHOLDS["warning"]):
            category = LatencyCategory.WARNING
        elif duration_ms >= self.thresholds.get("info", DEFAULT_THRESHOLDS["info"]):
            category = LatencyCategory.INFO
        elif duration_ms >= self.thresholds.get("trace", DEFAULT_THRESHOLDS["trace"]):
            category = LatencyCategory.TRACE
        else:
            category = LatencyCategory.EXCELLENT
        
        self._stage_profiles[stage_name].category = category
    
    def _get_resource_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        usage = ResourceUsage()
        
        if not self.enable_resource_monitoring:
            return usage
        
        try:
            # Get process info
            process = self._init_process
            
            # CPU usage
            usage.cpu_percent = process.cpu_percent(interval=None)
            
            # Memory usage in MB
            memory_info = process.memory_info()
            usage.memory_mb = memory_info.rss / (1024 * 1024)
            
            # Thread count
            usage.thread_count = len(process.threads())
            
            # I/O usage
            if hasattr(process, 'io_counters'):
                io_counters = process.io_counters()
                current_time = time.time()
                
                if self._io_counters_last and current_time > self._io_last_time:
                    # Calculate I/O rate in MB/s
                    time_delta = current_time - self._io_last_time
                    
                    read_bytes = io_counters.read_bytes - self._io_counters_last.read_bytes
                    write_bytes = io_counters.write_bytes - self._io_counters_last.write_bytes
                    
                    usage.io_read_mb = (read_bytes / (1024 * 1024)) / time_delta
                    usage.io_write_mb = (write_bytes / (1024 * 1024)) / time_delta
                
                # Update last values
                self._io_counters_last = io_counters
                self._io_last_time = current_time
                
        except Exception as e:
            logger.warning(f"Error getting resource usage: {str(e)}")
        
        return usage
    
    def get_stage_profile(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """
        Get profile information for a specific stage.
        
        Args:
            stage_name: Name of the stage to get profile for
            
        Returns:
            Stage profile dictionary or None if not found
        """
        with self._lock:
            if stage_name in self._stage_profiles:
                profile = self._stage_profiles[stage_name]
                # Compute statistics before returning
                profile.compute_statistics()
                return profile.to_dict()
            return None
    
    def get_all_stage_profiles(self) -> Dict[str, Dict[str, Any]]:
        """
        Get profile information for all stages.
        
        Returns:
            Dictionary of stage profiles
        """
        with self._lock:
            result = {}
            for name, profile in self._stage_profiles.items():
                # Compute statistics before returning
                profile.compute_statistics()
                result[name] = profile.to_dict()
            return result
    
    def get_request_profile(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get profile information for a specific request.
        
        Args:
            request_id: ID of the request to get profile for
            
        Returns:
            Request profile dictionary or None if not found
        """
        with self._lock:
            return self._request_profiles.get(request_id)
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """
        Get current scaling efficiency metrics.
        
        Returns:
            Dictionary of scaling metrics
        """
        with self._lock:
            return self._scaling_metrics.to_dict()
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Detect performance bottlenecks in the pipeline.
        
        Returns:
            List of bottleneck information with recommendations
        """
        logger.info("Starting bottleneck detection analysis")
        bottlenecks = []
        
        with self._lock:
            # Compute statistics for all stages
            logger.debug(f"Computing statistics for {len(self._stage_profiles)} stages")
            for name, profile in self._stage_profiles.items():
                logger.debug(f"Computing statistics for stage '{name}' ({profile.count} executions)")
                profile.compute_statistics()
            
            # Check for stages with critical or warning latency
            critical_stages = [
                s for s in self._stage_profiles.values() 
                if s.category in (LatencyCategory.CRITICAL, LatencyCategory.WARNING)
            ]
            
            # Log findings
            if not critical_stages:
                logger.info("No critical or warning stage latencies detected")
            else:
                logger.warning(f"Found {len(critical_stages)} stages with concerning latency")
            
            # Sort by avg time (descending)
            critical_stages.sort(key=lambda s: s.avg_time, reverse=True)
            
            # Calculate total execution count for frequency analysis
            total_executions = sum(p.count for p in self._stage_profiles.values())
            avg_executions_per_stage = total_executions / len(self._stage_profiles) if self._stage_profiles else 0
            logger.debug(f"Total executions across all stages: {total_executions}, avg per stage: {avg_executions_per_stage:.2f}")
            
            # Generate bottleneck information with recommendations
            for stage in critical_stages:
                logger.debug(f"Analyzing bottleneck: {stage.name} (avg: {stage.avg_time:.2f}ms, cat: {stage.category.value})")
                
                bottleneck = {
                    "stage_name": stage.name,
                    "category": stage.category.value,
                    "avg_time_ms": stage.avg_time,
                    "p95_time_ms": stage.p95_time,
                    "call_count": stage.count,
                    "variance": stage.std_dev,
                    "recommendations": []
                }
                
                # Log detailed metrics for important stages
                logger.debug(f"Stage '{stage.name}' metrics: " +
                           f"avg={stage.avg_time:.2f}ms, min={stage.min_time:.2f}ms, max={stage.max_time:.2f}ms, " +
                           f"p95={stage.p95_time or 0:.2f}ms, count={stage.count}")
                
                # Generate recommendations based on profile
                if stage.std_dev and stage.std_dev > (stage.avg_time / 2):
                    recommendation = "High variance detected. Consider investigating inconsistent execution paths or data-dependent performance."
                    logger.debug(f"Stage '{stage.name}' has high variance: std_dev={stage.std_dev:.2f}ms, avg={stage.avg_time:.2f}ms")
                    bottleneck["recommendations"].append(recommendation)
                
                if stage.resources.cpu_percent > 70:
                    recommendation = "High CPU usage. Consider optimization or parallelization."
                    logger.debug(f"Stage '{stage.name}' has high CPU usage: {stage.resources.cpu_percent:.1f}%")
                    bottleneck["recommendations"].append(recommendation)
                
                if stage.resources.memory_mb > 500:
                    recommendation = "High memory usage. Consider memory optimization or leaks."
                    logger.debug(f"Stage '{stage.name}' has high memory usage: {stage.resources.memory_mb:.1f}MB")
                    bottleneck["recommendations"].append(recommendation)
                
                if stage.count > (avg_executions_per_stage * 2) and stage.count > 5:
                    recommendation = "Stage called significantly more often than others. Consider caching or optimizing call pattern."
                    logger.debug(f"Stage '{stage.name}' is called frequently: {stage.count} times (avg: {avg_executions_per_stage:.2f})")
                    bottleneck["recommendations"].append(recommendation)
                
                # Check for I/O intensive operations
                if stage.resources.io_read_mb > 10 or stage.resources.io_write_mb > 10:
                    recommendation = "High I/O activity detected. Consider optimizing file/network operations or adding caching."
                    logger.debug(f"Stage '{stage.name}' has high I/O: read={stage.resources.io_read_mb:.1f}MB, write={stage.resources.io_write_mb:.1f}MB")
                    bottleneck["recommendations"].append(recommendation)
                
                # Check for error-prone stages
                if stage.errors > 0 and stage.count > 0:
                    error_rate = stage.errors / stage.count
                    if error_rate > 0.05:  # More than 5% error rate
                        recommendation = f"High error rate detected ({error_rate:.1%}). Consider improving error handling or investigating the root cause."
                        logger.debug(f"Stage '{stage.name}' has high error rate: {error_rate:.1%} ({stage.errors}/{stage.count})")
                        bottleneck["recommendations"].append(recommendation)
                
                # Add default recommendation if none generated
                if not bottleneck["recommendations"]:
                    recommendation = "Review implementation for optimization opportunities."
                    logger.debug(f"No specific recommendations for stage '{stage.name}', adding default")
                    bottleneck["recommendations"].append(recommendation)
                
                # Log recommended actions
                logger.info(f"Bottleneck identified: '{stage.name}' ({stage.category.value}, {stage.avg_time:.2f}ms)")
                for i, rec in enumerate(bottleneck["recommendations"], 1):
                    logger.info(f"  Recommendation {i}: {rec}")
                
                bottlenecks.append(bottleneck)
        
        # Log summary
        if bottlenecks:
            stages_list = ", ".join(b["stage_name"] for b in bottlenecks[:3])
            more_count = max(0, len(bottlenecks) - 3)
            more_text = f" and {more_count} more" if more_count > 0 else ""
            logger.warning(f"Performance bottlenecks identified in stages: {stages_list}{more_text}")
        else:
            logger.info("No performance bottlenecks identified")
            
        return bottlenecks
    
    def get_snapshot(self, request_id: Optional[str] = None) -> ProfileSnapshot:
        """
        Get a complete snapshot of performance metrics.
        
        Args:
            request_id: Optional request ID to include in the snapshot
            
        Returns:
            ProfileSnapshot object with all current metrics
        """
        with self._lock:
            # Get stage profiles
            stage_profiles = self.get_all_stage_profiles()
            
            # Get system info
            system_load = {}
            thread_count = 0
            if self.enable_resource_monitoring:
                try:
                    system_load = {
                        "cpu_percent": psutil.cpu_percent(interval=None),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_usage_percent": psutil.disk_usage('/').percent
                    }
                    thread_count = len(psutil.Process().threads())
                except Exception as e:
                    logger.warning(f"Error getting system load: {str(e)}")
            
            # Calculate total time
            total_time = sum(p.get("total_time", 0) for p in stage_profiles.values())
            
            # Calculate total CPU time
            total_cpu_time = sum(
                p.get("total_time", 0) * p.get("resources", {}).get("cpu_percent", 0) / 100
                for p in stage_profiles.values()
            )
            
            # Detect bottlenecks
            bottlenecks = self.detect_bottlenecks()
            
            # Create and return snapshot
            return ProfileSnapshot(
                timestamp=time.time(),
                request_id=request_id or "",
                stage_profiles=stage_profiles,
                system_load=system_load,
                thread_count=thread_count,
                total_time=total_time,
                total_cpu_time=total_cpu_time,
                bottlenecks=bottlenecks,
                thresholds=self.thresholds
            )
    
    def export_snapshot(self, filename: Optional[str] = None, request_id: Optional[str] = None) -> str:
        """
        Export a performance snapshot to a JSON file.
        
        Args:
            filename: Optional filename for the export
            request_id: Optional request ID to include in the snapshot
            
        Returns:
            Path to the exported file
        """
        logger.info(f"Exporting performance snapshot{' for request ' + request_id if request_id else ''}")
        
        try:
            # Get snapshot
            logger.debug("Generating performance snapshot")
            start_time = time.time()
            snapshot = self.get_snapshot(request_id)
            generation_time = (time.time() - start_time) * 1000.0
            logger.debug(f"Snapshot generated in {generation_time:.2f}ms")
            
            # Count the stages and other metrics
            stage_count = len(snapshot.stage_profiles)
            bottleneck_count = len(snapshot.bottlenecks)
            logger.debug(f"Snapshot contains {stage_count} stages and {bottleneck_count} bottlenecks")
            
            # Determine export path
            if not filename:
                timestamp = int(time.time())
                if request_id:
                    sanitized_req_id = request_id.replace('/', '_').replace(':', '_')
                    filename = f"acdek_perf_{sanitized_req_id}_{timestamp}.json"
                else:
                    filename = f"acdek_perf_{timestamp}.json"
                logger.debug(f"Generated filename: {filename}")
            
            # Resolve directory path
            directory = self.export_directory or os.path.join(os.getcwd(), "performance_data")
            logger.debug(f"Export directory: {directory}")
            
            # Ensure directory exists
            try:
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Ensured export directory exists: {directory}")
            except Exception as e:
                logger.error(f"Failed to create export directory {directory}: {e}")
                # Fall back to current working directory
                directory = os.getcwd()
                logger.warning(f"Falling back to current working directory: {directory}")
            
            # Construct full file path
            filepath = os.path.join(directory, filename)
            logger.debug(f"Full export path: {filepath}")
            
            # Write to file
            try:
                start_time = time.time()
                json_content = snapshot.to_json()
                json_size = len(json_content)
                logger.debug(f"Serialized snapshot to JSON: {json_size} bytes")
                
                with open(filepath, 'w') as f:
                    f.write(json_content)
                
                export_time = (time.time() - start_time) * 1000.0
                logger.debug(f"Wrote snapshot to file in {export_time:.2f}ms")
                
                # Verify file was written successfully
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    file_size = os.path.getsize(filepath)
                    logger.info(f"Exported performance snapshot to {filepath} ({file_size} bytes)")
                else:
                    logger.warning(f"Export file exists but may be empty: {filepath}")
                
                # Log summary of important metrics
                critical_stages = [name for name, profile in snapshot.stage_profiles.items() 
                                if profile.get("category") in ("critical", "warning")]
                if critical_stages:
                    logger.info(f"Exported snapshot contains {len(critical_stages)} concerning stages: {', '.join(critical_stages[:3])}")
                
                return filepath
                
            except Exception as e:
                logger.error(f"Failed to write snapshot to file {filepath}: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Failed to export performance snapshot: {e}")
            raise
    
    def reset(self):
        """Reset all profiling data."""
        with self._lock:
            self._stage_profiles = {}
            self._request_profiles = {}
            self._stage_dependencies = {}
            self._execution_stack = {}
            self._request_start_times = {}
            self._request_timings = []
            self._scaling_metrics = ScalingMetrics()
            self._concurrent_requests = 0
            
        logger.info("Performance profiler reset")
    
    def set_thresholds(self, thresholds: Dict[str, float]):
        """
        Update latency categorization thresholds.
        
        Args:
            thresholds: Dictionary of threshold values in milliseconds
        """
        with self._lock:
            self.thresholds.update(thresholds)
            
            # Recategorize existing stages
            for name, profile in self._stage_profiles.items():
                if profile.last_execution_time > 0:
                    self._categorize_latency(name, profile.last_execution_time)