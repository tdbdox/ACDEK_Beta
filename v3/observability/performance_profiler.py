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
from typing import Dict, List, Any, Optional, Tuple, Set, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
import psutil
import uuid
from enum import Enum

logger = logging.getLogger(__name__)

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
        self.instance_id = instance_id or str(uuid.uuid4())
        self.enable_resource_monitoring = enable_resource_monitoring
        self.max_history_size = max_history_size
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.export_directory = export_directory
        
        # Initialize storage
        self._lock = threading.RLock()
        self._stage_profiles: Dict[str, StageProfile] = {}
        self._request_profiles: Dict[str, Dict[str, Any]] = {}
        self._stage_dependencies: Dict[str, Set[str]] = {}
        self._execution_stack: Dict[int, List[str]] = {}
        
        # Scaling metrics
        self._scaling_metrics = ScalingMetrics()
        self._concurrent_requests = 0
        self._request_start_times: Dict[str, float] = {}
        self._request_timings: List[float] = []
        
        # Initialize system monitoring if enabled
        if self.enable_resource_monitoring:
            self._init_process = psutil.Process()
            self._io_counters_last = self._init_process.io_counters() if hasattr(self._init_process, 'io_counters') else None
            self._io_last_time = time.time()
        
        logger.info(f"PerformanceProfiler initialized with instance_id={self.instance_id}")
    
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
        
        # Get start resource usage if enabled
        start_resources = None
        if self.enable_resource_monitoring:
            start_resources = self._get_resource_usage()
        
        # Record entry in execution stack to track parent-child relationships
        parent_stage = None
        with self._lock:
            if thread_id in self._execution_stack:
                if self._execution_stack[thread_id]:
                    parent_stage = self._execution_stack[thread_id][-1]
            else:
                self._execution_stack[thread_id] = []
                
            if parent_stage:
                self._register_dependency(parent_stage, stage_name)
                
            self._execution_stack[thread_id].append(stage_name)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Execute the profiled code
            yield
        except Exception as e:
            # Record error but re-raise
            with self._lock:
                if stage_name in self._stage_profiles:
                    self._stage_profiles[stage_name].errors += 1
            raise
        finally:
            # End timing and calculate duration in milliseconds
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000.0
            
            # Get end resource usage if enabled
            end_resources = None
            if self.enable_resource_monitoring:
                end_resources = self._get_resource_usage()
                
            # Calculate resource delta
            resource_delta = None
            if start_resources and end_resources:
                resource_delta = ResourceUsage(
                    cpu_percent=end_resources.cpu_percent,
                    memory_mb=end_resources.memory_mb,
                    thread_count=end_resources.thread_count,
                    io_read_mb=end_resources.io_read_mb - start_resources.io_read_mb,
                    io_write_mb=end_resources.io_write_mb - start_resources.io_write_mb
                )
            
            # Record the profiling data
            with self._lock:
                # Update stage profile
                if stage_name not in self._stage_profiles:
                    self._stage_profiles[stage_name] = StageProfile(name=stage_name)
                
                # Add this execution to the profile
                self._stage_profiles[stage_name].add_execution(duration_ms, resource_delta)
                
                # Categorize the latency
                self._categorize_latency(stage_name, duration_ms)
                
                # Record in request profile if we have a request ID
                if request_id:
                    if request_id not in self._request_profiles:
                        self._request_profiles[request_id] = {
                            "stages": {},
                            "start_time": start_time,
                            "metadata": metadata or {}
                        }
                    
                    self._request_profiles[request_id]["stages"][stage_name] = {
                        "duration_ms": duration_ms,
                        "start_time": start_time,
                        "end_time": end_time
                    }
                
                # Remove stage from execution stack
                if thread_id in self._execution_stack:
                    if self._execution_stack[thread_id]:
                        self._execution_stack[thread_id].pop()
    
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
        
        # Increment concurrent request counter
        with self._lock:
            self._concurrent_requests += 1
            self._scaling_metrics.concurrent_requests = self._concurrent_requests
            self._request_start_times[req_id] = time.time()
        
        try:
            # Start request profiling
            with self.profile_stage("request_total", req_id, metadata):
                yield req_id
        except Exception:
            # Record error in scaling metrics
            with self._lock:
                self._scaling_metrics.error_rate = (
                    (self._scaling_metrics.error_rate * (self._concurrent_requests - 1) + 1.0) / 
                    self._concurrent_requests
                )
            raise
        finally:
            # Decrement concurrent request counter and update metrics
            with self._lock:
                # Calculate request duration
                if req_id in self._request_start_times:
                    duration = time.time() - self._request_start_times[req_id]
                    # Convert to milliseconds
                    duration_ms = duration * 1000.0
                    
                    # Add to request timings for throughput calculation
                    self._request_timings.append(duration_ms)
                    # Keep only the last 100 request timings
                    if len(self._request_timings) > 100:
                        self._request_timings = self._request_timings[-100:]
                    
                    # Update average latency in scaling metrics
                    if self._request_timings:
                        self._scaling_metrics.avg_latency_ms = sum(self._request_timings) / len(self._request_timings)
                    
                    # Calculate throughput (requests per second)
                    if self._scaling_metrics.avg_latency_ms > 0:
                        self._scaling_metrics.throughput_rps = 1000.0 / self._scaling_metrics.avg_latency_ms
                    
                    # Remove from start times
                    del self._request_start_times[req_id]
                
                # Update concurrent requests count
                self._concurrent_requests = max(0, self._concurrent_requests - 1)
                self._scaling_metrics.concurrent_requests = self._concurrent_requests
                
                # Update resource utilization if enabled
                if self.enable_resource_monitoring:
                    try:
                        self._scaling_metrics.resource_utilization = psutil.cpu_percent(interval=None) / 100.0
                    except Exception:
                        pass
                
                # Calculate efficiency score (throughput / resource utilization)
                if self._scaling_metrics.resource_utilization > 0:
                    self._scaling_metrics.efficiency_score = (
                        self._scaling_metrics.throughput_rps / 
                        self._scaling_metrics.resource_utilization
                    )
    
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
        bottlenecks = []
        
        with self._lock:
            # Compute statistics for all stages
            for profile in self._stage_profiles.values():
                profile.compute_statistics()
            
            # Check for stages with critical or warning latency
            critical_stages = [
                s for s in self._stage_profiles.values() 
                if s.category in (LatencyCategory.CRITICAL, LatencyCategory.WARNING)
            ]
            
            # Sort by avg time (descending)
            critical_stages.sort(key=lambda s: s.avg_time, reverse=True)
            
            # Generate bottleneck information with recommendations
            for stage in critical_stages:
                bottleneck = {
                    "stage_name": stage.name,
                    "category": stage.category.value,
                    "avg_time_ms": stage.avg_time,
                    "p95_time_ms": stage.p95_time,
                    "call_count": stage.count,
                    "variance": stage.std_dev,
                    "recommendations": []
                }
                
                # Generate recommendations based on profile
                if stage.std_dev and stage.std_dev > (stage.avg_time / 2):
                    bottleneck["recommendations"].append(
                        "High variance detected. Consider investigating inconsistent execution paths or data-dependent performance."
                    )
                
                if stage.resources.cpu_percent > 70:
                    bottleneck["recommendations"].append(
                        "High CPU usage. Consider optimization or parallelization."
                    )
                
                if stage.resources.memory_mb > 500:
                    bottleneck["recommendations"].append(
                        "High memory usage. Consider memory optimization or leaks."
                    )
                
                if stage.count > (sum(p.count for p in self._stage_profiles.values()) / len(self._stage_profiles) * 2):
                    bottleneck["recommendations"].append(
                        "Stage called significantly more often than others. Consider caching or optimizing call pattern."
                    )
                
                # Add default recommendation if none generated
                if not bottleneck["recommendations"]:
                    bottleneck["recommendations"].append(
                        "Review implementation for optimization opportunities."
                    )
                
                bottlenecks.append(bottleneck)
        
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
        # Get snapshot
        snapshot = self.get_snapshot(request_id)
        
        # Determine export path
        if not filename:
            timestamp = int(time.time())
            filename = f"acdek_perf_{timestamp}.json"
        
        directory = self.export_directory or os.path.join(os.getcwd(), "performance_data")
        os.makedirs(directory, exist_ok=True)
        
        filepath = os.path.join(directory, filename)
        
        # Write to file
        with open(filepath, 'w') as f:
            f.write(snapshot.to_json())
        
        logger.info(f"Exported performance snapshot to {filepath}")
        return filepath
    
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