"""
Enhanced metrics collector for ACDEK v3: Extends the base collector with distributed profiling.
"""
import threading
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from contextlib import contextmanager
import uuid
import os

from observability.metrics_collector import MetricsCollector
from observability.performance_profiler import PerformanceProfiler, StageProfile, ResourceUsage

logger = logging.getLogger(__name__)

@contextmanager
def TimingContext():
    """Context manager for timing code execution."""
    start = time.time()
    
    class Timer:
        def __init__(self):
            self.elapsed = 0.0
    
    timer = Timer()
    
    try:
        yield timer
    finally:
        timer.elapsed = time.time() - start

class EnhancedMetricsCollector(MetricsCollector):
    """
    Enhanced metrics collector with distributed profiling capabilities.
    
    This class extends the base MetricsCollector with more comprehensive
    timing and profiling capabilities, integrating with the PerformanceProfiler.
    """
    def __init__(
        self,
        service_name: str = "acdek",
        instance_id: Optional[str] = None,
        enable_distributed_tracing: bool = True,
        enable_profiling: bool = True,
        profiler_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the enhanced metrics collector.
        
        Args:
            service_name: Name of the service this collector is for
            instance_id: Unique identifier for this service instance
            enable_distributed_tracing: Whether to enable distributed tracing
            enable_profiling: Whether to enable detailed profiling
            profiler_config: Optional configuration for the performance profiler
        """
        super().__init__()
        
        self.service_name = service_name
        self.instance_id = instance_id or str(uuid.uuid4())
        self.enable_distributed_tracing = enable_distributed_tracing
        
        # Enhanced metrics storage
        self._histograms: Dict[str, Dict[str, Any]] = {}
        self._gauges: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}
        
        # Trace context
        self._trace_context: Dict[str, Dict[str, Any]] = {}
        
        # Initialize profiler if enabled
        self.enable_profiling = enable_profiling
        if enable_profiling:
            profiler_config = profiler_config or {}
            self.profiler = PerformanceProfiler(
                instance_id=self.instance_id,
                enable_resource_monitoring=profiler_config.get("enable_resource_monitoring", True),
                max_history_size=profiler_config.get("max_history_size", 1000),
                thresholds=profiler_config.get("thresholds"),
                export_directory=profiler_config.get("export_directory")
            )
        else:
            self.profiler = None
    
    def register_counter(self, name: str, description: str) -> None:
        """
        Register a counter metric.
        
        Args:
            name: Metric name
            description: Metric description
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = 0
    
    def register_gauge(self, name: str, description: str) -> None:
        """
        Register a gauge metric.
        
        Args:
            name: Metric name
            description: Metric description
        """
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = 0.0
    
    def register_histogram(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None
    ) -> None:
        """
        Register a histogram metric.
        
        Args:
            name: Metric name
            description: Metric description
            buckets: Optional bucket boundaries
        """
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = {
                    "count": 0,
                    "sum": 0.0,
                    "values": [],
                    "buckets": buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
                    "bucket_counts": {}
                }
                
                # Initialize bucket counts
                for bucket in self._histograms[name]["buckets"]:
                    self._histograms[name]["bucket_counts"][str(bucket)] = 0
    
    def counter_inc(self, name: str, value: int = 1) -> int:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            value: Amount to increment by (default: 1)
            
        Returns:
            New counter value
        """
        with self._lock:
            if name not in self._counters:
                self.register_counter(name, "")
            
            self._counters[name] += value
            return self._counters[name]
    
    def gauge_set(self, name: str, value: float) -> None:
        """
        Set a gauge metric to a specific value.
        
        Args:
            name: Metric name
            value: Value to set
        """
        with self._lock:
            if name not in self._gauges:
                self.register_gauge(name, "")
            
            self._gauges[name] = value
    
    def gauge_inc(self, name: str, value: float = 1.0) -> None:
        """
        Increment a gauge metric.
        
        Args:
            name: Metric name
            value: Amount to increment by (default: 1.0)
        """
        with self._lock:
            if name not in self._gauges:
                self.register_gauge(name, "")
            
            self._gauges[name] += value
    
    def gauge_dec(self, name: str, value: float = 1.0) -> None:
        """
        Decrement a gauge metric.
        
        Args:
            name: Metric name
            value: Amount to decrement by (default: 1.0)
        """
        with self._lock:
            if name not in self._gauges:
                self.register_gauge(name, "")
            
            self._gauges[name] -= value
    
    def histogram_observe(self, name: str, value: float) -> None:
        """
        Observe a value for a histogram metric.
        
        Args:
            name: Metric name
            value: Observed value
        """
        with self._lock:
            if name not in self._histograms:
                self.register_histogram(name, "")
            
            hist = self._histograms[name]
            hist["count"] += 1
            hist["sum"] += value
            hist["values"].append(value)
            
            # Keep only the last 1000 values
            if len(hist["values"]) > 1000:
                hist["values"] = hist["values"][-1000:]
            
            # Update bucket counts
            for bucket in hist["buckets"]:
                if value <= bucket:
                    hist["bucket_counts"][str(bucket)] += 1
    
    @contextmanager
    def time_execution(
        self,
        stage_name: str,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        record_histogram: bool = True,
        histogram_name: Optional[str] = None
    ):
        """
        Context manager for timing code execution with profiling integration.
        
        Args:
            stage_name: Name of the stage being timed
            request_id: Optional request identifier for correlation
            metadata: Optional metadata about the execution
            record_histogram: Whether to record in a histogram
            histogram_name: Custom histogram name (defaults to 'duration.{stage_name}')
        
        Yields:
            None
        """
        # Start timing
        start_time = time.time()
        
        # Use profiler if enabled
        profiler_ctx = None
        if self.enable_profiling and self.profiler:
            profiler_ctx = self.profiler.profile_stage(stage_name, request_id, metadata)
            profiler_ctx.__enter__()
        
        try:
            # Record stage invocation
            self.record_stage_invocation(stage_name)
            
            # Execute the timed code
            yield
        finally:
            # End timing and calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            # Record stage duration
            self.record_stage_duration(stage_name, duration)
            
            # Record in histogram if requested
            if record_histogram:
                hist_name = histogram_name or f"duration.{stage_name}"
                self.histogram_observe(hist_name, duration)
            
            # Exit profiler context if used
            if profiler_ctx:
                profiler_ctx.__exit__(None, None, None)
    
    @contextmanager
    def trace_request(
        self,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for tracing a request with profiling integration.
        
        Args:
            request_id: Optional request identifier (generated if not provided)
            metadata: Optional metadata about the request
        
        Yields:
            str: The request ID
        """
        # Generate request ID if not provided
        req_id = request_id or f"req_{uuid.uuid4()}"
        
        # Create trace context
        with self._lock:
            self._trace_context[req_id] = {
                "start_time": time.time(),
                "metadata": metadata or {},
                "stages": []
            }
        
        # Use profiler if enabled
        profiler_ctx = None
        if self.enable_profiling and self.profiler:
            profiler_ctx = self.profiler.profile_request(req_id, metadata)
            profiler_ctx.__enter__()
        
        try:
            # Execute the traced code
            yield req_id
        finally:
            # End tracing and calculate duration
            end_time = time.time()
            duration = end_time - self._trace_context[req_id]["start_time"]
            
            # Record request duration
            self.record_request(duration)
            
            # Update trace context
            with self._lock:
                self._trace_context[req_id]["end_time"] = end_time
                self._trace_context[req_id]["duration"] = duration
            
            # Exit profiler context if used
            if profiler_ctx:
                profiler_ctx.__exit__(None, None, None)
            
            # Clean up old trace contexts
            self._cleanup_trace_contexts()
    
    def _cleanup_trace_contexts(self):
        """Clean up old trace contexts to prevent memory leaks."""
        with self._lock:
            # Keep only the last 1000 trace contexts, based on start time
            if len(self._trace_context) > 1000:
                # Sort by start time (oldest first)
                sorted_contexts = sorted(
                    self._trace_context.items(),
                    key=lambda x: x[1].get("start_time", 0)
                )
                
                # Remove oldest contexts
                for req_id, _ in sorted_contexts[:-1000]:
                    del self._trace_context[req_id]
    
    def get_trace_context(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trace context for a request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Trace context or None if not found
        """
        with self._lock:
            return self._trace_context.get(request_id)
    
    def get_profiler(self) -> Optional[PerformanceProfiler]:
        """
        Get the performance profiler instance.
        
        Returns:
            PerformanceProfiler instance or None if profiling disabled
        """
        return self.profiler
    
    def snapshot(self) -> Dict[str, Any]:
        """
        Return a snapshot of current metrics as a dict.
        
        Returns:
            Metrics snapshot dictionary
        """
        base_snapshot = super().snapshot()
        
        with self._lock:
            # Add enhanced metrics
            enhanced_snapshot = {
                **base_snapshot,
                "histograms": {},
                "gauges": dict(self._gauges),
                "counters": dict(self._counters),
                "service_name": self.service_name,
                "instance_id": self.instance_id
            }
            
            # Format histogram data
            for name, hist in self._histograms.items():
                enhanced_snapshot["histograms"][name] = {
                    "count": hist["count"],
                    "sum": hist["sum"],
                    "avg": hist["sum"] / hist["count"] if hist["count"] > 0 else 0,
                    "bucket_counts": dict(hist["bucket_counts"]),
                    # Exclude full values list to keep snapshot size reasonable
                }
            
            # Add profiler data if available
            if self.enable_profiling and self.profiler:
                # Just include summary data, not the full profile
                stage_count = len(self.profiler._stage_profiles)
                bottlenecks = len(self.profiler.detect_bottlenecks())
                
                enhanced_snapshot["profiler"] = {
                    "enabled": True,
                    "stage_count": stage_count,
                    "bottlenecks": bottlenecks,
                    "instance_id": self.profiler.instance_id
                }
            else:
                enhanced_snapshot["profiler"] = {"enabled": False}
            
            return enhanced_snapshot
    
    def to_prometheus(self) -> str:
        """
        Format metrics for Prometheus exposition format.
        
        Returns:
            String in Prometheus exposition format
        """
        lines = []
        timestamp_ms = int(time.time() * 1000)
        
        # Format counters
        with self._lock:
            for name, value in self._counters.items():
                # Add metric metadata
                lines.append(f"# TYPE {name} counter")
                
                # Add metric value
                lines.append(f"{name} {value} {timestamp_ms}")
            
            # Format gauges
            for name, value in self._gauges.items():
                # Add metric metadata
                lines.append(f"# TYPE {name} gauge")
                
                # Add metric value
                lines.append(f"{name} {value} {timestamp_ms}")
            
            # Format histograms
            for name, hist in self._histograms.items():
                # Add metric metadata
                lines.append(f"# TYPE {name} histogram")
                
                # Add sum and count
                lines.append(f"{name}_sum {hist['sum']} {timestamp_ms}")
                lines.append(f"{name}_count {hist['count']} {timestamp_ms}")
                
                # Add buckets
                for bucket, count in hist["bucket_counts"].items():
                    lines.append(f"{name}_bucket{{le=\"{bucket}\"}} {count} {timestamp_ms}")
                
                # Add +Inf bucket
                lines.append(f"{name}_bucket{{le=\"+Inf\"}} {hist['count']} {timestamp_ms}")
        
        return "\n".join(lines)
    
    def export_to_file(
        self,
        format: str = "json",
        file_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Export metrics to a file.
        
        Args:
            format: Export format ('json' or 'prometheus')
            file_path: Path to export file (generated if None)
            
        Returns:
            Path to the exported file or None if export failed
        """
        if not file_path:
            timestamp = int(time.time())
            if format == "json":
                file_path = f"metrics_{self.service_name}_{timestamp}.json"
            else:
                file_path = f"metrics_{self.service_name}_{timestamp}.prom"
        
        try:
            with open(file_path, 'w') as f:
                if format == "json":
                    json.dump(self.snapshot(), f, indent=2)
                else:
                    f.write(self.to_prometheus())
            
            logger.info(f"Exported metrics to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            return None
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            super().reset()
            self._histograms = {}
            self._gauges = {}
            self._counters = {}
            self._trace_context = {}
            
            if self.enable_profiling and self.profiler:
                self.profiler.reset()
        
        logger.info(f"Reset all metrics for {self.service_name}")

# Enhanced TimingContext that integrates with the profiler
class ProfiledTimingContext:
    """Context manager for timing execution with profiling integration."""
    def __init__(
        self,
        stage_name: str,
        metrics_collector: Optional[EnhancedMetricsCollector] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize timing context.
        
        Args:
            stage_name: Name of the stage being timed
            metrics_collector: Optional metrics collector to use
            request_id: Optional request identifier
            metadata: Optional metadata about the execution
        """
        self.stage_name = stage_name
        self.metrics_collector = metrics_collector
        self.request_id = request_id
        self.metadata = metadata
        self.elapsed = 0.0
        self.start_time = None
    
    def __enter__(self):
        """Start timing execution."""
        self.start_time = time.time()
        
        # Integrate with metrics collector if provided
        if self.metrics_collector:
            self.metrics_collector.record_stage_invocation(self.stage_name)
            
            # Use profiler if available
            profiler = self.metrics_collector.get_profiler()
            if profiler:
                self._profiler_ctx = profiler.profile_stage(
                    self.stage_name,
                    self.request_id,
                    self.metadata
                )
                self._profiler_ctx.__enter__()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing execution and record result."""
        self.elapsed = time.time() - self.start_time
        
        # Integrate with metrics collector if provided
        if self.metrics_collector:
            self.metrics_collector.record_stage_duration(self.stage_name, self.elapsed)
            
            # Record histogram
            self.metrics_collector.histogram_observe(
                f"duration.{self.stage_name}",
                self.elapsed
            )
            
            # Exit profiler context if used
            if hasattr(self, '_profiler_ctx'):
                self._profiler_ctx.__exit__(exc_type, exc_val, exc_tb)

# Create a global instance for imports
enhanced_metrics_collector = EnhancedMetricsCollector()