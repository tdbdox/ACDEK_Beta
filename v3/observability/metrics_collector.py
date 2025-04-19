"""
Static metrics collector for ACDEK v3: tracks invocation counts and durations.
"""
import threading
import json
import time
from contextlib import contextmanager

class MetricsCollector:
    """Collects metrics for stages and requests."""
    def __init__(self, service_name=None):
        self._lock = threading.Lock()
        self._stage_counts = {}
        self._stage_timings = {}
        self._request_metrics = {"count": 0, "total_time": 0.0, "min": None, "max": None}
        self.service_name = service_name or "default"
        self.counters = {}
        self.gauges = {}
        self.histograms = {}

    def record_stage_invocation(self, stage_name):
        """Increment invocation count for a stage."""
        with self._lock:
            self._stage_counts[stage_name] = self._stage_counts.get(stage_name, 0) + 1

    def record_stage_duration(self, stage_name, duration):
        """Record duration for a stage execution."""
        with self._lock:
            stats = self._stage_timings.setdefault(stage_name, {"count": 0, "total_time": 0.0, "min": None, "max": None})
            stats["count"] += 1
            stats["total_time"] += duration
            stats["min"] = duration if stats["min"] is None or duration < stats["min"] else stats["min"]
            stats["max"] = duration if stats["max"] is None or duration > stats["max"] else stats["max"]

    def record_request(self, duration):
        """Record overall request duration."""
        with self._lock:
            m = self._request_metrics
            m["count"] += 1
            m["total_time"] += duration
            m["min"] = duration if m["min"] is None or duration < m["min"] else m["min"]
            m["max"] = duration if m["max"] is None or duration > m["max"] else m["max"]

    def snapshot(self):
        """Return a snapshot of current metrics as a dict."""
        with self._lock:
            # Stage timing summary
            stage_summary = {}
            for name, stats in self._stage_timings.items():
                count = stats["count"]
                total = stats["total_time"]
                avg = total / count if count else 0.0
                stage_summary[name] = {
                    "count": count,
                    "total_time": total,
                    "avg": avg,
                    "min": stats["min"],
                    "max": stats["max"]
                }
            # Request metrics summary
            rm = self._request_metrics
            req_count = rm["count"]
            req_total = rm["total_time"]
            req_avg = req_total / req_count if req_count else 0.0
            request_summary = {
                "count": req_count,
                "total_time": req_total,
                "avg": req_avg,
                "min": rm["min"],
                "max": rm["max"]
            }
            return {
                "stage_counts": dict(self._stage_counts),
                "stage_timings": stage_summary,
                "request_metrics": request_summary
            }

    def to_json(self):
        """Return metrics snapshot as JSON string."""
        return json.dumps(self.snapshot())
        
    def register_counter(self, name, description):
        """Register a counter metric."""
        with self._lock:
            self.counters[name] = {"value": 0, "description": description}
    
    def counter_inc(self, name, value=1):
        """Increment a counter by the specified value."""
        with self._lock:
            if name not in self.counters:
                self.register_counter(name, "")
            self.counters[name]["value"] += value
    
    def register_gauge(self, name, description):
        """Register a gauge metric."""
        with self._lock:
            self.gauges[name] = {"value": 0, "description": description}
    
    def gauge_set(self, name, value):
        """Set a gauge to the specified value."""
        with self._lock:
            if name not in self.gauges:
                self.register_gauge(name, "")
            self.gauges[name]["value"] = value
            
    def gauge_inc(self, name, value=1):
        """Increment a gauge by the specified value."""
        with self._lock:
            if name not in self.gauges:
                self.register_gauge(name, "")
            self.gauges[name]["value"] += value
            
    def gauge_dec(self, name, value=1):
        """Decrement a gauge by the specified value."""
        with self._lock:
            if name not in self.gauges:
                self.register_gauge(name, "")
            self.gauges[name]["value"] -= value
    
    def register_histogram(self, name, description, buckets=None):
        """Register a histogram metric."""
        with self._lock:
            self.histograms[name] = {
                "count": 0,
                "sum": 0,
                "values": [],
                "description": description,
                "buckets": buckets or [0.1, 0.5, 1.0, 5.0, 10.0]
            }
    
    def histogram_observe(self, name, value):
        """Observe a value for a histogram."""
        with self._lock:
            if name not in self.histograms:
                self.register_histogram(name, "")
            self.histograms[name]["count"] += 1
            self.histograms[name]["sum"] += value
            self.histograms[name]["values"].append(value)
            
    def get_profiler(self):
        """Get the performance profiler."""
        # Return a simple profiler that does nothing for now
        return SimpleProfiler()
        

class SimpleProfiler:
    """Simple profiler for testing."""
    def __init__(self):
        self.profiles = {}
        
    @contextmanager
    def profile_request(self, request_id, metadata=None):
        """Profile a request."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.profiles[request_id] = {
                "duration": duration,
                "metadata": metadata or {}
            }
            
    def get_snapshot(self):
        """Get a snapshot of the profiler."""
        return ProfileSnapshot(self.profiles)
        
    def export_snapshot(self, filename=None):
        """Export a snapshot to a file."""
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.profiles, f)
            return filename
        return None
        
        
class ProfileSnapshot:
    """Simple profile snapshot."""
    def __init__(self, profiles):
        self.profiles = profiles
        
    def to_dict(self):
        """Convert to dictionary."""
        return {"profiles": self.profiles}

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

# Module-level singleton
metrics_collector = MetricsCollector()