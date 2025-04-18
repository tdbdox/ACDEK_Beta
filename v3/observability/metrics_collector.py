"""
Static metrics collector for ACDEK v3: tracks invocation counts and durations.
"""
import threading
import json

class MetricsCollector:
    """Collects metrics for stages and requests."""
    def __init__(self):
        self._lock = threading.Lock()
        self._stage_counts = {}
        self._stage_timings = {}
        self._request_metrics = {"count": 0, "total_time": 0.0, "min": None, "max": None}

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

# Module-level singleton
metrics_collector = MetricsCollector()