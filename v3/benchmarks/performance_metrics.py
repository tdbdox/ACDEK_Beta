"""
Performance Metrics Scaffold

Provides context managers and aggregators for latency and retry metrics.
"""
import time
from contextlib import contextmanager

@contextmanager
def latency_timer(metric_name: str, metrics: Dict[str, float]):  # type: ignore
    """
    Context manager to measure latency for a specific metric.
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    metrics.setdefault('latency', {})[metric_name] = elapsed  # type: ignore

class RetryAggregator:
    """
    Aggregates retry counts and penalty metrics.
    """
    def __init__(self):
        self.counts: Dict[str, int] = {}
        self.penalties: Dict[str, float] = {}

    def record(self, plugin_name: str, retries: int, penalty: float) -> None:
        self.counts[plugin_name] = self.counts.get(plugin_name, 0) + retries
        self.penalties[plugin_name] = self.penalties.get(plugin_name, 0.0) + penalty

    def summary(self) -> Dict[str, Any]:  # type: ignore
        return {'counts': self.counts, 'penalties': self.penalties}

def write_metrics(metrics: Dict[str, Any], path: str, format: str = 'json') -> None:
    """
    Write metrics to disk in JSON or other formats.
    """
    # TODO: implement writing logic
    pass