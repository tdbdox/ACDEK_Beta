"""
Health monitor for ACDEK v3: provides health probes for core components.
"""
import json
from registry.plugin_registry import registry
from registry.registry_types import PluginType

class HealthMonitor:
    """Performs health checks for ACDEK components."""
    def __init__(self, kernel=None):
        self.kernel = kernel

    def check_plugin_registry(self):
        """Verify that plugin registry has loaded plugins."""
        try:
            registry_counts = {}
            for pt in PluginType:
                plugins = registry.list_plugins(pt)
                registry_counts[pt.value] = len(plugins) if plugins is not None else 0
            return {"status": "ok", "counts": registry_counts}
        except Exception as e:
            return {"status": "fail", "error": str(e)}

    def check_pipeline_stages(self):
        """Check enrichment pipeline stages initialization."""
        if not self.kernel:
            return {"status": "fail", "error": "No kernel instance provided"}
        pipeline = getattr(self.kernel, "enrichment_pipeline", None)
        if not pipeline or not hasattr(pipeline, "stages"):
            return {"status": "fail", "error": "Pipeline not initialized"}
        return {"status": "ok", "stage_count": len(pipeline.stages)}

    def check_fallback_system(self):
        """Check if fallback system is configured."""
        if not self.kernel:
            return {"status": "fail", "error": "No kernel instance provided"}
        fb = getattr(self.kernel, "fallback_chain", None)
        if fb:
            return {"status": "ok"}
        return {"status": "not_configured"}

    def get_health_report(self):
        """Aggregate health checks into a report."""
        return {
            "plugin_registry": self.check_plugin_registry(),
            "pipeline_stages": self.check_pipeline_stages(),
            "fallback_system": self.check_fallback_system(),
        }

    def to_json(self):
        """Return health report as JSON string."""
        return json.dumps(self.get_health_report())