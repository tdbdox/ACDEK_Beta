"""
Bottleneck detector for ACDEK v3: Identifies performance bottlenecks and suggests optimizations.
"""
import logging
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
import json
import os

from observability.performance_profiler import PerformanceProfiler

logger = logging.getLogger(__name__)

@dataclass
class BottleneckConfig:
    """Configuration for bottleneck detection thresholds and sensitivity."""
    critical_threshold_ms: float = 200.0  # Critical latency in ms
    warning_threshold_ms: float = 100.0   # Warning latency in ms
    high_variance_ratio: float = 2.0      # Ratio of std_dev/avg to flag high variance
    cpu_threshold: float = 70.0           # CPU usage percentage to flag high CPU
    memory_threshold_mb: float = 500.0    # Memory usage in MB to flag high memory
    io_threshold_mb: float = 50.0         # I/O usage in MB/s to flag high I/O
    frequency_ratio: float = 2.0          # Call frequency ratio vs avg to flag high frequency
    
    # Relative contribution thresholds
    relative_contribution_threshold: float = 0.15  # Stage uses 15%+ of total time
    
    # Scaling detection
    concurrency_threshold: int = 10       # Number of concurrent requests to analyze scaling
    efficiency_drop_threshold: float = 0.3  # Efficiency score drop to detect scaling issues

class BottleneckDetector:
    """
    Advanced bottleneck detection and optimization recommendation system.
    
    This class analyzes performance data to identify bottlenecks,
    diagnose root causes, and provide actionable optimization suggestions.
    """
    def __init__(
        self,
        profiler: PerformanceProfiler,
        config: Optional[BottleneckConfig] = None,
        export_directory: Optional[str] = None
    ):
        """
        Initialize the bottleneck detector.
        
        Args:
            profiler: Performance profiler instance to analyze
            config: Optional configuration for detection thresholds
            export_directory: Directory for exporting bottleneck reports
        """
        self.profiler = profiler
        self.config = config or BottleneckConfig()
        self.export_directory = export_directory or os.path.join(os.getcwd(), "bottleneck_reports")
        
        # Create export directory if it doesn't exist
        if self.export_directory:
            os.makedirs(self.export_directory, exist_ok=True)
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Detect performance bottlenecks and generate recommendations.
        
        Returns:
            List of bottleneck details with recommendations
        """
        # Get current performance data
        stage_profiles = self.profiler.get_all_stage_profiles()
        scaling_metrics = self.profiler.get_scaling_metrics()
        
        bottlenecks = []
        
        # Skip analysis if no data
        if not stage_profiles:
            return bottlenecks
        
        # Calculate total execution time across all stages
        total_time = sum(p.get("total_time", 0) for p in stage_profiles.values())
        
        # Skip if total time is zero (no meaningful data yet)
        if total_time == 0:
            return bottlenecks
        
        # Calculate average call count
        call_counts = [p.get("count", 0) for p in stage_profiles.values()]
        avg_call_count = sum(call_counts) / len(call_counts) if call_counts else 0
        
        # Analyze each stage for bottlenecks
        for name, profile in stage_profiles.items():
            # Skip stages with no executions
            if profile.get("count", 0) == 0:
                continue
            
            bottleneck = self._analyze_stage_bottleneck(name, profile, total_time, avg_call_count)
            if bottleneck:
                bottlenecks.append(bottleneck)
        
        # Add system-level bottlenecks
        system_bottlenecks = self._analyze_system_bottlenecks(scaling_metrics)
        bottlenecks.extend(system_bottlenecks)
        
        # Sort by severity and impact
        bottlenecks.sort(key=lambda b: (
            b.get("severity", 0),
            b.get("relative_impact", 0)
        ), reverse=True)
        
        return bottlenecks
    
    def _analyze_stage_bottleneck(
        self,
        stage_name: str,
        profile: Dict[str, Any],
        total_time: float,
        avg_call_count: float
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single stage for bottlenecks."""
        # Skip if not enough data
        if profile.get("count", 0) < 5:
            return None
        
        bottleneck_flags = []
        severity = 0
        recommendations = []
        
        # Check latency category
        category = profile.get("category", "trace")
        avg_time = profile.get("avg_time", 0)
        
        if category == "critical":
            bottleneck_flags.append("critical_latency")
            severity = 3
        elif category == "warning":
            bottleneck_flags.append("warning_latency")
            severity = 2
        
        # Check variance
        std_dev = profile.get("std_dev", 0)
        if std_dev and avg_time and (std_dev > avg_time * self.config.high_variance_ratio):
            bottleneck_flags.append("high_variance")
            recommendations.append(
                "High execution time variance detected. Investigate inconsistent execution paths, "
                "data-dependent performance issues, or resource contention."
            )
        
        # Check resource usage
        resources = profile.get("resources", {})
        
        if resources.get("cpu_percent", 0) > self.config.cpu_threshold:
            bottleneck_flags.append("high_cpu")
            recommendations.append(
                f"High CPU usage ({resources.get('cpu_percent', 0):.1f}%). Consider optimization, "
                "parallelization, or offloading work to background tasks."
            )
        
        if resources.get("memory_mb", 0) > self.config.memory_threshold_mb:
            bottleneck_flags.append("high_memory")
            recommendations.append(
                f"High memory usage ({resources.get('memory_mb', 0):.1f} MB). Check for memory leaks, "
                "large object allocations, or excessive caching."
            )
        
        if (resources.get("io_read_mb", 0) + resources.get("io_write_mb", 0)) > self.config.io_threshold_mb:
            bottleneck_flags.append("high_io")
            recommendations.append(
                f"High I/O usage (R: {resources.get('io_read_mb', 0):.1f} MB/s, "
                f"W: {resources.get('io_write_mb', 0):.1f} MB/s). Consider I/O optimization, "
                "caching, or asynchronous operations."
            )
        
        # Check call frequency
        call_count = profile.get("count", 0)
        if avg_call_count > 0 and call_count > avg_call_count * self.config.frequency_ratio:
            bottleneck_flags.append("high_frequency")
            recommendations.append(
                f"Called {call_count} times, which is {call_count/avg_call_count:.1f}x more frequently "
                "than average. Consider caching results or refactoring call patterns."
            )
        
        # Check relative contribution to total time
        relative_contribution = profile.get("total_time", 0) / total_time if total_time > 0 else 0
        if relative_contribution > self.config.relative_contribution_threshold:
            bottleneck_flags.append("high_contribution")
            recommendations.append(
                f"Consumes {relative_contribution*100:.1f}% of total processing time. "
                "Focus optimization efforts here for maximum impact."
            )
            # Increase severity based on contribution
            severity = max(severity, min(3, int(relative_contribution * 10)))
        
        # Only create bottleneck if we have flags
        if not bottleneck_flags:
            return None
        
        # Generate bottleneck report
        bottleneck = {
            "stage_name": stage_name,
            "category": category,
            "severity": severity,  # 0=info, 1=low, 2=medium, 3=high
            "avg_time_ms": avg_time,
            "p95_time_ms": profile.get("p95_time", avg_time),
            "p99_time_ms": profile.get("p99_time", avg_time),
            "call_count": call_count,
            "total_time_ms": profile.get("total_time", 0),
            "relative_impact": relative_contribution,
            "variance": std_dev,
            "flags": bottleneck_flags,
            "recommendations": recommendations,
            "timestamp": time.time()
        }
        
        # Add specific recommendations if none exist yet
        if not recommendations:
            bottleneck["recommendations"] = self._generate_specific_recommendations(
                stage_name, profile, bottleneck_flags
            )
        
        return bottleneck
    
    def _analyze_system_bottlenecks(self, scaling_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze system-level bottlenecks."""
        system_bottlenecks = []
        
        # Skip if not enough concurrent requests for scaling analysis
        if scaling_metrics.get("concurrent_requests", 0) < self.config.concurrency_threshold:
            return system_bottlenecks
        
        # Check for scaling efficiency issues
        efficiency = scaling_metrics.get("efficiency_score", 1.0)
        if efficiency < self.config.efficiency_drop_threshold:
            bottleneck = {
                "stage_name": "system_scaling",
                "category": "warning",
                "severity": 2,
                "flags": ["scaling_efficiency"],
                "avg_time_ms": scaling_metrics.get("avg_latency_ms", 0),
                "concurrent_requests": scaling_metrics.get("concurrent_requests", 0),
                "throughput_rps": scaling_metrics.get("throughput_rps", 0),
                "resource_utilization": scaling_metrics.get("resource_utilization", 0),
                "efficiency_score": efficiency,
                "relative_impact": 0.5,  # Assign medium impact
                "recommendations": [
                    f"Low scaling efficiency ({efficiency:.2f}). System is not scaling linearly with load.",
                    "Consider improving concurrency, reducing contention, or scaling horizontally.",
                    "Review database connections, thread pool sizes, and shared resource usage."
                ],
                "timestamp": time.time()
            }
            system_bottlenecks.append(bottleneck)
        
        # Check for high error rate
        error_rate = scaling_metrics.get("error_rate", 0)
        if error_rate > 0.05:  # More than 5% errors
            bottleneck = {
                "stage_name": "system_errors",
                "category": "critical" if error_rate > 0.1 else "warning",
                "severity": 3 if error_rate > 0.1 else 2,
                "flags": ["high_error_rate"],
                "error_rate": error_rate,
                "relative_impact": error_rate * 2,  # Higher impact for higher error rate
                "recommendations": [
                    f"High error rate ({error_rate*100:.1f}%). Review logs for exceptions and failures.",
                    "Consider circuit breakers or graceful degradation for failing components.",
                    "Add retry mechanisms with exponential backoff for transient failures."
                ],
                "timestamp": time.time()
            }
            system_bottlenecks.append(bottleneck)
        
        # Check for high resource utilization
        resource_util = scaling_metrics.get("resource_utilization", 0)
        if resource_util > 0.8:  # More than 80% resource utilization
            bottleneck = {
                "stage_name": "system_resources",
                "category": "warning",
                "severity": 2,
                "flags": ["high_resource_utilization"],
                "resource_utilization": resource_util,
                "relative_impact": resource_util,
                "recommendations": [
                    f"High resource utilization ({resource_util*100:.1f}%). System approaching capacity limits.",
                    "Consider scaling up/out, optimizing resource usage, or load shedding.",
                    "Review resource allocation and optimize high-usage components."
                ],
                "timestamp": time.time()
            }
            system_bottlenecks.append(bottleneck)
        
        return system_bottlenecks
    
    def _generate_specific_recommendations(
        self,
        stage_name: str,
        profile: Dict[str, Any],
        flags: List[str]
    ) -> List[str]:
        """Generate stage-specific optimization recommendations."""
        recommendations = []
        
        # Common recommendations based on stage name patterns
        if "embedding" in stage_name or "vector" in stage_name:
            recommendations.append(
                "Consider optimizing embedding operations: dimensionality reduction, "
                "caching embeddings, or using approximate nearest neighbors."
            )
        
        if "database" in stage_name or "db" in stage_name or "sql" in stage_name:
            recommendations.append(
                "Database operations can be optimized by adding indices, query optimization, "
                "connection pooling, or caching frequently accessed data."
            )
        
        if "http" in stage_name or "api" in stage_name or "request" in stage_name:
            recommendations.append(
                "Network operations can be optimized with connection pooling, keep-alive connections, "
                "parallel requests, or moving to asynchronous I/O patterns."
            )
        
        if "parse" in stage_name or "json" in stage_name or "xml" in stage_name:
            recommendations.append(
                "Parsing operations can be optimized with streaming parsers, schema validation skipping, "
                "or selective parsing of only required fields."
            )
        
        if "validation" in stage_name or "verify" in stage_name or "check" in stage_name:
            recommendations.append(
                "Consider optimizing validation: early termination on failure, caching validation results, "
                "or implementing progressive validation with most common checks first."
            )
        
        # Add default recommendation if none generated
        if not recommendations:
            recommendations.append(
                f"Optimize the {stage_name} implementation to reduce execution time or call frequency."
            )
            
            # Add general recommendation based on timing category
            if "critical_latency" in flags:
                recommendations.append(
                    "Critical latency detected. This stage requires immediate optimization attention."
                )
        
        return recommendations
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive bottleneck analysis report.
        
        Returns:
            Dictionary with bottleneck analysis and recommendations
        """
        bottlenecks = self.detect_bottlenecks()
        
        # Get profiling data
        stage_profiles = self.profiler.get_all_stage_profiles()
        scaling_metrics = self.profiler.get_scaling_metrics()
        
        # Generate stage timing summary
        stage_summary = []
        for name, profile in stage_profiles.items():
            stage_summary.append({
                "name": name,
                "avg_time_ms": profile.get("avg_time", 0),
                "total_time_ms": profile.get("total_time", 0),
                "call_count": profile.get("count", 0),
                "category": profile.get("category", "trace")
            })
        
        # Sort by total time (descending)
        stage_summary.sort(key=lambda s: s["total_time_ms"], reverse=True)
        
        # Generate optimization plan: prioritize bottlenecks into high/medium/low
        optimization_plan = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": []
        }
        
        for bottleneck in bottlenecks:
            if bottleneck["severity"] == 3:
                optimization_plan["high_priority"].append(bottleneck)
            elif bottleneck["severity"] == 2:
                optimization_plan["medium_priority"].append(bottleneck)
            else:
                optimization_plan["low_priority"].append(bottleneck)
        
        # Generate report
        report = {
            "timestamp": time.time(),
            "profiler_instance_id": self.profiler.instance_id,
            "bottlenecks": bottlenecks,
            "stage_summary": stage_summary,
            "scaling_metrics": scaling_metrics,
            "optimization_plan": optimization_plan,
            "system_recommendations": self._generate_system_recommendations(
                bottlenecks, scaling_metrics, stage_profiles
            )
        }
        
        return report
    
    def _generate_system_recommendations(
        self,
        bottlenecks: List[Dict[str, Any]],
        scaling_metrics: Dict[str, Any],
        stage_profiles: Dict[str, Any]
    ) -> List[str]:
        """Generate system-level recommendations."""
        recommendations = []
        
        # Count bottlenecks by severity
        severity_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for b in bottlenecks:
            severity_counts[b.get("severity", 0)] += 1
        
        # Critical bottlenecks found
        if severity_counts[3] > 0:
            recommendations.append(
                f"Critical bottlenecks detected ({severity_counts[3]}). "
                "Address high-priority issues first for maximum performance improvement."
            )
        
        # Check for concurrency issues
        if scaling_metrics.get("concurrent_requests", 0) > 10 and scaling_metrics.get("efficiency_score", 1.0) < 0.5:
            recommendations.append(
                "Poor scaling efficiency under concurrent load. "
                "Consider optimizing for concurrency, reducing shared resource contention, "
                "or implementing better parallelization."
            )
        
        # Check for I/O bound vs CPU bound
        io_bound_count = sum(1 for b in bottlenecks if "high_io" in b.get("flags", []))
        cpu_bound_count = sum(1 for b in bottlenecks if "high_cpu" in b.get("flags", []))
        
        if io_bound_count > cpu_bound_count:
            recommendations.append(
                "System appears to be I/O bound. Consider async I/O patterns, caching, "
                "or optimizing data access patterns."
            )
        elif cpu_bound_count > io_bound_count:
            recommendations.append(
                "System appears to be CPU bound. Consider algorithmic optimizations, "
                "parallelization, or offloading work to background tasks."
            )
        
        # Check for potential memory issues
        memory_issues = sum(1 for b in bottlenecks if "high_memory" in b.get("flags", []))
        if memory_issues > 0:
            recommendations.append(
                f"Memory usage issues detected in {memory_issues} components. "
                "Review for memory leaks, excessive object allocation, or "
                "inefficient data structures."
            )
        
        # Add general recommendations if few specific ones
        if len(recommendations) < 2:
            recommendations.append(
                "Regularly profile your application under production-like load to detect new bottlenecks "
                "as the system evolves."
            )
            
            recommendations.append(
                "Consider implementing performance budgets and automated performance regression testing "
                "to maintain performance over time."
            )
        
        return recommendations
    
    def export_report(self, filename: Optional[str] = None) -> str:
        """
        Export a bottleneck analysis report to a JSON file.
        
        Args:
            filename: Optional filename for the export
            
        Returns:
            Path to the exported file
        """
        # Generate report
        report = self.generate_report()
        
        # Determine export path
        if not filename:
            timestamp = int(time.time())
            filename = f"acdek_bottleneck_{timestamp}.json"
        
        filepath = os.path.join(self.export_directory, filename)
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Exported bottleneck report to {filepath}")
        return filepath