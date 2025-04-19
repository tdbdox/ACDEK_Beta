"""
Tests for ACDEK v3 performance measurement framework.
"""
import unittest
import time
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock

import sys
import os
# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observability.performance_profiler import (
    PerformanceProfiler, StageProfile, ResourceUsage, LatencyCategory
)
from observability.bottleneck_detector import BottleneckDetector, BottleneckConfig
from observability.performance_visualizer import PerformanceVisualizer
from observability.metrics_collector_enhanced import (
    EnhancedMetricsCollector, ProfiledTimingContext
)

class TestPerformanceProfiler(unittest.TestCase):
    """Test the PerformanceProfiler class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for exports
        self.temp_dir = tempfile.mkdtemp()
        
        # Create profiler with resource monitoring disabled for testing
        self.profiler = PerformanceProfiler(
            enable_resource_monitoring=False,
            export_directory=self.temp_dir
        )
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_profile_stage(self):
        """Test profiling a stage."""
        # Profile a test stage
        with self.profiler.profile_stage("test_stage", "test_request"):
            time.sleep(0.01)  # Small delay to ensure measurable duration
        
        # Check that the stage was recorded
        self.assertIn("test_stage", self.profiler._stage_profiles)
        
        # Check profiling data
        stage = self.profiler._stage_profiles["test_stage"]
        self.assertEqual(stage.name, "test_stage")
        self.assertEqual(stage.count, 1)
        self.assertGreater(stage.total_time, 0)
        self.assertGreater(stage.last_execution_time, 0)
        
        # Test with an error
        try:
            with self.profiler.profile_stage("error_stage", "test_request"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Check that the error was recorded
        self.assertIn("error_stage", self.profiler._stage_profiles)
        self.assertEqual(self.profiler._stage_profiles["error_stage"].errors, 1)
    
    def test_profile_request(self):
        """Test profiling a complete request."""
        # Profile a test request
        with self.profiler.profile_request("test_request") as req_id:
            # Profile some stages within the request
            with self.profiler.profile_stage("stage1", req_id):
                time.sleep(0.01)
            
            with self.profiler.profile_stage("stage2", req_id):
                time.sleep(0.01)
        
        # Check that stages were recorded
        self.assertIn("stage1", self.profiler._stage_profiles)
        self.assertIn("stage2", self.profiler._stage_profiles)
        
        # Check request total was recorded
        self.assertIn("request_total", self.profiler._stage_profiles)
        
        # Test scaling metrics
        scaling_metrics = self.profiler.get_scaling_metrics()
        self.assertEqual(scaling_metrics["concurrent_requests"], 0)  # Back to 0 after request
        self.assertGreater(scaling_metrics["avg_latency_ms"], 0)
        self.assertGreater(scaling_metrics["throughput_rps"], 0)
    
    def test_stage_dependency_tracking(self):
        """Test tracking dependencies between stages."""
        # Create a parent-child dependency
        thread_id = threading.get_ident()
        self.profiler._execution_stack[thread_id] = []
        
        with self.profiler.profile_stage("parent_stage"):
            with self.profiler.profile_stage("child_stage"):
                time.sleep(0.01)
        
        # Check dependency was recorded
        parent = self.profiler._stage_profiles["parent_stage"]
        child = self.profiler._stage_profiles["child_stage"]
        
        self.assertIn("child_stage", parent.child_stages)
        self.assertIn("parent_stage", child.parent_stages)
    
    def test_latency_categorization(self):
        """Test categorization of latency measurements."""
        # Test different latencies
        test_cases = [
            (5.0, LatencyCategory.EXCELLENT),  # Below trace threshold
            (15.0, LatencyCategory.TRACE),     # Above trace, below info
            (75.0, LatencyCategory.INFO),      # Above info, below warning
            (150.0, LatencyCategory.WARNING),  # Above warning, below critical
            (250.0, LatencyCategory.CRITICAL)  # Above critical
        ]
        
        for latency, expected_category in test_cases:
            stage_name = f"stage_{latency}"
            
            # Create stage profile
            self.profiler._stage_profiles[stage_name] = StageProfile(name=stage_name)
            
            # Categorize latency
            self.profiler._categorize_latency(stage_name, latency)
            
            # Check category
            self.assertEqual(
                self.profiler._stage_profiles[stage_name].category,
                expected_category
            )
    
    def test_detect_bottlenecks(self):
        """Test bottleneck detection."""
        # Create some test stages with different latencies
        test_stages = [
            ("fast_stage", 10.0, 100),    # Fast stage, many calls
            ("medium_stage", 50.0, 20),   # Medium stage, some calls
            ("slow_stage", 150.0, 5),     # Slow stage (warning), few calls
            ("critical_stage", 250.0, 3)  # Critical stage, very few calls
        ]
        
        for name, latency, count in test_stages:
            self.profiler._stage_profiles[name] = StageProfile(name=name)
            for _ in range(count):
                self.profiler._stage_profiles[name].add_execution(latency)
            self.profiler._categorize_latency(name, latency)
        
        # Detect bottlenecks
        bottlenecks = self.profiler.detect_bottlenecks()
        
        # Check results
        self.assertEqual(len(bottlenecks), 2)  # slow and critical stages
        
        # Bottlenecks should be sorted by category (severity)
        self.assertEqual(bottlenecks[0]["stage_name"], "critical_stage")
        self.assertEqual(bottlenecks[1]["stage_name"], "slow_stage")
        
        # Check recommendations
        self.assertGreater(len(bottlenecks[0]["recommendations"]), 0)
    
    def test_snapshot_and_export(self):
        """Test getting a snapshot and exporting it."""
        # Add some test data
        with self.profiler.profile_stage("test_stage"):
            time.sleep(0.01)
        
        # Get snapshot
        snapshot = self.profiler.get_snapshot()
        
        # Check snapshot content
        self.assertIn("test_stage", snapshot.stage_profiles)
        self.assertIsInstance(snapshot.system_load, dict)
        self.assertIsInstance(snapshot.bottlenecks, list)
        
        # Export snapshot
        export_path = self.profiler.export_snapshot(filename="test_snapshot.json")
        
        # Check export file exists and is valid JSON
        self.assertTrue(os.path.exists(export_path))
        
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        self.assertIn("stage_profiles", export_data)
        self.assertIn("test_stage", export_data["stage_profiles"])


class TestBottleneckDetector(unittest.TestCase):
    """Test the BottleneckDetector class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for exports
        self.temp_dir = tempfile.mkdtemp()
        
        # Create profiler with resource monitoring disabled for testing
        self.profiler = PerformanceProfiler(
            enable_resource_monitoring=False,
            export_directory=self.temp_dir
        )
        
        # Create detector
        self.detector = BottleneckDetector(
            self.profiler,
            export_directory=self.temp_dir
        )
        
        # Add some test data
        test_stages = [
            ("fast_stage", 10.0, 100),    # Fast stage, many calls
            ("medium_stage", 50.0, 20),   # Medium stage, some calls
            ("slow_stage", 150.0, 5),     # Slow stage (warning), few calls
            ("critical_stage", 250.0, 3)  # Critical stage, very few calls
        ]
        
        for name, latency, count in test_stages:
            self.profiler._stage_profiles[name] = StageProfile(name=name)
            for _ in range(count):
                self.profiler._stage_profiles[name].add_execution(latency)
            self.profiler._categorize_latency(name, latency)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_detect_bottlenecks(self):
        """Test bottleneck detection with custom config."""
        # Create custom config with lower thresholds
        config = BottleneckConfig(
            warning_threshold_ms=40.0,
            critical_threshold_ms=100.0
        )
        
        self.detector.config = config
        
        # Detect bottlenecks
        bottlenecks = self.detector.detect_bottlenecks()
        
        # Check results
        self.assertEqual(len(bottlenecks), 3)  # medium, slow, and critical stages
        
        # Check severity levels
        severity_map = {b["stage_name"]: b["severity"] for b in bottlenecks}
        self.assertEqual(severity_map["critical_stage"], 3)  # Critical
        self.assertGreaterEqual(severity_map["slow_stage"], 2)  # Warning or higher
        self.assertGreaterEqual(severity_map["medium_stage"], 1)  # At least low
    
    def test_generate_report(self):
        """Test generating a bottleneck report."""
        # Generate report
        report = self.detector.generate_report()
        
        # Check report structure
        self.assertIn("bottlenecks", report)
        self.assertIn("stage_summary", report)
        self.assertIn("system_recommendations", report)
        self.assertIn("optimization_plan", report)
        
        # Check bottlenecks are categorized by priority
        self.assertEqual(len(report["optimization_plan"]["high_priority"]), 1)  # critical_stage
        self.assertEqual(len(report["optimization_plan"]["medium_priority"]), 1)  # slow_stage
    
    def test_export_report(self):
        """Test exporting a bottleneck report."""
        # Export report
        export_path = self.detector.export_report(filename="test_bottleneck.json")
        
        # Check export file exists and is valid JSON
        self.assertTrue(os.path.exists(export_path))
        
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        self.assertIn("bottlenecks", export_data)
        self.assertIn("system_recommendations", export_data)


class TestEnhancedMetricsCollector(unittest.TestCase):
    """Test the EnhancedMetricsCollector class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for exports
        self.temp_dir = tempfile.mkdtemp()
        
        # Create enhanced metrics collector
        self.metrics = EnhancedMetricsCollector(
            service_name="test_service",
            enable_profiling=True,
            profiler_config={"export_directory": self.temp_dir}
        )
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_time_execution(self):
        """Test timing execution with the enhanced metrics collector."""
        # Time a stage
        with self.metrics.time_execution("test_stage", "test_request"):
            time.sleep(0.01)
        
        # Check stage metrics
        self.assertGreater(self.metrics._stage_timings.get("test_stage", {}).get("total_time", 0), 0)
        
        # Check histogram
        self.assertIn("duration.test_stage", self.metrics._histograms)
        self.assertEqual(self.metrics._histograms["duration.test_stage"]["count"], 1)
        
        # Check profiler integration
        profiler = self.metrics.get_profiler()
        self.assertIsNotNone(profiler)
        self.assertIn("test_stage", profiler._stage_profiles)
    
    def test_trace_request(self):
        """Test tracing a request with the enhanced metrics collector."""
        # Trace a request
        with self.metrics.trace_request("test_request") as req_id:
            # Time some stages
            with self.metrics.time_execution("stage1", req_id):
                time.sleep(0.01)
            
            with self.metrics.time_execution("stage2", req_id):
                time.sleep(0.01)
        
        # Check trace context
        trace = self.metrics.get_trace_context("test_request")
        self.assertIsNotNone(trace)
        self.assertIn("duration", trace)
        
        # Check profiler integration
        profiler = self.metrics.get_profiler()
        self.assertIn("stage1", profiler._stage_profiles)
        self.assertIn("stage2", profiler._stage_profiles)
    
    def test_profiled_timing_context(self):
        """Test the ProfiledTimingContext class."""
        # Create context
        with ProfiledTimingContext("test_context", self.metrics, "test_request"):
            time.sleep(0.01)
        
        # Check metrics
        self.assertGreater(self.metrics._stage_timings.get("test_context", {}).get("total_time", 0), 0)
        
        # Check profiler integration
        profiler = self.metrics.get_profiler()
        self.assertIn("test_context", profiler._stage_profiles)
    
    def test_metrics_export(self):
        """Test exporting metrics."""
        # Add some metrics
        self.metrics.counter_inc("test_counter", 5)
        self.metrics.gauge_set("test_gauge", 10.5)
        self.metrics.histogram_observe("test_histogram", 15.0)
        
        # Export metrics
        export_path = self.metrics.export_to_file(
            format="json",
            file_path=os.path.join(self.temp_dir, "test_metrics.json")
        )
        
        # Check export file exists and is valid JSON
        self.assertTrue(os.path.exists(export_path))
        
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        self.assertEqual(export_data["counters"]["test_counter"], 5)
        self.assertEqual(export_data["gauges"]["test_gauge"], 10.5)
        self.assertEqual(export_data["histograms"]["test_histogram"]["count"], 1)


class TestPerformanceVisualizer(unittest.TestCase):
    """Test the PerformanceVisualizer class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for exports
        self.temp_dir = tempfile.mkdtemp()
        
        # Create visualizer
        self.visualizer = PerformanceVisualizer(
            output_directory=self.temp_dir
        )
        
        # Create sample data
        self.profile_data = {
            "stage_profiles": {
                "fast_stage": {
                    "name": "fast_stage",
                    "avg_time": 10.0,
                    "total_time": 1000.0,
                    "count": 100,
                    "category": "excellent"
                },
                "medium_stage": {
                    "name": "medium_stage",
                    "avg_time": 50.0,
                    "total_time": 1000.0,
                    "count": 20,
                    "category": "info"
                },
                "slow_stage": {
                    "name": "slow_stage",
                    "avg_time": 150.0,
                    "total_time": 750.0,
                    "count": 5,
                    "category": "warning"
                },
                "critical_stage": {
                    "name": "critical_stage",
                    "avg_time": 250.0,
                    "total_time": 750.0,
                    "count": 3,
                    "category": "critical"
                }
            },
            "system_load": {
                "cpu_percent": 25.0,
                "memory_percent": 40.0
            },
            "request_metrics": {
                "avg": 450.0,
                "max": 750.0
            }
        }
        
        self.bottleneck_data = {
            "bottlenecks": [
                {
                    "stage_name": "critical_stage",
                    "category": "critical",
                    "severity": 3,
                    "avg_time_ms": 250.0,
                    "p95_time_ms": 300.0,
                    "call_count": 3,
                    "relative_impact": 0.3,
                    "recommendations": [
                        "Recommendation 1",
                        "Recommendation 2"
                    ]
                },
                {
                    "stage_name": "slow_stage",
                    "category": "warning",
                    "severity": 2,
                    "avg_time_ms": 150.0,
                    "p95_time_ms": 200.0,
                    "call_count": 5,
                    "relative_impact": 0.2,
                    "recommendations": [
                        "Recommendation 3",
                        "Recommendation 4"
                    ]
                }
            ],
            "system_recommendations": [
                "System recommendation 1",
                "System recommendation 2"
            ]
        }
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    @unittest.skipIf(os.environ.get('CI') == 'true', "Skip visualization tests in CI")
    def test_create_charts(self):
        """Test creating charts."""
        # Create stage timing chart
        stage_list = [{"name": name, **data} for name, data in self.profile_data["stage_profiles"].items()]
        
        chart_path = self.visualizer.create_stage_timing_chart(
            stage_list,
            filename="test_timing.png"
        )
        
        self.assertTrue(os.path.exists(chart_path))
        
        # Create relative timing chart
        chart_path = self.visualizer.create_relative_timing_chart(
            stage_list,
            filename="test_relative.png"
        )
        
        self.assertTrue(os.path.exists(chart_path))
        
        # Create bottleneck chart
        chart_path = self.visualizer.create_bottleneck_chart(
            self.bottleneck_data["bottlenecks"],
            filename="test_bottlenecks.png"
        )
        
        self.assertTrue(os.path.exists(chart_path))
    
    def test_create_dashboard(self):
        """Test creating a dashboard."""
        # Create dashboard
        dashboard_path = self.visualizer.create_dashboard(
            self.profile_data,
            self.bottleneck_data,
            filename="test_dashboard.html"
        )
        
        # Check dashboard file exists
        self.assertTrue(os.path.exists(dashboard_path))
        
        # Check dashboard content
        with open(dashboard_path, 'r') as f:
            content = f.read()
        
        # Check for key elements
        self.assertIn("critical_stage", content)
        self.assertIn("slow_stage", content)
        self.assertIn("Recommendation 1", content)
        self.assertIn("System recommendation 1", content)
    
    def test_create_cli_report(self):
        """Test creating a CLI report."""
        # Create CLI report
        report = self.visualizer.create_cli_report(
            self.profile_data,
            self.bottleneck_data
        )
        
        # Check report content
        self.assertIn("ACDEK PERFORMANCE REPORT", report)
        self.assertIn("critical_stage", report)
        self.assertIn("slow_stage", report)
        self.assertIn("Recommendation 1", report)
        self.assertIn("System recommendation 1", report)


if __name__ == '__main__':
    unittest.main()