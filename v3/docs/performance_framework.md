# ACDEK v3 Performance Measurement Framework

This document provides a comprehensive guide to the performance measurement framework in ACDEK v3, which enables detailed profiling, bottleneck detection, and performance visualization.

## Overview

The ACDEK v3 Performance Measurement Framework is designed to provide enterprise-grade performance monitoring and optimization capabilities. It enables developers and operations teams to:

1. **Measure execution time** across all ACDEK pipeline stages with microsecond precision
2. **Detect bottlenecks** automatically with intelligent analysis and recommendations
3. **Visualize performance metrics** through interactive dashboards and CLI reports
4. **Monitor performance in real-time** to catch regressions and optimize scaling

The framework is built to have minimal overhead (<5ms per request) and provides both programmatic and CLI interfaces for integration into development workflows, CI/CD pipelines, and production monitoring systems.

## Core Components

The framework consists of the following core components:

### 1. PerformanceProfiler

The `PerformanceProfiler` class is the foundation of the framework, providing:

- High-precision timing of execution stages
- Parent-child relationship tracking between stages
- Resource usage monitoring (CPU, memory, I/O)
- Automatic latency categorization
- Statistical analysis of performance data

```python
from observability.performance_profiler import PerformanceProfiler

# Create profiler instance
profiler = PerformanceProfiler(
    enable_resource_monitoring=True,
    export_directory="./performance_data"
)

# Profile a stage
with profiler.profile_stage("my_stage", request_id="req_123"):
    # Code to be profiled
    process_data()

# Profile a complete request
with profiler.profile_request(request_id="req_123") as req_id:
    # Request processing code
    with profiler.profile_stage("stage1", req_id):
        step1()
    with profiler.profile_stage("stage2", req_id):
        step2()

# Get performance snapshot
snapshot = profiler.get_snapshot()

# Export to file
profiler.export_snapshot(filename="performance_snapshot.json")
```

### 2. BottleneckDetector

The `BottleneckDetector` class analyzes performance data to identify bottlenecks and provide optimization recommendations:

- Intelligent bottleneck detection with configurable thresholds
- Severity classification (critical, warning, info)
- Stage-specific optimization recommendations
- System-level performance analysis
- Scaling efficiency metrics

```python
from observability.bottleneck_detector import BottleneckDetector, BottleneckConfig

# Create detector with custom thresholds
config = BottleneckConfig(
    warning_threshold_ms=100.0,  # Warning threshold in ms
    critical_threshold_ms=200.0   # Critical threshold in ms
)
detector = BottleneckDetector(profiler, config=config)

# Detect bottlenecks
bottlenecks = detector.detect_bottlenecks()

# Generate comprehensive report
report = detector.generate_report()

# Export report to file
detector.export_report(filename="bottleneck_report.json")
```

### 3. PerformanceVisualizer

The `PerformanceVisualizer` class creates visualizations and reports:

- Interactive HTML dashboards
- Stage timing charts
- Relative execution time breakdowns
- Bottleneck impact visualizations
- CLI-friendly text reports

```python
from observability.performance_visualizer import PerformanceVisualizer

# Create visualizer
visualizer = PerformanceVisualizer(
    output_directory="./performance_visualizations"
)

# Create dashboard
dashboard_path = visualizer.create_dashboard(
    profile_data=snapshot.to_dict(),
    bottleneck_data=report,
    filename="performance_dashboard.html"
)

# Generate CLI report
cli_report = visualizer.create_cli_report(
    profile_data=snapshot.to_dict(),
    bottleneck_data=report
)
print(cli_report)
```

### 4. EnhancedMetricsCollector

The `EnhancedMetricsCollector` class extends the base `MetricsCollector` with:

- Integration with the performance profiler
- Advanced histogram metrics
- Prometheus-compatible export
- Distributed trace context
- Request correlation

```python
from observability.metrics_collector_enhanced import (
    EnhancedMetricsCollector, ProfiledTimingContext
)

# Create enhanced metrics collector
metrics = EnhancedMetricsCollector(
    service_name="my_service",
    enable_profiling=True
)

# Time execution with metrics and profiling
with metrics.time_execution("my_operation", request_id="req_123"):
    # Code to be timed
    perform_operation()

# Use standalone timing context
with ProfiledTimingContext("another_operation", metrics, "req_123"):
    # More code to be timed
    another_operation()

# Export metrics
metrics.export_to_file(format="prometheus", file_path="metrics.prom")
```

### 5. CLI Tools

The framework includes a command-line interface for performance measurement:

```bash
# Run profiling
python performance_cli.py profile --duration=60 --interval=5 --export

# Analyze bottlenecks
python performance_cli.py analyze --input=profile_data.json --report-format=html

# Visualize performance data
python performance_cli.py visualize --input=profile_data.json --bottleneck-data=bottleneck_report.json

# Monitor performance in real-time
python performance_cli.py monitor --interval=5 --dashboard
```

## Integration with ACDEK

The framework is designed to integrate seamlessly with ACDEK v3:

### ACDEKKernel Integration

```python
from acdek.acdek_kernel import ACDEKKernel
from observability.performance_profiler import PerformanceProfiler

profiler = PerformanceProfiler()
kernel = ACDEKKernel()

# Process a request with profiling
with profiler.profile_request() as req_id:
    request = {
        "query": "What is the inventory status?",
        "context": {"request_id": req_id}
    }
    
    # Process with ACDEK kernel
    result = kernel.run(
        input_data=request,
        context=request["context"]
    )
```

### EnrichmentPipeline Integration

```python
from enrichment.enrichment_pipeline import EnrichmentPipeline
from observability.metrics_collector_enhanced import (
    EnhancedMetricsCollector, ProfiledTimingContext
)

# Create enhanced metrics collector
metrics = EnhancedMetricsCollector(service_name="enrichment_pipeline")

# Create pipeline with timing
pipeline = EnrichmentPipeline(
    stages=[...],
    execution_mode="adaptive"
)

# Execute pipeline with profiling
with metrics.trace_request("req_123") as req_id:
    with metrics.time_execution("pipeline_execute", req_id):
        result = pipeline.execute(input_data, context)
```

## Performance Overhead

The framework is designed to have minimal impact on production performance:

- Base profiling overhead: <1ms per profile point
- Resource monitoring overhead: ~2-3ms when enabled
- Memory usage: <10MB for storing 1000 profile points
- Total overhead per request: <5ms (without resource monitoring)

For production environments, you can selectively disable resource monitoring or reduce profiling granularity to reduce overhead further.

## Example Usage

A complete example demonstrating the performance framework is available in `/examples/performance_measurement_example.py`. This example shows:

- Profiling ACDEK pipeline execution
- Analyzing bottlenecks
- Creating visualizations
- Monitoring performance in real-time

```bash
# Run full example
python examples/performance_measurement_example.py --mode=all

# Run specific mode
python examples/performance_measurement_example.py --mode=profile --request-count=100
```

## Dashboard Features

The generated performance dashboard includes:

- **Overview**: High-level performance metrics and charts
- **Bottlenecks**: Detailed analysis of performance bottlenecks with recommendations
- **Stage Details**: Complete breakdown of all pipeline stages
- **Raw Data**: (Optional) Full raw performance data for advanced analysis

The dashboard is designed to be self-contained and can be shared with team members or included in CI/CD reports.

## Best Practices

For optimal results with the performance framework:

1. **Profile granularity**: Create profile points around meaningful operations, not too fine-grained or too coarse
2. **Naming conventions**: Use consistent, hierarchical naming for stages (e.g., `enrichment.vector_alignment`)
3. **Request correlation**: Always pass request IDs to enable end-to-end tracing
4. **Regular profiling**: Integrate profiling into CI/CD to catch performance regressions early
5. **Custom thresholds**: Adjust thresholds based on your specific performance requirements
6. **Production monitoring**: Use the monitoring mode with lower overhead settings in production

## Conclusion

The ACDEK v3 Performance Measurement Framework provides comprehensive tools for understanding, optimizing, and monitoring system performance. By integrating this framework into your development and operations workflows, you can ensure ACDEK meets enterprise-grade performance requirements and quickly identify opportunities for optimization.