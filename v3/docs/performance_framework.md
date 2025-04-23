# ACDEK v3 Performance Framework

This document provides a comprehensive guide to the performance framework in ACDEK v3, which includes both measurement tools and continuous monitoring capabilities.

## Overview

The ACDEK v3 Performance Framework consists of two integrated systems:

1. **Performance Measurement Framework** - For detailed profiling, bottleneck detection, and performance visualization.
2. **Performance Monitoring System** - For continuous performance monitoring, regression detection, and alerting.

Together, these systems provide enterprise-grade performance observability and optimization capabilities, enabling developers and operations teams to:

- Measure and optimize execution time across all ACDEK pipeline stages
- Detect performance bottlenecks automatically
- Monitor performance in real-time and detect regressions
- Visualize metrics through interactive dashboards
- Receive alerts on performance degradations

## Performance Measurement Framework

### Core Components

The measurement framework consists of the following core components:

#### 1. PerformanceProfiler

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

# Get performance snapshot
snapshot = profiler.get_snapshot()
```

#### 2. BottleneckDetector

The `BottleneckDetector` class analyzes performance data to identify bottlenecks and provide optimization recommendations:

```python
from observability.bottleneck_detector import BottleneckDetector, BottleneckConfig

# Create detector with custom thresholds
config = BottleneckConfig(
    warning_threshold_ms=100.0,
    critical_threshold_ms=200.0
)
detector = BottleneckDetector(profiler, config=config)

# Detect bottlenecks
bottlenecks = detector.detect_bottlenecks()
```

#### 3. PerformanceVisualizer

The `PerformanceVisualizer` class creates visualizations and reports:

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
```

#### 4. EnhancedMetricsCollector

The `EnhancedMetricsCollector` class extends the base `MetricsCollector` with:

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
```

## Performance Monitoring System

The Performance Monitoring System builds upon the measurement framework to provide continuous monitoring, regression detection, and alerting capabilities.

### Core Components

#### 1. Benchmark Runner

Located in `benchmarks/benchmark_run.py`, this component executes benchmarks with configurable parameters:

- Load testing with concurrent users
- Multi-stage testing (smoke, functional, performance, stress)
- Resource tracking for CPU, memory, and threads
- Detailed test results and performance metrics

```python
from benchmarks.benchmark_run import (
    run_benchmark, BenchmarkConfig, run_load_test, run_multi_stage_test
)

# Create configuration
config = BenchmarkConfig(
    domains=["medical", "legal", "financial"],
    include_v2=True,
    parallel=True
)

# Run benchmark
results = run_benchmark(config)

# Run load test
load_results = run_load_test(
    test_cases=test_cases,
    config=config,
    concurrent_users=10,
    ramp_up=30,
    duration=300
)
```

#### 2. Continuous Monitor

Located in `benchmarks/continuous_monitor.py`, this component provides:

- Daemon process for ongoing performance monitoring
- Configurable monitoring intervals
- Metrics collection and storage
- Integration with regression detection
- Alert management

```python
from benchmarks.continuous_monitor import ContinuousMonitor

# Create monitor
monitor = ContinuousMonitor(
    interval=3600,  # 1 hour between runs
    baseline_path="baseline.json",
    alert_threshold=0.05  # 5% regression threshold
)

# Start monitoring
monitor.start()

# Run once without starting daemon
result = monitor.run_once()

# Stop monitoring
monitor.stop()
```

#### 3. Regression Detector

Located in `benchmarks/regression_detector.py`, this component offers:

- Statistical detection of performance regressions
- Severity classification (minor, moderate, severe, critical)
- Domain impact analysis
- Trend analysis
- Baseline optimization

```python
from benchmarks.regression_detector import RegressionDetector

# Create detector
detector = RegressionDetector(
    significance_level=0.05,
    regression_thresholds={
        "minor": 0.02,  # 2% change
        "moderate": 0.05,  # 5% change
        "severe": 0.1,  # 10% change
        "critical": 0.2  # 20% change
    }
)

# Detect regressions
analysis = detector.detect_regressions(
    current_results=current_data,
    baseline_results=baseline_data
)

# Analyze trends
trend_analysis = detector.analyze_trend(historical_results)
```

#### 4. Performance Dashboard

Located in `benchmarks/performance_dashboard.py`, this component provides:

- Real-time visualization of performance metrics
- Interactive charts and graphs
- Regression visualization
- Alert management interface
- Domain-specific performance views

```python
from benchmarks.performance_dashboard import run_dashboard

# Run dashboard with monitor
run_dashboard(monitor, port=8080)
```

#### 5. API Endpoints

Located in `web/api/main.py`, the API server provides:

- RESTful API for performance data
- WebSocket connections for real-time updates
- Control interface for the monitoring system

```
# RESTful API Endpoints
GET /status - Get monitoring system status
GET /metrics - Get performance metrics
GET /alerts - Get active alerts
POST /monitor/command - Send commands to the monitoring system

# WebSocket API
Connect to /ws for real-time updates
```

## Performance Metrics

The framework tracks the following key metrics:

### Core Metrics
- **Hallucination Reduction**: Measures the effectiveness of ACDEK in reducing hallucinations
- **Latency Overhead**: Measures the additional processing time introduced by ACDEK
- **Confidence Score**: Measures the confidence in ACDEK's output
- **Success Rate**: Measures the percentage of successful pipeline executions

### Resource Metrics
- **CPU Usage**: Average and peak CPU utilization
- **Memory Usage**: Average and peak memory consumption
- **Thread Count**: Number of threads used during execution
- **Execution Time**: Total time to complete benchmarks

### Domain-specific Metrics
- Performance metrics broken down by domain (medical, legal, financial, etc.)
- Domain-specific hallucination reduction rates
- Domain-specific latency overhead

## Regression Detection

The regression detection system uses statistical analysis to identify performance degradations:

### Detection Methods
- **Delta Analysis**: Compares current values against baseline
- **Significance Testing**: Determines if changes are statistically significant
- **Trend Analysis**: Analyzes patterns over multiple runs

### Severity Classification
- **Minor**: Slight degradations with minimal impact
- **Moderate**: Notable degradations requiring attention
- **Severe**: Significant degradations requiring immediate action
- **Critical**: Extreme degradations potentially blocking deployment

## Alerts and Notifications

The alert system provides timely notifications of performance issues:

### Alert Types
- **Regression Alerts**: Triggered by detected performance regressions
- **Resource Alerts**: Triggered by excessive resource usage
- **Threshold Alerts**: Triggered when metrics exceed defined thresholds
- **System Alerts**: Triggered by monitoring system issues

### Notification Channels
- **Console**: Terminal output for local development
- **Email**: Email notifications for critical issues
- **Webhook**: Integration with external systems
- **Slack**: Team notifications for collaborative response
- **Dashboard**: Visual alerts in the monitoring dashboard

## Dashboard

The performance dashboard provides a unified view of system performance:

### Dashboard Components
- **Overview Panel**: High-level performance summary
- **Trend Analysis**: Historical performance trends
- **Regression Analysis**: Details of detected regressions
- **Resource Monitoring**: System resource usage metrics
- **Domain Analysis**: Domain-specific performance metrics
- **Alert Management**: Active and historical alerts

## Usage Guide

### Running Benchmarks
```bash
# Run a standard benchmark
python benchmark_run.py

# Run a load test with 10 concurrent users
python benchmark_run.py --load-test --users 10

# Run a multi-stage test (smoke, functional, performance, stress)
python benchmark_run.py --multi-stage
```

### Continuous Monitoring
```bash
# Start continuous monitoring with default settings
python continuous_monitor.py

# Start monitoring with custom interval and baseline
python continuous_monitor.py --interval 3600 --baseline baseline.json

# Start monitoring with dashboard
python continuous_monitor.py --dashboard
```

### Regression Detection
```bash
# Detect regressions between current and baseline results
python regression_detector.py --current results/latest.json --baseline results/baseline.json

# Analyze performance trends
python regression_detector.py --current results/latest.json --analyze-trend
```

### Dashboard
```bash
# Start the performance dashboard
python performance_dashboard.py

# Start dashboard on a specific port
python performance_dashboard.py --port 8080
```

## Performance Overhead

The framework is designed to have minimal impact on production performance:

- Base profiling overhead: <1ms per profile point
- Resource monitoring overhead: ~2-3ms when enabled
- Memory usage: <10MB for storing 1000 profile points
- Total overhead per request: <5ms (without resource monitoring)
- Continuous monitoring: Configurable intervals to manage impact

## Best Practices

For optimal results with the performance framework:

1. **Profile granularity**: Create profile points around meaningful operations, not too fine-grained or too coarse
2. **Naming conventions**: Use consistent, hierarchical naming for stages (e.g., `enrichment.vector_alignment`)
3. **Request correlation**: Always pass request IDs to enable end-to-end tracing
4. **Regular monitoring**: Run continuous monitoring in pre-production to catch performance regressions early
5. **Custom thresholds**: Adjust regression and alert thresholds based on your specific performance requirements
6. **Baseline management**: Regularly update baseline measurements after validated performance improvements
7. **Alert routing**: Configure appropriate alert channels based on severity and team responsibilities

## Example Usage

A complete example demonstrating the performance framework is available in `/examples/performance_measurement_example.py`. For monitoring examples, see `/examples/continuous_monitoring_example.py`.

## Conclusion

The ACDEK v3 Performance Framework provides comprehensive tools for understanding, optimizing, monitoring, and maintaining system performance. By integrating this framework into your development and operations workflows, you can ensure ACDEK meets enterprise-grade performance requirements and quickly identify opportunities for optimization.