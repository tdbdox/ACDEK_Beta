#!/usr/bin/env python
"""
ACDEK v3 Continuous Monitoring Example

This example demonstrates how to use the continuous monitoring and performance dashboard
features of the ACDEK framework to detect performance regressions and visualize metrics.

Features demonstrated:
1. Setting up continuous monitoring
2. Creating and using baselines
3. Detecting performance regressions
4. Generating alerts
5. Visualizing performance metrics
"""

import os
import sys
import argparse
import time
import json
import webbrowser
from pathlib import Path

# Add parent directory to path to allow importing from project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import ACDEK components
from benchmarks.continuous_monitor import ContinuousMonitor, Alert, AlertLevel
from benchmarks.regression_detector import RegressionDetector
from benchmarks.performance_dashboard import run_dashboard
from benchmarks.hallucination_benchmark import (
    BenchmarkConfig, Domain, TestCase, run_benchmark
)

# Constants
DEFAULT_OUTPUT_DIR = "./monitor_output"
DEFAULT_BASELINE_PATH = None  # Auto-generate baseline if none provided

def setup_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return Path(output_dir)

def create_baseline(output_dir):
    """Create a baseline by running a benchmark."""
    print("Creating baseline...")
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        domains=[Domain.MEDICAL, Domain.LEGAL, Domain.FINANCIAL],
        include_v2=True,
        parallel=True,
        max_workers=4,
        test_count=3,  # Use small number for example
        output_formats=["json"],
        output_dir=str(output_dir)
    )
    
    # Run benchmark to create baseline
    results = run_benchmark(config)
    
    # Save as baseline
    baseline_path = output_dir / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    
    print(f"Baseline created at {baseline_path}")
    return str(baseline_path)

def setup_monitor(output_dir, baseline_path=None, interval=60):
    """Set up continuous monitor."""
    # Create baseline if needed
    if baseline_path is None:
        baseline_path = create_baseline(output_dir)
    
    # Create monitor
    monitor = ContinuousMonitor(
        interval=interval,
        output_dir=str(output_dir),
        baseline_path=baseline_path,
        alert_threshold=0.05  # 5% regression threshold
    )
    
    print(f"Monitor created with {interval}s interval")
    return monitor

def demo_regression_detection(monitor):
    """Demonstrate regression detection."""
    print("\n=== Regression Detection Demo ===")
    
    # Run benchmark once
    result = monitor.run_once()
    print(f"Benchmark run completed (run_id: {result['run_id']})")
    
    # Print metrics
    if "metrics" in result:
        metrics = result["metrics"]
        print("\nPerformance Metrics:")
        if "metrics" in metrics:
            core_metrics = metrics["metrics"]
            for name, value in core_metrics.items():
                if isinstance(value, float):
                    if name.endswith("_rate") or name.endswith("reduction") or name.endswith("confidence"):
                        print(f"- {name}: {value*100:.2f}%")
                    else:
                        print(f"- {name}: {value:.2f}")
                else:
                    print(f"- {name}: {value}")
    
    # Print regression results
    if "regressions" in result:
        regressions = result["regressions"]
        has_regressions = regressions.get("has_regressions", False)
        
        if has_regressions:
            print("\nRegressions Detected:")
            severe_count = regressions.get("severe_count", 0)
            moderate_count = regressions.get("moderate_count", 0)
            minor_count = regressions.get("minor_count", 0)
            
            print(f"- {severe_count} severe")
            print(f"- {moderate_count} moderate")
            print(f"- {minor_count} minor")
            
            # Print details of severe regressions
            all_regressions = regressions.get("regressions", [])
            for regression in all_regressions:
                if "delta_percent" in regression and regression["delta_percent"] >= 0.1:  # 10% or more
                    print(f"\nSevere Regression in {regression['metric']}:")
                    print(f"- Baseline: {regression['baseline']:.4f}")
                    print(f"- Current: {regression['current']:.4f}")
                    print(f"- Change: {regression['delta']:+.4f} ({regression['delta_percent']*100:+.1f}%)")
                    if "domain" in regression and regression["domain"]:
                        print(f"- Domain: {regression['domain']}")
        else:
            print("\nNo regressions detected")
    
    return result

def demo_alert_system(monitor):
    """Demonstrate alert system."""
    print("\n=== Alert System Demo ===")
    
    # Get recent alerts
    alerts = monitor.get_recent_alerts(count=5)
    
    if not alerts:
        print("No recent alerts. Creating sample alerts...")
        
        # Create sample alerts
        alert_manager = monitor.alert_manager
        
        # Info alert
        alert_manager.send_alert(Alert(
            level=AlertLevel.INFO,
            message="Monitoring system started",
            metrics={"uptime": 0},
            source="demo"
        ))
        
        # Warning alert
        alert_manager.send_alert(Alert(
            level=AlertLevel.WARNING,
            message="Latency overhead increased by 7%",
            metrics={"latency_overhead": 45.3, "baseline_latency": 42.3},
            source="regression_detector"
        ))
        
        # Error alert
        alert_manager.send_alert(Alert(
            level=AlertLevel.ERROR,
            message="Hallucination reduction decreased by 12%",
            metrics={"hallucination_reduction": 0.68, "baseline_reduction": 0.78},
            source="regression_detector"
        ))
        
        # Get alerts again
        alerts = monitor.get_recent_alerts(count=5)
    
    # Display alerts
    print(f"Recent Alerts ({len(alerts)}):")
    for alert in alerts:
        level = alert.level if hasattr(alert, "level") else alert.get("level", "unknown")
        message = alert.message if hasattr(alert, "message") else alert.get("message", "No message")
        source = alert.source if hasattr(alert, "source") else alert.get("source", "unknown")
        
        print(f"\n[{level.upper()}] {message}")
        print(f"Source: {source}")
    
    return alerts

def run_continuous_monitor_demo(output_dir, baseline_path=None, interval=60, duration=300):
    """Run continuous monitoring demo."""
    print("\n=== Continuous Monitoring Demo ===")
    
    # Set up monitor
    monitor = setup_monitor(output_dir, baseline_path, interval)
    
    # Start monitoring
    print(f"Starting continuous monitoring for {duration} seconds...")
    monitor.start()
    
    try:
        # Wait for duration
        for i in range(duration // interval):
            time.sleep(interval)
            print(f"Elapsed time: {(i+1)*interval}s")
            
            # Get latest metrics
            latest_metrics = monitor.get_latest_metrics()
            if latest_metrics:
                metrics_dict = latest_metrics.to_dict() if hasattr(latest_metrics, "to_dict") else latest_metrics
                
                # Print summary
                if "metrics" in metrics_dict and "hallucination_reduction" in metrics_dict["metrics"]:
                    hall_red = metrics_dict["metrics"]["hallucination_reduction"] * 100
                    latency = metrics_dict["metrics"].get("latency_overhead", 0)
                    print(f"Latest metrics: {hall_red:.1f}% hallucination reduction, {latency:.1f}ms latency")
    
    except KeyboardInterrupt:
        print("\nMonitoring interrupted by user")
    
    finally:
        # Stop monitoring
        monitor.stop()
        print("Monitoring stopped")
    
    # Return monitor for dashboard
    return monitor

def run_dashboard_demo(monitor=None, output_dir=None, port=8080):
    """Run dashboard demo."""
    print("\n=== Dashboard Demo ===")
    
    if monitor is None and output_dir:
        # Create monitor from output
        monitor = setup_monitor(output_dir)
    
    if monitor is None:
        print("Error: No monitor available for dashboard")
        return
    
    print(f"Starting dashboard on port {port}...")
    print("Dashboard will open in your browser.")
    print("Press Ctrl+C to stop.")
    
    # Start dashboard
    try:
        run_dashboard(monitor, port=port)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ACDEK v3 Continuous Monitoring Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["regression", "alerts", "monitor", "dashboard", "all"],
        default="all",
        help="Demo mode to run"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_BASELINE_PATH,
        help="Path to baseline metrics file"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Interval between benchmark runs (seconds)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Duration of continuous monitoring (seconds)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for dashboard server"
    )
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    output_dir = setup_output_directory(args.output_dir)
    
    # Run selected demo mode
    if args.mode in ["regression", "all"]:
        monitor = setup_monitor(output_dir, args.baseline)
        demo_regression_detection(monitor)
    
    if args.mode in ["alerts", "all"]:
        monitor = setup_monitor(output_dir, args.baseline)
        demo_alert_system(monitor)
    
    if args.mode in ["monitor", "all"]:
        monitor = run_continuous_monitor_demo(
            output_dir,
            args.baseline,
            args.interval,
            args.duration
        )
    
    if args.mode in ["dashboard", "all"]:
        # If coming from monitor mode, reuse that monitor
        if args.mode == "all" or args.mode == "monitor":
            run_dashboard_demo(monitor, None, args.port)
        else:
            run_dashboard_demo(None, output_dir, args.port)

if __name__ == "__main__":
    main()