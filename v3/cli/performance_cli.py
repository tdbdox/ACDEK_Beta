"""
Performance CLI for ACDEK v3: Command-line interface for performance measurement.
"""
import os
import sys
import json
import time
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
import psutil
import threading
import signal
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from observability.performance_profiler import PerformanceProfiler
from observability.bottleneck_detector import BottleneckDetector, BottleneckConfig
from observability.performance_visualizer import PerformanceVisualizer
from acdek.acdek_kernel import ACDEKKernel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_cli")

class PerformanceCLI:
    """
    Command-line interface for ACDEK performance measurement.
    
    Provides tools for profiling, analyzing bottlenecks, and generating
    performance reports from the command line.
    """
    def __init__(self):
        """Initialize the CLI with default values."""
        self.profiler = None
        self.detector = None
        self.visualizer = None
        self.kernel = None
        self.monitor_active = False
        self.monitor_thread = None
    
    def setup_parser(self) -> argparse.ArgumentParser:
        """Set up command-line argument parser."""
        parser = argparse.ArgumentParser(
            description="ACDEK Performance Measurement CLI",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Add global options
        parser.add_argument(
            "--output-dir",
            type=str,
            default="./performance_data",
            help="Directory for output files"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
        
        # Add subcommands
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Profile command
        profile_parser = subparsers.add_parser(
            "profile",
            help="Profile ACDEK pipeline execution"
        )
        profile_parser.add_argument(
            "--duration",
            type=int,
            default=60,
            help="Duration to profile in seconds"
        )
        profile_parser.add_argument(
            "--interval",
            type=int,
            default=5,
            help="Snapshot interval in seconds"
        )
        profile_parser.add_argument(
            "--export",
            action="store_true",
            help="Export profile data after collection"
        )
        
        # Analyze command
        analyze_parser = subparsers.add_parser(
            "analyze",
            help="Analyze performance data for bottlenecks"
        )
        analyze_parser.add_argument(
            "--input",
            type=str,
            help="Input file with performance data to analyze"
        )
        analyze_parser.add_argument(
            "--threshold",
            type=float,
            default=100.0,
            help="Warning threshold in milliseconds"
        )
        analyze_parser.add_argument(
            "--report-format",
            choices=["cli", "html", "json"],
            default="cli",
            help="Output format for the report"
        )
        
        # Visualize command
        visualize_parser = subparsers.add_parser(
            "visualize",
            help="Create visualizations from performance data"
        )
        visualize_parser.add_argument(
            "--input",
            type=str,
            required=True,
            help="Input file with performance data to visualize"
        )
        visualize_parser.add_argument(
            "--bottleneck-data",
            type=str,
            help="Input file with bottleneck data to include"
        )
        visualize_parser.add_argument(
            "--output",
            type=str,
            default="performance_dashboard.html",
            help="Output filename for the dashboard"
        )
        
        # Monitor command
        monitor_parser = subparsers.add_parser(
            "monitor",
            help="Continuously monitor ACDEK performance"
        )
        monitor_parser.add_argument(
            "--interval",
            type=int,
            default=5,
            help="Monitoring interval in seconds"
        )
        monitor_parser.add_argument(
            "--duration",
            type=int,
            default=0,
            help="Monitoring duration in seconds (0 = indefinite)"
        )
        monitor_parser.add_argument(
            "--dashboard",
            action="store_true",
            help="Continuously update a dashboard"
        )
        monitor_parser.add_argument(
            "--dashboard-interval",
            type=int,
            default=30,
            help="Dashboard update interval in seconds"
        )
        
        return parser
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """
        Run the CLI with the given arguments.
        
        Args:
            args: Command-line arguments (uses sys.argv if None)
            
        Returns:
            Exit code (0 for success, non-zero for errors)
        """
        parser = self.setup_parser()
        parsed_args = parser.parse_args(args)
        
        # Configure logging
        if parsed_args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Create output directory if needed
        os.makedirs(parsed_args.output_dir, exist_ok=True)
        
        # Handle commands
        try:
            if parsed_args.command == "profile":
                return self.handle_profile(parsed_args)
            elif parsed_args.command == "analyze":
                return self.handle_analyze(parsed_args)
            elif parsed_args.command == "visualize":
                return self.handle_visualize(parsed_args)
            elif parsed_args.command == "monitor":
                return self.handle_monitor(parsed_args)
            else:
                parser.print_help()
                return 1
        except KeyboardInterrupt:
            logger.info("Operation interrupted by user")
            return 130  # Standard exit code for SIGINT
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            if parsed_args.verbose:
                logger.exception("Detailed error information:")
            return 1
    
    def handle_profile(self, args) -> int:
        """Handle the profile command."""
        logger.info(f"Profiling ACDEK for {args.duration} seconds")
        
        # Initialize profiler
        self.profiler = PerformanceProfiler(
            enable_resource_monitoring=True,
            export_directory=args.output_dir
        )
        
        # Initialize kernel (if not already done)
        if not self.kernel:
            logger.info("Initializing ACDEK kernel")
            self.kernel = ACDEKKernel()
        
        # Set up signal handler
        def signal_handler(sig, frame):
            logger.info("Stopping profiling due to signal")
            self.cleanup()
            sys.exit(0)
        
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Profile for the specified duration
            start_time = time.time()
            snapshot_times = []
            snapshots = []
            
            while (time.time() - start_time) < args.duration:
                # Sleep until next snapshot interval
                next_snapshot = start_time + len(snapshot_times) * args.interval
                sleep_time = max(0, next_snapshot - time.time())
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Take snapshot
                snapshot = self.profiler.get_snapshot()
                snapshots.append(snapshot)
                snapshot_times.append(time.time())
                
                logger.info(f"Snapshot {len(snapshot_times)} captured - "
                           f"{len(snapshot.stage_profiles)} stages, "
                           f"{len(snapshot.bottlenecks)} bottlenecks detected")
            
            logger.info(f"Profiling complete - {len(snapshots)} snapshots captured")
            
            # Export data if requested
            if args.export:
                export_path = os.path.join(
                    args.output_dir,
                    f"profile_{int(time.time())}.json"
                )
                
                with open(export_path, 'w') as f:
                    export_data = {
                        "snapshots": [s.to_dict() for s in snapshots],
                        "snapshot_times": snapshot_times,
                        "metadata": {
                            "duration": args.duration,
                            "interval": args.interval,
                            "timestamp": time.time()
                        }
                    }
                    json.dump(export_data, f, indent=2)
                
                logger.info(f"Profile data exported to {export_path}")
            
            # Show final snapshot summary
            final_snapshot = snapshots[-1] if snapshots else None
            if final_snapshot:
                bottlenecks = final_snapshot.bottlenecks
                
                if bottlenecks:
                    logger.info("Top bottlenecks detected:")
                    sorted_bottlenecks = sorted(
                        bottlenecks,
                        key=lambda b: b.get("severity", 0),
                        reverse=True
                    )[:3]  # Top 3
                    
                    for i, b in enumerate(sorted_bottlenecks):
                        logger.info(f"{i+1}. {b.get('stage_name')} - "
                                   f"{b.get('avg_time_ms', 0):.1f} ms avg, "
                                   f"severity: {b.get('severity', 0)}")
            
            return 0
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
    
    def handle_analyze(self, args) -> int:
        """Handle the analyze command."""
        # Load input data if specified
        input_data = None
        if args.input:
            logger.info(f"Loading performance data from {args.input}")
            with open(args.input, 'r') as f:
                input_data = json.load(f)
            
            # Check if we have snapshot data
            if "snapshots" in input_data:
                # Use the last snapshot
                logger.info(f"Using latest of {len(input_data['snapshots'])} snapshots")
                snapshot_data = input_data["snapshots"][-1]
            else:
                # Assume the file contains a single snapshot
                snapshot_data = input_data
        else:
            # If no input specified, create a new profiler and take a snapshot
            logger.info("No input specified, analyzing current state")
            
            # Initialize profiler if needed
            if not self.profiler:
                self.profiler = PerformanceProfiler(
                    enable_resource_monitoring=True,
                    export_directory=args.output_dir
                )
            
            # Initialize kernel if needed
            if not self.kernel:
                logger.info("Initializing ACDEK kernel")
                self.kernel = ACDEKKernel()
            
            # Take snapshot
            snapshot = self.profiler.get_snapshot()
            snapshot_data = snapshot.to_dict()
        
        # Initialize bottleneck detector with custom threshold
        config = BottleneckConfig(
            warning_threshold_ms=args.threshold,
            critical_threshold_ms=args.threshold * 2
        )
        
        # Create a temporary profiler for analysis if needed
        if not self.profiler:
            temp_profiler = PerformanceProfiler(
                enable_resource_monitoring=False,
                export_directory=args.output_dir
            )
            
            # Populate with snapshot data
            for name, profile in snapshot_data.get("stage_profiles", {}).items():
                if profile.get("count", 0) > 0:
                    temp_profiler._stage_profiles[name] = profile
            
            self.detector = BottleneckDetector(
                temp_profiler,
                config=config,
                export_directory=args.output_dir
            )
        else:
            self.detector = BottleneckDetector(
                self.profiler,
                config=config,
                export_directory=args.output_dir
            )
        
        # Generate report
        bottleneck_report = self.detector.generate_report()
        
        # Output report in requested format
        if args.report_format == "cli":
            # Initialize visualizer for CLI report
            if not self.visualizer:
                self.visualizer = PerformanceVisualizer(
                    output_directory=args.output_dir
                )
            
            cli_report = self.visualizer.create_cli_report(
                snapshot_data,
                bottleneck_report
            )
            print(cli_report)
            
        elif args.report_format == "html":
            # Initialize visualizer if needed
            if not self.visualizer:
                self.visualizer = PerformanceVisualizer(
                    output_directory=args.output_dir
                )
            
            # Generate HTML dashboard
            dashboard_path = self.visualizer.create_dashboard(
                snapshot_data,
                bottleneck_report,
                filename=f"bottleneck_report_{int(time.time())}.html"
            )
            
            logger.info(f"HTML report generated at {dashboard_path}")
            
        elif args.report_format == "json":
            # Export bottleneck report as JSON
            output_path = os.path.join(
                args.output_dir,
                f"bottleneck_report_{int(time.time())}.json"
            )
            
            with open(output_path, 'w') as f:
                json.dump(bottleneck_report, f, indent=2)
            
            logger.info(f"JSON report exported to {output_path}")
        
        return 0
    
    def handle_visualize(self, args) -> int:
        """Handle the visualize command."""
        # Load input performance data
        logger.info(f"Loading performance data from {args.input}")
        with open(args.input, 'r') as f:
            input_data = json.load(f)
        
        # Check if we have snapshot data
        if "snapshots" in input_data:
            # Use the last snapshot
            logger.info(f"Using latest of {len(input_data['snapshots'])} snapshots")
            snapshot_data = input_data["snapshots"][-1]
        else:
            # Assume the file contains a single snapshot
            snapshot_data = input_data
        
        # Load bottleneck data if specified
        bottleneck_data = None
        if args.bottleneck_data:
            logger.info(f"Loading bottleneck data from {args.bottleneck_data}")
            with open(args.bottleneck_data, 'r') as f:
                bottleneck_data = json.load(f)
        else:
            # If no bottleneck data specified, create minimal structure
            bottleneck_data = {
                "bottlenecks": [],
                "system_recommendations": [
                    "No bottleneck analysis provided. Run 'analyze' command to generate recommendations."
                ]
            }
        
        # Initialize visualizer
        if not self.visualizer:
            self.visualizer = PerformanceVisualizer(
                output_directory=args.output_dir
            )
        
        # Create dashboard
        dashboard_path = self.visualizer.create_dashboard(
            snapshot_data,
            bottleneck_data,
            filename=args.output
        )
        
        logger.info(f"Dashboard created at {dashboard_path}")
        return 0
    
    def handle_monitor(self, args) -> int:
        """Handle the monitor command."""
        logger.info("Starting performance monitoring")
        
        # Initialize profiler if needed
        if not self.profiler:
            self.profiler = PerformanceProfiler(
                enable_resource_monitoring=True,
                export_directory=args.output_dir
            )
        
        # Initialize detector if needed
        if not self.detector:
            self.detector = BottleneckDetector(
                self.profiler,
                export_directory=args.output_dir
            )
        
        # Initialize visualizer if needed
        if args.dashboard and not self.visualizer:
            self.visualizer = PerformanceVisualizer(
                output_directory=args.output_dir
            )
        
        # Initialize kernel if needed
        if not self.kernel:
            logger.info("Initializing ACDEK kernel")
            self.kernel = ACDEKKernel()
        
        # Set up monitoring state
        self.monitor_active = True
        monitor_start = time.time()
        last_dashboard_update = 0
        
        # Set up signal handler
        def signal_handler(sig, frame):
            logger.info("Stopping monitoring due to signal")
            self.monitor_active = False
        
        original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Start monitoring loop
            while self.monitor_active:
                # Check if duration exceeded
                if args.duration > 0 and (time.time() - monitor_start) >= args.duration:
                    logger.info(f"Monitoring duration of {args.duration}s reached")
                    break
                
                # Take snapshot
                snapshot = self.profiler.get_snapshot()
                bottlenecks = self.detector.detect_bottlenecks()
                
                # Print monitoring information
                system_load = snapshot.system_load
                thread_count = snapshot.thread_count
                
                logger.info(f"[{datetime.now().strftime('%H:%M:%S')}] "
                           f"CPU: {system_load.get('cpu_percent', 0):.1f}% | "
                           f"Memory: {system_load.get('memory_percent', 0):.1f}% | "
                           f"Threads: {thread_count} | "
                           f"Bottlenecks: {len(bottlenecks)}")
                
                # Show critical bottlenecks if any
                critical_bottlenecks = [b for b in bottlenecks if b.get("severity", 0) >= 2]
                if critical_bottlenecks:
                    logger.warning(f"{len(critical_bottlenecks)} critical/warning bottlenecks detected:")
                    for i, b in enumerate(critical_bottlenecks[:3]):  # Top 3
                        severity = "CRITICAL" if b.get("severity", 0) == 3 else "WARNING"
                        logger.warning(f"  {i+1}. [{severity}] {b.get('stage_name')}: "
                                      f"{b.get('avg_time_ms', 0):.1f} ms avg")
                
                # Update dashboard if requested
                if args.dashboard and (time.time() - last_dashboard_update) >= args.dashboard_interval:
                    bottleneck_report = self.detector.generate_report()
                    
                    dashboard_path = self.visualizer.create_dashboard(
                        snapshot.to_dict(),
                        bottleneck_report,
                        filename="live_dashboard.html"
                    )
                    
                    logger.info(f"Dashboard updated at {dashboard_path}")
                    last_dashboard_update = time.time()
                
                # Sleep until next interval
                time.sleep(args.interval)
            
            logger.info("Monitoring complete")
            return 0
            
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
            self.monitor_active = False
    
    def cleanup(self):
        """Clean up resources."""
        if self.monitor_active:
            self.monitor_active = False
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=1.0)

def main():
    """Main entry point."""
    cli = PerformanceCLI()
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())