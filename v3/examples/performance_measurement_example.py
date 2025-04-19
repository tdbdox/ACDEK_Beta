"""
Performance Measurement Framework Example for ACDEK v3.

This example demonstrates how to use the performance measurement framework
to profile, analyze, and visualize ACDEK v3 performance characteristics.

Usage:
  python performance_measurement_example.py

Options:
  --mode=profile    Run pipeline with profiling
  --mode=analyze    Analyze performance bottlenecks
  --mode=monitor    Continuously monitor performance
  --mode=visualize  Create visualization dashboard
  --mode=all        Run all modes in sequence
"""
import sys
import os
import time
import json
import random
import argparse
import logging
from typing import Dict, List, Any, Optional
import threading

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acdek.acdek_kernel import ACDEKKernel
from observability.performance_profiler import PerformanceProfiler
from observability.bottleneck_detector import BottleneckDetector, BottleneckConfig
from observability.performance_visualizer import PerformanceVisualizer
from observability.metrics_collector_enhanced import EnhancedMetricsCollector, ProfiledTimingContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("performance_example")

class PerformanceExample:
    """
    Example application demonstrating the ACDEK v3 performance measurement framework.
    """
    def __init__(self, output_dir: str = "./performance_data"):
        """
        Initialize the example.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.kernel = None
        self.profiler = None
        self.detector = None
        self.visualizer = None
        self.metrics = None
    
    def setup_components(self):
        """Set up ACDEK components and performance measurement tools."""
        logger.info("Setting up ACDEK and performance tools")
        
        # Initialize metrics collector
        self.metrics = EnhancedMetricsCollector(
            service_name="performance_example",
            enable_profiling=True,
            profiler_config={"export_directory": self.output_dir}
        )
        
        # Initialize performance profiler
        self.profiler = self.metrics.get_profiler()
        
        # Initialize bottleneck detector
        self.detector = BottleneckDetector(
            self.profiler,
            export_directory=self.output_dir
        )
        
        # Initialize visualizer
        self.visualizer = PerformanceVisualizer(
            output_directory=self.output_dir,
            dashboard_title="ACDEK Performance Example Dashboard"
        )
        
        # Initialize ACDEK kernel
        self.kernel = ACDEKKernel(
            config={
                "fallback_mode": "lenient",
                "parallel_execution": True
            }
        )
        
        logger.info("Components initialized successfully")
    
    def generate_test_request(self, complexity: str = "medium") -> Dict[str, Any]:
        """
        Generate a test request for ACDEK.
        
        Args:
            complexity: Request complexity level ('simple', 'medium', 'complex')
            
        Returns:
            Request payload dictionary
        """
        # Base request data
        request = {
            "query": "What is the current inventory status?",
            "context": {},
            "parameters": {
                "confidence_threshold": 0.7,
                "max_results": 5
            }
        }
        
        # Adjust complexity
        if complexity == "simple":
            pass  # Use default simple request
        elif complexity == "medium":
            # Add more context
            request["context"] = {
                "user_id": "user_123",
                "tenant_id": "tenant_abc",
                "session_data": {
                    "previous_queries": ["What products are on sale?"],
                    "user_preferences": {"category": "electronics"}
                }
            }
            request["query"] = "What is the inventory status for laptops and monitors at the Seattle store?"
        elif complexity == "complex":
            # Add extensive context and parameters
            request["context"] = {
                "user_id": "user_456",
                "tenant_id": "tenant_def",
                "session_data": {
                    "previous_queries": [
                        "What products are on sale?",
                        "Do you have laptops in stock?",
                        "What about monitors?"
                    ],
                    "user_preferences": {
                        "category": "electronics",
                        "brands": ["Dell", "HP", "Lenovo"],
                        "price_range": {"min": 500, "max": 2000}
                    },
                    "location": {
                        "city": "Seattle",
                        "store_id": "SEA-001",
                        "department": "Business Solutions"
                    }
                },
                "permissions": ["inventory_read", "pricing_read", "special_offers"],
                "authentication": {
                    "level": "authenticated",
                    "method": "oauth",
                    "expiry": time.time() + 3600
                }
            }
            request["parameters"] = {
                "confidence_threshold": 0.9,
                "max_results": 20,
                "include_related_products": True,
                "check_availability": True,
                "filter_in_stock_only": True,
                "sort_by": "price",
                "sort_order": "asc",
                "apply_user_preferences": True,
                "expand_product_details": True
            }
            request["query"] = (
                "What is the current inventory status for Dell and HP laptops "
                "with 16GB RAM and at least 512GB SSD storage at the Seattle "
                "Business Solutions store, and are there any compatible monitors "
                "on sale this week that would qualify for the bundle discount?"
            )
        
        return request
    
    def run_profile_mode(self, request_count: int = 50):
        """
        Run ACDEK with performance profiling.
        
        Args:
            request_count: Number of requests to process
        """
        logger.info(f"Running profile mode with {request_count} requests")
        
        if not self.kernel or not self.profiler:
            self.setup_components()
        
        # Process multiple requests of varying complexity
        for i in range(request_count):
            # Select random complexity
            complexity = random.choice(["simple", "medium", "complex"])
            
            # Generate request
            request = self.generate_test_request(complexity)
            request_id = f"req_{i}"
            
            # Add request identifier
            context = request.get("context", {})
            context["request_id"] = request_id
            request["context"] = context
            
            # Process request with profiling
            with self.profiler.profile_request(request_id, {"complexity": complexity}):
                logger.info(f"Processing request {i+1}/{request_count} ({complexity} complexity)")
                
                # Process with ACDEK kernel
                with ProfiledTimingContext("request_total", self.metrics, request_id):
                    result = self.kernel.run(
                        input_data=request,
                        context=context
                    )
                
                # Check result
                success = result.get("success", False)
                confidence = result.get("confidence", {}).get("aggregate_score", 0)
                logger.info(f"Request {i+1} completed: success={success}, confidence={confidence:.2f}")
        
        # Export results
        logger.info("Exporting profile data")
        self.profiler.export_snapshot(filename="profile_results.json")
        
        # Generate bottleneck report
        bottleneck_report = self.detector.generate_report()
        bottleneck_path = self.detector.export_report(filename="bottleneck_report.json")
        
        logger.info(f"Profiling complete. Reports saved to:")
        logger.info(f"- Profile data: {os.path.join(self.output_dir, 'profile_results.json')}")
        logger.info(f"- Bottleneck report: {bottleneck_path}")
    
    def run_analyze_mode(self, input_path: Optional[str] = None):
        """
        Analyze performance bottlenecks.
        
        Args:
            input_path: Optional path to profile data (uses recent data if None)
        """
        logger.info("Running analyze mode")
        
        if not self.detector:
            self.setup_components()
        
        # Load profile data if specified
        if input_path:
            logger.info(f"Loading profile data from {input_path}")
            with open(input_path, 'r') as f:
                snapshot_data = json.load(f)
            
            # TODO: Restore snapshot data to profiler
            bottleneck_report = self.detector.generate_report()
        else:
            # Use current profiler data
            logger.info("Analyzing current profiler data")
            bottleneck_report = self.detector.generate_report()
        
        # Export report
        bottleneck_path = self.detector.export_report(filename="bottleneck_analysis.json")
        
        # Display bottleneck summary
        bottlenecks = bottleneck_report.get("bottlenecks", [])
        logger.info(f"Found {len(bottlenecks)} bottlenecks")
        
        for i, bottleneck in enumerate(bottlenecks[:5]):  # Show top 5
            severity = "CRITICAL" if bottleneck.get("severity") == 3 else (
                "WARNING" if bottleneck.get("severity") == 2 else "INFO"
            )
            
            logger.info(f"Bottleneck {i+1}: {bottleneck.get('stage_name')} [{severity}]")
            logger.info(f"  Avg Time: {bottleneck.get('avg_time_ms', 0):.1f} ms")
            logger.info(f"  Impact: {bottleneck.get('relative_impact', 0) * 100:.1f}%")
            
            # Show recommendations
            logger.info("  Recommendations:")
            for j, rec in enumerate(bottleneck.get("recommendations", [])[:2]):  # Top 2 recommendations
                logger.info(f"    {j+1}. {rec}")
        
        # Show system recommendations
        logger.info("System Recommendations:")
        for i, rec in enumerate(bottleneck_report.get("system_recommendations", [])):
            logger.info(f"  {i+1}. {rec}")
        
        logger.info(f"Analysis complete. Report saved to: {bottleneck_path}")
    
    def run_monitor_mode(self, duration: int = 60, interval: int = 5):
        """
        Continuously monitor ACDEK performance.
        
        Args:
            duration: Monitoring duration in seconds
            interval: Monitoring interval in seconds
        """
        logger.info(f"Running monitor mode for {duration} seconds")
        
        if not self.kernel or not self.profiler:
            self.setup_components()
        
        # Set up monitoring state
        monitor_active = True
        monitor_start = time.time()
        last_update = 0
        
        # Create a thread for generating requests
        def request_generator():
            while monitor_active and (time.time() - monitor_start) < duration:
                # Generate and process request
                complexity = random.choice(["simple", "medium", "complex"])
                request = self.generate_test_request(complexity)
                request_id = f"req_{int(time.time() * 1000)}"
                
                # Add request identifier
                context = request.get("context", {})
                context["request_id"] = request_id
                request["context"] = context
                
                # Process with ACDEK kernel
                with self.profiler.profile_request(request_id, {"complexity": complexity}):
                    self.kernel.run(
                        input_data=request,
                        context=context
                    )
                
                # Add random delay between requests
                time.sleep(random.uniform(0.5, 2.0))
        
        # Start request generator thread
        generator_thread = threading.Thread(target=request_generator)
        generator_thread.daemon = True
        generator_thread.start()
        
        try:
            # Start monitoring loop
            while (time.time() - monitor_start) < duration:
                # Take snapshot if interval elapsed
                current_time = time.time()
                if (current_time - last_update) >= interval:
                    # Get performance snapshot
                    snapshot = self.profiler.get_snapshot()
                    bottlenecks = self.detector.detect_bottlenecks()
                    
                    # Print monitoring information
                    system_load = snapshot.system_load
                    thread_count = snapshot.thread_count
                    
                    logger.info(f"[{time.strftime('%H:%M:%S')}] "
                               f"CPU: {system_load.get('cpu_percent', 0):.1f}% | "
                               f"Memory: {system_load.get('memory_percent', 0):.1f}% | "
                               f"Threads: {thread_count} | "
                               f"Stages: {len(snapshot.stage_profiles)} | "
                               f"Bottlenecks: {len(bottlenecks)}")
                    
                    # Show critical bottlenecks if any
                    critical_bottlenecks = [b for b in bottlenecks if b.get("severity", 0) >= 2]
                    if critical_bottlenecks:
                        logger.info(f"{len(critical_bottlenecks)} critical/warning bottlenecks detected")
                    
                    # Update dashboard
                    if (current_time - monitor_start) % 15 < interval:  # Update every ~15 seconds
                        bottleneck_report = self.detector.generate_report()
                        
                        dashboard_path = self.visualizer.create_dashboard(
                            snapshot.to_dict(),
                            bottleneck_report,
                            filename="live_monitor.html"
                        )
                        
                        logger.info(f"Dashboard updated at {dashboard_path}")
                    
                    last_update = current_time
                
                # Sleep briefly
                time.sleep(0.5)
            
            # Export final results
            logger.info("Exporting final monitoring data")
            self.profiler.export_snapshot(filename="monitor_results.json")
            bottleneck_report = self.detector.generate_report()
            self.detector.export_report(filename="monitor_bottlenecks.json")
            
            # Create final dashboard
            dashboard_path = self.visualizer.create_dashboard(
                self.profiler.get_snapshot().to_dict(),
                bottleneck_report,
                filename="monitor_dashboard.html"
            )
            
            logger.info(f"Monitoring complete. Dashboard saved to: {dashboard_path}")
            
        finally:
            # Clean up
            monitor_active = False
            if generator_thread.is_alive():
                generator_thread.join(timeout=1.0)
    
    def run_visualize_mode(
        self,
        profile_path: Optional[str] = None,
        bottleneck_path: Optional[str] = None
    ):
        """
        Create performance visualizations.
        
        Args:
            profile_path: Path to profile data file
            bottleneck_path: Path to bottleneck report file
        """
        logger.info("Running visualize mode")
        
        if not self.visualizer:
            self.setup_components()
        
        # Load profile data
        profile_data = None
        if profile_path:
            logger.info(f"Loading profile data from {profile_path}")
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
        else:
            # Check for default profile file
            default_path = os.path.join(self.output_dir, "profile_results.json")
            if os.path.exists(default_path):
                logger.info(f"Loading profile data from {default_path}")
                with open(default_path, 'r') as f:
                    profile_data = json.load(f)
            else:
                # Use current profiler data
                logger.info("Using current profiler data")
                if self.profiler:
                    profile_data = self.profiler.get_snapshot().to_dict()
        
        # Check if we have profile data
        if not profile_data:
            logger.error("No profile data available. Run profile mode first.")
            return
        
        # Load bottleneck data
        bottleneck_data = None
        if bottleneck_path:
            logger.info(f"Loading bottleneck data from {bottleneck_path}")
            with open(bottleneck_path, 'r') as f:
                bottleneck_data = json.load(f)
        else:
            # Check for default bottleneck file
            default_path = os.path.join(self.output_dir, "bottleneck_report.json")
            if os.path.exists(default_path):
                logger.info(f"Loading bottleneck data from {default_path}")
                with open(default_path, 'r') as f:
                    bottleneck_data = json.load(f)
            else:
                # Generate bottleneck report
                logger.info("Generating bottleneck report")
                if self.detector:
                    bottleneck_data = self.detector.generate_report()
        
        # Check if we have bottleneck data
        if not bottleneck_data:
            logger.warning("No bottleneck data available. Using empty report.")
            bottleneck_data = {
                "bottlenecks": [],
                "system_recommendations": [
                    "No bottleneck analysis available. Run analyze mode first."
                ]
            }
        
        # Create visualizations
        logger.info("Creating performance dashboard")
        dashboard_path = self.visualizer.create_dashboard(
            profile_data,
            bottleneck_data,
            filename="performance_dashboard.html",
            include_raw_data=True
        )
        
        logger.info(f"Visualization complete. Dashboard saved to: {dashboard_path}")
    
    def run_all_modes(self):
        """Run all modes in sequence."""
        logger.info("Running all modes in sequence")
        
        # Set up components
        self.setup_components()
        
        # Run profile mode with fewer requests
        self.run_profile_mode(request_count=20)
        
        # Run analyze mode
        self.run_analyze_mode()
        
        # Run visualize mode
        self.run_visualize_mode()
        
        # Run monitor mode for a short duration
        self.run_monitor_mode(duration=30, interval=5)
        
        logger.info("All modes completed successfully")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ACDEK Performance Measurement Example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        choices=["profile", "analyze", "monitor", "visualize", "all"],
        default="all",
        help="Mode to run"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./performance_data",
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--request-count",
        type=int,
        default=50,
        help="Number of requests to process in profile mode"
    )
    
    parser.add_argument(
        "--monitor-duration",
        type=int,
        default=60,
        help="Duration to monitor in seconds"
    )
    
    parser.add_argument(
        "--profile-input",
        type=str,
        help="Input file with profile data for analyze/visualize modes"
    )
    
    parser.add_argument(
        "--bottleneck-input",
        type=str,
        help="Input file with bottleneck data for visualize mode"
    )
    
    args = parser.parse_args()
    
    # Run example
    example = PerformanceExample(output_dir=args.output_dir)
    
    try:
        if args.mode == "profile":
            example.run_profile_mode(request_count=args.request_count)
        elif args.mode == "analyze":
            example.run_analyze_mode(input_path=args.profile_input)
        elif args.mode == "monitor":
            example.run_monitor_mode(duration=args.monitor_duration)
        elif args.mode == "visualize":
            example.run_visualize_mode(
                profile_path=args.profile_input,
                bottleneck_path=args.bottleneck_input
            )
        elif args.mode == "all":
            example.run_all_modes()
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.exception("Detailed error information:")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())