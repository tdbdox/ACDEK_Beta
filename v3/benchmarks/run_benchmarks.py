#!/usr/bin/env python3
"""
ACDEK Benchmark Suite Runner

This script provides a command-line interface for running the complete ACDEK benchmark
suite, including the general benchmark, the grounding benchmark, and generating
comprehensive reports and visualizations.

Usage:
    python run_benchmarks.py --all  # Run all benchmarks
    python run_benchmarks.py --general  # Run only general benchmark
    python run_benchmarks.py --grounding  # Run only grounding benchmark
    python run_benchmarks.py --dashboard  # Generate dashboard from previous runs
"""

import os
import sys
import argparse
import logging
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_runner.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = Path("./benchmarks/results")
DEFAULT_MODELS = ["gpt-4"]
DEFAULT_DOMAINS = ["medical", "financial", "legal"]


def run_general_benchmark(args: argparse.Namespace) -> str:
    """
    Run the general benchmark suite.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Path to the results file
    """
    logger.info("Running general benchmark suite...")
    start_time = time.time()
    
    # Build command
    cmd = [
        "python", "benchmark_runner.py",
        "--models"] + args.models + [
        "--domains"] + args.domains + [
        "--output-dir", args.output_dir
    ]
    
    if args.parallel:
        cmd.append("--parallel")
        cmd.extend(["--max-workers", str(args.max_workers)])
    
    # Run the benchmark
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        
        # Extract results file path from output
        output_lines = result.stdout.strip().split('\n')
        results_file = None
        
        for line in reversed(output_lines):
            if "Results exported to" in line:
                parts = line.split()
                if parts[-1].endswith(".json"):
                    results_file = parts[-1]
                break
        
        if not results_file:
            # Try to find the most recent results file
            results_dir = Path(args.output_dir)
            json_files = list(results_dir.glob("benchmark_results_*.json"))
            if json_files:
                results_file = str(sorted(json_files, key=os.path.getmtime)[-1])
        
        logger.info(f"General benchmark completed in {time.time() - start_time:.2f}s")
        logger.info(f"Results saved to {results_file}")
        
        return results_file
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running general benchmark: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return ""


def run_grounding_benchmark(args: argparse.Namespace) -> str:
    """
    Run the grounding benchmark suite.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Path to the results file
    """
    logger.info("Running grounding benchmark suite...")
    start_time = time.time()
    
    # Build command
    output_file = f"grounding_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        # Import the module
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from llm_grounding_benchmark import GroundingBenchmark, BenchmarkConfig
        
        # Create config
        config = BenchmarkConfig(
            model=args.models[0],  # Use the first model
            domains=args.domains,
            parallel=args.parallel,
            max_workers=args.max_workers,
            output_dir=args.output_dir,
            output_file=output_file
        )
        
        # Run benchmark
        benchmark = GroundingBenchmark(config)
        benchmark.run_benchmark()
        
        results_file = os.path.join(args.output_dir, output_file)
        
        logger.info(f"Grounding benchmark completed in {time.time() - start_time:.2f}s")
        logger.info(f"Results saved to {results_file}")
        
        return results_file
        
    except Exception as e:
        logger.error(f"Error running grounding benchmark: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return ""


def generate_dashboard(args: argparse.Namespace, results_files: List[str]) -> List[str]:
    """
    Generate dashboards for benchmark results.
    
    Args:
        args: Command-line arguments
        results_files: List of paths to results files
        
    Returns:
        List of paths to generated dashboards
    """
    dashboards = []
    
    for results_file in results_files:
        if not results_file or not os.path.exists(results_file):
            logger.warning(f"Results file not found: {results_file}")
            continue
            
        logger.info(f"Generating dashboard for {results_file}...")
        
        # Build command
        cmd = [
            "python", "benchmark_dashboard.py",
            "--input", results_file,
            "--output-dir", os.path.join(args.output_dir, "dashboards")
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=True
            )
            
            # Extract dashboard path from output
            output_lines = result.stdout.strip().split('\n')
            dashboard_path = None
            
            for line in output_lines:
                if "Dashboard generated" in line:
                    parts = line.split(": ")
                    if len(parts) > 1:
                        dashboard_path = parts[1]
                    break
            
            if dashboard_path:
                dashboards.append(dashboard_path)
                logger.info(f"Dashboard generated: {dashboard_path}")
            else:
                logger.warning(f"Could not determine dashboard path for {results_file}")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Error generating dashboard: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
    
    return dashboards


def run_combined_analysis(results_files: List[str], output_dir: str) -> str:
    """
    Run a combined analysis across multiple benchmark results.
    
    Args:
        results_files: List of paths to results files
        output_dir: Directory to save the combined analysis
        
    Returns:
        Path to the combined analysis report
    """
    logger.info("Running combined analysis...")
    
    if not results_files:
        logger.warning("No results files available for combined analysis")
        return ""
    
    # Load all results
    all_results = []
    for file_path in results_files:
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
                all_results.append(results)
                logger.info(f"Loaded results from {file_path}")
        except Exception as e:
            logger.error(f"Error loading results from {file_path}: {str(e)}")
    
    if not all_results:
        logger.error("No results could be loaded for combined analysis")
        return ""
    
    # Create combined report
    combined_report = {
        "timestamp": datetime.now().isoformat(),
        "source_files": results_files,
        "summary": {},
        "comparisons": []
    }
    
    # Extract summary metrics
    general_metrics = {}
    grounding_metrics = {}
    
    for results in all_results:
        if "summary_metrics" in results:
            if "overall" in results["summary_metrics"]:
                # Grounding benchmark results
                if "avg_factuality_improvement" in results["summary_metrics"]["overall"]:
                    grounding_metrics = results["summary_metrics"]["overall"]
                # General benchmark results
                elif "improvements" in results["summary_metrics"]["overall"]:
                    general_metrics = results["summary_metrics"]["overall"]
    
    # Combine metrics
    combined_report["summary"] = {
        "general": general_metrics,
        "grounding": grounding_metrics,
        "combined_score": (
            general_metrics.get("improvements", {}).get("overall", 0) * 0.5 +
            grounding_metrics.get("avg_factuality_improvement", 0) * 0.5
        ) if general_metrics and grounding_metrics else None
    }
    
    # Save combined report
    output_path = os.path.join(output_dir, f"combined_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, 'w') as f:
        json.dump(combined_report, f, indent=2)
    
    logger.info(f"Combined analysis saved to {output_path}")
    
    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ACDEK Benchmark Suite Runner")
    
    # Benchmark selection
    benchmark_group = parser.add_argument_group("Benchmark Selection")
    benchmark_group.add_argument("--all", action="store_true", help="Run all benchmarks")
    benchmark_group.add_argument("--general", action="store_true", help="Run general benchmark")
    benchmark_group.add_argument("--grounding", action="store_true", help="Run grounding benchmark")
    benchmark_group.add_argument("--dashboard", action="store_true", help="Generate dashboards from results")
    benchmark_group.add_argument("--combined", action="store_true", help="Generate combined analysis")
    
    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--models", nargs="+", default=DEFAULT_MODELS, 
                             help="LLM models to test")
    config_group.add_argument("--domains", nargs="+", default=DEFAULT_DOMAINS,
                             help="Domains to test")
    config_group.add_argument("--parallel", action="store_true", help="Run benchmarks in parallel")
    config_group.add_argument("--max-workers", type=int, default=4, 
                             help="Maximum number of parallel workers")
    config_group.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                             help="Directory to store results")
    
    # Additional options
    parser.add_argument("--input-files", nargs="+", help="Input files for dashboard generation")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if any benchmark is selected
    if not (args.all or args.general or args.grounding or args.dashboard or args.combined):
        parser.print_help()
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track results files
    results_files = []
    
    # Run general benchmark
    if args.all or args.general:
        general_results = run_general_benchmark(args)
        if general_results:
            results_files.append(general_results)
    
    # Run grounding benchmark
    if args.all or args.grounding:
        grounding_results = run_grounding_benchmark(args)
        if grounding_results:
            results_files.append(grounding_results)
    
    # Add input files if specified
    if args.input_files:
        for file_path in args.input_files:
            if os.path.exists(file_path):
                results_files.append(file_path)
            else:
                logger.warning(f"Input file not found: {file_path}")
    
    # Generate dashboards
    dashboards = []
    if args.all or args.dashboard:
        dashboards = generate_dashboard(args, results_files)
    
    # Run combined analysis
    combined_report = ""
    if args.all or args.combined:
        combined_report = run_combined_analysis(results_files, args.output_dir)
    
    # Print summary
    print("\nBenchmark Suite Summary:")
    print(f"General benchmark: {'✓' if args.all or args.general else '✗'}")
    print(f"Grounding benchmark: {'✓' if args.all or args.grounding else '✗'}")
    print(f"Dashboard generation: {'✓' if args.all or args.dashboard else '✗'}")
    print(f"Combined analysis: {'✓' if args.all or args.combined else '✗'}")
    
    print("\nResults:")
    for result_file in results_files:
        print(f"- {result_file}")
    
    if dashboards:
        print("\nDashboards:")
        for dashboard in dashboards:
            print(f"- {dashboard}")
    
    if combined_report:
        print(f"\nCombined analysis: {combined_report}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())