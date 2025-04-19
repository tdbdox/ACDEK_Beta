#!/usr/bin/env python3
"""
ACDEK Benchmark Runner

This script provides a command-line interface for running the ACDEK benchmarking suite
with customizable configurations for comparing raw LLMs and ACDEK-enhanced systems.

Usage:
    python benchmark_runner.py --models gpt-4 --domains medical financial --output-dir results
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from acdek_benchmark_suite import (
    ACDEKBenchmarkSuite, 
    BenchmarkSuiteConfig,
    Domain
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_logs.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = Path("./benchmarks/results")
DEFAULT_MODELS = ["gpt-4"]
DEFAULT_DOMAINS = [domain.value for domain in Domain]
DEFAULT_OUTPUT_FORMATS = ["json", "csv"]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ACDEK Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Models configuration
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="LLM models to test (e.g., gpt-4, gpt-3.5-turbo)"
    )
    
    # Domain configuration
    parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=DEFAULT_DOMAINS,
        help=f"Domains to test from: {', '.join(DEFAULT_DOMAINS)}"
    )
    
    # Execution configuration
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run benchmarks in parallel"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel workers"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store benchmark results"
    )
    
    parser.add_argument(
        "--output-formats",
        type=str,
        nargs="+",
        default=DEFAULT_OUTPUT_FORMATS,
        help="Output formats (json, csv)"
    )
    
    # Specific focus areas
    parser.add_argument(
        "--dimensions",
        type=str,
        nargs="+",
        default=["factuality", "hallucination", "uncertainty", "execution_fidelity"],
        help="Evaluation dimensions to focus on"
    )
    
    # Additional options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def validate_config(args) -> bool:
    """
    Validate the configuration options.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        True if configuration is valid, False otherwise
    """
    valid = True
    
    # Validate domains
    valid_domains = [domain.value for domain in Domain]
    for domain in args.domains:
        if domain not in valid_domains:
            logger.error(f"Invalid domain: {domain}")
            logger.info(f"Valid domains are: {', '.join(valid_domains)}")
            valid = False
    
    # Validate output formats
    valid_formats = ["json", "csv", "html"]
    for fmt in args.output_formats:
        if fmt.lower() not in valid_formats:
            logger.error(f"Invalid output format: {fmt}")
            logger.info(f"Valid formats are: {', '.join(valid_formats)}")
            valid = False
    
    # Validate dimensions
    valid_dimensions = ["factuality", "hallucination", "uncertainty", "execution_fidelity", "all"]
    for dim in args.dimensions:
        if dim.lower() not in valid_dimensions:
            logger.error(f"Invalid evaluation dimension: {dim}")
            logger.info(f"Valid dimensions are: {', '.join(valid_dimensions)}")
            valid = False
    
    return valid


def configure_logging(verbose: bool):
    """
    Configure logging level based on verbosity.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    else:
        logging.getLogger().setLevel(logging.INFO)


def create_benchmark_config(args) -> BenchmarkSuiteConfig:
    """
    Create a benchmark configuration from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        BenchmarkSuiteConfig object
    """
    return BenchmarkSuiteConfig(
        llm_models=args.models,
        domains=args.domains,
        parallel=args.parallel,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        output_formats=args.output_formats,
        evaluation_dimensions=args.dimensions
    )


def run_benchmark(config: BenchmarkSuiteConfig) -> int:
    """
    Run the benchmark suite with the given configuration.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Create and run benchmark suite
        benchmark_suite = ACDEKBenchmarkSuite(config)
        results = benchmark_suite.run_benchmark()
        
        # Print summary
        overall = results.summary_metrics["overall"]
        print("\nBenchmark Summary:")
        print(f"Total test cases: {overall['case_count']}")
        print(f"Overall improvement: {overall['improvements']['overall']:.4f}")
        print(f"Factuality improvement: {overall['improvements']['factuality']:.4f}")
        print(f"Latency overhead: {overall['latency']['avg_overhead_seconds']:.3f}s")
        print(f"Success rate: {overall['success_rates']['overall_success']:.1%}")
        
        # Print comparative metrics
        comp_metrics = results.comparative_metrics
        print("\nComparative Metrics:")
        print(f"Hallucination reduction rate: {comp_metrics.get('hallucination_reduction_rate', 0):.2%}")
        print(f"Raw hallucination count: {comp_metrics.get('raw_hallucination_count', 0)}")
        print(f"ACDEK hallucination count: {comp_metrics.get('acdek_hallucination_count', 0)}")
        
        improvement_summary = comp_metrics.get("improvement_summary", {})
        if improvement_summary:
            improved_percent = improvement_summary.get("cases_improved", 0) / max(1, improvement_summary.get("total_cases", 1))
            success_percent = improvement_summary.get("overall_success_cases", 0) / max(1, improvement_summary.get("total_cases", 1))
            
            print(f"Cases improved: {improved_percent:.2%}")
            print(f"Overall success: {success_percent:.2%}")
        
        print(f"\nResults exported to {config.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running benchmark: {str(e)}")
        return 1


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Configure logging
    configure_logging(args.verbose)
    
    # Validate configuration
    if not validate_config(args):
        logger.error("Invalid configuration. Exiting.")
        return 1
    
    # Create benchmark configuration
    config = create_benchmark_config(args)
    
    # Log configuration
    logger.info(f"Running benchmark with configuration:")
    logger.info(f"  Models: {config.llm_models}")
    logger.info(f"  Domains: {config.domains}")
    logger.info(f"  Parallel: {config.parallel}")
    logger.info(f"  Max workers: {config.max_workers}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info(f"  Output formats: {config.output_formats}")
    logger.info(f"  Evaluation dimensions: {config.evaluation_dimensions}")
    
    # Run benchmark
    return run_benchmark(config)


if __name__ == "__main__":
    sys.exit(main())