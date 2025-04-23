#!/usr/bin/env python
"""
ACDEK v3 Benchmark Runner

This script provides a command-line interface for running the ACDEK v3 benchmarks
with various configuration options for comparing raw LLM, v2, and v3 outputs.

Example usage:
    python benchmark_run.py --output-dir results --include-v2 --parallel
"""

import argparse
import logging
import os
import sys
import time
import json
import uuid
import psutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

from config.domain_config import Domain

from benchmarks.hallucination_benchmark import (
    BenchmarkConfig,
    run_benchmark,
    generate_test_cases,
    TestCase
)
from benchmarks.analysis_tools import (
    export_report,
    precision_recall_f1,
    aggregate_retry_fallback_rates,
    compare_versions,
    load_data
)
from observability.performance_profiler import (
    PerformanceProfiler,
    ResourceUsage,
    ProfileSnapshot
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
DEFAULT_MAX_WORKERS = 4
DEFAULT_FORMATS = ["json", "csv", "html"]

# Load testing constants
DEFAULT_CONCURRENT_USERS = 1
DEFAULT_RAMP_UP_SECONDS = 10
DEFAULT_DURATION_SECONDS = 60
DEFAULT_THINK_TIME_SECONDS = 1.0

# Resource tracking constants
DEFAULT_RESOURCE_TRACKING_INTERVAL = 1.0  # seconds

# Complexity levels for test cases
class ComplexityLevel:
    """Complexity levels for test cases."""
    LOW = 1
    MEDIUM = 2 
    HIGH = 3
    EXTREME = 4

# Mapping of textual complexity to enum values
COMPLEXITY_MAP = {
    "low": ComplexityLevel.LOW,
    "medium": ComplexityLevel.MEDIUM,
    "high": ComplexityLevel.HIGH,
    "extreme": ComplexityLevel.EXTREME
}

# Test stages for multi-stage testing
class TestStage:
    """Test stages for multi-stage testing."""
    SMOKE = "smoke"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    STRESS = "stress"
    RESILIENCE = "resilience"


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ACDEK v3 Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store benchmark results"
    )
    
    parser.add_argument(
        "--include-v2",
        action="store_true",
        help="Include v2 in the comparison"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run benchmarks in parallel"
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Maximum number of parallel workers"
    )
    
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=DEFAULT_FORMATS,
        help="Output formats (json, csv, html)"
    )
    
    # Test case filtering
    filter_group = parser.add_argument_group('Test Case Filtering')
    
    filter_group.add_argument(
        "--domain",
        type=str,
        choices=[d.value for d in Domain],
        help="Filter test cases by domain"
    )
    
    filter_group.add_argument(
        "--case-id",
        type=str,
        help="Run only the specified test case ID"
    )
    
    filter_group.add_argument(
        "--tag",
        type=str,
        help="Filter test cases by tag"
    )
    
    filter_group.add_argument(
        "--difficulty",
        type=int,
        choices=range(1, 6),
        help="Filter test cases by difficulty level (1-5)"
    )
    
    filter_group.add_argument(
        "--complexity",
        type=str,
        choices=["low", "medium", "high", "extreme"],
        help="Filter test cases by computational complexity"
    )
    
    filter_group.add_argument(
        "--resource-intensive",
        action="store_true",
        help="Include resource-intensive test cases"
    )
    
    # Load testing configuration
    load_group = parser.add_argument_group('Load Testing')
    
    load_group.add_argument(
        "--load-test",
        action="store_true",
        help="Enable load testing mode"
    )
    
    load_group.add_argument(
        "--concurrent-users",
        type=int,
        default=DEFAULT_CONCURRENT_USERS,
        help="Number of concurrent users to simulate in load testing"
    )
    
    load_group.add_argument(
        "--ramp-up",
        type=int,
        default=DEFAULT_RAMP_UP_SECONDS,
        help="Ramp-up period in seconds for load test"
    )
    
    load_group.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_DURATION_SECONDS,
        help="Duration of load test in seconds"
    )
    
    load_group.add_argument(
        "--think-time",
        type=float,
        default=DEFAULT_THINK_TIME_SECONDS,
        help="Think time between requests in seconds"
    )
    
    load_group.add_argument(
        "--constant-rate",
        type=int,
        help="Constant rate of requests per second (overrides concurrent-users)"
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis Options')
    
    analysis_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show test cases that would be run without executing them"
    )
    
    analysis_group.add_argument(
        "--compare-run",
        type=str,
        help="Path to previous benchmark results to compare against"
    )
    
    analysis_group.add_argument(
        "--detect-regressions",
        action="store_true",
        help="Enable automatic regression detection"
    )
    
    analysis_group.add_argument(
        "--regression-threshold",
        type=float,
        default=0.05,
        help="Threshold for regression detection (0.0-1.0)"
    )
    
    analysis_group.add_argument(
        "--extended-metrics",
        action="store_true",
        help="Generate extended metrics and analysis"
    )
    
    analysis_group.add_argument(
        "--resource-tracking",
        action="store_true",
        help="Enable detailed resource usage tracking (CPU, memory, I/O)"
    )
    
    analysis_group.add_argument(
        "--report-title",
        type=str,
        default="ACDEK v3 Benchmark Results",
        help="Title for the benchmark report"
    )
    
    analysis_group.add_argument(
        "--focus-areas",
        type=str,
        nargs="+",
        choices=["hallucination", "latency", "confidence", "resources", "all"],
        default=["all"],
        help="Focus areas for analysis"
    )
    
    analysis_group.add_argument(
        "--alert-on-regression",
        action="store_true",
        help="Send alerts if regressions are detected"
    )
    
    analysis_group.add_argument(
        "--alert-config",
        type=str,
        help="Path to alert configuration file"
    )
    
    # Multi-stage testing
    stage_group = parser.add_argument_group('Multi-stage Testing')
    
    stage_group.add_argument(
        "--multi-stage",
        action="store_true",
        help="Enable multi-stage test execution"
    )
    
    stage_group.add_argument(
        "--stages",
        type=str,
        nargs="+",
        choices=["smoke", "functional", "performance", "stress", "resilience"],
        default=["functional"],
        help="Test stages to run in multi-stage mode"
    )
    
    return parser.parse_args()


def filter_test_cases(
    test_cases: List[TestCase], 
    domain: Optional[str] = None,
    case_id: Optional[str] = None,
    tag: Optional[str] = None,
    difficulty: Optional[int] = None,
    complexity: Optional[str] = None,
    resource_intensive: bool = False
) -> List[TestCase]:
    """
    Filter test cases based on criteria.
    
    Args:
        test_cases: List of test cases to filter
        domain: Filter by domain
        case_id: Filter by specific case ID
        tag: Filter by tag
        difficulty: Filter by difficulty level (1-5)
        complexity: Filter by computational complexity (low, medium, high, extreme)
        resource_intensive: Include resource-intensive test cases
        
    Returns:
        Filtered list of test cases
    """
    filtered_cases = test_cases
    
    if domain:
        try:
            domain_enum = Domain(domain)
            filtered_cases = [tc for tc in filtered_cases if tc.domain == domain_enum]
        except ValueError:
            logger.warning(f"Invalid domain: {domain}")
    
    if case_id:
        filtered_cases = [tc for tc in filtered_cases if tc.id == case_id]
    
    if tag:
        filtered_cases = [tc for tc in filtered_cases if tag in tc.tags]
    
    if difficulty:
        filtered_cases = [tc for tc in filtered_cases if tc.difficulty == difficulty]
    
    # Filter by computational complexity
    if complexity:
        complexity_level = COMPLEXITY_MAP.get(complexity, ComplexityLevel.LOW)
        
        # Check if the test case has a complexity attribute, otherwise use a default mapping
        # based on difficulty level (as a fallback)
        filtered_cases = [
            tc for tc in filtered_cases if 
            (hasattr(tc, 'complexity') and tc.complexity == complexity_level) or
            (not hasattr(tc, 'complexity') and tc.difficulty >= complexity_level)
        ]
    
    # Filter out resource-intensive tests unless explicitly requested
    if not resource_intensive:
        filtered_cases = [
            tc for tc in filtered_cases if 
            not (hasattr(tc, 'resource_intensive') and tc.resource_intensive)
        ]
    
    return filtered_cases


def print_test_case_info(test_cases: List[TestCase]) -> None:
    """Print information about test cases."""
    print(f"\nSelected Test Cases ({len(test_cases)}):")
    
    for i, tc in enumerate(test_cases, 1):
        print(f"{i}. [{tc.id}] {tc.prompt[:80]}{'...' if len(tc.prompt) > 80 else ''}")
        
        # Show basic info
        basic_info = f"Domain: {tc.domain.value}, Difficulty: {tc.difficulty}, Tags: {', '.join(tc.tags)}"
        print(f"   {basic_info}")
        
        # Show complexity and resource info if available
        complexity_str = f"Complexity: {getattr(tc, 'complexity', 'N/A')}"
        resource_intensive = getattr(tc, 'resource_intensive', False)
        resource_str = "Resource-Intensive: Yes" if resource_intensive else "Resource-Intensive: No"
        
        print(f"   {complexity_str}, {resource_str}")
        print(f"   Facts: {len(tc.ground_truth.facts)}, Constraints: {len(tc.ground_truth.constraints)}")
        print()


def generate_comparison_report(current_results_path: str, previous_results_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Generate a comparison report between current and previous benchmark runs.
    
    Args:
        current_results_path: Path to current benchmark results JSON
        previous_results_path: Path to previous benchmark results JSON
        output_dir: Directory to save comparison report
        
    Returns:
        Dictionary with comparison metrics
    """
    logger.info(f"Generating comparison between {current_results_path} and {previous_results_path}")
    
    # Load results
    current_results = load_data(current_results_path, "json")
    previous_results = load_data(previous_results_path, "json")
    
    # Compare results
    comparison = compare_versions(previous_results, current_results)
    
    # Calculate improvement metrics
    improvements = {
        "hallucination_reduction": {
            "previous": previous_results.get("summary_metrics", {}).get("hallucination", {}).get("avg_v3_reduction_rate", 0),
            "current": current_results.get("summary_metrics", {}).get("hallucination", {}).get("avg_v3_reduction_rate", 0),
            "delta": 0,
            "percent_change": 0
        },
        "latency": {
            "previous": previous_results.get("summary_metrics", {}).get("latency", {}).get("avg_v3_overhead_ms", 0),
            "current": current_results.get("summary_metrics", {}).get("latency", {}).get("avg_v3_overhead_ms", 0),
            "delta": 0,
            "percent_change": 0
        },
        "confidence": {
            "previous": previous_results.get("summary_metrics", {}).get("confidence", {}).get("avg_v3_confidence", 0),
            "current": current_results.get("summary_metrics", {}).get("confidence", {}).get("avg_v3_confidence", 0),
            "delta": 0,
            "percent_change": 0
        },
        "resources": {
            "previous": previous_results.get("summary_metrics", {}).get("resources", {}).get("avg_cpu_usage", 0),
            "current": current_results.get("summary_metrics", {}).get("resources", {}).get("avg_cpu_usage", 0),
            "delta": 0,
            "percent_change": 0
        }
    }
    
    # Calculate deltas and percent changes
    for metric in improvements:
        prev = improvements[metric]["previous"]
        curr = improvements[metric]["current"]
        
        if prev != 0:
            delta = curr - prev
            improvements[metric]["delta"] = delta
            improvements[metric]["percent_change"] = (delta / prev) * 100
    
    # Generate case-by-case comparison
    case_comparisons = {}
    current_cases = {c.get("case_id"): c for c in current_results.get("comparisons", [])}
    previous_cases = {c.get("case_id"): c for c in previous_results.get("comparisons", [])}
    
    # Find common cases
    common_case_ids = set(current_cases.keys()) & set(previous_cases.keys())
    
    for case_id in common_case_ids:
        curr_case = current_cases[case_id]
        prev_case = previous_cases[case_id]
        
        # Calculate metrics
        hallucination_delta = (
            curr_case.get("metrics", {}).get("hallucination", {}).get("v3_reduction_rate", 0) -
            prev_case.get("metrics", {}).get("hallucination", {}).get("v3_reduction_rate", 0)
        )
        
        latency_delta = (
            curr_case.get("metrics", {}).get("latency", {}).get("v3_overhead_ms", 0) -
            prev_case.get("metrics", {}).get("latency", {}).get("v3_overhead_ms", 0)
        )

        # Check for resource metrics
        resource_delta = 0
        if "resources" in curr_case.get("metrics", {}) and "resources" in prev_case.get("metrics", {}):
            curr_resource = curr_case["metrics"]["resources"].get("cpu_percent", 0)
            prev_resource = prev_case["metrics"]["resources"].get("cpu_percent", 0)
            resource_delta = curr_resource - prev_resource
        
        case_comparisons[case_id] = {
            "hallucination_delta": hallucination_delta,
            "latency_delta": latency_delta,
            "resource_delta": resource_delta,
            "improved": hallucination_delta > 0 and latency_delta <= 0 and resource_delta <= 0
        }
    
    # Compile comparison report
    comparison_report = {
        "improvements": improvements,
        "case_comparisons": case_comparisons,
        "common_cases_count": len(common_case_ids),
        "improved_cases_count": sum(1 for c in case_comparisons.values() if c["improved"]),
        "regression_cases_count": sum(1 for c in case_comparisons.values() if not c["improved"]),
        "overall_improvement": (
            improvements["hallucination_reduction"]["delta"] > 0 and
            improvements["latency"]["delta"] <= 0
        ),
        "timestamp": datetime.now().isoformat(),
        "run_ids": {
            "current": os.path.basename(current_results_path),
            "previous": os.path.basename(previous_results_path)
        }
    }
    
    # Export comparison report
    comparison_path = os.path.join(output_dir, f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison_report, f, indent=2)
    
    logger.info(f"Comparison report saved to {comparison_path}")
    
    return comparison_report


def print_comparison_summary(comparison: Dict[str, Any]) -> None:
    """Print a summary of the comparison report."""
    print("\nBenchmark Comparison Summary:")
    print(f"Common test cases: {comparison['common_cases_count']}")
    print(f"Improved cases: {comparison['improved_cases_count']}")
    print(f"Regression cases: {comparison['regression_cases_count']}")
    
    hall_delta = comparison["improvements"]["hallucination_reduction"]["delta"]
    hall_percent = comparison["improvements"]["hallucination_reduction"]["percent_change"]
    print(f"Hallucination reduction change: {hall_delta:.4f} ({hall_percent:+.2f}%)")
    
    latency_delta = comparison["improvements"]["latency"]["delta"]
    latency_percent = comparison["improvements"]["latency"]["percent_change"]
    print(f"Latency overhead change: {latency_delta:.2f}ms ({latency_percent:+.2f}%)")
    
    conf_delta = comparison["improvements"]["confidence"]["delta"]
    conf_percent = comparison["improvements"]["confidence"]["percent_change"]
    print(f"Confidence score change: {conf_delta:.4f} ({conf_percent:+.2f}%)")
    
    # Print resource metrics if available
    if "resources" in comparison["improvements"]:
        resource_delta = comparison["improvements"]["resources"]["delta"]
        resource_percent = comparison["improvements"]["resources"]["percent_change"]
        print(f"Resource usage change: {resource_delta:.2f}% ({resource_percent:+.2f}%)")
    
    print(f"Overall improvement: {'✅' if comparison['overall_improvement'] else '❌'}")


def generate_extended_metrics(results_path: str, output_dir: str, focus_areas: List[str]) -> Dict[str, Any]:
    """
    Generate extended metrics and analysis from benchmark results.
    
    Args:
        results_path: Path to benchmark results JSON
        output_dir: Directory to save extended metrics
        focus_areas: Areas to focus analysis on
        
    Returns:
        Dictionary with extended metrics
    """
    logger.info(f"Generating extended metrics for {results_path}")
    
    # Load results
    results = load_data(results_path, "json")
    
    extended_metrics = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "source_results": results_path,
            "focus_areas": focus_areas
        },
        "metrics": {}
    }
    
    # Generate domain-specific metrics
    domain_metrics = {}
    comparisons = results.get("comparisons", [])
    test_cases = results.get("test_cases", [])
    
    # Map case_ids to domains
    case_domains = {}
    for tc in test_cases:
        case_domains[tc.get("id")] = tc.get("domain")
    
    # Group comparisons by domain
    domain_comparisons = {}
    for comp in comparisons:
        case_id = comp.get("case_id")
        domain = case_domains.get(case_id, "unknown")
        
        if domain not in domain_comparisons:
            domain_comparisons[domain] = []
        
        domain_comparisons[domain].append(comp)
    
    # Calculate domain-specific metrics
    for domain, domain_comps in domain_comparisons.items():
        hall_rates = [c.get("metrics", {}).get("hallucination", {}).get("v3_reduction_rate", 0) for c in domain_comps]
        latencies = [c.get("metrics", {}).get("latency", {}).get("v3_overhead_ms", 0) for c in domain_comps]
        confidences = [c.get("metrics", {}).get("confidence", {}).get("v3_confidence", 0) for c in domain_comps 
                      if c.get("metrics", {}).get("confidence", {}).get("v3_confidence", 0) is not None]
        
        # Extract resource metrics if available and in focus areas
        resources = []
        if "resources" in focus_areas or "all" in focus_areas:
            resources = [
                c.get("metrics", {}).get("resources", {}) for c in domain_comps
                if "resources" in c.get("metrics", {})
            ]
        
        domain_metrics[domain] = {
            "case_count": len(domain_comps),
            "avg_hallucination_reduction": sum(hall_rates) / len(hall_rates) if hall_rates else 0,
            "avg_latency_overhead": sum(latencies) / len(latencies) if latencies else 0,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "min_hallucination_reduction": min(hall_rates) if hall_rates else 0,
            "max_hallucination_reduction": max(hall_rates) if hall_rates else 0,
            "min_latency_overhead": min(latencies) if latencies else 0,
            "max_latency_overhead": max(latencies) if latencies else 0
        }
        
        # Add resource metrics if available
        if resources:
            cpu_percentages = [r.get("cpu_percent", 0) for r in resources if "cpu_percent" in r]
            memory_usages = [r.get("memory_mb", 0) for r in resources if "memory_mb" in r]
            
            if cpu_percentages:
                domain_metrics[domain]["avg_cpu_usage"] = sum(cpu_percentages) / len(cpu_percentages)
                domain_metrics[domain]["max_cpu_usage"] = max(cpu_percentages)
            
            if memory_usages:
                domain_metrics[domain]["avg_memory_usage"] = sum(memory_usages) / len(memory_usages)
                domain_metrics[domain]["max_memory_usage"] = max(memory_usages)
    
    extended_metrics["metrics"]["domain_analysis"] = domain_metrics
    
    # Generate difficulty-based metrics if "all" or "hallucination" in focus areas
    if "all" in focus_areas or "hallucination" in focus_areas:
        difficulty_metrics = {}
        
        # Map case_ids to difficulties
        case_difficulties = {}
        for tc in test_cases:
            case_difficulties[tc.get("id")] = tc.get("difficulty", 1)
        
        # Group comparisons by difficulty
        difficulty_comparisons = {}
        for comp in comparisons:
            case_id = comp.get("case_id")
            difficulty = case_difficulties.get(case_id, 1)
            
            if difficulty not in difficulty_comparisons:
                difficulty_comparisons[difficulty] = []
            
            difficulty_comparisons[difficulty].append(comp)
        
        # Calculate difficulty-specific metrics
        for difficulty, diff_comps in difficulty_comparisons.items():
            hall_rates = [c.get("metrics", {}).get("hallucination", {}).get("v3_reduction_rate", 0) for c in diff_comps]
            
            difficulty_metrics[difficulty] = {
                "case_count": len(diff_comps),
                "avg_hallucination_reduction": sum(hall_rates) / len(hall_rates) if hall_rates else 0,
                "min_hallucination_reduction": min(hall_rates) if hall_rates else 0,
                "max_hallucination_reduction": max(hall_rates) if hall_rates else 0
            }
        
        extended_metrics["metrics"]["difficulty_analysis"] = difficulty_metrics
    
    # Generate complexity-based metrics if available and "all" or "resources" in focus areas
    if ("all" in focus_areas or "resources" in focus_areas) and any(hasattr(tc, 'complexity') for tc in test_cases):
        complexity_metrics = {}
        
        # Map case_ids to complexity
        case_complexity = {}
        for tc in test_cases:
            if hasattr(tc, 'complexity'):
                case_complexity[tc.get("id")] = tc.complexity
            else:
                # Default mapping based on difficulty if complexity not specified
                case_complexity[tc.get("id")] = max(1, tc.get("difficulty", 1) - 1)
        
        # Group comparisons by complexity
        complexity_comparisons = {}
        for comp in comparisons:
            case_id = comp.get("case_id")
            complexity = case_complexity.get(case_id, 1)
            
            if complexity not in complexity_comparisons:
                complexity_comparisons[complexity] = []
            
            complexity_comparisons[complexity].append(comp)
        
        # Calculate complexity-specific metrics
        for complexity, comp_comps in complexity_comparisons.items():
            # Extract resource metrics
            resources = [
                c.get("metrics", {}).get("resources", {}) for c in comp_comps
                if "resources" in c.get("metrics", {})
            ]
            
            cpu_percentages = [r.get("cpu_percent", 0) for r in resources if "cpu_percent" in r]
            memory_usages = [r.get("memory_mb", 0) for r in resources if "memory_mb" in r]
            latencies = [c.get("metrics", {}).get("latency", {}).get("v3_overhead_ms", 0) for c in comp_comps]
            
            complexity_name = next((k for k, v in COMPLEXITY_MAP.items() if v == complexity), str(complexity))
            
            complexity_metrics[complexity_name] = {
                "case_count": len(comp_comps),
                "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
                "avg_cpu_usage": sum(cpu_percentages) / len(cpu_percentages) if cpu_percentages else 0,
                "avg_memory_usage": sum(memory_usages) / len(memory_usages) if memory_usages else 0,
                "max_cpu_usage": max(cpu_percentages) if cpu_percentages else 0,
                "max_memory_usage": max(memory_usages) if memory_usages else 0
            }
        
        extended_metrics["metrics"]["complexity_analysis"] = complexity_metrics
    
    # Generate hallucination type analysis if "all" or "hallucination" in focus areas
    if "all" in focus_areas or "hallucination" in focus_areas:
        hallucination_types = {}
        
        # Count hallucination types
        for comp in comparisons:
            raw_issues = comp.get("raw_issues", [])
            v3_issues = comp.get("v3_issues", [])
            
            # Count types in raw issues
            for issue in raw_issues:
                issue_type = issue.get("type", "unknown")
                if issue_type not in hallucination_types:
                    hallucination_types[issue_type] = {"raw_count": 0, "v3_count": 0}
                
                hallucination_types[issue_type]["raw_count"] += 1
            
            # Count types in v3 issues
            for issue in v3_issues:
                issue_type = issue.get("type", "unknown")
                if issue_type not in hallucination_types:
                    hallucination_types[issue_type] = {"raw_count": 0, "v3_count": 0}
                
                hallucination_types[issue_type]["v3_count"] += 1
        
        # Calculate reduction rates by type
        for hall_type, counts in hallucination_types.items():
            raw_count = counts["raw_count"]
            v3_count = counts["v3_count"]
            
            if raw_count > 0:
                reduction_rate = 1.0 - (v3_count / raw_count)
            else:
                reduction_rate = 0.0
            
            hallucination_types[hall_type]["reduction_rate"] = reduction_rate
        
        extended_metrics["metrics"]["hallucination_type_analysis"] = hallucination_types
    
    # Export extended metrics
    extended_path = os.path.join(output_dir, f"extended_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(extended_path, "w") as f:
        json.dump(extended_metrics, f, indent=2)
    
    logger.info(f"Extended metrics saved to {extended_path}")
    
    return extended_metrics


def track_resource_usage(
    output_dir: str,
    interval: float = DEFAULT_RESOURCE_TRACKING_INTERVAL,
    stop_event: threading.Event = None
) -> str:
    """
    Track resource usage during benchmark execution.
    
    Args:
        output_dir: Directory to save resource tracking data
        interval: Sampling interval in seconds
        stop_event: Event to signal when to stop tracking
        
    Returns:
        Path to the generated resource tracking file
    """
    logger.info(f"Starting resource usage tracking with interval {interval}s")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"resource_tracking_{timestamp}.json")
    
    # Get process
    process = psutil.Process()
    
    # Initialize tracking data
    tracking_data = {
        "meta": {
            "start_time": timestamp,
            "sampling_interval": interval,
            "process_id": process.pid,
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "total_memory_mb": psutil.virtual_memory().total / (1024 * 1024)
            }
        },
        "samples": []
    }
    
    try:
        # Start tracking loop
        while not (stop_event and stop_event.is_set()):
            # Get sample time
            sample_time = time.time()
            
            try:
                # Get CPU and memory usage
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                
                # Get thread count
                thread_count = len(process.threads())
                
                # Get I/O counters if available
                io_read_mb = 0
                io_write_mb = 0
                
                if hasattr(process, 'io_counters'):
                    io_counters = process.io_counters()
                    io_read_mb = io_counters.read_bytes / (1024 * 1024)
                    io_write_mb = io_counters.write_bytes / (1024 * 1024)
                
                # Get system-wide info
                system_cpu = psutil.cpu_percent(interval=0.1)
                system_memory = psutil.virtual_memory().percent
                
                # Add sample
                tracking_data["samples"].append({
                    "timestamp": sample_time,
                    "cpu_percent": cpu_percent,
                    "memory_mb": memory_mb,
                    "thread_count": thread_count,
                    "io_read_mb": io_read_mb,
                    "io_write_mb": io_write_mb,
                    "system_cpu": system_cpu,
                    "system_memory": system_memory
                })
                
                # Sleep for the interval
                if stop_event:
                    stop_event.wait(interval)
                else:
                    time.sleep(interval)
                
            except Exception as e:
                logger.warning(f"Error collecting resource sample: {str(e)}")
                time.sleep(interval)
    
    finally:
        # Add end time
        tracking_data["meta"]["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
        tracking_data["meta"]["sample_count"] = len(tracking_data["samples"])
        
        # Calculate summary metrics
        if tracking_data["samples"]:
            cpu_percentages = [sample["cpu_percent"] for sample in tracking_data["samples"]]
            memory_mbs = [sample["memory_mb"] for sample in tracking_data["samples"]]
            thread_counts = [sample["thread_count"] for sample in tracking_data["samples"]]
            
            tracking_data["summary"] = {
                "avg_cpu_percent": sum(cpu_percentages) / len(cpu_percentages),
                "max_cpu_percent": max(cpu_percentages),
                "avg_memory_mb": sum(memory_mbs) / len(memory_mbs),
                "max_memory_mb": max(memory_mbs),
                "avg_thread_count": sum(thread_counts) / len(thread_counts),
                "max_thread_count": max(thread_counts)
            }
        
        # Write to file
        with open(output_path, "w") as f:
            json.dump(tracking_data, f, indent=2)
        
        logger.info(f"Resource tracking data saved to {output_path}")
        return output_path


def run_load_test(
    test_cases: List[TestCase],
    config: BenchmarkConfig,
    concurrent_users: int = DEFAULT_CONCURRENT_USERS,
    ramp_up: int = DEFAULT_RAMP_UP_SECONDS, 
    duration: int = DEFAULT_DURATION_SECONDS,
    think_time: float = DEFAULT_THINK_TIME_SECONDS,
    constant_rate: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a load test with the specified test cases.
    
    Args:
        test_cases: Test cases to run
        config: Benchmark configuration
        concurrent_users: Number of concurrent users to simulate
        ramp_up: Ramp-up period in seconds
        duration: Test duration in seconds
        think_time: Think time between requests in seconds
        constant_rate: Constant rate of requests per second (overrides concurrent users)
        
    Returns:
        Dictionary with load test results
    """
    logger.info(f"Starting load test with {concurrent_users} concurrent users")
    logger.info(f"Ramp-up: {ramp_up}s, Duration: {duration}s, Think time: {think_time}s")
    
    # Initialize load test metrics
    load_test_metrics = {
        "config": {
            "concurrent_users": concurrent_users,
            "ramp_up": ramp_up,
            "duration": duration,
            "think_time": think_time,
            "constant_rate": constant_rate,
            "test_case_count": len(test_cases)
        },
        "results": [],
        "summary": {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0,
            "min_response_time": None,
            "max_response_time": None,
            "throughput": 0,
            "start_time": datetime.now().isoformat(),
            "end_time": None
        }
    }
    
    # Initialize profiler for resource tracking
    profiler = PerformanceProfiler(
        instance_id=f"load_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        enable_resource_monitoring=True
    )
    
    # Create ThreadPoolExecutor for concurrent users
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        # Store futures
        futures = []
        
        # Track test state
        start_time = time.time()
        stop_flag = threading.Event()
        
        # User worker function
        def user_worker(user_id: int):
            # Calculate start delay for ramp-up
            if ramp_up > 0 and concurrent_users > 1:
                start_delay = (user_id / concurrent_users) * ramp_up
                time.sleep(start_delay)
            
            # Initialize user metrics
            user_results = []
            request_count = 0
            
            # Run until stop flag is set
            while not stop_flag.is_set():
                # Select a test case (round-robin)
                test_case = test_cases[request_count % len(test_cases)]
                
                # Generate unique request ID
                request_id = f"user{user_id}_req{request_count}_{str(uuid.uuid4())[:8]}"
                
                try:
                    # Profile the request execution
                    with profiler.profile_request(request_id=request_id) as req_id:
                        # Execute benchmark for this test case
                        start = time.time()
                        result = run_benchmark(config._replace(test_cases=[test_case]))
                        end = time.time()
                        
                        # Calculate response time
                        response_time = (end - start) * 1000.0  # ms
                        
                        # Record result
                        user_results.append({
                            "request_id": request_id,
                            "test_case_id": test_case.id,
                            "response_time": response_time,
                            "success": True,
                            "timestamp": time.time()
                        })
                        
                except Exception as e:
                    # Record failure
                    logger.warning(f"Request {request_id} failed: {str(e)}")
                    user_results.append({
                        "request_id": request_id,
                        "test_case_id": test_case.id,
                        "response_time": 0,
                        "success": False,
                        "error": str(e),
                        "timestamp": time.time()
                    })
                
                # Increment request counter
                request_count += 1
                
                # Think time between requests (if not using constant rate)
                if constant_rate is None and think_time > 0 and not stop_flag.is_set():
                    time.sleep(think_time)
                elif constant_rate is not None:
                    # Calculate sleep time to maintain constant rate
                    # We need to ensure requests/second across all users equals the constant rate
                    target_interval = 1.0 / constant_rate
                    actual_users = min(concurrent_users, int((time.time() - start_time) / ramp_up * concurrent_users)) if ramp_up > 0 else concurrent_users
                    if actual_users > 0:
                        user_interval = target_interval * actual_users
                        time.sleep(max(0, user_interval))
            
            return user_results
        
        # Submit user workers
        for user_id in range(concurrent_users):
            future = executor.submit(user_worker, user_id)
            futures.append(future)
        
        try:
            # Wait for the specified duration
            time.sleep(duration)
            
            # Signal workers to stop
            stop_flag.set()
            
            # Wait for all futures to complete
            all_results = []
            for future in futures:
                user_results = future.result()
                all_results.extend(user_results)
            
            # Compile results
            load_test_metrics["results"] = all_results
            load_test_metrics["summary"]["total_requests"] = len(all_results)
            load_test_metrics["summary"]["successful_requests"] = sum(1 for r in all_results if r["success"])
            load_test_metrics["summary"]["failed_requests"] = sum(1 for r in all_results if not r["success"])
            
            # Calculate response time metrics
            successful_times = [r["response_time"] for r in all_results if r["success"]]
            if successful_times:
                load_test_metrics["summary"]["avg_response_time"] = sum(successful_times) / len(successful_times)
                load_test_metrics["summary"]["min_response_time"] = min(successful_times)
                load_test_metrics["summary"]["max_response_time"] = max(successful_times)
            
            # Calculate throughput
            end_time = time.time()
            test_duration = end_time - start_time
            load_test_metrics["summary"]["throughput"] = len(all_results) / test_duration if test_duration > 0 else 0
            load_test_metrics["summary"]["end_time"] = datetime.now().isoformat()
            
            # Add profiler metrics
            try:
                profiler_snapshot = profiler.get_snapshot()
                load_test_metrics["resource_metrics"] = {
                    "system_load": profiler_snapshot.system_load,
                    "thread_count": profiler_snapshot.thread_count,
                    "bottlenecks": profiler_snapshot.bottlenecks,
                    "scaling_metrics": profiler.get_scaling_metrics()
                }
            except Exception as e:
                logger.warning(f"Error capturing profiler metrics: {str(e)}")
            
            # Export results
            output_dir = config.output_dir
            load_test_path = os.path.join(output_dir, f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(load_test_path, "w") as f:
                json.dump(load_test_metrics, f, indent=2)
            
            logger.info(f"Load test results saved to {load_test_path}")
            
        except KeyboardInterrupt:
            logger.info("Load test interrupted by user")
            stop_flag.set()
    
    return load_test_metrics


def print_load_test_summary(metrics: Dict[str, Any]) -> None:
    """Print a summary of load test results."""
    summary = metrics["summary"]
    config = metrics["config"]
    
    print("\nLoad Test Summary:")
    print(f"Configuration: {config['concurrent_users']} users, {config['duration']}s duration, {config['ramp_up']}s ramp-up")
    
    print(f"\nResults:")
    print(f"  Total Requests: {summary['total_requests']}")
    print(f"  Successful: {summary['successful_requests']} ({summary['successful_requests']/max(1, summary['total_requests']):.1%})")
    print(f"  Failed: {summary['failed_requests']} ({summary['failed_requests']/max(1, summary['total_requests']):.1%})")
    
    if summary['avg_response_time']:
        print(f"\nResponse Times:")
        print(f"  Average: {summary['avg_response_time']:.2f} ms")
        print(f"  Min: {summary['min_response_time']:.2f} ms")
        print(f"  Max: {summary['max_response_time']:.2f} ms")
    
    print(f"\nPerformance:")
    print(f"  Throughput: {summary['throughput']:.2f} requests/second")
    
    if "resource_metrics" in metrics:
        resources = metrics["resource_metrics"]
        print(f"\nResource Usage:")
        if "system_load" in resources:
            system_load = resources["system_load"]
            print(f"  CPU: {system_load.get('cpu_percent', 'N/A')}%")
            print(f"  Memory: {system_load.get('memory_percent', 'N/A')}%")
        
        if "bottlenecks" in resources and resources["bottlenecks"]:
            print(f"\nBottlenecks Detected:")
            for bottleneck in resources["bottlenecks"][:3]:  # Show top 3
                print(f"  - {bottleneck['stage_name']}: {bottleneck['avg_time_ms']:.2f} ms")
                if bottleneck["recommendations"]:
                    print(f"    Recommendation: {bottleneck['recommendations'][0]}")


def print_extended_metrics_summary(extended_metrics: Dict[str, Any]) -> None:
    """Print a summary of the extended metrics."""
    print("\nExtended Metrics Summary:")
    
    # Domain analysis
    domain_analysis = extended_metrics.get("metrics", {}).get("domain_analysis", {})
    if domain_analysis:
        print("\nDomain Analysis:")
        for domain, metrics in domain_analysis.items():
            print(f"- {domain}:")
            print(f"  Cases: {metrics['case_count']}")
            print(f"  Hallucination Reduction: {metrics['avg_hallucination_reduction']:.2%}")
            print(f"  Latency Overhead: {metrics['avg_latency_overhead']:.2f}ms")
            if 'avg_confidence' in metrics:
                print(f"  Confidence: {metrics['avg_confidence']:.2f}")
            
            # Print resource metrics if available
            if 'avg_cpu_usage' in metrics:
                print(f"  CPU Usage: {metrics['avg_cpu_usage']:.1f}% (Max: {metrics['max_cpu_usage']:.1f}%)")
            if 'avg_memory_usage' in metrics:
                print(f"  Memory Usage: {metrics['avg_memory_usage']:.1f} MB (Max: {metrics['max_memory_usage']:.1f} MB)")
    
    # Difficulty analysis
    difficulty_analysis = extended_metrics.get("metrics", {}).get("difficulty_analysis", {})
    if difficulty_analysis:
        print("\nDifficulty Analysis:")
        for difficulty, metrics in sorted(difficulty_analysis.items()):
            print(f"- Level {difficulty}:")
            print(f"  Cases: {metrics['case_count']}")
            print(f"  Hallucination Reduction: {metrics['avg_hallucination_reduction']:.2%}")
    
    # Complexity analysis
    complexity_analysis = extended_metrics.get("metrics", {}).get("complexity_analysis", {})
    if complexity_analysis:
        print("\nComplexity Analysis:")
        for complexity, metrics in complexity_analysis.items():
            print(f"- {complexity.capitalize()}:")
            print(f"  Cases: {metrics['case_count']}")
            print(f"  Latency: {metrics['avg_latency']:.2f}ms")
            print(f"  CPU Usage: {metrics['avg_cpu_usage']:.1f}% (Max: {metrics['max_cpu_usage']:.1f}%)")
            print(f"  Memory Usage: {metrics['avg_memory_usage']:.1f} MB (Max: {metrics['max_memory_usage']:.1f} MB)")
    
    # Hallucination type analysis
    hallucination_types = extended_metrics.get("metrics", {}).get("hallucination_type_analysis", {})
    if hallucination_types:
        print("\nHallucination Type Analysis:")
        for hall_type, metrics in hallucination_types.items():
            print(f"- {hall_type}:")
            print(f"  Raw Count: {metrics['raw_count']}")
            print(f"  V3 Count: {metrics['v3_count']}")
            print(f"  Reduction Rate: {metrics['reduction_rate']:.2%}")


def run_multi_stage_test(
    test_cases: List[TestCase],
    config: BenchmarkConfig,
    stages: List[str],
    output_dir: str,
    resource_tracking: bool = False
) -> Dict[str, Any]:
    """
    Run a multi-stage test with the specified test cases.
    
    Args:
        test_cases: Base test cases to run
        config: Benchmark configuration
        stages: List of test stages to run (smoke, functional, performance, stress, resilience)
        output_dir: Directory to save results
        resource_tracking: Whether to track resource usage during the test
        
    Returns:
        Dictionary with multi-stage test results
    """
    logger.info(f"Starting multi-stage test with stages: {', '.join(stages)}")
    
    # Initialize multi-stage test results
    multi_stage_results = {
        "meta": {
            "start_time": datetime.now().isoformat(),
            "stages": stages,
            "test_case_count": len(test_cases)
        },
        "stage_results": {},
        "summary": {}
    }
    
    # Create stage-specific configurations
    stage_configs = {}
    
    # Smoke test: Run a minimal subset of tests quickly
    if TestStage.SMOKE in stages:
        # Select a small subset (10% or at least 1) of test cases for quick validation
        smoke_count = max(1, int(len(test_cases) * 0.1))
        smoke_cases = test_cases[:smoke_count]
        
        stage_configs[TestStage.SMOKE] = config._replace(
            test_cases=smoke_cases,
            output_dir=os.path.join(output_dir, "smoke")
        )
    
    # Functional test: Run all test cases with standard configuration
    if TestStage.FUNCTIONAL in stages:
        stage_configs[TestStage.FUNCTIONAL] = config._replace(
            output_dir=os.path.join(output_dir, "functional")
        )
    
    # Performance test: Run with resource tracking and performance focus
    if TestStage.PERFORMANCE in stages:
        # Use the full test set but with performance tracking
        performance_config = config._replace(
            output_dir=os.path.join(output_dir, "performance")
        )
        stage_configs[TestStage.PERFORMANCE] = performance_config
    
    # Stress test: Run with high concurrency and load
    if TestStage.STRESS in stages:
        # Use a subset focused on more complex test cases
        stress_cases = [
            tc for tc in test_cases if 
            (hasattr(tc, 'complexity') and tc.complexity >= ComplexityLevel.HIGH) or
            tc.difficulty >= 4
        ]
        
        # If no high-complexity cases found, use all cases
        if not stress_cases:
            stress_cases = test_cases
        
        stage_configs[TestStage.STRESS] = config._replace(
            test_cases=stress_cases,
            output_dir=os.path.join(output_dir, "stress")
        )
    
    # Resilience test: Run with fault injection
    if TestStage.RESILIENCE in stages:
        # Use all test cases but with resilience configuration
        resilience_config = config._replace(
            output_dir=os.path.join(output_dir, "resilience")
        )
        stage_configs[TestStage.RESILIENCE] = resilience_config
    
    # Create directories for each stage
    for stage in stage_configs:
        os.makedirs(stage_configs[stage].output_dir, exist_ok=True)
    
    # Track start time
    multi_stage_start = time.time()
    
    # Run each stage
    for stage in stages:
        if stage not in stage_configs:
            logger.warning(f"Stage {stage} not configured, skipping")
            continue
        
        logger.info(f"Running {stage} stage...")
        stage_start = time.time()
        
        try:
            # Start resource tracking for this stage if enabled
            resource_tracking_thread = None
            resource_tracking_stop = threading.Event()
            resource_tracking_path = None
            
            if resource_tracking and stage in [TestStage.PERFORMANCE, TestStage.STRESS]:
                resource_tracking_thread = threading.Thread(
                    target=track_resource_usage,
                    args=(stage_configs[stage].output_dir, DEFAULT_RESOURCE_TRACKING_INTERVAL, resource_tracking_stop)
                )
                resource_tracking_thread.daemon = True
                resource_tracking_thread.start()
            
            # Run the appropriate test based on stage
            if stage == TestStage.STRESS:
                # Run as a load test
                results = run_load_test(
                    test_cases=stage_configs[stage].test_cases,
                    config=stage_configs[stage],
                    concurrent_users=DEFAULT_CONCURRENT_USERS * 2,  # Higher concurrency for stress test
                    duration=DEFAULT_DURATION_SECONDS * 1.5,  # Longer duration for stress test
                    ramp_up=DEFAULT_RAMP_UP_SECONDS
                )
            elif stage == TestStage.RESILIENCE:
                # TODO: Implement fault injection for resilience testing
                # For now, run standard benchmark
                results = run_benchmark(stage_configs[stage])
            else:
                # Run standard benchmark
                results = run_benchmark(stage_configs[stage])
            
            # Stop resource tracking
            if resource_tracking_thread:
                resource_tracking_stop.set()
                resource_tracking_thread.join(timeout=5)
            
            # Calculate stage duration
            stage_duration = time.time() - stage_start
            
            # Store stage results
            multi_stage_results["stage_results"][stage] = {
                "success": True,
                "duration_seconds": stage_duration,
                "results_path": os.path.join(stage_configs[stage].output_dir, f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"),
                "resource_tracking_path": resource_tracking_path
            }
            
            # Add stage-specific metrics
            if stage == TestStage.SMOKE:
                # Simple pass/fail for smoke tests
                multi_stage_results["stage_results"][stage]["metrics"] = {
                    "pass_rate": results.summary_metrics["success_metrics"]["overall_success"],
                    "case_count": len(stage_configs[stage].test_cases)
                }
            elif stage == TestStage.STRESS:
                # Add load test metrics
                multi_stage_results["stage_results"][stage]["metrics"] = {
                    "throughput": results["summary"]["throughput"],
                    "avg_response_time": results["summary"]["avg_response_time"],
                    "success_rate": results["summary"]["successful_requests"] / max(1, results["summary"]["total_requests"])
                }
            else:
                # Add standard benchmark metrics
                multi_stage_results["stage_results"][stage]["metrics"] = {
                    "hallucination_reduction": results.summary_metrics["hallucination"]["avg_v3_reduction_rate"],
                    "latency_overhead": results.summary_metrics["latency"]["avg_v3_overhead_ms"],
                    "success_rate": results.summary_metrics["success_metrics"]["overall_success"]
                }
        
        except Exception as e:
            logger.error(f"Error in {stage} stage: {str(e)}")
            multi_stage_results["stage_results"][stage] = {
                "success": False,
                "error": str(e),
                "duration_seconds": time.time() - stage_start
            }
    
    # Calculate overall duration
    multi_stage_duration = time.time() - multi_stage_start
    multi_stage_results["meta"]["end_time"] = datetime.now().isoformat()
    multi_stage_results["meta"]["duration_seconds"] = multi_stage_duration
    
    # Calculate summary metrics
    successful_stages = sum(1 for stage_result in multi_stage_results["stage_results"].values() if stage_result["success"])
    multi_stage_results["summary"] = {
        "stage_count": len(stages),
        "successful_stages": successful_stages,
        "success_rate": successful_stages / len(stages),
        "total_duration": multi_stage_duration
    }
    
    # Generate overall assessment
    if successful_stages == len(stages):
        multi_stage_results["summary"]["assessment"] = "All stages completed successfully."
    elif successful_stages >= len(stages) * 0.75:
        multi_stage_results["summary"]["assessment"] = "Most stages completed successfully."
    elif successful_stages >= len(stages) * 0.5:
        multi_stage_results["summary"]["assessment"] = "Some stages failed."
    else:
        multi_stage_results["summary"]["assessment"] = "Multiple stage failures detected."
    
    # Export multi-stage results
    multi_stage_path = os.path.join(output_dir, f"multi_stage_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(multi_stage_path, "w") as f:
        json.dump(multi_stage_results, f, indent=2)
    
    logger.info(f"Multi-stage test results saved to {multi_stage_path}")
    
    return multi_stage_results


def print_multi_stage_summary(multi_stage_results: Dict[str, Any]) -> None:
    """Print a summary of multi-stage test results."""
    summary = multi_stage_results["summary"]
    meta = multi_stage_results["meta"]
    stage_results = multi_stage_results["stage_results"]
    
    print("\nMulti-Stage Test Summary:")
    print(f"Stages Run: {', '.join(meta['stages'])}")
    print(f"Total Duration: {summary['total_duration']:.2f} seconds")
    print(f"Success Rate: {summary['success_rate']:.0%} ({summary['successful_stages']}/{summary['stage_count']} stages)")
    print(f"Assessment: {summary['assessment']}")
    
    print("\nStage Results:")
    for stage, result in stage_results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        duration = f"{result['duration_seconds']:.2f}s"
        
        print(f"- {stage}: {status} ({duration})")
        
        if result["success"] and "metrics" in result:
            metrics = result["metrics"]
            
            if stage == TestStage.SMOKE:
                print(f"  Pass Rate: {metrics['pass_rate']:.0%}")
            elif stage == TestStage.STRESS:
                print(f"  Throughput: {metrics['throughput']:.2f} req/s")
                print(f"  Avg Response Time: {metrics['avg_response_time']:.2f} ms")
                print(f"  Success Rate: {metrics['success_rate']:.0%}")
            else:
                print(f"  Hallucination Reduction: {metrics['hallucination_reduction']:.2%}")
                print(f"  Latency Overhead: {metrics['latency_overhead']:.2f} ms")
                print(f"  Success Rate: {metrics['success_rate']:.0%}")


def main():
    """Main function to run benchmarks from command-line."""
    args = parse_args()
    
    # Generate test cases
    test_cases = generate_test_cases()
    logger.info(f"Generated {len(test_cases)} test cases")
    
    # Apply filters
    filtered_cases = filter_test_cases(
        test_cases,
        domain=args.domain,
        case_id=args.case_id,
        tag=args.tag,
        difficulty=args.difficulty,
        complexity=args.complexity,
        resource_intensive=args.resource_intensive
    )
    
    if not filtered_cases:
        logger.error("No test cases match the specified filters")
        return 1
    
    logger.info(f"Selected {len(filtered_cases)} test cases after filtering")
    
    # Print test case info in dry run mode
    if args.dry_run:
        print_test_case_info(filtered_cases)
        return 0
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        test_cases=filtered_cases,
        include_v2=args.include_v2,
        parallel=args.parallel,
        max_workers=args.max_workers,
        output_formats=args.formats,
        output_dir=args.output_dir
    )
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Print configuration summary
    print("\nBenchmark Configuration:")
    print(f"- Test Cases: {len(config.test_cases)}")
    print(f"- Include v2: {config.include_v2}")
    print(f"- Parallel Execution: {config.parallel}")
    print(f"- Max Workers: {config.max_workers}")
    print(f"- Output Formats: {', '.join(config.output_formats)}")
    print(f"- Output Directory: {config.output_dir}")
    
    # Initialize results variable
    results = None
    resource_tracking_path = None
    
    # Start resource tracking if enabled
    resource_tracking_thread = None
    resource_tracking_stop = threading.Event()
    
    if args.resource_tracking:
        resource_tracking_thread = threading.Thread(
            target=track_resource_usage,
            args=(config.output_dir, DEFAULT_RESOURCE_TRACKING_INTERVAL, resource_tracking_stop)
        )
        resource_tracking_thread.daemon = True
        resource_tracking_thread.start()
        print("\nResource tracking enabled")
    
    try:
        # Handle different test modes
        if args.load_test:
            print("\nRunning in load test mode... (this may take a while)")
            
            # Run load test
            load_test_results = run_load_test(
                test_cases=filtered_cases,
                config=config,
                concurrent_users=args.concurrent_users,
                ramp_up=args.ramp_up,
                duration=args.duration,
                think_time=args.think_time,
                constant_rate=args.constant_rate
            )
            
            # Print load test summary
            print_load_test_summary(load_test_results)
            
        elif args.multi_stage:
            print(f"\nRunning multi-stage test with stages: {', '.join(args.stages)}... (this may take a while)")
            
            # Run multi-stage test
            multi_stage_results = run_multi_stage_test(
                test_cases=filtered_cases,
                config=config,
                stages=args.stages,
                output_dir=config.output_dir,
                resource_tracking=args.resource_tracking
            )
            
            # Print multi-stage summary
            print_multi_stage_summary(multi_stage_results)
            
        else:
            print("\nRunning benchmarks... (this may take a while)")
            
            # Record start time
            start_time = time.time()
            
            # Run standard benchmark
            results = run_benchmark(config)
            
            # Record end time
            end_time = time.time()
            total_time = end_time - start_time
            
            # Print summary
            print("\nBenchmark Summary:")
            print(f"Total test cases: {results.summary_metrics['case_count']}")
            print(f"Execution time: {total_time:.2f} seconds")
            print(f"Average hallucination reduction rate: {results.summary_metrics['hallucination']['avg_v3_reduction_rate']:.2%}")
            print(f"Average latency overhead: {results.summary_metrics['latency']['avg_v3_overhead_ms']:.2f} ms")
            
            print("\nSuccess Metrics:")
            hallucination_success = results.summary_metrics['success_metrics']['hallucination_reduction_success']
            latency_success = results.summary_metrics['success_metrics']['latency_overhead_success']
            overall_success = results.summary_metrics['success_metrics']['overall_success']
            
            print(f"Hallucination Reduction Success: {'✅' if hallucination_success else '❌'} " +
                f"({results.summary_metrics['hallucination']['avg_v3_reduction_rate']:.2%} vs 80% target)")
            
            print(f"Latency Overhead Success: {'✅' if latency_success else '❌'} " +
                f"({results.summary_metrics['latency']['avg_v3_overhead_ms']:.2f}ms vs 250ms target)")
            
            print(f"Overall Success: {'✅' if overall_success else '❌'}")
            
            if 'domain_metrics' in results.summary_metrics:
                print("\nDomain-Specific Metrics:")
                for domain, metrics in results.summary_metrics['domain_metrics'].items():
                    print(f"- {domain}: {metrics['avg_reduction_rate']:.2%} reduction rate, " +
                        f"{metrics['avg_latency_overhead_ms']:.2f}ms overhead")
            
            print(f"\nResults exported to {config.output_dir}")
    
    finally:
        # Stop resource tracking
        if resource_tracking_thread:
            resource_tracking_stop.set()
            resource_tracking_thread.join(timeout=5)
            print("Resource tracking stopped")
    
    # Find the results JSON file
    results_files = [f for f in os.listdir(config.output_dir) 
                    if f.startswith("benchmark_results_") and f.endswith(".json")]
    results_files.sort(reverse=True)  # Get the most recent one
    
    if results_files:
        current_results_path = os.path.join(config.output_dir, results_files[0])
        
        # Compare with previous run if requested
        if args.compare_run:
            try:
                comparison = generate_comparison_report(current_results_path, args.compare_run, config.output_dir)
                print_comparison_summary(comparison)
                
                # Detect and report regressions if enabled
                if args.detect_regressions:
                    regression_count = comparison["regression_cases_count"]
                    regression_threshold = args.regression_threshold
                    
                    regression_detected = regression_count > 0
                    regression_significant = (regression_count / comparison["common_cases_count"]) > regression_threshold
                    
                    if regression_detected:
                        print(f"\nRegression Analysis:")
                        print(f"Found {regression_count} potential regression cases.")
                        
                        if regression_significant:
                            print(f"⚠️ WARNING: Significant regression detected, exceeding threshold of {regression_threshold:.1%}")
                            
                            # Send regression alert if enabled
                            if args.alert_on_regression:
                                print("Sending regression alert...")
                                # TODO: Implement alert sender
                        else:
                            print(f"ℹ️ Minor regression detected, below threshold of {regression_threshold:.1%}")
            except Exception as e:
                logger.error(f"Error generating comparison report: {str(e)}")
        
        # Generate extended metrics if requested
        if args.extended_metrics:
            try:
                focus_areas = args.focus_areas
                if "all" in focus_areas:
                    focus_areas = ["hallucination", "latency", "confidence", "resources"]
                
                extended_metrics = generate_extended_metrics(current_results_path, config.output_dir, focus_areas)
                print_extended_metrics_summary(extended_metrics)
            except Exception as e:
                logger.error(f"Error generating extended metrics: {str(e)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())