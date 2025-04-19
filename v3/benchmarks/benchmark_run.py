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
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Set

from benchmarks.hallucination_benchmark import (
    BenchmarkConfig,
    run_benchmark,
    generate_test_cases,
    TestCase,
    Domain
)
from benchmarks.analysis_tools import (
    export_report,
    precision_recall_f1,
    aggregate_retry_fallback_rates,
    compare_versions,
    load_data
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ACDEK v3 Benchmark Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
    
    parser.add_argument(
        "--domain",
        type=str,
        choices=[d.value for d in Domain],
        help="Filter test cases by domain"
    )
    
    parser.add_argument(
        "--case-id",
        type=str,
        help="Run only the specified test case ID"
    )
    
    parser.add_argument(
        "--tag",
        type=str,
        help="Filter test cases by tag"
    )
    
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=range(1, 6),
        help="Filter test cases by difficulty level (1-5)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show test cases that would be run without executing them"
    )
    
    parser.add_argument(
        "--compare-run",
        type=str,
        help="Path to previous benchmark results to compare against"
    )
    
    parser.add_argument(
        "--extended-metrics",
        action="store_true",
        help="Generate extended metrics and analysis"
    )
    
    parser.add_argument(
        "--report-title",
        type=str,
        default="ACDEK v3 Benchmark Results",
        help="Title for the benchmark report"
    )
    
    parser.add_argument(
        "--focus-areas",
        type=str,
        nargs="+",
        choices=["hallucination", "latency", "confidence", "all"],
        default=["all"],
        help="Focus areas for analysis"
    )
    
    return parser.parse_args()


def filter_test_cases(
    test_cases: List[TestCase], 
    domain: Optional[str] = None,
    case_id: Optional[str] = None,
    tag: Optional[str] = None,
    difficulty: Optional[int] = None
) -> List[TestCase]:
    """Filter test cases based on criteria."""
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
    
    return filtered_cases


def print_test_case_info(test_cases: List[TestCase]) -> None:
    """Print information about test cases."""
    print(f"\nSelected Test Cases ({len(test_cases)}):")
    
    for i, tc in enumerate(test_cases, 1):
        print(f"{i}. [{tc.id}] {tc.prompt[:80]}{'...' if len(tc.prompt) > 80 else ''}")
        print(f"   Domain: {tc.domain.value}, Difficulty: {tc.difficulty}, Tags: {', '.join(tc.tags)}")
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
        
        case_comparisons[case_id] = {
            "hallucination_delta": hallucination_delta,
            "latency_delta": latency_delta,
            "improved": hallucination_delta > 0 and latency_delta <= 0
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
        )
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
    
    # Difficulty analysis
    difficulty_analysis = extended_metrics.get("metrics", {}).get("difficulty_analysis", {})
    if difficulty_analysis:
        print("\nDifficulty Analysis:")
        for difficulty, metrics in sorted(difficulty_analysis.items()):
            print(f"- Level {difficulty}:")
            print(f"  Cases: {metrics['case_count']}")
            print(f"  Hallucination Reduction: {metrics['avg_hallucination_reduction']:.2%}")
    
    # Hallucination type analysis
    hallucination_types = extended_metrics.get("metrics", {}).get("hallucination_type_analysis", {})
    if hallucination_types:
        print("\nHallucination Type Analysis:")
        for hall_type, metrics in hallucination_types.items():
            print(f"- {hall_type}:")
            print(f"  Raw Count: {metrics['raw_count']}")
            print(f"  V3 Count: {metrics['v3_count']}")
            print(f"  Reduction Rate: {metrics['reduction_rate']:.2%}")


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
        difficulty=args.difficulty
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
    
    # Print configuration summary
    print("\nBenchmark Configuration:")
    print(f"- Test Cases: {len(config.test_cases)}")
    print(f"- Include v2: {config.include_v2}")
    print(f"- Parallel Execution: {config.parallel}")
    print(f"- Max Workers: {config.max_workers}")
    print(f"- Output Formats: {', '.join(config.output_formats)}")
    print(f"- Output Directory: {config.output_dir}")
    print("\nRunning benchmarks... (this may take a while)")
    
    # Record start time
    start_time = time.time()
    
    # Run benchmark
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
            except Exception as e:
                logger.error(f"Error generating comparison report: {str(e)}")
        
        # Generate extended metrics if requested
        if args.extended_metrics:
            try:
                focus_areas = args.focus_areas
                if "all" in focus_areas:
                    focus_areas = ["hallucination", "latency", "confidence"]
                
                extended_metrics = generate_extended_metrics(current_results_path, config.output_dir, focus_areas)
                print_extended_metrics_summary(extended_metrics)
            except Exception as e:
                logger.error(f"Error generating extended metrics: {str(e)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())