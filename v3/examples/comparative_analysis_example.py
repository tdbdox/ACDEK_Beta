#!/usr/bin/env python
"""
ACDEK v3 Comparative Analysis Example

This example demonstrates how to use the ComparativeAnalyzer to:
1. Load benchmark data from existing results
2. Standardize metrics across v2, v3, and raw LLM outputs
3. Detect performance regressions
4. Normalize scores across different LLM providers
5. Generate exportable benchmark reports

Usage:
    python comparative_analysis_example.py [--benchmark-file PATH] [--output-format FORMAT]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path to allow importing from project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.comparative_analyzer import (
    ComparativeAnalyzer,
    OutputFormat,
    BenchmarkComparison
)

DEFAULT_BENCHMARK_FILE = Path("./benchmarks/results/benchmark_results_latest.json")
DEFAULT_OUTPUT_DIR = Path("./benchmarks/reports")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ACDEK v3 Comparative Analysis Example")
    
    parser.add_argument(
        "--benchmark-file",
        type=str,
        default=str(DEFAULT_BENCHMARK_FILE),
        help="Path to benchmark results file to analyze"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output reports"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=[fmt.value for fmt in OutputFormat],
        default="html",
        help="Output format for the report"
    )
    
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize metrics by LLM provider"
    )
    
    return parser.parse_args()

def main():
    """Run the comparative analysis example."""
    args = parse_args()
    
    print(f"ACDEK v3 Comparative Analysis Example")
    print(f"=====================================")
    
    # Check if benchmark file exists or create synthetic data for demo
    benchmark_data = {}
    benchmark_file = Path(args.benchmark_file)
    
    if benchmark_file.exists():
        print(f"Loading benchmark data from {benchmark_file}")
        try:
            with open(benchmark_file, 'r') as f:
                benchmark_data = json.load(f)
        except Exception as e:
            print(f"Error loading benchmark file: {e}")
            print("Generating synthetic data for demonstration instead.")
            benchmark_data = generate_synthetic_data()
    else:
        print(f"Benchmark file not found: {benchmark_file}")
        print("Generating synthetic data for demonstration.")
        benchmark_data = generate_synthetic_data()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the analyzer
    analyzer = ComparativeAnalyzer(
        results_dir=benchmark_file.parent,
        output_dir=output_dir
    )
    
    # Analyze the benchmark data
    print(f"Analyzing benchmark data...")
    comparison = analyzer.analyze_benchmark_data(
        benchmark_data,
        normalize_by_provider=args.normalize
    )
    
    # Print summary information
    print_summary(comparison)
    
    # Export the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_format = OutputFormat(args.output_format)
    output_file = output_dir / f"comparative_analysis_{timestamp}.{output_format.value}"
    
    print(f"Exporting {output_format.value.upper()} report...")
    report_path = analyzer.export_comparison_report(
        comparison,
        output_format=output_format,
        output_file=output_file
    )
    
    print(f"Report generated successfully: {report_path}")
    return 0

def print_summary(comparison: BenchmarkComparison):
    """Print a summary of the comparison results."""
    print("\nComparison Summary:")
    print(f"------------------")
    print(f"Total test cases: {comparison.metadata['case_count']}")
    
    # Print v3 performance metrics
    v3_metrics = comparison.summary["v3"]
    print(f"\nACDEK v3 Performance:")
    print(f"  Hallucination Reduction: {v3_metrics['avg_hallucination_reduction']:.2%}")
    print(f"  Latency: {v3_metrics['avg_latency_ms']:.2f} ms")
    print(f"  Confidence Score: {v3_metrics['avg_confidence_score']:.2f}")
    print(f"  Combined Score: {v3_metrics['avg_combined_score']:.2f}")
    
    # Print v2 comparison if available
    v2_comp = comparison.summary["v2_comparison"]
    if v2_comp["has_v2_data"]:
        print(f"\nComparison with ACDEK v2:")
        print(f"  Cases with v2 data: {v2_comp['comparison_count']}")
        print(f"  Regression cases: {v2_comp['regression_count']} ({v2_comp['regression_percent']:.2%})")
        
        deltas = v2_comp["avg_deltas"]
        print(f"  Hallucination Reduction delta: {deltas['hallucination_reduction']:.4f}")
        print(f"  Latency delta: {deltas['latency_ms']:.2f} ms")
        print(f"  Confidence delta: {deltas['confidence_score']:.4f}")
    
    # Print domain breakdown
    print(f"\nPerformance by Domain:")
    for domain, data in comparison.domains.items():
        print(f"  {domain}: {data['case_count']} cases, {data['v3']['hallucination_reduction']:.2%} reduction")
    
    # Print regression information if any
    if comparison.regression_cases:
        print(f"\nDetected {len(comparison.regression_cases)} regression cases")
    
    print()

def generate_synthetic_data():
    """Generate synthetic benchmark data for demonstration purposes."""
    return {
        "summary_metrics": {
            "hallucination": {
                "total_raw_count": 150,
                "total_v3_count": 30,
                "total_v2_count": 45,
                "avg_v3_reduction_rate": 0.8,
                "avg_v2_reduction_rate": 0.7
            },
            "latency": {
                "avg_raw_latency_ms": 200.0,
                "avg_v3_latency_ms": 250.0,
                "avg_v2_latency_ms": 240.0,
                "avg_v3_overhead_ms": 50.0,
                "avg_v2_overhead_ms": 40.0
            },
            "confidence": {
                "avg_v3_confidence": 0.9,
                "avg_v2_confidence": 0.85,
                "avg_raw_confidence": 0.7
            }
        },
        "comparisons": [
            # Financial domain cases
            {
                "case_id": "financial_case_1",
                "metrics": {
                    "hallucination": {
                        "raw_count": 20,
                        "v3_count": 4,
                        "v2_count": 6,
                        "v3_reduction_rate": 0.8,
                        "v2_reduction_rate": 0.7
                    },
                    "latency": {
                        "raw_latency_ms": 200.0,
                        "v3_latency_ms": 250.0,
                        "v2_latency_ms": 240.0,
                        "v3_overhead_ms": 50.0,
                        "v2_overhead_ms": 40.0
                    },
                    "confidence": {
                        "v3_confidence": 0.9,
                        "v2_confidence": 0.85
                    }
                },
                "raw_llm": {"confidence_score": 0.7}
            },
            {
                "case_id": "financial_case_2",
                "metrics": {
                    "hallucination": {
                        "raw_count": 15,
                        "v3_count": 3,
                        "v2_count": 4,
                        "v3_reduction_rate": 0.8,
                        "v2_reduction_rate": 0.73
                    },
                    "latency": {
                        "raw_latency_ms": 190.0,
                        "v3_latency_ms": 230.0,
                        "v2_latency_ms": 235.0,
                        "v3_overhead_ms": 40.0,
                        "v2_overhead_ms": 45.0
                    },
                    "confidence": {
                        "v3_confidence": 0.88,
                        "v2_confidence": 0.83
                    }
                },
                "raw_llm": {"confidence_score": 0.68}
            },
            # Medical domain cases
            {
                "case_id": "medical_case_1",
                "metrics": {
                    "hallucination": {
                        "raw_count": 25,
                        "v3_count": 4,
                        "v2_count": 7,
                        "v3_reduction_rate": 0.84,
                        "v2_reduction_rate": 0.72
                    },
                    "latency": {
                        "raw_latency_ms": 210.0,
                        "v3_latency_ms": 260.0,
                        "v2_latency_ms": 250.0,
                        "v3_overhead_ms": 50.0,
                        "v2_overhead_ms": 40.0
                    },
                    "confidence": {
                        "v3_confidence": 0.92,
                        "v2_confidence": 0.87
                    }
                },
                "raw_llm": {"confidence_score": 0.73}
            },
            # Legal domain cases
            {
                "case_id": "legal_case_1",
                "metrics": {
                    "hallucination": {
                        "raw_count": 30,
                        "v3_count": 5,
                        "v2_count": 8,
                        "v3_reduction_rate": 0.83,
                        "v2_reduction_rate": 0.73
                    },
                    "latency": {
                        "raw_latency_ms": 220.0,
                        "v3_latency_ms": 280.0,
                        "v2_latency_ms": 270.0,
                        "v3_overhead_ms": 60.0,
                        "v2_overhead_ms": 50.0
                    },
                    "confidence": {
                        "v3_confidence": 0.89,
                        "v2_confidence": 0.84
                    }
                },
                "raw_llm": {"confidence_score": 0.71}
            },
            # Technical domain cases (regression case)
            {
                "case_id": "technical_case_1",
                "metrics": {
                    "hallucination": {
                        "raw_count": 18,
                        "v3_count": 7,
                        "v2_count": 5,
                        "v3_reduction_rate": 0.61,  # Worse than v2
                        "v2_reduction_rate": 0.72
                    },
                    "latency": {
                        "raw_latency_ms": 180.0,
                        "v3_latency_ms": 260.0,
                        "v2_latency_ms": 220.0,
                        "v3_overhead_ms": 80.0,  # Worse than v2
                        "v2_overhead_ms": 40.0
                    },
                    "confidence": {
                        "v3_confidence": 0.79,  # Worse than v2
                        "v2_confidence": 0.87
                    }
                },
                "raw_llm": {"confidence_score": 0.69}
            }
        ],
        "test_cases": [
            {"id": "financial_case_1", "domain": "financial", "model_info": {"provider": "openai"}},
            {"id": "financial_case_2", "domain": "financial", "model_info": {"provider": "openai"}},
            {"id": "medical_case_1", "domain": "medical", "model_info": {"provider": "anthropic"}},
            {"id": "legal_case_1", "domain": "legal", "model_info": {"provider": "anthropic"}},
            {"id": "technical_case_1", "domain": "technical", "model_info": {"provider": "google"}}
        ]
    }

if __name__ == "__main__":
    sys.exit(main())