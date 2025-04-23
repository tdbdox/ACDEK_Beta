#!/usr/bin/env python
"""
ACDEK v3 Integrated Benchmark Example

This example demonstrates a complete workflow using the enhanced benchmark suite:
1. Configure and run the ACDEK benchmark suite
2. Apply domain-specific threshold configuration
3. Analyze results with statistical significance testing
4. Generate advanced visualizations
5. Export comprehensive reports

Usage:
    python integrated_benchmark_example.py [--models MODEL_LIST] [--domains DOMAIN_LIST]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple

# Add parent directory to path to allow importing from project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import benchmark components
from benchmarks.acdek_benchmark_suite import (
    ACDEKBenchmarkSuite,
    BenchmarkSuiteConfig,
    Domain
)
from benchmarks.comparative_analyzer import (
    ComparativeAnalyzer,
    OutputFormat,
    MetricType,
    VersionMetrics
)
from benchmarks.threshold_config import (
    ThresholdConfig,
    Domain as ThresholdDomain,
    MetricType as ThresholdMetricType,
    ThresholdSet
)
from benchmarks.advanced_visualizations import AdvancedVisualization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integrated_benchmark.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = Path("./benchmark_reports")
DEFAULT_MODELS = ["gpt-4"]
DEFAULT_DOMAINS = [domain.value for domain in Domain]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ACDEK v3 Integrated Benchmark Example")
    
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated list of LLM models to benchmark"
    )
    
    parser.add_argument(
        "--domains",
        type=str,
        default=",".join([d for d in DEFAULT_DOMAINS if d != "general"]),
        help="Comma-separated list of domains to test"
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
        "--parallel",
        action="store_true",
        default=True,
        help="Run benchmarks in parallel"
    )
    
    parser.add_argument(
        "--significance-level",
        type=float,
        default=0.05,
        help="Statistical significance level (default: 0.05)"
    )
    
    parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="Optimize thresholds based on benchmark results"
    )
    
    return parser.parse_args()


def main():
    """Run the integrated benchmark workflow."""
    args = parse_args()
    
    print(f"ACDEK v3 Integrated Benchmark Example")
    print(f"=====================================\n")
    
    # Parse model and domain lists
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    domains = [domain.strip() for domain in args.domains.split(",") if domain.strip()]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Step 1: Configure and run the benchmark suite
    print(f"Step 1: Configuring and running benchmark suite...")
    results = run_benchmark_suite(models, domains, output_dir, args.parallel)
    
    # Step 2: Apply domain-specific threshold configuration
    print(f"\nStep 2: Applying domain-specific thresholds...")
    thresholds = configure_thresholds(results, optimize=args.optimize_thresholds)
    
    # Step 3: Analyze results with statistical significance testing
    print(f"\nStep 3: Analyzing results with statistical significance testing...")
    analysis = analyze_results(results, thresholds, 
                             significance_level=args.significance_level)
    
    # Step 4: Generate advanced visualizations
    print(f"\nStep 4: Generating advanced visualizations...")
    visualization_paths = generate_visualizations(analysis, output_dir, timestamp)
    
    # Step 5: Export comprehensive reports
    print(f"\nStep 5: Exporting comprehensive reports...")
    report_path = export_report(
        results, 
        analysis, 
        visualization_paths,
        output_dir, 
        timestamp, 
        format=OutputFormat(args.output_format)
    )
    
    print(f"\nIntegrated benchmark workflow complete!")
    print(f"Report generated at: {report_path}")
    
    return 0


def run_benchmark_suite(
    models: List[str],
    domains: List[str],
    output_dir: Path,
    parallel: bool = True
) -> Dict[str, Any]:
    """Run the benchmark suite and return results."""
    # Configure the benchmark suite
    config = BenchmarkSuiteConfig(
        llm_models=models,
        domains=domains,
        parallel=parallel,
        max_workers=4,
        output_dir=str(output_dir),
        output_formats=["json", "csv"]
    )
    
    # Create and run benchmark suite
    benchmark_suite = ACDEKBenchmarkSuite(config)
    results = benchmark_suite.run_benchmark()
    
    # Print basic summary
    overall = results.summary_metrics["overall"]
    print(f"Benchmark completed with {overall['case_count']} test cases across {len(models)} models")
    print(f"Overall improvement: {overall['improvements']['overall']:.4f}")
    print(f"Factuality improvement: {overall['improvements']['factuality']:.4f}")
    print(f"Latency overhead: {overall['latency']['avg_overhead_seconds']:.3f}s")
    print(f"Success rate: {overall['success_rates']['overall_success']:.1%}")
    
    return results.to_dict()


def configure_thresholds(
    benchmark_results: Dict[str, Any],
    optimize: bool = False
) -> ThresholdConfig:
    """Configure domain-specific thresholds for analysis."""
    # Create base threshold configuration
    threshold_config = ThresholdConfig()
    
    # Extract domains from results
    domains = set()
    for result in benchmark_results.get("results", []):
        if "test_case" in result and "domain" in result["test_case"]:
            domains.add(result["test_case"]["domain"])
    
    print(f"Configuring thresholds for domains: {', '.join(domains)}")
    
    # If optimization requested, use benchmark data to optimize thresholds
    if optimize and benchmark_results:
        for domain in domains:
            try:
                # Try to convert to enum
                domain_enum = ThresholdDomain(domain)
                
                # Optimize factuality threshold
                optimized = threshold_config.optimize_threshold(
                    domain=domain,
                    metric_type=ThresholdMetricType.FACTUALITY,
                    benchmark_data=benchmark_results
                )
                
                print(f"Optimized {domain} factuality threshold: {optimized.threshold}")
                
                # Optimize hallucination threshold
                optimized = threshold_config.optimize_threshold(
                    domain=domain,
                    metric_type=ThresholdMetricType.HALLUCINATION,
                    benchmark_data=benchmark_results
                )
                
                print(f"Optimized {domain} hallucination threshold: {optimized.threshold}")
                
            except ValueError:
                print(f"Warning: Domain {domain} not recognized for threshold optimization")
    else:
        print("Using default thresholds (optimization disabled)")
    
    return threshold_config


def analyze_results(
    benchmark_results: Dict[str, Any],
    thresholds: ThresholdConfig,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """Analyze benchmark results with statistical significance testing."""
    # Create analyzer
    analyzer = ComparativeAnalyzer()
    
    # Prepare data for analysis
    formatted_results = benchmark_results.copy()
    
    # Extract v3 metrics
    v3_metrics = analyzer.extract_version_metrics(formatted_results, version="v3")
    
    # Extract raw metrics (if available)
    raw_metrics = analyzer.extract_version_metrics(formatted_results, version="raw")
    
    # Run statistical analysis
    analysis_results = analyzer.analyze_with_statistical_significance(
        v3_metrics=v3_metrics, 
        raw_metrics=raw_metrics,
        significance_level=significance_level
    )
    
    # Apply domain-specific analysis
    domain_analysis = analyzer.perform_domain_specific_analysis(
        v3_metrics, 
        raw_metrics,
        thresholds
    )
    
    # Print summary of statistical testing
    significance_results = analysis_results.get("significance_tests", {})
    for metric, result in significance_results.items():
        is_significant = result.get("is_significant", False)
        p_value = result.get("p_value", 1.0)
        print(f"Metric: {metric} - Statistically significant: {is_significant} (p={p_value:.4f})")
    
    # Combined results
    results = {
        "statistical_analysis": analysis_results,
        "domain_analysis": domain_analysis,
        "v3_metrics": v3_metrics,
        "raw_metrics": raw_metrics
    }
    
    return results


def generate_visualizations(
    analysis_results: Dict[str, Any],
    output_dir: Path,
    timestamp: str
) -> Dict[str, Path]:
    """Generate advanced visualizations from analysis results."""
    # Create visualizations directory
    viz_dir = output_dir / "visualizations" / timestamp
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization handler
    visualizer = AdvancedVisualization(output_dir=viz_dir)
    
    # Extract metrics from analysis results
    v3_metrics = analysis_results.get("v3_metrics", {})
    raw_metrics = analysis_results.get("raw_metrics", {})
    domain_analysis = analysis_results.get("domain_analysis", {})
    
    # Initialize results dictionary to track generated visualizations
    visualization_paths = {}
    
    # Create radar chart for v3 vs raw performance
    print(f"Generating radar chart...")
    radar_data = {
        "ACDEK v3": v3_metrics.get("aggregated", {}),
        "Raw LLM": raw_metrics.get("aggregated", {})
    }
    
    labels = ["Factuality", "Hallucination", "Constraint", "Latency", "Confidence"]
    radar_path = visualizer.create_radar_chart(
        data=radar_data,
        labels=labels,
        title="ACDEK v3 vs Raw LLM Performance"
    )
    visualization_paths["radar_chart"] = radar_path
    
    # Create heatmap for domain performance
    print(f"Generating performance heatmap...")
    heatmap_data = {}
    domains = domain_analysis.keys()
    metrics = ["factuality", "hallucination", "constraint", "latency", "confidence"]
    
    for domain in domains:
        domain_metrics = {}
        for metric in metrics:
            if metric in domain_analysis.get(domain, {}).get("v3", {}):
                domain_metrics[metric] = domain_analysis[domain]["v3"][metric]
            else:
                domain_metrics[metric] = 0.0
        heatmap_data[domain] = domain_metrics
    
    heatmap_path = visualizer.create_heatmap(
        data=heatmap_data,
        title="Performance by Domain"
    )
    visualization_paths["domain_heatmap"] = heatmap_path
    
    # Generate ROC curve for hallucination detection
    print(f"Generating ROC curve...")
    # This would typically use real TPR/FPR data, but we'll use synthetic data for the example
    roc_data = {
        "ACDEK v3": {"tpr": [0, 0.2, 0.5, 0.8, 0.9, 0.95, 1.0], "fpr": [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]},
        "Raw LLM": {"tpr": [0, 0.1, 0.3, 0.6, 0.7, 0.85, 1.0], "fpr": [0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]}
    }
    
    roc_path = visualizer.create_roc_curve(
        data=roc_data,
        title="Hallucination Detection Performance"
    )
    visualization_paths["roc_curve"] = roc_path
    
    # Generate statistical significance bar chart
    print(f"Generating statistical analysis chart...")
    significance_data = analysis_results.get("statistical_analysis", {}).get("significance_tests", {})
    
    sig_chart_data = {
        "Metric": [],
        "P-Value": [],
        "Significant": []
    }
    
    for metric, result in significance_data.items():
        sig_chart_data["Metric"].append(metric)
        sig_chart_data["P-Value"].append(result.get("p_value", 1.0))
        sig_chart_data["Significant"].append(result.get("is_significant", False))
    
    # Convert data format for bar chart with error
    bar_data = {"Statistics": {}}
    for i, metric in enumerate(sig_chart_data["Metric"]):
        p_value = sig_chart_data["P-Value"][i]
        # Format as (value, lower_bound, upper_bound) for error bars
        bar_data["Statistics"][metric] = (p_value, max(0, p_value-0.01), min(1.0, p_value+0.01))
    
    sig_chart_path = visualizer.create_bar_with_error(
        data=bar_data,
        title="Statistical Significance by Metric"
    )
    visualization_paths["significance_chart"] = sig_chart_path
    
    # Create an interactive dashboard
    print(f"Generating interactive dashboard...")
    dashboard_path = visualizer.create_html_dashboard(
        title="ACDEK v3 Benchmark Dashboard",
        charts=visualization_paths,
        metadata={"Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    )
    visualization_paths["dashboard"] = dashboard_path
    
    return visualization_paths


def export_report(
    benchmark_results: Dict[str, Any],
    analysis_results: Dict[str, Any],
    visualization_paths: Dict[str, Path],
    output_dir: Path,
    timestamp: str,
    format: OutputFormat = OutputFormat.HTML
) -> Path:
    """Export a comprehensive report with all results and visualizations."""
    # Create analyzer for report export
    analyzer = ComparativeAnalyzer(output_dir=output_dir)
    
    # Combine results for report
    combined_results = {
        "benchmark_results": benchmark_results,
        "analysis_results": analysis_results,
        "visualization_paths": {k: str(v) for k, v in visualization_paths.items()}
    }
    
    # Export the report
    report_name = f"integrated_benchmark_report_{timestamp}"
    report_path = output_dir / f"{report_name}.{format.value}"
    
    # Use the analyzer to export a properly formatted report
    exported_path = analyzer.export_analysis_report(
        data=combined_results,
        output_format=format,
        output_file=report_path,
        include_visualizations=True
    )
    
    # Generate PDF version if requested
    if format != OutputFormat.PDF:
        print("Generating PDF version of the report...")
        visualizer = AdvancedVisualization(output_dir=output_dir)
        
        # Create list of charts to include
        charts = [str(path) for path in visualization_paths.values()]
        
        # Generate PDF report
        pdf_path = visualizer.generate_pdf_report(
            title="ACDEK v3 Benchmark Report",
            charts=charts,
            metadata={"Summary": f"Benchmark results for {len(benchmark_results.get('results', []))} test cases"}
        )
        
        print(f"PDF report generated at: {pdf_path}")
    
    return exported_path


if __name__ == "__main__":
    sys.exit(main())
