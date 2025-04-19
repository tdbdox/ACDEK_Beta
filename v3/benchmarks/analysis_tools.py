"""
Benchmark Analysis Tools

Provides comparative analysis, metrics computations, and report formatting.
"""
import json
import csv
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data(path: str, format: str = "json") -> Dict[str, Any]:
    """
    Load data from file in specified format.
    
    Args:
        path: Path to the data file
        format: Format of the file ('json' or 'csv')
        
    Returns:
        Loaded data as dictionary or list
    """
    if format == "json":
        with open(path, 'r') as f:
            return json.load(f)
    elif format == "csv":
        with open(path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)
    return {}

def compare_versions(v2_data: Dict[str, Any], v3_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare benchmark results between two versions.
    
    Args:
        v2_data: Dictionary containing data from the first version
        v3_data: Dictionary containing data from the second version
        
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {
        "metadata": {
            "comparison_date": datetime.now().isoformat(),
            "v2_source": getattr(v2_data, "meta", {}).get("source_results", "unknown"),
            "v3_source": getattr(v3_data, "meta", {}).get("source_results", "unknown")
        },
        "metrics": {
            "hallucination": {},
            "latency": {},
            "confidence": {}
        },
        "case_by_case": {}
    }
    
    # Extract summary metrics
    v2_summary = v2_data.get("summary_metrics", {})
    v3_summary = v3_data.get("summary_metrics", {})
    
    # Compare hallucination metrics
    v2_hall = v2_summary.get("hallucination", {})
    v3_hall = v3_summary.get("hallucination", {})
    
    if v2_hall and v3_hall:
        comparison["metrics"]["hallucination"] = {
            "v2_avg_reduction_rate": v2_hall.get("avg_v3_reduction_rate", 0),
            "v3_avg_reduction_rate": v3_hall.get("avg_v3_reduction_rate", 0),
            "delta": v3_hall.get("avg_v3_reduction_rate", 0) - v2_hall.get("avg_v3_reduction_rate", 0),
            "v2_total_count": v2_hall.get("total_v3_count", 0),
            "v3_total_count": v3_hall.get("total_v3_count", 0),
            "count_delta": v3_hall.get("total_v3_count", 0) - v2_hall.get("total_v3_count", 0)
        }
    
    # Compare latency metrics
    v2_latency = v2_summary.get("latency", {})
    v3_latency = v3_summary.get("latency", {})
    
    if v2_latency and v3_latency:
        comparison["metrics"]["latency"] = {
            "v2_avg_overhead_ms": v2_latency.get("avg_v3_overhead_ms", 0),
            "v3_avg_overhead_ms": v3_latency.get("avg_v3_overhead_ms", 0),
            "delta": v3_latency.get("avg_v3_overhead_ms", 0) - v2_latency.get("avg_v3_overhead_ms", 0),
            "percent_change": ((v3_latency.get("avg_v3_overhead_ms", 0) - v2_latency.get("avg_v3_overhead_ms", 0)) / 
                              v2_latency.get("avg_v3_overhead_ms", 1)) * 100 if v2_latency.get("avg_v3_overhead_ms", 0) else 0
        }
    
    # Compare confidence metrics
    v2_conf = v2_summary.get("confidence", {})
    v3_conf = v3_summary.get("confidence", {})
    
    if v2_conf and v3_conf:
        comparison["metrics"]["confidence"] = {
            "v2_avg_confidence": v2_conf.get("avg_v3_confidence", 0),
            "v3_avg_confidence": v3_conf.get("avg_v3_confidence", 0),
            "delta": v3_conf.get("avg_v3_confidence", 0) - v2_conf.get("avg_v3_confidence", 0),
            "percent_change": ((v3_conf.get("avg_v3_confidence", 0) - v2_conf.get("avg_v3_confidence", 0)) / 
                              v2_conf.get("avg_v3_confidence", 1)) * 100 if v2_conf.get("avg_v3_confidence", 0) else 0
        }
    
    # Case-by-case comparison
    v2_comparisons = {c.get("case_id"): c for c in v2_data.get("comparisons", [])}
    v3_comparisons = {c.get("case_id"): c for c in v3_data.get("comparisons", [])}
    
    # Find common cases
    common_cases = set(v2_comparisons.keys()) & set(v3_comparisons.keys())
    
    for case_id in common_cases:
        v2_case = v2_comparisons[case_id]
        v3_case = v3_comparisons[case_id]
        
        # Extract metrics for comparison
        v2_hall_metrics = v2_case.get("metrics", {}).get("hallucination", {})
        v3_hall_metrics = v3_case.get("metrics", {}).get("hallucination", {})
        
        v2_latency_metrics = v2_case.get("metrics", {}).get("latency", {})
        v3_latency_metrics = v3_case.get("metrics", {}).get("latency", {})
        
        # Create case comparison
        case_comparison = {
            "hallucination": {
                "v2_reduction_rate": v2_hall_metrics.get("v3_reduction_rate", 0),
                "v3_reduction_rate": v3_hall_metrics.get("v3_reduction_rate", 0),
                "delta": v3_hall_metrics.get("v3_reduction_rate", 0) - v2_hall_metrics.get("v3_reduction_rate", 0)
            },
            "latency": {
                "v2_overhead_ms": v2_latency_metrics.get("v3_overhead_ms", 0),
                "v3_overhead_ms": v3_latency_metrics.get("v3_overhead_ms", 0),
                "delta": v3_latency_metrics.get("v3_overhead_ms", 0) - v2_latency_metrics.get("v3_overhead_ms", 0)
            },
            "improved": False
        }
        
        # Determine if case improved overall
        hall_improved = case_comparison["hallucination"]["delta"] > 0
        latency_improved = case_comparison["latency"]["delta"] <= 0
        
        case_comparison["improved"] = hall_improved and latency_improved
        
        comparison["case_by_case"][case_id] = case_comparison
    
    # Overall improvement metrics
    comparison["overall"] = {
        "case_count": len(common_cases),
        "improved_count": sum(1 for c in comparison["case_by_case"].values() if c["improved"]),
        "regression_count": sum(1 for c in comparison["case_by_case"].values() if not c["improved"]),
        "hall_improved": comparison["metrics"]["hallucination"].get("delta", 0) > 0,
        "latency_improved": comparison["metrics"]["latency"].get("delta", 0) <= 0,
        "overall_improved": (comparison["metrics"]["hallucination"].get("delta", 0) > 0 and 
                           comparison["metrics"]["latency"].get("delta", 0) <= 0)
    }
    
    return comparison

def precision_recall_f1(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    Each result dict: {'true_positive': int, 'false_positive': int, 'false_negative': int}
    
    Args:
        results: List of dictionaries with true_positive, false_positive, and false_negative
        
    Returns:
        Dictionary with precision, recall, and F1 metrics
    """
    tp = sum(r.get('true_positive', 0) for r in results)
    fp = sum(r.get('false_positive', 0) for r in results)
    fn = sum(r.get('false_negative', 0) for r in results)
    
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    
    return {
        'precision': precision, 
        'recall': recall, 
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }

def aggregate_retry_fallback_rates(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate retry and fallback rates across benchmark runs.
    Each result: {'retry_count': int, 'fallback_count': int}
    
    Args:
        results: List of dictionaries with retry_count and fallback_count
        
    Returns:
        Dictionary with aggregated retry and fallback metrics
    """
    if not results:
        return {
            'average_retries': 0.0,
            'average_fallbacks': 0.0,
            'max_retries': 0,
            'max_fallbacks': 0,
            'total_retries': 0,
            'total_fallbacks': 0
        }
    
    total_retries = sum(r.get('retry_count', 0) for r in results)
    total_fallbacks = sum(r.get('fallback_count', 0) for r in results)
    max_retries = max(r.get('retry_count', 0) for r in results)
    max_fallbacks = max(r.get('fallback_count', 0) for r in results)
    runs = len(results)
    
    return {
        'average_retries': total_retries / runs,
        'average_fallbacks': total_fallbacks / runs,
        'max_retries': max_retries,
        'max_fallbacks': max_fallbacks,
        'total_retries': total_retries,
        'total_fallbacks': total_fallbacks
    }

def export_report(data: Dict[str, Any], path: str, format: str = "json") -> None:
    """
    Export analysis report to various formats.
    
    Args:
        data: Data to export
        path: Path to save the report
        format: Format to export to ('json', 'csv', 'html')
    """
    if format == "json":
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    elif format == "csv":
        # For CSV, we need to flatten the data structure
        flattened_data = _flatten_dict(data)
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(["Metric", "Value"])
            
            # Write data rows
            for key, value in flattened_data.items():
                writer.writerow([key, value])
    elif format == "html":
        # Create a simple HTML report
        html_content = _generate_html_report(data)
        
        with open(path, 'w') as f:
            f.write(html_content)

def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single-level dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Key from parent dictionary
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def _generate_html_report(data: Dict[str, Any]) -> str:
    """
    Generate HTML report from data.
    
    Args:
        data: Data to include in the report
        
    Returns:
        HTML string for the report
    """
    # Simple HTML template for report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Benchmark Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ margin-bottom: 20px; padding: 10px; border: 1px solid #eee; border-radius: 5px; }}
            .metric {{ margin: 5px 0; }}
            .key {{ font-weight: bold; }}
            .value {{ color: #444; }}
            .improved {{ color: green; }}
            .regression {{ color: red; }}
            .neutral {{ color: gray; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Benchmark Analysis Report</h1>
        <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    """
    
    # Add sections based on the data structure
    if "metrics" in data:
        html += """
        <div class="section">
            <h2>Summary Metrics</h2>
        """
        
        for category, metrics in data["metrics"].items():
            html += f"""
            <h3>{category.capitalize()}</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            """
            
            for metric, value in metrics.items():
                # Format based on metric type
                if isinstance(value, float):
                    if "percent" in metric or "rate" in metric:
                        formatted_value = f"{value:.2%}"
                    else:
                        formatted_value = f"{value:.2f}"
                else:
                    formatted_value = value
                
                # Apply color based on delta metrics
                css_class = "neutral"
                if "delta" in metric:
                    if category == "latency" and value <= 0:
                        css_class = "improved"
                    elif category != "latency" and value > 0:
                        css_class = "improved"
                    elif value != 0:
                        css_class = "regression"
                
                html += f"""
                <tr>
                    <td class="key">{metric.replace('_', ' ').title()}</td>
                    <td class="value {css_class}">{formatted_value}</td>
                </tr>
                """
            
            html += """
            </table>
            """
        
        html += """
        </div>
        """
    
    # Add case-by-case section if available
    if "case_by_case" in data and data["case_by_case"]:
        html += """
        <div class="section">
            <h2>Case-by-Case Comparison</h2>
            <table>
                <tr>
                    <th>Case ID</th>
                    <th>Hall. Reduction Δ</th>
                    <th>Latency Overhead Δ</th>
                    <th>Overall</th>
                </tr>
        """
        
        for case_id, case_data in data["case_by_case"].items():
            hall_delta = case_data.get("hallucination", {}).get("delta", 0)
            latency_delta = case_data.get("latency", {}).get("delta", 0)
            improved = case_data.get("improved", False)
            
            hall_class = "improved" if hall_delta > 0 else "regression" if hall_delta < 0 else "neutral"
            latency_class = "improved" if latency_delta <= 0 else "regression"
            overall_class = "improved" if improved else "regression"
            
            html += f"""
            <tr>
                <td>{case_id}</td>
                <td class="{hall_class}">{hall_delta:.2%}</td>
                <td class="{latency_class}">{latency_delta:.2f} ms</td>
                <td class="{overall_class}">{"✓" if improved else "✗"}</td>
            </tr>
            """
        
        html += """
            </table>
        </div>
        """
    
    # Add overall summary if available
    if "overall" in data:
        overall = data["overall"]
        html += """
        <div class="section">
            <h2>Overall Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        for metric, value in overall.items():
            if metric == "overall_improved":
                formatted_value = "✓" if value else "✗"
                css_class = "improved" if value else "regression"
            elif metric.endswith("_improved"):
                formatted_value = "✓" if value else "✗"
                css_class = "improved" if value else "regression"
            elif isinstance(value, float):
                formatted_value = f"{value:.2f}"
                css_class = "neutral"
            else:
                formatted_value = value
                css_class = "neutral"
            
            html += f"""
            <tr>
                <td class="key">{metric.replace('_', ' ').title()}</td>
                <td class="value {css_class}">{formatted_value}</td>
            </tr>
            """
        
        html += """
            </table>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    return html

def generate_charts(data: Dict[str, Any], output_dir: str) -> List[str]:
    """
    Generate charts from benchmark data and save to output directory.
    
    Args:
        data: Benchmark data to visualize
        output_dir: Directory to save charts
        
    Returns:
        List of paths to generated chart files
    """
    os.makedirs(output_dir, exist_ok=True)
    chart_paths = []
    
    # Extract relevant data
    comparisons = data.get("comparisons", [])
    test_cases = data.get("test_cases", [])
    summary = data.get("summary_metrics", {})
    
    # Only create charts if we have data
    if not comparisons:
        return chart_paths
    
    # Create domain comparison chart
    if "domain_metrics" in summary:
        domain_chart_path = os.path.join(output_dir, f"domain_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        domains = []
        reduction_rates = []
        latency_overheads = []
        
        for domain, metrics in summary["domain_metrics"].items():
            domains.append(domain)
            reduction_rates.append(metrics.get("avg_reduction_rate", 0) * 100)  # Convert to percentage
            latency_overheads.append(metrics.get("avg_latency_overhead_ms", 0))
        
        fig, ax1 = plt.figure(figsize=(10, 6)), plt.gca()
        
        x = np.arange(len(domains))
        width = 0.35
        
        # Create bars for reduction rates
        bars1 = ax1.bar(x - width/2, reduction_rates, width, label='Hallucination Reduction (%)')
        ax1.set_ylabel('Reduction Rate (%)')
        ax1.set_title('Domain Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(domains, rotation=45, ha='right')
        
        # Create second y-axis for latency
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, latency_overheads, width, color='orange', label='Latency Overhead (ms)')
        ax2.set_ylabel('Latency Overhead (ms)')
        
        # Add legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(domain_chart_path)
        plt.close()
        
        chart_paths.append(domain_chart_path)
    
    # Create hallucination reduction chart
    hall_chart_path = os.path.join(output_dir, f"hallucination_reduction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    case_ids = []
    raw_counts = []
    v3_counts = []
    
    for comp in comparisons:
        case_ids.append(comp.get("case_id", "unknown"))
        raw_counts.append(comp.get("metrics", {}).get("hallucination", {}).get("raw_count", 0))
        v3_counts.append(comp.get("metrics", {}).get("hallucination", {}).get("v3_count", 0))
    
    # Limit to 10 cases for readability
    if len(case_ids) > 10:
        case_ids = case_ids[:10]
        raw_counts = raw_counts[:10]
        v3_counts = v3_counts[:10]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(case_ids))
    width = 0.35
    
    plt.bar(x - width/2, raw_counts, width, label='Raw LLM Issues')
    plt.bar(x + width/2, v3_counts, width, label='ACDEK v3 Issues')
    
    plt.ylabel('Hallucination Issues Count')
    plt.title('Hallucination Reduction by Test Case')
    plt.xticks(x, case_ids, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(hall_chart_path)
    plt.close()
    
    chart_paths.append(hall_chart_path)
    
    # Create latency comparison chart
    latency_chart_path = os.path.join(output_dir, f"latency_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    
    case_ids = []
    raw_latencies = []
    v3_latencies = []
    v2_latencies = []
    
    for comp in comparisons:
        case_ids.append(comp.get("case_id", "unknown"))
        raw_latencies.append(comp.get("metrics", {}).get("latency", {}).get("raw_latency_ms", 0))
        v3_latencies.append(comp.get("metrics", {}).get("latency", {}).get("v3_latency_ms", 0))
        
        # Add v2 if available
        v2_latency = comp.get("metrics", {}).get("latency", {}).get("v2_latency_ms")
        v2_latencies.append(v2_latency if v2_latency is not None else 0)
    
    # Limit to 10 cases for readability
    if len(case_ids) > 10:
        case_ids = case_ids[:10]
        raw_latencies = raw_latencies[:10]
        v3_latencies = v3_latencies[:10]
        v2_latencies = v2_latencies[:10]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(case_ids))
    width = 0.25
    
    plt.bar(x - width, raw_latencies, width, label='Raw LLM Latency (ms)')
    
    # Only include v2 if we have data
    has_v2 = any(v2 > 0 for v2 in v2_latencies)
    if has_v2:
        plt.bar(x, v2_latencies, width, label='ACDEK v2 Latency (ms)')
        plt.bar(x + width, v3_latencies, width, label='ACDEK v3 Latency (ms)')
    else:
        plt.bar(x + width/2, v3_latencies, width, label='ACDEK v3 Latency (ms)')
    
    plt.ylabel('Latency (ms)')
    plt.title('Latency Comparison by Test Case')
    plt.xticks(x, case_ids, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(latency_chart_path)
    plt.close()
    
    chart_paths.append(latency_chart_path)
    
    return chart_paths

def export_html_dashboard(
    data: Dict[str, Any], 
    output_path: str,
    title: str = "ACDEK v3 Benchmark Dashboard",
    chart_dir: Optional[str] = None
) -> str:
    """
    Export an HTML dashboard with interactive components.
    
    Args:
        data: Benchmark data to include in the dashboard
        output_path: Path to save the HTML dashboard
        title: Title for the dashboard
        chart_dir: Directory to save chart images (if None, will create charts directory next to output_path)
        
    Returns:
        Path to the generated HTML dashboard
    """
    # Default chart directory if not specified
    if not chart_dir:
        chart_dir = os.path.join(os.path.dirname(output_path), "charts")
    
    # Generate charts
    chart_paths = generate_charts(data, chart_dir)
    
    # Extract data for dashboard
    summary = data.get("summary_metrics", {})
    comparisons = data.get("comparisons", [])
    
    # Calculate high-level metrics
    hall_reduction = summary.get("hallucination", {}).get("avg_v3_reduction_rate", 0)
    latency_overhead = summary.get("latency", {}).get("avg_v3_overhead_ms", 0)
    success = summary.get("success_metrics", {}).get("overall_success", False)
    
    # Create a relative path for charts
    chart_rel_dir = os.path.relpath(chart_dir, os.path.dirname(output_path))
    chart_rel_paths = [os.path.join(chart_rel_dir, os.path.basename(p)) for p in chart_paths]
    
    # Simple HTML template for the dashboard
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
            .header {{ background: #333; color: white; padding: 20px; margin-bottom: 20px; }}
            .header h1 {{ margin: 0; }}
            .summary-cards {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
            .card {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; flex: 1; min-width: 200px; }}
            .card h2 {{ margin-top: 0; color: #333; font-size: 18px; }}
            .card .value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
            .card .success {{ color: #4CAF50; }}
            .card .failure {{ color: #F44336; }}
            .chart-container {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
            .chart-container h2 {{ margin-top: 0; color: #333; font-size: 18px; }}
            .chart {{ width: 100%; max-width: 100%; height: auto; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .improved {{ color: #4CAF50; }}
            .regression {{ color: #F44336; }}
            .filters {{ background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
            .filters h2 {{ margin-top: 0; font-size: 18px; }}
            .filters select {{ padding: 8px; margin-right: 10px; }}
            .footer {{ text-align: center; margin-top: 40px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="container">
                <h1>{title}</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
        </div>
        
        <div class="container">
            <div class="summary-cards">
                <div class="card">
                    <h2>Hallucination Reduction</h2>
                    <div class="value {('success' if hall_reduction >= 0.8 else 'failure')}">{hall_reduction:.2%}</div>
                    <p>Target: 80%</p>
                </div>
                
                <div class="card">
                    <h2>Latency Overhead</h2>
                    <div class="value {('success' if latency_overhead <= 250 else 'failure')}">{latency_overhead:.2f} ms</div>
                    <p>Target: 250ms</p>
                </div>
                
                <div class="card">
                    <h2>Overall Success</h2>
                    <div class="value {('success' if success else 'failure')}">{"✓" if success else "✗"}</div>
                    <p>Meets all targets</p>
                </div>
                
                <div class="card">
                    <h2>Test Cases</h2>
                    <div class="value">{len(comparisons)}</div>
                    <p>Total cases run</p>
                </div>
            </div>
    """
    
    # Add charts if available
    if chart_paths:
        for chart_rel_path in chart_rel_paths:
            chart_name = os.path.basename(chart_rel_path).split('_')[0].capitalize()
            
            html += f"""
            <div class="chart-container">
                <h2>{chart_name} Analysis</h2>
                <img src="{chart_rel_path}" alt="{chart_name} chart" class="chart">
            </div>
            """
    
    # Add case-by-case table
    if comparisons:
        html += """
        <div class="chart-container">
            <h2>Case-by-Case Results</h2>
            <div class="filters">
                <h2>Filters</h2>
                <select id="domainFilter">
                    <option value="all">All Domains</option>
        """
        
        # Get unique domains
        domains = set()
        for comp in comparisons:
            case_id = comp.get("case_id", "")
            domain = next((tc.get("domain") for tc in data.get("test_cases", []) if tc.get("id") == case_id), "unknown")
            domains.add(domain)
        
        for domain in sorted(domains):
            html += f'<option value="{domain}">{domain}</option>'
        
        html += """
                </select>
                <select id="sortBy">
                    <option value="case_id">Sort by ID</option>
                    <option value="reduction">Sort by Reduction</option>
                    <option value="latency">Sort by Latency</option>
                </select>
            </div>
            
            <table id="resultsTable">
                <thead>
                    <tr>
                        <th>Case ID</th>
                        <th>Domain</th>
                        <th>Raw Issues</th>
                        <th>V3 Issues</th>
                        <th>Reduction</th>
                        <th>Latency</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for comp in comparisons:
            case_id = comp.get("case_id", "")
            domain = next((tc.get("domain") for tc in data.get("test_cases", []) if tc.get("id") == case_id), "unknown")
            raw_count = comp.get("metrics", {}).get("hallucination", {}).get("raw_count", 0)
            v3_count = comp.get("metrics", {}).get("hallucination", {}).get("v3_count", 0)
            reduction = comp.get("metrics", {}).get("hallucination", {}).get("v3_reduction_rate", 0)
            latency = comp.get("metrics", {}).get("latency", {}).get("v3_latency_ms", 0)
            
            reduction_class = "improved" if reduction >= 0.8 else "regression"
            latency_class = "improved" if latency <= 250 else "regression"
            
            html += f"""
                <tr data-domain="{domain}">
                    <td>{case_id}</td>
                    <td>{domain}</td>
                    <td>{raw_count}</td>
                    <td>{v3_count}</td>
                    <td class="{reduction_class}">{reduction:.2%}</td>
                    <td class="{latency_class}">{latency:.2f} ms</td>
                </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
        
        <script>
            // Simple filter and sort functionality
            document.getElementById('domainFilter').addEventListener('change', filterTable);
            document.getElementById('sortBy').addEventListener('change', sortTable);
            
            function filterTable() {
                const domain = document.getElementById('domainFilter').value;
                const rows = document.querySelectorAll('#resultsTable tbody tr');
                
                rows.forEach(row => {
                    if (domain === 'all' || row.getAttribute('data-domain') === domain) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });
            }
            
            function sortTable() {
                const sortBy = document.getElementById('sortBy').value;
                const tbody = document.querySelector('#resultsTable tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                
                rows.sort((a, b) => {
                    if (sortBy === 'case_id') {
                        return a.cells[0].textContent.localeCompare(b.cells[0].textContent);
                    } else if (sortBy === 'reduction') {
                        return parseFloat(b.cells[4].textContent) - parseFloat(a.cells[4].textContent);
                    } else if (sortBy === 'latency') {
                        return parseFloat(a.cells[5].textContent) - parseFloat(b.cells[5].textContent);
                    }
                    return 0;
                });
                
                // Remove all rows
                while (tbody.firstChild) {
                    tbody.removeChild(tbody.firstChild);
                }
                
                // Add sorted rows
                rows.forEach(row => {
                    tbody.appendChild(row);
                });
            }
        </script>
        """
    
    # Add footer
    html += """
            <div class="footer">
                <p>ACDEK v3 Benchmark Dashboard | Generated by analysis_tools.py</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path