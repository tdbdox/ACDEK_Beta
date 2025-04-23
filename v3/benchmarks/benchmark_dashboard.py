#!/usr/bin/env python3
"""
ACDEK Benchmark Dashboard

Generates interactive visualization dashboards from benchmark results.
This script can process both general benchmarking results and specialized
grounding benchmark results.

Usage:
    python benchmark_dashboard.py --input results/benchmark_results_20250418.json --output dashboard.html
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = Path("./benchmarks/results/dashboards")
CHART_STYLES = {
    "colors": {
        "primary": "#2196F3",
        "secondary": "#4CAF50",
        "accent": "#9C27B0",
        "warning": "#FFC107",
        "error": "#F44336",
        "neutral": "#607D8B"
    },
    "font_size": {
        "title": 16,
        "axis": 12,
        "ticks": 10,
        "legend": 10
    }
}


def load_benchmark_results(file_path: str) -> Dict[str, Any]:
    """
    Load benchmark results from a JSON file.
    
    Args:
        file_path: Path to the benchmark results file
        
    Returns:
        Dictionary with benchmark results
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded benchmark results from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading benchmark results from {file_path}: {str(e)}")
        return {}


def detect_result_type(results: Dict[str, Any]) -> str:
    """
    Detect the type of benchmark results.
    
    Args:
        results: Benchmark results
        
    Returns:
        Type of benchmark results ("general" or "grounding")
    """
    # Check for grounding-specific fields
    if "standard_response" in results.get("results", [{}])[0] if results.get("results") else {}:
        return "grounding"
    
    # Check for general benchmark fields
    if "acdek_response" in results.get("results", [{}])[0] if results.get("results") else {}:
        return "general"
    
    # Default to general
    return "general"


class DashboardGenerator:
    """
    Base class for dashboard generators.
    """
    
    def __init__(self, results: Dict[str, Any], output_dir: str):
        """
        Initialize the dashboard generator.
        
        Args:
            results: Benchmark results
            output_dir: Directory to save generated files
        """
        self.results = results
        self.output_dir = output_dir
        self.charts_dir = os.path.join(output_dir, "charts")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_dashboard(self) -> str:
        """
        Generate the dashboard.
        
        Returns:
            Path to the generated dashboard file
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _generate_html_content(self, chart_paths: Dict[str, str], summary: Dict[str, Any]) -> str:
        """
        Generate HTML content for the dashboard.
        
        Args:
            chart_paths: Dictionary mapping chart names to file paths
            summary: Summary metrics
            
        Returns:
            HTML content
        """
        raise NotImplementedError("Subclasses must implement this method")


class GeneralBenchmarkDashboard(DashboardGenerator):
    """
    Dashboard generator for general benchmark results.
    """
    
    def generate_dashboard(self) -> str:
        """
        Generate the dashboard.
        
        Returns:
            Path to the generated dashboard file
        """
        # Generate charts
        chart_paths = self._generate_charts()
        
        # Extract summary metrics
        summary = self.results.get("summary_metrics", {})
        
        # Generate HTML content
        html_content = self._generate_html_content(chart_paths, summary)
        
        # Write HTML file
        output_path = os.path.join(self.output_dir, f"benchmark_dashboard_{self.timestamp}.html")
        with open(output_path, "w") as f:
            f.write(html_content)
            
        logger.info(f"Generated dashboard at {output_path}")
        return output_path
    
    def _generate_charts(self) -> Dict[str, str]:
        """
        Generate charts for the dashboard.
        
        Returns:
            Dictionary mapping chart names to file paths
        """
        chart_paths = {}
        
        # Overall improvement chart
        overall_chart_path = os.path.join(self.charts_dir, f"overall_improvement_{self.timestamp}.png")
        self._create_overall_improvement_chart(overall_chart_path)
        chart_paths["overall"] = os.path.relpath(overall_chart_path, self.output_dir)
        
        # Domain comparison chart
        domain_chart_path = os.path.join(self.charts_dir, f"domain_comparison_{self.timestamp}.png")
        self._create_domain_comparison_chart(domain_chart_path)
        chart_paths["domain"] = os.path.relpath(domain_chart_path, self.output_dir)
        
        # Factuality improvement chart
        factuality_chart_path = os.path.join(self.charts_dir, f"factuality_improvement_{self.timestamp}.png")
        self._create_factuality_improvement_chart(factuality_chart_path)
        chart_paths["factuality"] = os.path.relpath(factuality_chart_path, self.output_dir)
        
        # Latency comparison chart
        latency_chart_path = os.path.join(self.charts_dir, f"latency_comparison_{self.timestamp}.png")
        self._create_latency_comparison_chart(latency_chart_path)
        chart_paths["latency"] = os.path.relpath(latency_chart_path, self.output_dir)
        
        return chart_paths
    
    def _create_overall_improvement_chart(self, output_path: str) -> None:
        """
        Create chart showing overall improvement.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(10, 6))
        
        # Extract data
        by_model = self.results.get("summary_metrics", {}).get("by_model", {})
        models = list(by_model.keys())
        improvements = [model_data.get("improvements", {}).get("overall", 0) for model_data in by_model.values()]
        
        # Create bar chart
        colors = [CHART_STYLES["colors"]["primary"] if imp > 0 else CHART_STYLES["colors"]["error"] 
                 for imp in improvements]
        plt.bar(models, improvements, color=colors)
        
        # Add horizontal line at zero
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Format chart
        plt.title('Overall Improvement by Model (ACDEK vs Raw LLM)', fontsize=CHART_STYLES["font_size"]["title"])
        plt.xlabel('Model', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.ylabel('Score Improvement', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _create_domain_comparison_chart(self, output_path: str) -> None:
        """
        Create chart comparing performance across domains.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(10, 6))
        
        # Extract data
        by_domain = self.results.get("summary_metrics", {}).get("by_domain", {})
        domains = list(by_domain.keys())
        improvements = [domain_data.get("avg_improvement", 0) for domain_data in by_domain.values()]
        success_rates = [domain_data.get("success_rate", 0) for domain_data in by_domain.values()]
        
        # Create grouped bar chart
        x = np.arange(len(domains))
        width = 0.35
        
        plt.bar(x - width/2, improvements, width, label='Avg Improvement', 
               color=CHART_STYLES["colors"]["primary"])
        plt.bar(x + width/2, success_rates, width, label='Success Rate', 
               color=CHART_STYLES["colors"]["secondary"])
        
        # Format chart
        plt.title('Performance by Domain', fontsize=CHART_STYLES["font_size"]["title"])
        plt.xlabel('Domain', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.ylabel('Score', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.xticks(x, domains, fontsize=CHART_STYLES["font_size"]["ticks"])
        plt.legend(fontsize=CHART_STYLES["font_size"]["legend"])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _create_factuality_improvement_chart(self, output_path: str) -> None:
        """
        Create chart showing factuality improvement.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(10, 6))
        
        # Extract data
        results_list = self.results.get("results", [])
        if not results_list:
            # Create empty chart
            plt.text(0.5, 0.5, "No factuality data available", 
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=100)
            plt.close()
            return
        
        # Group by test case
        test_cases = []
        raw_scores = []
        acdek_scores = []
        
        for result in results_list[:10]:  # Limit to first 10 for readability
            test_case = result.get("test_case", {})
            test_cases.append(test_case.get("id", "unknown"))
            
            metrics = result.get("metrics", {})
            raw_scores.append(metrics.get("factuality", {}).get("raw_score", 0))
            acdek_scores.append(metrics.get("factuality", {}).get("acdek_score", 0))
        
        # Create grouped bar chart
        x = np.arange(len(test_cases))
        width = 0.35
        
        plt.bar(x - width/2, raw_scores, width, label='Raw LLM', 
               color=CHART_STYLES["colors"]["warning"])
        plt.bar(x + width/2, acdek_scores, width, label='ACDEK', 
               color=CHART_STYLES["colors"]["primary"])
        
        # Format chart
        plt.title('Factuality Score Comparison', fontsize=CHART_STYLES["font_size"]["title"])
        plt.xlabel('Test Case', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.ylabel('Factuality Score', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.xticks(x, test_cases, rotation=45, ha='right', fontsize=CHART_STYLES["font_size"]["ticks"])
        plt.legend(fontsize=CHART_STYLES["font_size"]["legend"])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _create_latency_comparison_chart(self, output_path: str) -> None:
        """
        Create chart showing latency comparison.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(10, 6))
        
        # Extract data
        by_model = self.results.get("summary_metrics", {}).get("by_model", {})
        models = list(by_model.keys())
        overheads = [model_data.get("latency", {}).get("avg_overhead_seconds", 0) for model_data in by_model.values()]
        acceptable_rates = [model_data.get("success_rates", {}).get("acceptable_latency", 0) 
                          for model_data in by_model.values()]
        
        # Create dual-axis chart
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Primary y-axis for latency overhead
        color = CHART_STYLES["colors"]["primary"]
        ax1.set_xlabel('Model', fontsize=CHART_STYLES["font_size"]["axis"])
        ax1.set_ylabel('Latency Overhead (seconds)', color=color, fontsize=CHART_STYLES["font_size"]["axis"])
        ax1.bar(models, overheads, color=color, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Secondary y-axis for acceptable rate
        ax2 = ax1.twinx()
        color = CHART_STYLES["colors"]["secondary"]
        ax2.set_ylabel('Acceptable Latency Rate', color=color, fontsize=CHART_STYLES["font_size"]["axis"])
        ax2.plot(models, acceptable_rates, 'o-', color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1.1)
        
        # Format chart
        plt.title('Latency Overhead Analysis', fontsize=CHART_STYLES["font_size"]["title"])
        ax1.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _generate_html_content(self, chart_paths: Dict[str, str], summary: Dict[str, Any]) -> str:
        """
        Generate HTML content for the dashboard.
        
        Args:
            chart_paths: Dictionary mapping chart names to file paths
            summary: Summary metrics
            
        Returns:
            HTML content
        """
        overall = summary.get("overall", {})
        by_model = summary.get("by_model", {})
        by_domain = summary.get("by_domain", {})
        
        # Format metrics for display
        case_count = overall.get("case_count", 0)
        factuality_improvement = overall.get("improvements", {}).get("factuality", 0)
        overall_improvement = overall.get("improvements", {}).get("overall", 0)
        latency_overhead = overall.get("latency", {}).get("avg_overhead_seconds", 0)
        success_rate = overall.get("success_rates", {}).get("overall_success", 0)
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ACDEK Benchmark Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #333; }}
                .card {{ background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
                .metric-card {{ background-color: #f5f5f5; border-radius: 8px; padding: 15px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #F44336; }}
                .neutral {{ color: #2196F3; }}
                .chart-container {{ margin-bottom: 20px; }}
                .chart {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f5f5f5; }}
                .footer {{ text-align: center; margin-top: 40px; color: #666; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ACDEK Benchmark Results</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="card">
                    <h2>Summary</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Test Cases</div>
                            <div class="metric-value neutral">{case_count}</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Factuality Improvement</div>
                            <div class="metric-value {'positive' if factuality_improvement > 0 else 'negative'}">
                                {factuality_improvement:.4f}
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Overall Improvement</div>
                            <div class="metric-value {'positive' if overall_improvement > 0 else 'negative'}">
                                {overall_improvement:.4f}
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Latency Overhead</div>
                            <div class="metric-value {'negative' if latency_overhead > 1.0 else 'neutral'}">
                                {latency_overhead:.3f}s
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Success Rate</div>
                            <div class="metric-value {'positive' if success_rate > 0.7 else 'neutral'}">
                                {success_rate:.1%}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Performance Visualization</h2>
                    
                    <div class="chart-container">
                        <h3>Overall Improvement by Model</h3>
                        <img src="{chart_paths.get('overall', '')}" alt="Overall Improvement Chart" class="chart">
                    </div>
                    
                    <div class="chart-container">
                        <h3>Factuality Improvement</h3>
                        <img src="{chart_paths.get('factuality', '')}" alt="Factuality Improvement Chart" class="chart">
                    </div>
                    
                    <div class="chart-container">
                        <h3>Performance by Domain</h3>
                        <img src="{chart_paths.get('domain', '')}" alt="Domain Performance Chart" class="chart">
                    </div>
                    
                    <div class="chart-container">
                        <h3>Latency Analysis</h3>
                        <img src="{chart_paths.get('latency', '')}" alt="Latency Analysis Chart" class="chart">
                    </div>
                </div>
                
                <div class="card">
                    <h2>Model Performance</h2>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Cases</th>
                            <th>Overall Improvement</th>
                            <th>Factuality Improvement</th>
                            <th>Latency Overhead</th>
                            <th>Success Rate</th>
                        </tr>
        """
        
        # Add rows for each model
        for model_name, metrics in by_model.items():
            overall_imp = metrics.get("improvements", {}).get("overall", 0)
            factuality_imp = metrics.get("improvements", {}).get("factuality", 0)
            latency_oh = metrics.get("latency", {}).get("avg_overhead_seconds", 0)
            success_r = metrics.get("success_rates", {}).get("overall_success", 0)
            
            html += f"""
                        <tr>
                            <td>{model_name}</td>
                            <td>{metrics.get('case_count', 0)}</td>
                            <td class="{'positive' if overall_imp > 0 else 'negative'}">{overall_imp:.4f}</td>
                            <td class="{'positive' if factuality_imp > 0 else 'negative'}">{factuality_imp:.4f}</td>
                            <td class="{'negative' if latency_oh > 1.0 else 'neutral'}">{latency_oh:.3f}s</td>
                            <td class="{'positive' if success_r > 0.7 else 'neutral'}">{success_r:.1%}</td>
                        </tr>
            """
        
        # Continue HTML
        html += """
                    </table>
                </div>
                
                <div class="card">
                    <h2>Domain Performance</h2>
                    <table>
                        <tr>
                            <th>Domain</th>
                            <th>Cases</th>
                            <th>Avg Improvement</th>
                            <th>Success Rate</th>
                        </tr>
        """
        
        # Add rows for each domain
        for domain, metrics in by_domain.items():
            avg_imp = metrics.get("avg_improvement", 0)
            success_r = metrics.get("success_rate", 0)
            
            html += f"""
                        <tr>
                            <td>{domain}</td>
                            <td>{metrics.get('case_count', 0)}</td>
                            <td class="{'positive' if avg_imp > 0 else 'negative'}">{avg_imp:.4f}</td>
                            <td class="{'positive' if success_r > 0.7 else 'neutral'}">{success_r:.1%}</td>
                        </tr>
            """
        
        # Finish HTML
        html += """
                    </table>
                </div>
                
                <div class="footer">
                    ACDEK Benchmark Dashboard | Generated by ACDEK Benchmarking Suite
                </div>
            </div>
        </body>
        </html>
        """
        
        return html


class GroundingBenchmarkDashboard(DashboardGenerator):
    """
    Dashboard generator for grounding benchmark results.
    """
    
    def generate_dashboard(self) -> str:
        """
        Generate the dashboard.
        
        Returns:
            Path to the generated dashboard file
        """
        # Generate charts
        chart_paths = self._generate_charts()
        
        # Extract summary metrics
        summary = self.results.get("summary_metrics", {})
        
        # Generate HTML content
        html_content = self._generate_html_content(chart_paths, summary)
        
        # Write HTML file
        output_path = os.path.join(self.output_dir, f"grounding_dashboard_{self.timestamp}.html")
        with open(output_path, "w") as f:
            f.write(html_content)
            
        logger.info(f"Generated dashboard at {output_path}")
        return output_path
    
    def _generate_charts(self) -> Dict[str, str]:
        """
        Generate charts for the dashboard.
        
        Returns:
            Dictionary mapping chart names to file paths
        """
        chart_paths = {}
        
        # Factuality comparison chart
        factuality_chart_path = os.path.join(self.charts_dir, f"factuality_comparison_{self.timestamp}.png")
        self._create_factuality_comparison_chart(factuality_chart_path)
        chart_paths["factuality"] = os.path.relpath(factuality_chart_path, self.output_dir)
        
        # Domain performance chart
        domain_chart_path = os.path.join(self.charts_dir, f"domain_performance_{self.timestamp}.png")
        self._create_domain_performance_chart(domain_chart_path)
        chart_paths["domain"] = os.path.relpath(domain_chart_path, self.output_dir)
        
        # Hallucination detection accuracy chart
        detection_chart_path = os.path.join(self.charts_dir, f"detection_accuracy_{self.timestamp}.png")
        self._create_detection_accuracy_chart(detection_chart_path)
        chart_paths["detection"] = os.path.relpath(detection_chart_path, self.output_dir)
        
        # Difficulty analysis chart
        difficulty_chart_path = os.path.join(self.charts_dir, f"difficulty_analysis_{self.timestamp}.png")
        self._create_difficulty_analysis_chart(difficulty_chart_path)
        chart_paths["difficulty"] = os.path.relpath(difficulty_chart_path, self.output_dir)
        
        # Domain-specific threshold comparison chart
        threshold_chart_path = os.path.join(self.charts_dir, f"domain_thresholds_{self.timestamp}.png")
        self._create_domain_threshold_chart(threshold_chart_path)
        chart_paths["thresholds"] = os.path.relpath(threshold_chart_path, self.output_dir)
        
        # Retry metrics chart
        retry_chart_path = os.path.join(self.charts_dir, f"retry_metrics_{self.timestamp}.png")
        self._create_retry_metrics_chart(retry_chart_path)
        chart_paths["retries"] = os.path.relpath(retry_chart_path, self.output_dir)
        
        # Trend analysis chart
        trend_chart_path = os.path.join(self.charts_dir, f"trend_analysis_{self.timestamp}.png")
        self._create_trend_analysis_chart(trend_chart_path)
        chart_paths["trends"] = os.path.relpath(trend_chart_path, self.output_dir)
        
        return chart_paths
        
    def _create_domain_threshold_chart(self, output_path: str) -> None:
        """
        Create chart showing performance against domain-specific thresholds.
        
        Args:
            output_path: Path to save the chart
        """
        from benchmarks.performance_metrics import DOMAIN_THRESHOLDS
        
        plt.figure(figsize=(12, 6))
        
        # Extract data from results
        by_domain = self.results.get("summary_metrics", {}).get("by_domain", {})
        domains = list(by_domain.keys())
        
        if not domains:
            # Create empty chart
            plt.text(0.5, 0.5, "No domain data available", 
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=100)
            plt.close()
            return
        
        # Prepare factuality data
        actual_values = []
        threshold_values = []
        
        for domain in domains:
            # Get actual factuality score
            domain_data = by_domain.get(domain, {})
            factuality_score = domain_data.get("avg_accuracy_factuality", 0)
            actual_values.append(factuality_score)
            
            # Get threshold for this domain
            domain_threshold = DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS.get("general", {}))
            threshold_value = domain_threshold.get("factuality", 0.7)  # Default to 0.7
            threshold_values.append(threshold_value)
        
        # Create grouped bar chart
        x = np.arange(len(domains))
        width = 0.35
        
        plt.bar(x - width/2, actual_values, width, label='Actual Factuality', 
               color=CHART_STYLES["colors"]["primary"])
        plt.bar(x + width/2, threshold_values, width, label='Domain Threshold', 
               color=CHART_STYLES["colors"]["accent"], alpha=0.7)
        
        # Add goal lines connecting threshold points
        plt.plot(x + width/2, threshold_values, 'o--', color=CHART_STYLES["colors"]["accent"], 
                alpha=0.8, linewidth=1.5)
        
        # Format chart
        plt.title('Factuality Scores vs. Domain-Specific Thresholds', fontsize=CHART_STYLES["font_size"]["title"])
        plt.xlabel('Domain', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.ylabel('Factuality Score', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.xticks(x, domains, fontsize=CHART_STYLES["font_size"]["ticks"])
        plt.legend(fontsize=CHART_STYLES["font_size"]["legend"])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _create_retry_metrics_chart(self, output_path: str) -> None:
        """
        Create chart showing retry metrics and penalties.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(10, 6))
        
        # Extract data from results
        retry_metrics = self.results.get("summary_metrics", {}).get("retries", {})
        
        if not retry_metrics:
            # Create empty chart
            plt.text(0.5, 0.5, "No retry metrics available", 
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=100)
            plt.close()
            return
        
        # Prepare data
        plugins = list(retry_metrics.get("counts", {}).keys())
        counts = [retry_metrics.get("counts", {}).get(plugin, 0) for plugin in plugins]
        penalties = [retry_metrics.get("penalties", {}).get(plugin, 0) for plugin in plugins]
        
        # Create dual-axis chart
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Primary y-axis for retry counts
        color = CHART_STYLES["colors"]["primary"]
        ax1.set_xlabel('Plugin', fontsize=CHART_STYLES["font_size"]["axis"])
        ax1.set_ylabel('Retry Counts', color=color, fontsize=CHART_STYLES["font_size"]["axis"])
        ax1.bar(plugins, counts, color=color, alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Secondary y-axis for penalties
        ax2 = ax1.twinx()
        color = CHART_STYLES["colors"]["warning"]
        ax2.set_ylabel('Penalty Score', color=color, fontsize=CHART_STYLES["font_size"]["axis"])
        ax2.plot(plugins, penalties, 'o-', color=color, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add trend indicators if available
        if "trends" in retry_metrics:
            for i, plugin in enumerate(plugins):
                trend = retry_metrics.get("trends", {}).get(plugin, "stable")
                
                if trend == "increasing":
                    ax1.annotate('↑', (i, counts[i]), xytext=(0, 10), 
                                textcoords='offset points', ha='center', 
                                color=CHART_STYLES["colors"]["error"], fontsize=16)
                elif trend == "decreasing":
                    ax1.annotate('↓', (i, counts[i]), xytext=(0, 10), 
                                textcoords='offset points', ha='center', 
                                color=CHART_STYLES["colors"]["secondary"], fontsize=16)
        
        # Format chart
        plt.title('Retry Metrics by Plugin', fontsize=CHART_STYLES["font_size"]["title"])
        plt.xticks(rotation=45, ha='right', fontsize=CHART_STYLES["font_size"]["ticks"])
        fig.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _create_trend_analysis_chart(self, output_path: str) -> None:
        """
        Create chart showing performance trends over time.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(10, 6))
        
        # Extract trend data from results
        trend_data = self.results.get("summary_metrics", {}).get("trends", {})
        
        if not trend_data or "windows" not in trend_data or not trend_data["windows"]:
            # Create empty chart
            plt.text(0.5, 0.5, "No trend data available", 
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=100)
            plt.close()
            return
        
        # Prepare data
        windows = trend_data.get("windows", [])
        times = []
        factuality_scores = []
        latency_values = []
        
        for window in windows:
            # Convert timestamps to datetime for better x-axis display
            if "start_time" in window:
                times.append(datetime.fromtimestamp(window["start_time"]))
            else:
                continue
                
            factuality_scores.append(window.get("avg_factuality", 0))
            latency_values.append(window.get("avg_latency", 0))
        
        # Create dual-axis chart
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Primary y-axis for factuality
        color = CHART_STYLES["colors"]["primary"]
        ax1.set_xlabel('Time', fontsize=CHART_STYLES["font_size"]["axis"])
        ax1.set_ylabel('Factuality Score', color=color, fontsize=CHART_STYLES["font_size"]["axis"])
        ax1.plot(times, factuality_scores, 'o-', color=color, linewidth=2, label='Factuality')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Secondary y-axis for latency
        ax2 = ax1.twinx()
        color = CHART_STYLES["colors"]["warning"]
        ax2.set_ylabel('Latency (seconds)', color=color, fontsize=CHART_STYLES["font_size"]["axis"])
        ax2.plot(times, latency_values, 's-', color=color, linewidth=2, label='Latency')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add trend indicators
        fact_trend = trend_data.get("factuality_trend", "stable")
        latency_trend = trend_data.get("latency_trend", "stable")
        
        trend_text = f"Factuality: {fact_trend.title()}, Latency: {latency_trend.title()}"
        plt.figtext(0.5, 0.01, trend_text, ha='center', fontsize=12)
        
        # Format chart
        plt.title('Performance Trends Over Time', fontsize=CHART_STYLES["font_size"]["title"])
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
                   bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=CHART_STYLES["font_size"]["legend"])
        
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for trend text
        
        # Format x-axis as dates
        fig.autofmt_xdate()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _create_factuality_comparison_chart(self, output_path: str) -> None:
        """
        Create chart comparing factuality scores.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(10, 6))
        
        # Extract data from results
        results_list = self.results.get("results", [])
        if not results_list:
            # Create empty chart
            plt.text(0.5, 0.5, "No factuality data available", 
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=100)
            plt.close()
            return
        
        # Prepare data
        case_ids = []
        standard_scores = []
        accuracy_scores = []
        
        for result in results_list[:10]:  # Limit to first 10 for readability
            case_ids.append(result.get("test_case", {}).get("id", "unknown"))
            standard_scores.append(result.get("metrics", {}).get("standard_factuality_score", 0))
            accuracy_scores.append(result.get("metrics", {}).get("accuracy_factuality_score", 0))
        
        # Create grouped bar chart
        x = np.arange(len(case_ids))
        width = 0.35
        
        plt.bar(x - width/2, standard_scores, width, label='Standard Response', 
               color=CHART_STYLES["colors"]["warning"])
        plt.bar(x + width/2, accuracy_scores, width, label='Accuracy-Focused Response', 
               color=CHART_STYLES["colors"]["primary"])
        
        # Format chart
        plt.title('Factuality Score Comparison', fontsize=CHART_STYLES["font_size"]["title"])
        plt.xlabel('Test Case', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.ylabel('Factuality Score', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.xticks(x, case_ids, rotation=45, ha='right', fontsize=CHART_STYLES["font_size"]["ticks"])
        plt.legend(fontsize=CHART_STYLES["font_size"]["legend"])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _create_domain_performance_chart(self, output_path: str) -> None:
        """
        Create chart showing performance by domain.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(10, 6))
        
        # Extract data
        by_domain = self.results.get("summary_metrics", {}).get("by_domain", {})
        domains = list(by_domain.keys())
        standard_scores = [domain_data.get("avg_standard_factuality", 0) for domain_data in by_domain.values()]
        accuracy_scores = [domain_data.get("avg_accuracy_factuality", 0) for domain_data in by_domain.values()]
        improvements = [domain_data.get("avg_improvement", 0) for domain_data in by_domain.values()]
        
        # Create grouped bar chart
        x = np.arange(len(domains))
        width = 0.25
        
        plt.bar(x - width, standard_scores, width, label='Standard Response', 
               color=CHART_STYLES["colors"]["warning"])
        plt.bar(x, accuracy_scores, width, label='Accuracy-Focused Response', 
               color=CHART_STYLES["colors"]["primary"])
        plt.bar(x + width, improvements, width, label='Improvement', 
               color=CHART_STYLES["colors"]["secondary"])
        
        # Format chart
        plt.title('Performance by Domain', fontsize=CHART_STYLES["font_size"]["title"])
        plt.xlabel('Domain', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.ylabel('Score', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.xticks(x, domains, fontsize=CHART_STYLES["font_size"]["ticks"])
        plt.legend(fontsize=CHART_STYLES["font_size"]["legend"])
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _create_detection_accuracy_chart(self, output_path: str) -> None:
        """
        Create chart showing hallucination detection accuracy.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(10, 6))
        
        # Extract data
        overall = self.results.get("summary_metrics", {}).get("overall", {})
        
        # Prepare data
        categories = ['Standard Response', 'Accuracy-Focused Response']
        accuracies = [
            overall.get("standard_detection_accuracy", 0),
            overall.get("accuracy_detection_accuracy", 0)
        ]
        
        # Create bar chart
        plt.bar(categories, accuracies, color=[
            CHART_STYLES["colors"]["warning"],
            CHART_STYLES["colors"]["primary"]
        ])
        
        # Add threshold line
        plt.axhline(y=0.7, color=CHART_STYLES["colors"]["accent"], linestyle='--', 
                   alpha=0.7, label='Target Accuracy')
        
        # Format chart
        plt.title('Hallucination Detection Accuracy', fontsize=CHART_STYLES["font_size"]["title"])
        plt.ylabel('Accuracy', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.ylim(0, 1.0)
        plt.legend(fontsize=CHART_STYLES["font_size"]["legend"])
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, accuracy in enumerate(accuracies):
            plt.text(i, accuracy + 0.02, f"{accuracy:.1%}", 
                    ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _create_difficulty_analysis_chart(self, output_path: str) -> None:
        """
        Create chart showing performance by difficulty.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(10,.6))
        
        # Extract data
        by_difficulty = self.results.get("summary_metrics", {}).get("by_difficulty", {})
        difficulties = sorted(by_difficulty.keys())
        
        # Check if we have data
        if not difficulties:
            # Create empty chart
            plt.text(0.5, 0.5, "No difficulty data available", 
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=100)
            plt.close()
            return
        
        # Prepare data
        standard_scores = [by_difficulty[d].get("avg_standard_factuality", 0) for d in difficulties]
        accuracy_scores = [by_difficulty[d].get("avg_accuracy_factuality", 0) for d in difficulties]
        improvements = [by_difficulty[d].get("avg_improvement", 0) for d in difficulties]
        
        # Create line chart
        plt.figure(figsize=(10, 6))
        x = [int(d) for d in difficulties]  # Convert to integers for the x-axis
        
        plt.plot(x, standard_scores, 'o-', label='Standard Response', 
                color=CHART_STYLES["colors"]["warning"], linewidth=2)
        plt.plot(x, accuracy_scores, 's-', label='Accuracy-Focused Response', 
                color=CHART_STYLES["colors"]["primary"], linewidth=2)
        plt.plot(x, improvements, '^-', label='Improvement', 
                color=CHART_STYLES["colors"]["secondary"], linewidth=2)
        
        # Format chart
        plt.title('Performance by Difficulty Level', fontsize=CHART_STYLES["font_size"]["title"])
        plt.xlabel('Difficulty (1-5)', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.ylabel('Score', fontsize=CHART_STYLES["font_size"]["axis"])
        plt.xticks(x, fontsize=CHART_STYLES["font_size"]["ticks"])
        plt.legend(fontsize=CHART_STYLES["font_size"]["legend"])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _generate_html_content(self, chart_paths: Dict[str, str], summary: Dict[str, Any]) -> str:
        """
        Generate HTML content for the dashboard.
        
        Args:
            chart_paths: Dictionary mapping chart names to file paths
            summary: Summary metrics
            
        Returns:
            HTML content
        """
        overall = summary.get("overall", {})
        by_domain = summary.get("by_domain", {})
        by_difficulty = summary.get("by_difficulty", {})
        retry_metrics = summary.get("retries", {})
        trend_data = summary.get("trends", {})
        
        # Format metrics for display
        case_count = overall.get("case_count", 0)
        avg_standard_factuality = overall.get("avg_standard_factuality", 0)
        avg_accuracy_factuality = overall.get("avg_accuracy_factuality", 0)
        avg_improvement = overall.get("avg_factuality_improvement", 0)
        standard_detection = overall.get("standard_detection_accuracy", 0)
        accuracy_detection = overall.get("accuracy_detection_accuracy", 0)
        avg_latency = overall.get("avg_latency", 0)
        
        # Get trend indicators
        factuality_trend = trend_data.get("factuality_trend", "stable")
        latency_trend = trend_data.get("latency_trend", "stable")
        overall_trend = trend_data.get("trend", "stable")
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ACDEK Grounding Benchmark Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #333; }}
                .card {{ background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
                .metric-card {{ background-color: #f5f5f5; border-radius: 8px; padding: 15px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #F44336; }}
                .neutral {{ color: #2196F3; }}
                .warning {{ color: #FFC107; }}
                .chart-container {{ margin-bottom: 20px; }}
                .chart {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .charts-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f5f5f5; }}
                .footer {{ text-align: center; margin-top: 40px; color: #666; font-size: 14px; }}
                .trend-indicator {{ font-size: 18px; margin-left: 5px; }}
                .trend-up {{ color: #4CAF50; }}
                .trend-down {{ color: #F44336; }}
                .trend-stable {{ color: #FFC107; }}
                .tab-container {{ margin-bottom: 20px; }}
                .tab-buttons {{ display: flex; margin-bottom: 15px; }}
                .tab-button {{ padding: 10px 15px; cursor: pointer; background-color: #f2f2f2; border: none; margin-right: 5px; border-radius: 5px 5px 0 0; }}
                .tab-button.active {{ background-color: #2196F3; color: white; }}
                .tab-content {{ display: none; }}
                .tab-content.active {{ display: block; }}
            </style>
            <script>
                function openTab(evt, tabName) {{
                    // Hide all tab content
                    var tabContents = document.getElementsByClassName("tab-content");
                    for (var i = 0; i < tabContents.length; i++) {{
                        tabContents[i].style.display = "none";
                    }}
                    
                    // Remove active class from all tab buttons
                    var tabButtons = document.getElementsByClassName("tab-button");
                    for (var i = 0; i < tabButtons.length; i++) {{
                        tabButtons[i].className = tabButtons[i].className.replace(" active", "");
                    }}
                    
                    // Show the current tab and add active class to the button
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <h1>ACDEK Grounding Benchmark Results</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="card">
                    <h2>Summary</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Test Cases</div>
                            <div class="metric-value neutral">{case_count}</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Standard Factuality</div>
                            <div class="metric-value neutral">{avg_standard_factuality:.4f}</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Accuracy-Focused Factuality</div>
                            <div class="metric-value neutral">{avg_accuracy_factuality:.4f}
                                <span class="trend-indicator {{'trend-up' if factuality_trend == 'improving' else 'trend-down' if factuality_trend == 'declining' else 'trend-stable'}}">
                                    {{'↑' if factuality_trend == 'improving' else '↓' if factuality_trend == 'declining' else '→'}}
                                </span>
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Factuality Improvement</div>
                            <div class="metric-value {'positive' if avg_improvement > 0 else 'negative'}">
                                {avg_improvement:.4f}
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Detection Accuracy</div>
                            <div class="metric-value {'positive' if accuracy_detection > 0.7 else 'neutral'}">
                                {accuracy_detection:.1%}
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Avg Latency</div>
                            <div class="metric-value {'positive' if avg_latency < 0.2 else 'warning' if avg_latency < 0.5 else 'negative'}">
                                {avg_latency:.3f}s
                                <span class="trend-indicator {{'trend-up' if latency_trend == 'improving' else 'trend-down' if latency_trend == 'declining' else 'trend-stable'}}">
                                    {{'↑' if latency_trend == 'improving' else '↓' if latency_trend == 'declining' else '→'}}
                                </span>
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Overall Trend</div>
                            <div class="metric-value {'positive' if overall_trend == 'improving' else 'negative' if overall_trend == 'declining' else 'warning'}">
                                {overall_trend.title()}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Performance Visualization</h2>
                    
                    <div class="tab-container">
                        <div class="tab-buttons">
                            <button class="tab-button active" onclick="openTab(event, 'factuality-tab')">Factuality</button>
                            <button class="tab-button" onclick="openTab(event, 'domain-tab')">Domain Analysis</button>
                            <button class="tab-button" onclick="openTab(event, 'retry-tab')">Retry & Trend</button>
                        </div>
                        
                        <div id="factuality-tab" class="tab-content active">
                            <div class="charts-grid">
                                <div class="chart-container">
                                    <h3>Factuality Score Comparison</h3>
                                    <img src="{chart_paths.get('factuality', '')}" alt="Factuality Comparison Chart" class="chart">
                                </div>
                                
                                <div class="chart-container">
                                    <h3>Hallucination Detection Accuracy</h3>
                                    <img src="{chart_paths.get('detection', '')}" alt="Detection Accuracy Chart" class="chart">
                                </div>
                                
                                <div class="chart-container">
                                    <h3>Performance by Difficulty</h3>
                                    <img src="{chart_paths.get('difficulty', '')}" alt="Difficulty Analysis Chart" class="chart">
                                </div>
                                
                                <div class="chart-container">
                                    <h3>Performance Trends Over Time</h3>
                                    <img src="{chart_paths.get('trends', '')}" alt="Trend Analysis Chart" class="chart">
                                </div>
                            </div>
                        </div>
                        
                        <div id="domain-tab" class="tab-content">
                            <div class="charts-grid">
                                <div class="chart-container">
                                    <h3>Performance by Domain</h3>
                                    <img src="{chart_paths.get('domain', '')}" alt="Domain Performance Chart" class="chart">
                                </div>
                                
                                <div class="chart-container">
                                    <h3>Domain-Specific Thresholds</h3>
                                    <img src="{chart_paths.get('thresholds', '')}" alt="Domain Threshold Chart" class="chart">
                                </div>
                            </div>
                        </div>
                        
                        <div id="retry-tab" class="tab-content">
                            <div class="charts-grid">
                                <div class="chart-container">
                                    <h3>Retry Metrics by Plugin</h3>
                                    <img src="{chart_paths.get('retries', '')}" alt="Retry Metrics Chart" class="chart">
                                </div>
                                
                                <div class="chart-container">
                                    <h3>Performance Trends Over Time</h3>
                                    <img src="{chart_paths.get('trends', '')}" alt="Trend Analysis Chart" class="chart">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Domain Analysis</h2>
                    <table>
                        <tr>
                            <th>Domain</th>
                            <th>Cases</th>
                            <th>Standard Factuality</th>
                            <th>Accuracy-Focused Factuality</th>
                            <th>Improvement</th>
                            <th>Threshold</th>
                            <th>Status</th>
                        </tr>
        """
        
        # Add rows for each domain
        from benchmarks.performance_metrics import DOMAIN_THRESHOLDS
        
        for domain, metrics in by_domain.items():
            std_factuality = metrics.get("avg_standard_factuality", 0)
            acc_factuality = metrics.get("avg_accuracy_factuality", 0)
            improvement = metrics.get("avg_improvement", 0)
            
            # Get domain threshold
            domain_threshold = DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS.get("general", {}))
            threshold_value = domain_threshold.get("factuality", 0.7)  # Default to 0.7
            
            # Determine status
            if acc_factuality >= threshold_value:
                status = "✅ Meets Threshold"
                status_class = "positive"
            else:
                gap = threshold_value - acc_factuality
                if gap < 0.05:
                    status = f"⚠️ Close (-{gap:.3f})"
                    status_class = "warning"
                else:
                    status = f"❌ Below (-{gap:.3f})"
                    status_class = "negative"
            
            html += f"""
                        <tr>
                            <td>{domain}</td>
                            <td>{metrics.get('case_count', 0)}</td>
                            <td>{std_factuality:.4f}</td>
                            <td>{acc_factuality:.4f}</td>
                            <td class="{'positive' if improvement > 0 else 'negative'}">{improvement:.4f}</td>
                            <td>{threshold_value:.4f}</td>
                            <td class="{status_class}">{status}</td>
                        </tr>
            """
        
        # Continue HTML
        html += """
                    </table>
                </div>
                
                <div class="card">
                    <h2>Difficulty Analysis</h2>
                    <table>
                        <tr>
                            <th>Difficulty</th>
                            <th>Cases</th>
                            <th>Standard Factuality</th>
                            <th>Accuracy-Focused Factuality</th>
                            <th>Improvement</th>
                        </tr>
        """
        
        # Add rows for each difficulty level
        for difficulty in sorted(by_difficulty.keys()):
            metrics = by_difficulty[difficulty]
            std_factuality = metrics.get("avg_standard_factuality", 0)
            acc_factuality = metrics.get("avg_accuracy_factuality", 0)
            improvement = metrics.get("avg_improvement", 0)
            
            html += f"""
                        <tr>
                            <td>{difficulty}</td>
                            <td>{metrics.get('case_count', 0)}</td>
                            <td>{std_factuality:.4f}</td>
                            <td>{acc_factuality:.4f}</td>
                            <td class="{'positive' if improvement > 0 else 'negative'}">{improvement:.4f}</td>
                        </tr>
            """
        
        # Add retry metrics section if available
        if retry_metrics:
            html += """
                    </table>
                </div>
                
                <div class="card">
                    <h2>Retry Metrics</h2>
                    <table>
                        <tr>
                            <th>Plugin</th>
                            <th>Count</th>
                            <th>Penalty</th>
                            <th>Trend</th>
                        </tr>
            """
            
            # Add rows for each plugin
            for plugin, count in retry_metrics.get("counts", {}).items():
                penalty = retry_metrics.get("penalties", {}).get(plugin, 0.0)
                trend = retry_metrics.get("trends", {}).get(plugin, "stable")
                
                # Determine trend indicator
                if trend == "increasing":
                    trend_indicator = "↑"
                    trend_class = "negative"
                elif trend == "decreasing":
                    trend_indicator = "↓"
                    trend_class = "positive"
                else:
                    trend_indicator = "→"
                    trend_class = "neutral"
                
                html += f"""
                            <tr>
                                <td>{plugin}</td>
                                <td>{count}</td>
                                <td>{penalty:.3f}</td>
                                <td class="{trend_class}">{trend.title()} {trend_indicator}</td>
                            </tr>
                """
        
        # Add trend analysis section if available
        if trend_data and "windows" in trend_data and trend_data["windows"]:
            html += """
                    </table>
                </div>
                
                <div class="card">
                    <h2>Trend Analysis</h2>
                    <table>
                        <tr>
                            <th>Time Window</th>
                            <th>Run Count</th>
                            <th>Avg Factuality</th>
                            <th>Avg Latency</th>
                        </tr>
            """
            
            # Add rows for each time window
            for window in trend_data.get("windows", []):
                if "start_time" in window:
                    time_str = datetime.fromtimestamp(window["start_time"]).strftime("%Y-%m-%d %H:%M")
                else:
                    time_str = "Unknown"
                
                run_count = window.get("run_count", 0)
                avg_factuality = window.get("avg_factuality", 0)
                avg_latency = window.get("avg_latency", 0)
                
                html += f"""
                            <tr>
                                <td>{time_str}</td>
                                <td>{run_count}</td>
                                <td>{avg_factuality:.4f}</td>
                                <td>{avg_latency:.3f}s</td>
                            </tr>
                """
        
        # Finish HTML
        html += """
                    </table>
                </div>
                
                <div class="footer">
                    ACDEK Grounding Benchmark Dashboard | Generated by ACDEK Benchmarking Suite
                </div>
            </div>
            
            <script>
                // Set the first tab as active by default
                document.getElementsByClassName("tab-button")[0].click();
            </script>
        </body>
        </html>
        """
        
        return html


class IntegratedDashboard:
    """
    Integrated dashboard that combines hallucination and performance metrics.
    """
    
    def __init__(self, hallucination_results_path: str, performance_results_path: str, output_dir: str):
        """
        Initialize with paths to both hallucination and performance results.
        
        Args:
            hallucination_results_path: Path to hallucination benchmark results
            performance_results_path: Path to performance metrics results
            output_dir: Directory to save integrated dashboard
        """
        self.hallucination_results_path = hallucination_results_path
        self.performance_results_path = performance_results_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.charts_dir = os.path.join(output_dir, "charts")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
        # Load results
        self.hallucination_results = load_benchmark_results(hallucination_results_path)
        self.performance_results = self._load_performance_results(performance_results_path)
    
    def _load_performance_results(self, path: str) -> Dict[str, Any]:
        """Load performance metrics results."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            logger.info(f"Successfully loaded performance metrics from {path}")
            return data
        except Exception as e:
            logger.error(f"Error loading performance metrics from {path}: {str(e)}")
            return {}
    
    def generate_dashboard(self) -> str:
        """
        Generate integrated dashboard combining hallucination and performance metrics.
        
        Returns:
            Path to the generated dashboard
        """
        # First generate individual dashboards
        hallucination_type = detect_result_type(self.hallucination_results)
        
        if hallucination_type == "grounding":
            hallucination_dashboard = GroundingBenchmarkDashboard(self.hallucination_results, self.output_dir)
        else:
            hallucination_dashboard = GeneralBenchmarkDashboard(self.hallucination_results, self.output_dir)
        
        # Generate individual dashboards
        hallucination_path = hallucination_dashboard.generate_dashboard()
        
        # Generate integrated metrics
        integrated_metrics = self._integrate_metrics()
        
        # Generate integrated charts
        chart_paths = self._generate_integrated_charts()
        
        # Generate HTML
        output_path = os.path.join(self.output_dir, f"integrated_dashboard_{self.timestamp}.html")
        html_content = self._generate_html_content(chart_paths, integrated_metrics)
        
        with open(output_path, "w") as f:
            f.write(html_content)
        
        logger.info(f"Generated integrated dashboard at {output_path}")
        
        return output_path
    
    def _integrate_metrics(self) -> Dict[str, Any]:
        """
        Integrate hallucination and performance metrics into a single dataset.
        
        Returns:
            Integrated metrics dictionary
        """
        integrated = {
            "meta": {
                "generated_at": datetime.now().isoformat(),
                "hallucination_source": self.hallucination_results_path,
                "performance_source": self.performance_results_path
            },
            "overall": {},
            "by_domain": {}
        }
        
        # Extract hallucination metrics
        hall_overall = self.hallucination_results.get("summary_metrics", {}).get("overall", {})
        hall_by_domain = self.hallucination_results.get("summary_metrics", {}).get("by_domain", {})
        
        # Extract performance metrics
        perf_metrics = self.performance_results.get("metrics", {})
        domain_performance = perf_metrics.get("domain_analysis", {})
        
        # Integrate overall metrics
        integrated["overall"] = {
            "case_count": hall_overall.get("case_count", 0),
            "factuality": {
                "standard": hall_overall.get("avg_standard_factuality", 0),
                "accuracy": hall_overall.get("avg_accuracy_factuality", 0),
                "improvement": hall_overall.get("avg_factuality_improvement", 0)
            },
            "detection": {
                "accuracy": hall_overall.get("accuracy_detection_accuracy", 0)
            },
            "performance": {
                "latency": perf_metrics.get("overall_latency", 0),
                "retry_penalty": perf_metrics.get("overall_retry_penalty", 0)
            },
            "trends": {
                "factuality": perf_metrics.get("factuality_trend", "stable"),
                "latency": perf_metrics.get("latency_trend", "stable"),
                "overall": perf_metrics.get("overall_trend", "stable")
            }
        }
        
        # Integrate domain metrics
        domains = set(list(hall_by_domain.keys()) + list(domain_performance.keys()))
        
        for domain in domains:
            hall_domain = hall_by_domain.get(domain, {})
            perf_domain = domain_performance.get(domain, {})
            
            integrated["by_domain"][domain] = {
                "case_count": hall_domain.get("case_count", 0),
                "factuality": {
                    "standard": hall_domain.get("avg_standard_factuality", 0),
                    "accuracy": hall_domain.get("avg_accuracy_factuality", 0),
                    "improvement": hall_domain.get("avg_improvement", 0)
                },
                "performance": {
                    "latency": perf_domain.get("avg_latency_overhead", 0),
                    "retry_penalty": perf_domain.get("avg_penalty", 0),
                }
            }
            
            # Add domain-specific thresholds
            from benchmarks.performance_metrics import DOMAIN_THRESHOLDS
            domain_threshold = DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS.get("general", {}))
            
            integrated["by_domain"][domain]["thresholds"] = {
                "factuality": domain_threshold.get("factuality", 0.7),
                "latency": domain_threshold.get("latency", 200.0),
                "retry_penalty": domain_threshold.get("retry_penalty", 0.8)
            }
        
        return integrated
    
    def _generate_integrated_charts(self) -> Dict[str, str]:
        """
        Generate charts that integrate hallucination and performance metrics.
        
        Returns:
            Dictionary mapping chart names to file paths
        """
        chart_paths = {}
        
        # Generate domain-specific integrated chart
        integrated_domain_path = os.path.join(self.charts_dir, f"integrated_domain_{self.timestamp}.png")
        self._create_integrated_domain_chart(integrated_domain_path)
        chart_paths["integrated_domain"] = os.path.relpath(integrated_domain_path, self.output_dir)
        
        # Generate threshold comparison chart
        threshold_chart_path = os.path.join(self.charts_dir, f"threshold_comparison_{self.timestamp}.png")
        self._create_threshold_comparison_chart(threshold_chart_path)
        chart_paths["threshold_comparison"] = os.path.relpath(threshold_chart_path, self.output_dir)
        
        # Generate performance impact chart
        performance_impact_path = os.path.join(self.charts_dir, f"performance_impact_{self.timestamp}.png")
        self._create_performance_impact_chart(performance_impact_path)
        chart_paths["performance_impact"] = os.path.relpath(performance_impact_path, self.output_dir)
        
        return chart_paths
    
    def _create_integrated_domain_chart(self, output_path: str) -> None:
        """
        Create chart showing integrated domain metrics.
        
        Args:
            output_path: Path to save the chart
        """
        from benchmarks.performance_metrics import DOMAIN_THRESHOLDS
        
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        integrated = self._integrate_metrics()
        domains = list(integrated["by_domain"].keys())
        
        if not domains:
            plt.text(0.5, 0.5, "No domain data available", 
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=100)
            plt.close()
            return
        
        # Extract metrics for each domain
        factuality_scores = []
        latency_values = []
        threshold_values = []
        
        for domain in domains:
            domain_data = integrated["by_domain"][domain]
            factuality_scores.append(domain_data["factuality"]["accuracy"])
            latency_values.append(domain_data["performance"]["latency"])
            threshold_values.append(domain_data["thresholds"]["factuality"])
        
        # Normalize latency for plotting on same scale
        max_latency = max(latency_values) if latency_values else 1.0
        normalized_latency = [value / max_latency for value in latency_values]
        
        # Setup figure with dual y-axes
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Define x positions
        x = np.arange(len(domains))
        width = 0.25
        
        # Primary axis for factuality scores and thresholds
        ax1.set_xlabel('Domain', fontsize=12)
        ax1.set_ylabel('Factuality Score', fontsize=12, color=CHART_STYLES["colors"]["primary"])
        ax1.bar(x - width, factuality_scores, width, label='Factuality Score', 
               color=CHART_STYLES["colors"]["primary"])
        ax1.bar(x, threshold_values, width, label='Factuality Threshold', 
               color=CHART_STYLES["colors"]["accent"], alpha=0.7)
        ax1.tick_params(axis='y', labelcolor=CHART_STYLES["colors"]["primary"])
        ax1.set_ylim(0, 1.0)
        
        # Secondary axis for latency
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalized Latency', fontsize=12, color=CHART_STYLES["colors"]["warning"])
        ax2.bar(x + width, normalized_latency, width, label='Normalized Latency', 
               color=CHART_STYLES["colors"]["warning"])
        ax2.tick_params(axis='y', labelcolor=CHART_STYLES["colors"]["warning"])
        ax2.set_ylim(0, 1.0)
        
        # Add label for actual latency values
        for i, value in enumerate(latency_values):
            ax2.annotate(f"{value:.2f}ms", 
                        xy=(x[i] + width, normalized_latency[i]), 
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9)
        
        # Set title and format
        plt.title('Integrated Domain Metrics: Factuality vs. Performance', fontsize=16)
        plt.xticks(x, domains, fontsize=10)
        
        # Create combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
        
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _create_threshold_comparison_chart(self, output_path: str) -> None:
        """
        Create chart comparing actual metrics to thresholds across domains.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        integrated = self._integrate_metrics()
        domains = list(integrated["by_domain"].keys())
        
        if not domains:
            plt.text(0.5, 0.5, "No domain data available", 
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=100)
            plt.close()
            return
        
        # Calculate threshold achievement percentages
        factuality_pcts = []
        latency_pcts = []
        retry_pcts = []
        
        for domain in domains:
            domain_data = integrated["by_domain"][domain]
            
            # Factuality (higher is better)
            fact_score = domain_data["factuality"]["accuracy"]
            fact_threshold = domain_data["thresholds"]["factuality"]
            fact_pct = min(100, (fact_score / fact_threshold) * 100) if fact_threshold > 0 else 0
            factuality_pcts.append(fact_pct)
            
            # Latency (lower is better, so invert)
            latency = domain_data["performance"]["latency"]
            latency_threshold = domain_data["thresholds"]["latency"]
            # Higher percentage is better (means using less of allowed latency)
            latency_pct = min(100, (1 - (latency / latency_threshold)) * 100) if latency_threshold > 0 else 0
            latency_pcts.append(latency_pct)
            
            # Retry penalty (lower is better, so invert)
            retry = domain_data["performance"]["retry_penalty"]
            retry_threshold = domain_data["thresholds"]["retry_penalty"]
            # Higher percentage is better (means using less of allowed penalty)
            retry_pct = min(100, (1 - (retry / retry_threshold)) * 100) if retry_threshold > 0 else 0
            retry_pcts.append(retry_pct)
        
        # Create grouped bar chart
        x = np.arange(len(domains))
        width = 0.25
        
        plt.bar(x - width, factuality_pcts, width, label='Factuality Score', 
               color=CHART_STYLES["colors"]["primary"])
        plt.bar(x, latency_pcts, width, label='Latency Efficiency', 
               color=CHART_STYLES["colors"]["warning"])
        plt.bar(x + width, retry_pcts, width, label='Retry Efficiency', 
               color=CHART_STYLES["colors"]["accent"])
        
        # Add goal line at 100%
        plt.axhline(y=100, color='gray', linestyle='--', alpha=0.7)
        
        # Format chart
        plt.xlabel('Domain', fontsize=12)
        plt.ylabel('Threshold Achievement (%)', fontsize=12)
        plt.title('Threshold Achievement by Domain', fontsize=16)
        plt.xticks(x, domains, fontsize=10)
        plt.ylim(0, 120)  # Leave room above 100% line
        plt.legend(fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, v in enumerate(factuality_pcts):
            plt.text(i - width, v + 3, f"{v:.0f}%", ha='center', fontsize=9)
        
        for i, v in enumerate(latency_pcts):
            plt.text(i, v + 3, f"{v:.0f}%", ha='center', fontsize=9)
            
        for i, v in enumerate(retry_pcts):
            plt.text(i + width, v + 3, f"{v:.0f}%", ha='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _create_performance_impact_chart(self, output_path: str) -> None:
        """
        Create chart showing how factuality improvements impact performance.
        
        Args:
            output_path: Path to save the chart
        """
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        integrated = self._integrate_metrics()
        domains = list(integrated["by_domain"].keys())
        
        if not domains:
            plt.text(0.5, 0.5, "No domain data available", 
                    ha='center', va='center', fontsize=14)
            plt.savefig(output_path, dpi=100)
            plt.close()
            return
        
        # Extract improvement and performance impact
        factuality_improvements = []
        latency_impacts = []
        labels = []
        
        for domain in domains:
            domain_data = integrated["by_domain"][domain]
            improvement = domain_data["factuality"]["improvement"]
            latency = domain_data["performance"]["latency"]
            
            factuality_improvements.append(improvement)
            latency_impacts.append(latency)
            labels.append(domain)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        
        # Define colors based on improvement/latency balance
        colors = []
        for i, (imp, lat) in enumerate(zip(factuality_improvements, latency_impacts)):
            # Good balance: high improvement, low latency
            if imp > 0.1 and lat < 150:
                colors.append(CHART_STYLES["colors"]["secondary"])  # Green
            # High improvement but high latency
            elif imp > 0.1:
                colors.append(CHART_STYLES["colors"]["warning"])  # Yellow
            # Low improvement and high latency
            else:
                colors.append(CHART_STYLES["colors"]["error"])  # Red
        
        # Create scatter plot
        scatter = plt.scatter(factuality_improvements, latency_impacts, 
                            s=100, alpha=0.7, c=colors)
        
        # Add labels to points
        for i, label in enumerate(labels):
            plt.annotate(label, 
                        (factuality_improvements[i], latency_impacts[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=9)
        
        # Add quadrant lines
        avg_improvement = sum(factuality_improvements) / len(factuality_improvements) if factuality_improvements else 0
        avg_latency = sum(latency_impacts) / len(latency_impacts) if latency_impacts else 0
        
        plt.axvline(x=avg_improvement, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=avg_latency, color='gray', linestyle='--', alpha=0.5)
        
        # Label quadrants
        plt.text(max(factuality_improvements) * 0.8, min(latency_impacts) * 1.1, 
                "Ideal", fontsize=9, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                                                           fc="lightgreen", alpha=0.5))
        
        plt.text(min(factuality_improvements) * 0.8, min(latency_impacts) * 1.1, 
                "Low Impact", fontsize=9, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                                                               fc="lightyellow", alpha=0.5))
        
        plt.text(max(factuality_improvements) * 0.8, max(latency_impacts) * 0.9, 
                "High Resource", fontsize=9, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                                                                  fc="lightyellow", alpha=0.5))
        
        plt.text(min(factuality_improvements) * 0.8, max(latency_impacts) * 0.9, 
                "Poor", fontsize=9, ha='center', bbox=dict(boxstyle="round,pad=0.3", 
                                                          fc="lightcoral", alpha=0.5))
        
        # Format chart
        plt.xlabel('Factuality Improvement', fontsize=12)
        plt.ylabel('Latency Impact (ms)', fontsize=12)
        plt.title('Performance Impact of Factuality Improvements by Domain', fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def _generate_html_content(self, chart_paths: Dict[str, str], metrics: Dict[str, Any]) -> str:
        """
        Generate HTML content for the integrated dashboard.
        
        Args:
            chart_paths: Dictionary mapping chart names to file paths
            metrics: Integrated metrics dictionary
            
        Returns:
            HTML content
        """
        # Extract metrics
        overall = metrics.get("overall", {})
        by_domain = metrics.get("by_domain", {})
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ACDEK Integrated Benchmark Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1, h2, h3 {{ color: #333; }}
                .card {{ background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; margin-bottom: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
                .metric-card {{ background-color: #f5f5f5; border-radius: 8px; padding: 15px; text-align: center; }}
                .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #F44336; }}
                .neutral {{ color: #2196F3; }}
                .warning {{ color: #FFC107; }}
                .chart-container {{ margin-bottom: 20px; }}
                .chart {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .charts-row {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
                .chart-card {{ flex: 1; min-width: 45%; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f5f5f5; }}
                .footer {{ text-align: center; margin-top: 40px; color: #666; font-size: 14px; }}
                .trend-indicator {{ font-size: 18px; margin-left: 5px; }}
                .trend-up {{ color: #4CAF50; }}
                .trend-down {{ color: #F44336; }}
                .trend-stable {{ color: #FFC107; }}
                .tab-container {{ margin-bottom: 20px; }}
                .tab-buttons {{ display: flex; margin-bottom: 15px; }}
                .tab-button {{ padding: 10px 15px; cursor: pointer; background-color: #f2f2f2; border: none; margin-right: 5px; border-radius: 5px 5px 0 0; }}
                .tab-button.active {{ background-color: #2196F3; color: white; }}
                .tab-content {{ display: none; }}
                .tab-content.active {{ display: block; }}
            </style>
            <script>
                function openTab(evt, tabName) {{
                    var tabContents = document.getElementsByClassName("tab-content");
                    for (var i = 0; i < tabContents.length; i++) {{
                        tabContents[i].style.display = "none";
                    }}
                    
                    var tabButtons = document.getElementsByClassName("tab-button");
                    for (var i = 0; i < tabButtons.length; i++) {{
                        tabButtons[i].className = tabButtons[i].className.replace(" active", "");
                    }}
                    
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }}
            </script>
        </head>
        <body>
            <div class="container">
                <h1>ACDEK Integrated Benchmark Dashboard</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="card">
                    <h2>Summary</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Test Cases</div>
                            <div class="metric-value neutral">{overall.get("case_count", 0)}</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Factuality Score</div>
                            <div class="metric-value neutral">{overall.get("factuality", {}).get("accuracy", 0):.4f}</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Factuality Improvement</div>
                            <div class="metric-value {'positive' if overall.get("factuality", {}).get("improvement", 0) > 0 else 'negative'}">
                                {overall.get("factuality", {}).get("improvement", 0):.4f}
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Detection Accuracy</div>
                            <div class="metric-value {'positive' if overall.get("detection", {}).get("accuracy", 0) > 0.7 else 'neutral'}">
                                {overall.get("detection", {}).get("accuracy", 0):.1%}
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Latency</div>
                            <div class="metric-value {'positive' if overall.get("performance", {}).get("latency", 0) < 150 else 'warning' if overall.get("performance", {}).get("latency", 0) < 250 else 'negative'}">
                                {overall.get("performance", {}).get("latency", 0):.2f}ms
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Overall Trend</div>
                            <div class="metric-value {'positive' if overall.get("trends", {}).get("overall", "stable") == 'improving' else 'negative' if overall.get("trends", {}).get("overall", "stable") == 'declining' else 'warning'}">
                                {overall.get("trends", {}).get("overall", "stable").title()}
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Integrated Performance Visualization</h2>
                    
                    <div class="tab-container">
                        <div class="tab-buttons">
                            <button class="tab-button active" onclick="openTab(event, 'integrated-tab')">Integrated Metrics</button>
                            <button class="tab-button" onclick="openTab(event, 'threshold-tab')">Threshold Analysis</button>
                            <button class="tab-button" onclick="openTab(event, 'impact-tab')">Performance Impact</button>
                        </div>
                        
                        <div id="integrated-tab" class="tab-content active">
                            <div class="chart-container">
                                <h3>Integrated Domain Metrics</h3>
                                <img src="{chart_paths.get('integrated_domain', '')}" alt="Integrated Domain Metrics" class="chart">
                                <p class="chart-description">
                                    This chart shows the relationship between factuality scores and performance metrics across different domains.
                                    The bars represent factuality scores, domain-specific thresholds, and normalized latency impact.
                                </p>
                            </div>
                        </div>
                        
                        <div id="threshold-tab" class="tab-content">
                            <div class="chart-container">
                                <h3>Threshold Achievement by Domain</h3>
                                <img src="{chart_paths.get('threshold_comparison', '')}" alt="Threshold Comparison Chart" class="chart">
                                <p class="chart-description">
                                    This chart shows how well each domain is performing against its domain-specific thresholds.
                                    100% means the domain fully meets its threshold requirements. Higher percentages are better.
                                </p>
                            </div>
                        </div>
                        
                        <div id="impact-tab" class="tab-content">
                            <div class="chart-container">
                                <h3>Performance Impact Analysis</h3>
                                <img src="{chart_paths.get('performance_impact', '')}" alt="Performance Impact Chart" class="chart">
                                <p class="chart-description">
                                    This chart plots factuality improvement against latency impact for each domain.
                                    The ideal quadrant has high factuality improvement with low latency impact.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Domain Analysis</h2>
                    <table>
                        <tr>
                            <th>Domain</th>
                            <th>Cases</th>
                            <th>Factuality Score</th>
                            <th>Factuality Threshold</th>
                            <th>Improvement</th>
                            <th>Latency (ms)</th>
                            <th>Status</th>
                        </tr>
        """
        
        # Add rows for each domain
        for domain, domain_data in by_domain.items():
            factuality = domain_data.get("factuality", {}).get("accuracy", 0)
            threshold = domain_data.get("thresholds", {}).get("factuality", 0.7)
            improvement = domain_data.get("factuality", {}).get("improvement", 0)
            latency = domain_data.get("performance", {}).get("latency", 0)
            
            # Determine status
            if factuality >= threshold and latency < domain_data.get("thresholds", {}).get("latency", 200):
                status = "✅ Optimal"
                status_class = "positive"
            elif factuality >= threshold:
                status = "⚠️ High Latency"
                status_class = "warning"
            elif latency < domain_data.get("thresholds", {}).get("latency", 200):
                status = "⚠️ Low Factuality"
                status_class = "warning"
            else:
                status = "❌ Below Thresholds"
                status_class = "negative"
            
            html += f"""
                        <tr>
                            <td>{domain}</td>
                            <td>{domain_data.get("case_count", 0)}</td>
                            <td>{factuality:.4f}</td>
                            <td>{threshold:.4f}</td>
                            <td class="{'positive' if improvement > 0 else 'negative'}">{improvement:.4f}</td>
                            <td>{latency:.2f}</td>
                            <td class="{status_class}">{status}</td>
                        </tr>
            """
        
        # Finish HTML
        html += """
                    </table>
                </div>
                
                <div class="card">
                    <h2>Integration Notes</h2>
                    <p>This dashboard integrates metrics from both hallucination benchmarks and performance metrics to provide a comprehensive view of system behavior across different domains.</p>
                    <ul>
                        <li><strong>Factuality Score:</strong> Measures how well the system adheres to factual information</li>
                        <li><strong>Factuality Threshold:</strong> Domain-specific minimum acceptable factuality score</li>
                        <li><strong>Latency:</strong> Overhead introduced by factuality checking and other performance aspects</li>
                        <li><strong>Threshold Achievement:</strong> How well the system meets domain-specific requirements</li>
                        <li><strong>Performance Impact:</strong> Trade-off between factuality improvement and computational cost</li>
                    </ul>
                </div>
                
                <div class="footer">
                    ACDEK Integrated Benchmark Dashboard | Generated by ACDEK Benchmarking Suite
                </div>
            </div>
            
            <script>
                // Set the first tab as active by default
                document.getElementsByClassName("tab-button")[0].click();
            </script>
        </body>
        </html>
        """
        
        return html


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="ACDEK Benchmark Dashboard Generator")
    parser.add_argument("--input", required=True, help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save dashboard files")
    parser.add_argument("--type", choices=["auto", "general", "grounding", "integrated"], default="auto", 
                      help="Type of benchmark results (auto detects if not specified)")
    parser.add_argument("--performance-input", help="Path to performance metrics JSON file (for integrated dashboard)")
    args = parser.parse_args()
    
    # Check for integrated dashboard request
    if args.type == "integrated":
        if not args.performance_input:
            logger.error("--performance-input is required for integrated dashboard")
            return 1
        
        integrated_dashboard = IntegratedDashboard(args.input, args.performance_input, args.output_dir)
        dashboard_path = integrated_dashboard.generate_dashboard()
        print(f"Integrated dashboard generated: {dashboard_path}")
        return 0
    
    # Standard dashboard generation
    results = load_benchmark_results(args.input)
    if not results:
        logger.error("Failed to load benchmark results. Exiting.")
        return 1
    
    # Detect type if auto
    if args.type == "auto":
        benchmark_type = detect_result_type(results)
        logger.info(f"Detected benchmark type: {benchmark_type}")
    else:
        benchmark_type = args.type
    
    # Create appropriate dashboard generator
    if benchmark_type == "grounding":
        dashboard_generator = GroundingBenchmarkDashboard(results, args.output_dir)
    else:
        dashboard_generator = GeneralBenchmarkDashboard(results, args.output_dir)
    
    # Generate dashboard
    dashboard_path = dashboard_generator.generate_dashboard()
    
    print(f"Dashboard generated: {dashboard_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())