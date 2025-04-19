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
        
        return chart_paths
    
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
        
        # Format metrics for display
        case_count = overall.get("case_count", 0)
        avg_standard_factuality = overall.get("avg_standard_factuality", 0)
        avg_accuracy_factuality = overall.get("avg_accuracy_factuality", 0)
        avg_improvement = overall.get("avg_factuality_improvement", 0)
        standard_detection = overall.get("standard_detection_accuracy", 0)
        accuracy_detection = overall.get("accuracy_detection_accuracy", 0)
        
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
                            <div class="metric-value neutral">{avg_accuracy_factuality:.4f}</div>
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
                    </div>
                </div>
                
                <div class="card">
                    <h2>Performance Visualization</h2>
                    
                    <div class="chart-container">
                        <h3>Factuality Score Comparison</h3>
                        <img src="{chart_paths.get('factuality', '')}" alt="Factuality Comparison Chart" class="chart">
                    </div>
                    
                    <div class="chart-container">
                        <h3>Performance by Domain</h3>
                        <img src="{chart_paths.get('domain', '')}" alt="Domain Performance Chart" class="chart">
                    </div>
                    
                    <div class="chart-container">
                        <h3>Hallucination Detection Accuracy</h3>
                        <img src="{chart_paths.get('detection', '')}" alt="Detection Accuracy Chart" class="chart">
                    </div>
                    
                    <div class="chart-container">
                        <h3>Performance by Difficulty</h3>
                        <img src="{chart_paths.get('difficulty', '')}" alt="Difficulty Analysis Chart" class="chart">
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
                        </tr>
        """
        
        # Add rows for each domain
        for domain, metrics in by_domain.items():
            std_factuality = metrics.get("avg_standard_factuality", 0)
            acc_factuality = metrics.get("avg_accuracy_factuality", 0)
            improvement = metrics.get("avg_improvement", 0)
            
            html += f"""
                        <tr>
                            <td>{domain}</td>
                            <td>{metrics.get('case_count', 0)}</td>
                            <td>{std_factuality:.4f}</td>
                            <td>{acc_factuality:.4f}</td>
                            <td class="{'positive' if improvement > 0 else 'negative'}">{improvement:.4f}</td>
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
        
        # Finish HTML
        html += """
                    </table>
                </div>
                
                <div class="footer">
                    ACDEK Grounding Benchmark Dashboard | Generated by ACDEK Benchmarking Suite
                </div>
            </div>
        </body>
        </html>
        """
        
        return html


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="ACDEK Benchmark Dashboard Generator")
    parser.add_argument("--input", required=True, help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to save dashboard files")
    parser.add_argument("--type", choices=["auto", "general", "grounding"], default="auto", 
                      help="Type of benchmark results (auto detects if not specified)")
    args = parser.parse_args()
    
    # Load benchmark results
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