"""
ACDEK v3 Hallucination Metrics Dashboard

Interactive dashboard for visualizing hallucination metrics collected by the
hallucination metrics collection system. Provides comprehensive visualization
of factuality scores, drift detection, confidence correlation, and aggregate
benchmark results.
"""

import os
import json
import logging
import glob
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import base64
from io import BytesIO

# Import hallucination metrics components for consistency
from enrichment.hallucination_metrics import (
    FactualityGrade,
    DriftSeverity,
    ConfidenceCorrelation,
    HallucinationCategory,
    DEFAULT_METRICS_COLLECTOR
)

# Import comparative analyzer for integrated analysis
try:
    from benchmarks.comparative_analyzer import ComparativeAnalyzer, OutputFormat
    COMPARATIVE_ANALYZER_AVAILABLE = True
except ImportError:
    COMPARATIVE_ANALYZER_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_METRICS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "enrichment/hallucination_results"
)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "results"
)

# Ensure output directory exists
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)


class HallucinationDashboard:
    """
    Dashboard for visualizing hallucination metrics results.
    
    Generates interactive HTML dashboards with comprehensive visualization
    of factuality scores, semantic drift, confidence correlation, and
    detailed breakdowns of hallucination types.
    """
    
    def __init__(
        self,
        metrics_dir: str = DEFAULT_METRICS_DIR,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        title: str = "ACDEK v3 Hallucination Metrics Dashboard"
    ):
        """
        Initialize the hallucination dashboard.
        
        Args:
            metrics_dir: Directory containing hallucination metrics JSON files
            output_dir: Directory to save dashboard output
            title: Dashboard title
        """
        self.metrics_dir = metrics_dir
        self.output_dir = output_dir
        self.title = title
        
        # Ensure directories exist
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_metrics_files(self, pattern: str = "*.json") -> List[Dict[str, Any]]:
        """
        Load hallucination metrics from JSON files.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            List of loaded metrics data
        """
        metrics_files = glob.glob(os.path.join(self.metrics_dir, pattern))
        metrics_data = []
        
        for file_path in metrics_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    metrics_data.append(data)
            except Exception as e:
                logger.error(f"Error loading metrics file {file_path}: {str(e)}")
                continue
        
        logger.info(f"Loaded {len(metrics_data)} metrics files from {self.metrics_dir}")
        return metrics_data
    
    def generate_factuality_chart(self, metrics_data: List[Dict[str, Any]]) -> str:
        """
        Generate factuality distribution chart.
        
        Args:
            metrics_data: List of metrics data
            
        Returns:
            Base64-encoded PNG image
        """
        # Count factuality grades
        grade_counts = {
            "high": 0,
            "moderate": 0,
            "low": 0,
            "unfactual": 0
        }
        
        for data in metrics_data:
            grade = data.get("factuality_grade", "unfactual")
            if grade in grade_counts:
                grade_counts[grade] += 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Define colors for each grade
        colors = {
            "high": "#4CAF50",  # Green
            "moderate": "#2196F3",  # Blue
            "low": "#FFC107",  # Amber
            "unfactual": "#F44336"  # Red
        }
        
        # Create bar chart
        labels = list(grade_counts.keys())
        values = list(grade_counts.values())
        bar_colors = [colors[grade] for grade in labels]
        
        ax.bar(labels, values, color=bar_colors)
        
        # Add labels and title
        ax.set_title("Factuality Grade Distribution")
        ax.set_xlabel("Factuality Grade")
        ax.set_ylabel("Count")
        
        # Add count labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.5, str(v), ha='center')
        
        # Encode as base64
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def generate_drift_chart(self, metrics_data: List[Dict[str, Any]]) -> str:
        """
        Generate semantic drift severity chart.
        
        Args:
            metrics_data: List of metrics data
            
        Returns:
            Base64-encoded PNG image
        """
        # Count drift severities
        severity_counts = {
            "none": 0,
            "minor": 0,
            "moderate": 0,
            "severe": 0
        }
        
        for data in metrics_data:
            severity = data.get("semantic_drift", {}).get("severity", "none")
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Define colors for each severity
        colors = {
            "none": "#4CAF50",  # Green
            "minor": "#8BC34A",  # Light Green
            "moderate": "#FFC107",  # Amber
            "severe": "#F44336"  # Red
        }
        
        # Create bar chart
        labels = list(severity_counts.keys())
        values = list(severity_counts.values())
        bar_colors = [colors[severity] for severity in labels]
        
        ax.bar(labels, values, color=bar_colors)
        
        # Add labels and title
        ax.set_title("Semantic Drift Severity Distribution")
        ax.set_xlabel("Drift Severity")
        ax.set_ylabel("Count")
        
        # Add count labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v + 0.5, str(v), ha='center')
        
        # Encode as base64
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def generate_confidence_chart(self, metrics_data: List[Dict[str, Any]]) -> str:
        """
        Generate confidence correlation chart.
        
        Args:
            metrics_data: List of metrics data
            
        Returns:
            Base64-encoded PNG image
        """
        # Extract confidence data
        confidence_data = []
        for data in metrics_data:
            confidence_analysis = data.get("confidence_analysis", {})
            if confidence_analysis:
                confidence_data.append({
                    "reported": confidence_analysis.get("reported_confidence", 0),
                    "measured": confidence_analysis.get("measured_factuality", 0),
                    "correlation": confidence_analysis.get("correlation_score", 0),
                    "type": confidence_analysis.get("correlation_type", "unknown")
                })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Define colors for each correlation type
        colors = {
            "accurate": "#4CAF50",  # Green
            "overconfident": "#F44336",  # Red
            "underconfident": "#2196F3",  # Blue
            "confidently_wrong": "#9C27B0",  # Purple
            "confidently_right": "#009688",  # Teal
            "unknown": "#9E9E9E"  # Grey
        }
        
        # Plot correlation scatter
        for point in confidence_data:
            ax.scatter(
                point["reported"],
                point["measured"],
                color=colors.get(point["type"], colors["unknown"]),
                alpha=0.7,
                s=50
            )
        
        # Add diagonal line for perfect correlation
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Add labels and title
        ax.set_title("Confidence vs. Factuality Correlation")
        ax.set_xlabel("Reported Confidence")
        ax.set_ylabel("Measured Factuality")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, label=label, markersize=8)
            for label, color in colors.items()
            if any(point["type"] == label for point in confidence_data)
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        # Encode as base64
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def generate_issues_chart(self, metrics_data: List[Dict[str, Any]]) -> str:
        """
        Generate hallucination issues category chart.
        
        Args:
            metrics_data: List of metrics data
            
        Returns:
            Base64-encoded PNG image
        """
        # Count issue categories
        category_counts = {}
        
        for data in metrics_data:
            issues = data.get("detected_issues", [])
            for issue in issues:
                category = issue.get("category", "unknown")
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar chart
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        
        # Sort by count
        sorted_indices = np.argsort(counts)
        categories = [categories[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # Define colors
        cmap = plt.cm.get_cmap('viridis', len(categories))
        colors = [cmap(i) for i in range(len(categories))]
        
        # Create horizontal bar chart
        ax.barh(categories, counts, color=colors)
        
        # Add labels and title
        ax.set_title("Hallucination Issue Categories")
        ax.set_xlabel("Count")
        ax.set_ylabel("Category")
        
        # Add count labels
        for i, v in enumerate(counts):
            ax.text(v + 0.5, i, str(v), va='center')
        
        # Encode as base64
        buf = BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format='png')
        plt.close(fig)
        
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    def generate_summary_stats(self, metrics_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics from metrics data.
        
        Args:
            metrics_data: List of metrics data
            
        Returns:
            Dictionary with summary statistics
        """
        if not metrics_data:
            return {
                "total_count": 0,
                "avg_factuality_score": 0,
                "avg_drift_score": 0,
                "avg_confidence_correlation": 0,
                "hallucination_reduction_rate": 0
            }
        
        # Calculate basic stats
        total_count = len(metrics_data)
        
        # Factuality stats
        factuality_scores = [data.get("factuality_score", 0) for data in metrics_data]
        avg_factuality_score = sum(factuality_scores) / total_count if factuality_scores else 0
        
        high_factuality_count = sum(1 for score in factuality_scores if score >= 0.9)
        moderate_factuality_count = sum(1 for score in factuality_scores if 0.7 <= score < 0.9)
        low_factuality_count = sum(1 for score in factuality_scores if 0.5 <= score < 0.7)
        unfactual_count = sum(1 for score in factuality_scores if score < 0.5)
        
        # Drift stats
        drift_scores = [data.get("semantic_drift", {}).get("drift_score", 0) for data in metrics_data]
        avg_drift_score = sum(drift_scores) / total_count if drift_scores else 0
        
        # Confidence correlation stats
        correlation_scores = [
            data.get("confidence_analysis", {}).get("correlation_score", 0) 
            for data in metrics_data
        ]
        avg_correlation_score = sum(correlation_scores) / total_count if correlation_scores else 0
        
        # Calculate hallucination reduction rate
        # This requires comparison data which might not be available
        # For now, we'll estimate based on percentage of high factuality responses
        hallucination_reduction_rate = high_factuality_count / total_count if total_count else 0
        
        return {
            "total_count": total_count,
            "avg_factuality_score": avg_factuality_score,
            "high_factuality_count": high_factuality_count,
            "moderate_factuality_count": moderate_factuality_count,
            "low_factuality_count": low_factuality_count,
            "unfactual_count": unfactual_count,
            "avg_drift_score": avg_drift_score,
            "avg_confidence_correlation": avg_correlation_score,
            "hallucination_reduction_rate": hallucination_reduction_rate
        }
    
    def generate_dashboard_html(self, metrics_data: List[Dict[str, Any]]) -> str:
        """
        Generate complete HTML dashboard.
        
        Args:
            metrics_data: List of metrics data
            
        Returns:
            HTML content as string
        """
        # Generate charts
        factuality_chart = self.generate_factuality_chart(metrics_data)
        drift_chart = self.generate_drift_chart(metrics_data)
        confidence_chart = self.generate_confidence_chart(metrics_data)
        issues_chart = self.generate_issues_chart(metrics_data)
        
        # Generate summary stats
        stats = self.generate_summary_stats(metrics_data)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f8f9fa;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #343a40;
                    color: white;
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                .header h1 {{
                    margin: 0;
                }}
                .summary-cards {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: 0 -10px 20px;
                }}
                .card {{
                    flex: 1 0 200px;
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin: 10px;
                    padding: 15px;
                }}
                .card h3 {{
                    margin-top: 0;
                    color: #495057;
                }}
                .card .value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #343a40;
                }}
                .card .subtitle {{
                    font-size: 14px;
                    color: #6c757d;
                }}
                .chart-container {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    padding: 20px;
                }}
                .chart-container h2 {{
                    margin-top: 0;
                    color: #495057;
                }}
                .chart {{
                    width: 100%;
                    max-width: 100%;
                }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                .metrics-table th, .metrics-table td {{
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #dee2e6;
                }}
                .metrics-table th {{
                    background-color: #e9ecef;
                    font-weight: bold;
                }}
                .metrics-table tr:hover {{
                    background-color: #f8f9fa;
                }}
                .footer {{
                    text-align: center;
                    padding: 20px;
                    color: #6c757d;
                    font-size: 14px;
                }}
                .high-value {{ color: #4CAF50; }}
                .moderate-value {{ color: #2196F3; }}
                .low-value {{ color: #FFC107; }}
                .poor-value {{ color: #F44336; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>{self.title}</h1>
                    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
            </div>
            
            <div class="container">
                <div class="summary-cards">
                    <div class="card">
                        <h3>Total Responses</h3>
                        <div class="value">{stats["total_count"]}</div>
                        <div class="subtitle">Analyzed responses</div>
                    </div>
                    
                    <div class="card">
                        <h3>Avg Factuality</h3>
                        <div class="value {self._get_value_class(stats["avg_factuality_score"])}">{stats["avg_factuality_score"]:.2f}</div>
                        <div class="subtitle">0.0 - 1.0 scale</div>
                    </div>
                    
                    <div class="card">
                        <h3>Hallucination Reduction</h3>
                        <div class="value {self._get_value_class(stats["hallucination_reduction_rate"])}">{stats["hallucination_reduction_rate"]:.2%}</div>
                        <div class="subtitle">Estimated reduction rate</div>
                    </div>
                    
                    <div class="card">
                        <h3>Avg Drift Score</h3>
                        <div class="value {self._get_value_class(1.0 - stats["avg_drift_score"])}">{stats["avg_drift_score"]:.2f}</div>
                        <div class="subtitle">Lower is better</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h2>Factuality Grade Distribution</h2>
                    <img src="{factuality_chart}" class="chart" alt="Factuality Grade Distribution">
                </div>
                
                <div class="chart-container">
                    <h2>Semantic Drift Severity</h2>
                    <img src="{drift_chart}" class="chart" alt="Semantic Drift Severity">
                </div>
                
                <div class="chart-container">
                    <h2>Confidence vs. Factuality Correlation</h2>
                    <img src="{confidence_chart}" class="chart" alt="Confidence vs. Factuality">
                </div>
                
                <div class="chart-container">
                    <h2>Hallucination Issue Categories</h2>
                    <img src="{issues_chart}" class="chart" alt="Hallucination Issues">
                </div>
                
                <div class="chart-container">
                    <h2>Recent Metrics</h2>
                    <table class="metrics-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Timestamp</th>
                                <th>Factuality</th>
                                <th>Drift</th>
                                <th>Confidence</th>
                                <th>Issues</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Add recent metrics rows (most recent 10)
        sorted_metrics = sorted(
            metrics_data,
            key=lambda x: x.get("timestamp", ""),
            reverse=True
        )
        
        for data in sorted_metrics[:10]:
            factuality = data.get("factuality_score", 0)
            drift = data.get("semantic_drift", {}).get("drift_score", 0)
            confidence = data.get("confidence_analysis", {}).get("correlation_score", 0)
            issue_count = len(data.get("detected_issues", []))
            
            html_content += f"""
                            <tr>
                                <td>{data.get("id", "unknown")[:8]}...</td>
                                <td>{data.get("timestamp", "unknown")[:16]}</td>
                                <td class="{self._get_value_class(factuality)}">{factuality:.2f}</td>
                                <td class="{self._get_value_class(1.0 - drift)}">{drift:.2f}</td>
                                <td class="{self._get_value_class(confidence)}">{confidence:.2f}</td>
                                <td>{issue_count}</td>
                            </tr>
            """
        
        html_content += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="footer">
                <p>ACDEK v3 Hallucination Metrics Dashboard | Generated by HallucinationDashboard</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _get_value_class(self, value: float) -> str:
        """
        Get CSS class for a value based on its quality.
        
        Args:
            value: Value to classify
            
        Returns:
            CSS class name
        """
        if value >= 0.9:
            return "high-value"
        elif value >= 0.7:
            return "moderate-value"
        elif value >= 0.5:
            return "low-value"
        else:
            return "poor-value"
    
    def generate_dashboard(self, output_file: Optional[str] = None) -> str:
        """
        Generate and save the dashboard.
        
        Args:
            output_file: Optional output file path
            
        Returns:
            Path to the generated dashboard HTML file
        """
        # Load metrics data
        metrics_data = self.load_metrics_files()
        
        if not metrics_data:
            logger.warning(f"No metrics data found in {self.metrics_dir}")
            # Create empty dashboard with warning
            metrics_data = []
        
        # Generate dashboard HTML
        html_content = self.generate_dashboard_html(metrics_data)
        
        # Generate output file path if not provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"hallucination_dashboard_{timestamp}.html")
        
        # Save dashboard
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard generated at {output_file}")
        return output_file


def main():
    """
    Main function to generate dashboard from command line.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate hallucination metrics dashboard")
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=DEFAULT_METRICS_DIR,
        help=f"Directory containing metrics files (default: {DEFAULT_METRICS_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for dashboard output (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="ACDEK v3 Hallucination Metrics Dashboard",
        help="Dashboard title"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file path (default: auto-generated)"
    )
    parser.add_argument(
        "--comparative",
        action="store_true",
        help="Generate additional comparative analysis if benchmark data is available"
    )
    parser.add_argument(
        "--benchmark-file",
        type=str,
        help="Path to benchmark results file for comparative analysis"
    )
    parser.add_argument(
        "--compare-format",
        type=str,
        choices=["json", "html", "markdown", "csv", "yaml"],
        default="html",
        help="Output format for comparative analysis report"
    )
    
    args = parser.parse_args()
    
    # Create dashboard
    dashboard = HallucinationDashboard(
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir,
        title=args.title
    )
    
    # Generate dashboard
    output_file = dashboard.generate_dashboard(args.output_file)
    
    print(f"Dashboard generated: {output_file}")
    
    # Generate comparative analysis if requested and available
    if args.comparative and COMPARATIVE_ANALYZER_AVAILABLE:
        try:
            if args.benchmark_file:
                benchmark_file = args.benchmark_file
            else:
                # Try to find the most recent benchmark result file
                result_files = list(Path(args.output_dir).parent.glob("benchmark_results_*.json"))
                if result_files:
                    benchmark_file = str(sorted(result_files, key=os.path.getmtime, reverse=True)[0])
                else:
                    print("No benchmark file found for comparative analysis")
                    return
            
            # Run comparative analysis
            print(f"Generating comparative analysis from {benchmark_file}")
            analyzer = ComparativeAnalyzer(
                results_dir=os.path.dirname(benchmark_file),
                output_dir=args.output_dir
            )
            
            # Load benchmark data
            with open(benchmark_file, 'r') as f:
                benchmark_data = json.load(f)
            
            # Analyze data
            comparison = analyzer.analyze_benchmark_data(benchmark_data)
            
            # Export report
            output_format = OutputFormat(args.compare_format)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            compare_file = os.path.join(args.output_dir, f"comparative_analysis_{timestamp}.{args.compare_format}")
            
            report_path = analyzer.export_comparison_report(
                comparison,
                output_format=output_format,
                output_file=compare_file
            )
            
            print(f"Comparative analysis report generated: {report_path}")
        except Exception as e:
            print(f"Error generating comparative analysis: {str(e)}")
    elif args.comparative and not COMPARATIVE_ANALYZER_AVAILABLE:
        print("Comparative analyzer is not available. Install or import the module to enable this feature.")


if __name__ == "__main__":
    main()