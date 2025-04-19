"""
Performance visualization tools for ACDEK v3: Creates performance dashboards and reports.
"""
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

class PerformanceVisualizer:
    """
    Creates visualizations for performance data.
    
    This class generates graphs, charts, and HTML dashboards for ACDEK
    performance data to aid in analysis and optimization.
    """
    def __init__(
        self,
        output_directory: Optional[str] = None,
        dashboard_title: str = "ACDEK Performance Dashboard",
        include_css: bool = True
    ):
        """
        Initialize the performance visualizer.
        
        Args:
            output_directory: Directory to save visualizations and dashboard
            dashboard_title: Title for the generated dashboard
            include_css: Whether to include CSS styles in the dashboard
        """
        self.output_directory = output_directory or os.path.join(os.getcwd(), "performance_visualizations")
        self.dashboard_title = dashboard_title
        self.include_css = include_css
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)
        
        # Copy CSS file if including CSS
        if self.include_css:
            self._ensure_css()
    
    def _ensure_css(self):
        """Create or copy CSS file for dashboard styling."""
        css_path = os.path.join(self.output_directory, "dashboard.css")
        
        # Check if CSS file exists, create if not
        if not os.path.exists(css_path):
            with open(css_path, 'w') as f:
                f.write("""
/* ACDEK Performance Dashboard Styles */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f9f9f9;
}

h1, h2, h3, h4 {
    color: #2c3e50;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

h1 {
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}

.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.timestamp {
    color: #7f8c8d;
    font-size: 0.9em;
}

.visualization-container {
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    margin-bottom: 20px;
    padding: 15px;
    overflow: hidden;
}

.visualization-title {
    font-weight: bold;
    margin-bottom: 10px;
    color: #2980b9;
}

.chart-container {
    width: 100%;
    margin-bottom: 20px;
}

.chart-container img {
    width: 100%;
    height: auto;
    margin-top: 10px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

th, td {
    padding: 12px 15px;
    text-align: left;
    border-bottom: 1px solid #e1e1e1;
}

th {
    background-color: #f2f2f2;
    font-weight: bold;
}

tr:hover {
    background-color: #f5f5f5;
}

.critical {
    background-color: #ffebee;
    color: #c62828;
}

.warning {
    background-color: #fff8e1;
    color: #f57f17;
}

.info {
    background-color: #e3f2fd;
    color: #1565c0;
}

.excellent {
    background-color: #e8f5e9;
    color: #2e7d32;
}

.metric-group {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    margin-bottom: 20px;
}

.metric-card {
    flex: 1;
    min-width: 200px;
    padding: 15px;
    background-color: white;
    border-radius: 5px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
}

.metric-title {
    font-size: 0.9em;
    color: #7f8c8d;
    margin-bottom: 5px;
}

.metric-value {
    font-size: 1.8em;
    font-weight: bold;
    color: #2c3e50;
}

.metric-unit {
    font-size: 0.8em;
    color: #95a5a6;
}

.bottleneck-card {
    border-left: 4px solid #e74c3c;
    padding-left: 15px;
    margin-bottom: 15px;
}

.bottleneck-stage {
    font-weight: bold;
    margin-bottom: 5px;
}

.bottleneck-metrics {
    color: #7f8c8d;
    font-size: 0.9em;
    margin-bottom: 5px;
}

.recommendations {
    background-color: #f8f9fa;
    padding: 10px 15px;
    border-radius: 3px;
    margin-top: 10px;
}

.recommendations h4 {
    margin-top: 0;
    margin-bottom: 10px;
    font-size: 1em;
}

.recommendations ul {
    margin: 0;
    padding-left: 20px;
}

.tab-container {
    margin-top: 20px;
}

.tab-buttons {
    display: flex;
    border-bottom: 1px solid #ddd;
}

.tab-button {
    padding: 10px 20px;
    background-color: #f1f1f1;
    border: none;
    cursor: pointer;
    margin-right: 5px;
    border-radius: 5px 5px 0 0;
}

.tab-button.active {
    background-color: #3498db;
    color: white;
}

.tab-content {
    padding: 15px;
    border: 1px solid #ddd;
    border-top: none;
    display: none;
}

.tab-content.active {
    display: block;
}

@media (max-width: 768px) {
    .metric-group {
        flex-direction: column;
    }
    
    .metric-card {
        width: 100%;
    }
}
                """)
    
    def create_stage_timing_chart(
        self,
        stage_data: List[Dict[str, Any]],
        filename: str = "stage_timing.png",
        title: str = "Stage Execution Times",
        top_n: int = 10
    ) -> str:
        """
        Create a bar chart of stage execution times.
        
        Args:
            stage_data: List of stage profile data
            filename: Output filename
            title: Chart title
            top_n: Number of top stages to show
            
        Returns:
            Path to the created file
        """
        # Sort stages by avg_time (descending) and take top N
        sorted_stages = sorted(
            stage_data,
            key=lambda x: x.get("avg_time", 0),
            reverse=True
        )[:top_n]
        
        # Skip if no stages
        if not sorted_stages:
            return ""
        
        # Extract data
        stage_names = [s.get("name", "Unknown") for s in sorted_stages]
        avg_times = [s.get("avg_time", 0) for s in sorted_stages]
        
        # Create colors based on category
        colors = []
        for stage in sorted_stages:
            category = stage.get("category", "trace")
            if category == "critical":
                colors.append("#e74c3c")  # red
            elif category == "warning":
                colors.append("#f39c12")  # orange
            elif category == "info":
                colors.append("#3498db")  # blue
            elif category == "excellent":
                colors.append("#2ecc71")  # green
            else:
                colors.append("#95a5a6")  # gray
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(stage_names))
        plt.barh(y_pos, avg_times, align='center', color=colors)
        plt.yticks(y_pos, stage_names)
        
        # Add labels and title
        plt.xlabel('Execution Time (ms)')
        plt.title(title)
        
        # Add grid
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add values at end of bars
        for i, v in enumerate(avg_times):
            plt.text(v + 0.1, i, f"{v:.1f} ms", va='center')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to file
        file_path = os.path.join(self.output_directory, filename)
        plt.savefig(file_path, dpi=100)
        plt.close()
        
        return file_path
    
    def create_relative_timing_chart(
        self,
        stage_data: List[Dict[str, Any]],
        filename: str = "relative_timing.png",
        title: str = "Relative Stage Execution Time",
        min_percentage: float = 1.0  # Minimum percentage to show
    ) -> str:
        """
        Create a pie chart of relative execution times.
        
        Args:
            stage_data: List of stage profile data
            filename: Output filename
            title: Chart title
            min_percentage: Minimum percentage to include as separate slice
            
        Returns:
            Path to the created file
        """
        # Calculate total time
        total_time = sum(s.get("total_time", 0) for s in stage_data)
        
        # Skip if total time is zero
        if total_time == 0:
            return ""
        
        # Calculate percentages and filter stages
        stage_percentages = []
        stage_names = []
        stage_colors = []
        other_percentage = 0
        
        for stage in stage_data:
            name = stage.get("name", "Unknown")
            time = stage.get("total_time", 0)
            percentage = (time / total_time) * 100
            
            # Get color based on category
            category = stage.get("category", "trace")
            if category == "critical":
                color = "#e74c3c"  # red
            elif category == "warning":
                color = "#f39c12"  # orange
            elif category == "info":
                color = "#3498db"  # blue
            elif category == "excellent":
                color = "#2ecc71"  # green
            else:
                color = "#95a5a6"  # gray
            
            # Include as separate slice if above threshold
            if percentage >= min_percentage:
                stage_percentages.append(percentage)
                stage_names.append(name)
                stage_colors.append(color)
            else:
                other_percentage += percentage
        
        # Add "Other" category if needed
        if other_percentage > 0:
            stage_percentages.append(other_percentage)
            stage_names.append("Other")
            stage_colors.append("#bdc3c7")  # light gray
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create pie chart
        plt.pie(
            stage_percentages,
            labels=stage_names,
            autopct='%1.1f%%',
            startangle=90,
            colors=stage_colors,
            shadow=False,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        plt.axis('equal')
        
        # Add title
        plt.title(title)
        
        # Save to file
        file_path = os.path.join(self.output_directory, filename)
        plt.savefig(file_path, dpi=100)
        plt.close()
        
        return file_path
    
    def create_bottleneck_chart(
        self,
        bottlenecks: List[Dict[str, Any]],
        filename: str = "bottlenecks.png",
        title: str = "Performance Bottlenecks"
    ) -> str:
        """
        Create a chart visualizing bottlenecks by severity and impact.
        
        Args:
            bottlenecks: List of bottleneck data
            filename: Output filename
            title: Chart title
            
        Returns:
            Path to the created file
        """
        # Skip if no bottlenecks
        if not bottlenecks:
            return ""
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Extract data
        names = [b.get("stage_name", "Unknown") for b in bottlenecks]
        impact = [b.get("relative_impact", 0) * 100 for b in bottlenecks]  # Convert to percentage
        severity = [b.get("severity", 0) for b in bottlenecks]
        
        # Define colors by severity
        severity_colors = {
            0: "#95a5a6",  # gray
            1: "#3498db",  # blue
            2: "#f39c12",  # orange
            3: "#e74c3c",  # red
        }
        
        colors = [severity_colors.get(s, "#95a5a6") for s in severity]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(names))
        plt.barh(y_pos, impact, align='center', color=colors)
        plt.yticks(y_pos, names)
        
        # Add labels and title
        plt.xlabel('Relative Impact (%)')
        plt.title(title)
        
        # Add grid
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add values at end of bars
        for i, v in enumerate(impact):
            plt.text(v + 0.1, i, f"{v:.1f}%", va='center')
        
        # Add severity legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color=severity_colors[3], label='Critical'),
            plt.Rectangle((0, 0), 1, 1, color=severity_colors[2], label='Warning'),
            plt.Rectangle((0, 0), 1, 1, color=severity_colors[1], label='Low'),
            plt.Rectangle((0, 0), 1, 1, color=severity_colors[0], label='Info')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to file
        file_path = os.path.join(self.output_directory, filename)
        plt.savefig(file_path, dpi=100)
        plt.close()
        
        return file_path
    
    def create_scaling_chart(
        self,
        scaling_data: List[Dict[str, Any]],
        filename: str = "scaling.png",
        title: str = "Scaling Efficiency"
    ) -> str:
        """
        Create a chart visualizing scaling efficiency metrics.
        
        Args:
            scaling_data: List of scaling metrics over time
            filename: Output filename
            title: Chart title
            
        Returns:
            Path to the created file
        """
        # Skip if not enough data points
        if len(scaling_data) < 2:
            return ""
        
        # Create figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        
        # Extract data
        timestamps = [d.get("timestamp", 0) for d in scaling_data]
        timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
        
        concurrent = [d.get("concurrent_requests", 0) for d in scaling_data]
        latency = [d.get("avg_latency_ms", 0) for d in scaling_data]
        throughput = [d.get("throughput_rps", 0) for d in scaling_data]
        
        # Calculate scaling efficiency (latency vs. concurrency)
        # Ideal scaling: latency stays constant as concurrency increases
        ideal_latency = latency[0] if latency else 0
        efficiency = []
        
        for i, concurrency in enumerate(concurrent):
            if concurrency == 0 or i == 0:
                efficiency.append(1.0)  # Default to 1.0
            else:
                # Calculate efficiency: ideal_latency / actual_latency
                # Normalized to range 0-1 where 1 is perfect scaling
                efficiency.append(
                    min(1.0, ideal_latency / latency[i]) if latency[i] > 0 else 1.0
                )
        
        # Plot concurrency
        ax1.plot(timestamps, concurrent, 'b-', label='Concurrent Requests')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Concurrent Requests', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot latency on the same axis but different color
        ax1.plot(timestamps, latency, 'r-', label='Avg Latency (ms)')
        
        # Plot throughput and efficiency on second y-axis
        ax2.plot(timestamps, throughput, 'g-', label='Throughput (rps)')
        ax2.plot(timestamps, efficiency, 'm--', label='Scaling Efficiency')
        ax2.set_ylabel('Throughput / Efficiency', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Add title
        plt.title(title)
        
        # Add grid
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save to file
        file_path = os.path.join(self.output_directory, filename)
        plt.savefig(file_path, dpi=100)
        plt.close()
        
        return file_path
    
    def create_dashboard(
        self,
        profile_data: Dict[str, Any],
        bottleneck_data: Dict[str, Any],
        filename: str = "performance_dashboard.html",
        include_raw_data: bool = True
    ) -> str:
        """
        Create an HTML dashboard with performance visualizations.
        
        Args:
            profile_data: Performance profile data
            bottleneck_data: Bottleneck analysis data
            filename: Output filename
            include_raw_data: Whether to include raw data in the dashboard
            
        Returns:
            Path to the created dashboard file
        """
        # Extract data
        stage_profiles = profile_data.get("stage_profiles", {})
        stage_list = [{"name": name, **data} for name, data in stage_profiles.items()]
        
        # Create charts
        stage_timing_chart = self.create_stage_timing_chart(
            stage_list,
            filename="stage_timing.png",
            title="Top 10 Stages by Execution Time"
        )
        
        relative_timing_chart = self.create_relative_timing_chart(
            stage_list,
            filename="relative_timing.png",
            title="Relative Stage Execution Time"
        )
        
        bottlenecks = bottleneck_data.get("bottlenecks", [])
        bottleneck_chart = self.create_bottleneck_chart(
            bottlenecks,
            filename="bottlenecks.png",
            title="Performance Bottlenecks by Impact"
        )
        
        # Start HTML content
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.dashboard_title}</title>
    {"<link rel='stylesheet' href='dashboard.css'>" if self.include_css else ""}
    <style>
        {'' if self.include_css else '''
        body { font-family: sans-serif; line-height: 1.5; max-width: 1200px; margin: 0 auto; padding: 20px; }
        .visualization-container { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        .critical { background-color: #ffebee; }
        .warning { background-color: #fff8e1; }
        .tab-buttons { display: flex; border-bottom: 1px solid #ddd; }
        .tab-button { padding: 10px 20px; background-color: #f1f1f1; border: none; cursor: pointer; }
        .tab-button.active { background-color: #4CAF50; color: white; }
        .tab-content { display: none; padding: 15px; border: 1px solid #ddd; border-top: none; }
        .tab-content.active { display: block; }
        '''} 
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>{self.dashboard_title}</h1>
        <div class="timestamp">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>

    <div class="metric-group">
        <div class="metric-card">
            <div class="metric-title">Total Stages</div>
            <div class="metric-value">{len(stage_profiles)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Critical Bottlenecks</div>
            <div class="metric-value">{sum(1 for b in bottlenecks if b.get('severity', 0) == 3)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Avg Response Time</div>
            <div class="metric-value">
                {profile_data.get('request_metrics', {}).get('avg', 0):.1f}
                <span class="metric-unit">ms</span>
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Max Response Time</div>
            <div class="metric-value">
                {profile_data.get('request_metrics', {}).get('max', 0):.1f}
                <span class="metric-unit">ms</span>
            </div>
        </div>
    </div>

    <div class="tab-container">
        <div class="tab-buttons">
            <button class="tab-button active" onclick="openTab(event, 'tab-overview')">Overview</button>
            <button class="tab-button" onclick="openTab(event, 'tab-bottlenecks')">Bottlenecks</button>
            <button class="tab-button" onclick="openTab(event, 'tab-stages')">Stage Details</button>
            {'''<button class="tab-button" onclick="openTab(event, 'tab-raw')">Raw Data</button>''' if include_raw_data else ''}
        </div>

        <div id="tab-overview" class="tab-content active">
            <div class="visualization-container">
                <div class="visualization-title">Stage Execution Times</div>
                <div class="chart-container">
                    <img src="stage_timing.png" alt="Stage Execution Times">
                </div>
            </div>

            <div class="visualization-container">
                <div class="visualization-title">Relative Stage Execution Time</div>
                <div class="chart-container">
                    <img src="relative_timing.png" alt="Relative Stage Execution Time">
                </div>
            </div>
        </div>

        <div id="tab-bottlenecks" class="tab-content">
            <div class="visualization-container">
                <div class="visualization-title">Performance Bottlenecks</div>
                <div class="chart-container">
                    <img src="bottlenecks.png" alt="Performance Bottlenecks">
                </div>
            </div>

            <div class="visualization-container">
                <div class="visualization-title">Bottleneck Details</div>
"""
        
        # Add bottleneck details
        if bottlenecks:
            for bottleneck in bottlenecks:
                severity_class = ""
                if bottleneck.get("severity") == 3:
                    severity_class = "critical"
                elif bottleneck.get("severity") == 2:
                    severity_class = "warning"
                
                html_content += f"""
                <div class="bottleneck-card {severity_class}">
                    <div class="bottleneck-stage">{bottleneck.get("stage_name", "Unknown")}</div>
                    <div class="bottleneck-metrics">
                        Avg: {bottleneck.get("avg_time_ms", 0):.1f} ms | 
                        P95: {bottleneck.get("p95_time_ms", 0):.1f} ms | 
                        Calls: {bottleneck.get("call_count", 0)} | 
                        Impact: {bottleneck.get("relative_impact", 0) * 100:.1f}%
                    </div>
                    <div class="recommendations">
                        <h4>Recommendations:</h4>
                        <ul>
"""
                
                # Add recommendations
                recommendations = bottleneck.get("recommendations", [])
                for recommendation in recommendations:
                    html_content += f"                            <li>{recommendation}</li>\n"
                
                html_content += """
                        </ul>
                    </div>
                </div>
"""
        else:
            html_content += "<p>No bottlenecks detected.</p>"
        
        html_content += """
            </div>

            <div class="visualization-container">
                <div class="visualization-title">System Recommendations</div>
                <ul>
"""
        
        # Add system recommendations
        system_recommendations = bottleneck_data.get("system_recommendations", [])
        for recommendation in system_recommendations:
            html_content += f"                    <li>{recommendation}</li>\n"
        
        html_content += """
                </ul>
            </div>
        </div>

        <div id="tab-stages" class="tab-content">
            <div class="visualization-container">
                <div class="visualization-title">Stage Execution Details</div>
                <table>
                    <thead>
                        <tr>
                            <th>Stage</th>
                            <th>Avg Time (ms)</th>
                            <th>P95 (ms)</th>
                            <th>Max (ms)</th>
                            <th>Calls</th>
                            <th>Category</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        # Add stage details
        sorted_stages = sorted(
            [{"name": name, **data} for name, data in stage_profiles.items()],
            key=lambda x: x.get("avg_time", 0),
            reverse=True
        )
        
        for stage in sorted_stages:
            category = stage.get("category", "trace")
            html_content += f"""
                        <tr class="{category}">
                            <td>{stage.get("name", "Unknown")}</td>
                            <td>{stage.get("avg_time", 0):.2f}</td>
                            <td>{stage.get("p95_time", 0):.2f}</td>
                            <td>{stage.get("max_time", 0):.2f}</td>
                            <td>{stage.get("count", 0)}</td>
                            <td>{category}</td>
                        </tr>
"""
        
        html_content += """
                    </tbody>
                </table>
            </div>
        </div>
"""
        
        # Add raw data tab if requested
        if include_raw_data:
            html_content += """
        <div id="tab-raw" class="tab-content">
            <div class="visualization-container">
                <div class="visualization-title">Raw Profile Data</div>
                <pre style="overflow: auto; max-height: 400px;">
"""
            
            # Add raw profile data as formatted JSON
            html_content += json.dumps(profile_data, indent=2)
            
            html_content += """
                </pre>
            </div>

            <div class="visualization-container">
                <div class="visualization-title">Raw Bottleneck Data</div>
                <pre style="overflow: auto; max-height: 400px;">
"""
            
            # Add raw bottleneck data as formatted JSON
            html_content += json.dumps(bottleneck_data, indent=2)
            
            html_content += """
                </pre>
            </div>
        </div>
"""
        
        # Add JavaScript for tab functionality
        html_content += """
    </div>

    <script>
        function openTab(evt, tabName) {
            var i, tabContent, tabButtons;
            
            // Hide all tab content
            tabContent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabContent.length; i++) {
                tabContent[i].className = tabContent[i].className.replace(" active", "");
            }
            
            // Deactivate all tab buttons
            tabButtons = document.getElementsByClassName("tab-button");
            for (i = 0; i < tabButtons.length; i++) {
                tabButtons[i].className = tabButtons[i].className.replace(" active", "");
            }
            
            // Show the current tab and activate the button
            document.getElementById(tabName).className += " active";
            evt.currentTarget.className += " active";
        }
    </script>
</body>
</html>
"""
        
        # Write to file
        file_path = os.path.join(self.output_directory, filename)
        with open(file_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Created performance dashboard at {file_path}")
        return file_path
    
    def create_cli_report(
        self,
        profile_data: Dict[str, Any],
        bottleneck_data: Dict[str, Any]
    ) -> str:
        """
        Create a textual CLI report for performance data.
        
        Args:
            profile_data: Performance profile data
            bottleneck_data: Bottleneck analysis data
            
        Returns:
            String containing the formatted CLI report
        """
        # Extract data
        stage_profiles = profile_data.get("stage_profiles", {})
        bottlenecks = bottleneck_data.get("bottlenecks", [])
        
        # Start report
        report = f"""
=============================================================
ACDEK PERFORMANCE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
=============================================================

SUMMARY:
- Total Stages: {len(stage_profiles)}
- Critical Bottlenecks: {sum(1 for b in bottlenecks if b.get('severity', 0) == 3)}
- Warning Bottlenecks: {sum(1 for b in bottlenecks if b.get('severity', 0) == 2)}
- Avg Response Time: {profile_data.get('request_metrics', {}).get('avg', 0):.1f} ms
- Max Response Time: {profile_data.get('request_metrics', {}).get('max', 0):.1f} ms

TOP BOTTLENECKS:
"""
        
        # Add bottleneck details
        if bottlenecks:
            # Sort by severity and impact
            sorted_bottlenecks = sorted(
                bottlenecks,
                key=lambda b: (b.get("severity", 0), b.get("relative_impact", 0)),
                reverse=True
            )
            
            for i, bottleneck in enumerate(sorted_bottlenecks[:5]):  # Show top 5
                severity_label = "CRITICAL" if bottleneck.get("severity") == 3 else (
                    "WARNING" if bottleneck.get("severity") == 2 else "INFO"
                )
                
                report += f"""
{i+1}. {bottleneck.get("stage_name", "Unknown")} [{severity_label}]
   - Avg Time: {bottleneck.get("avg_time_ms", 0):.1f} ms
   - P95: {bottleneck.get("p95_time_ms", 0):.1f} ms
   - Calls: {bottleneck.get("call_count", 0)}
   - Impact: {bottleneck.get("relative_impact", 0) * 100:.1f}%
   
   Recommendations:
"""
                
                # Add recommendations
                recommendations = bottleneck.get("recommendations", [])
                for recommendation in recommendations:
                    report += f"   * {recommendation}\n"
        else:
            report += "No bottlenecks detected.\n"
        
        # Add system recommendations
        report += """
SYSTEM RECOMMENDATIONS:
"""
        
        system_recommendations = bottleneck_data.get("system_recommendations", [])
        for i, recommendation in enumerate(system_recommendations):
            report += f"{i+1}. {recommendation}\n"
        
        # Add top stage details
        report += """
TOP 10 SLOWEST STAGES:
"""
        
        # Sort stages by avg_time
        sorted_stages = sorted(
            [{"name": name, **data} for name, data in stage_profiles.items()],
            key=lambda x: x.get("avg_time", 0),
            reverse=True
        )
        
        # Format as table
        report += "\n{:<30} {:<10} {:<10} {:<10} {:<10}\n".format(
            "Stage", "Avg (ms)", "P95 (ms)", "Max (ms)", "Calls"
        )
        report += "-" * 70 + "\n"
        
        for stage in sorted_stages[:10]:  # Show top 10
            report += "{:<30} {:<10.2f} {:<10.2f} {:<10.2f} {:<10}\n".format(
                stage.get("name", "Unknown")[:30],
                stage.get("avg_time", 0),
                stage.get("p95_time", 0),
                stage.get("max_time", 0),
                stage.get("count", 0)
            )
        
        return report