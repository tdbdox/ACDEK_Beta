"""
Performance Metrics Framework

Provides tools for measuring, aggregating, and exporting performance metrics
across benchmark runs. This module includes:

1. Context managers for latency measurement
2. Aggregators for retry counts and penalty metrics
3. Domain-specific thresholds and weighting
4. Export utilities for multiple formats (JSON, CSV, HTML)
"""
import time
import os
import json
import csv
import statistics
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

# Domain-specific thresholds
DOMAIN_THRESHOLDS = {
    "medical": {
        "factuality": 0.75,
        "latency": 200.0,
        "retry_penalty": 0.8
    },
    "legal": {
        "factuality": 0.72,
        "latency": 250.0,
        "retry_penalty": 0.9
    },
    "financial": {
        "factuality": 0.68,
        "latency": 180.0,
        "retry_penalty": 0.7
    },
    "general": {
        "factuality": 0.70,  # Global default
        "latency": 200.0,
        "retry_penalty": 0.8
    }
}

# Domain-specific weights for aggregate scoring
DOMAIN_WEIGHTS = {
    "medical": {
        "factuality": 0.7,
        "latency": 0.15,
        "retry": 0.15
    },
    "legal": {
        "factuality": 0.8,
        "latency": 0.1,
        "retry": 0.1
    },
    "financial": {
        "factuality": 0.6,
        "latency": 0.2,
        "retry": 0.2
    },
    "general": {
        "factuality": 0.65,
        "latency": 0.2,
        "retry": 0.15
    }
}

@contextmanager
def latency_timer(metric_name: str, metrics: Dict[str, Any]) -> None:
    """
    Context manager to measure latency for a specific metric.
    
    Args:
        metric_name: Name of the metric being measured
        metrics: Dictionary to store the metric result
        
    Example:
        ```python
        metrics = {}
        with latency_timer("api_call", metrics):
            response = api.call()
        
        print(f"API call took {metrics['latency']['api_call']} seconds")
        ```
    """
    start = time.time()
    yield
    elapsed = time.time() - start
    
    # Store in the metrics dictionary
    if 'latency' not in metrics:
        metrics['latency'] = {}
    
    metrics['latency'][metric_name] = elapsed

class RetryAggregator:
    """
    Aggregates retry counts and penalty metrics with trend analysis.
    
    This enhanced version includes:
    - Trend tracking over time windows
    - Threshold alerts for abnormal retry rates
    - Domain-specific penalty weights
    """
    def __init__(self, window_size: int = 10, alert_threshold: float = 1.5):
        self.counts: Dict[str, int] = {}
        self.penalties: Dict[str, float] = {}
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Trend tracking
        self.history: Dict[str, List[Tuple[int, float]]] = {}
        self.alerts: Dict[str, List[str]] = {}
        self.timestamp_history: Dict[str, List[float]] = {}
    
    def record(self, plugin_name: str, retries: int, penalty: float, domain: str = "general") -> None:
        """
        Record retries for a plugin with associated penalty.
        
        Args:
            plugin_name: Name of the plugin
            retries: Number of retries
            penalty: Penalty value associated with retries
            domain: Domain context for domain-specific weighting
        """
        # Apply domain-specific penalty weight
        domain_weight = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS["general"])
        adjusted_penalty = penalty * domain_weight["retry"]
        
        # Update counts and penalties
        self.counts[plugin_name] = self.counts.get(plugin_name, 0) + retries
        self.penalties[plugin_name] = self.penalties.get(plugin_name, 0.0) + adjusted_penalty
        
        # Update history for trend analysis
        if plugin_name not in self.history:
            self.history[plugin_name] = []
            self.timestamp_history[plugin_name] = []
        
        self.history[plugin_name].append((retries, adjusted_penalty))
        self.timestamp_history[plugin_name].append(time.time())
        
        # Keep only the last window_size entries
        if len(self.history[plugin_name]) > self.window_size:
            self.history[plugin_name] = self.history[plugin_name][-self.window_size:]
            self.timestamp_history[plugin_name] = self.timestamp_history[plugin_name][-self.window_size:]
        
        # Check for alerts
        self._check_for_alerts(plugin_name)
    
    def _check_for_alerts(self, plugin_name: str) -> None:
        """
        Check for abnormal retry patterns and generate alerts.
        
        Args:
            plugin_name: Plugin to check
        """
        if plugin_name not in self.history or len(self.history[plugin_name]) < 2:
            return
        
        # Calculate recent average
        recent_retries = [r[0] for r in self.history[plugin_name][-3:]]
        if not recent_retries:
            return
        
        recent_avg = sum(recent_retries) / len(recent_retries)
        
        # Calculate historical average (excluding most recent)
        hist_retries = [r[0] for r in self.history[plugin_name][:-3]]
        if not hist_retries:
            return
        
        hist_avg = sum(hist_retries) / len(hist_retries)
        
        # Check if recent is significantly higher than historical
        if hist_avg > 0 and recent_avg / hist_avg > self.alert_threshold:
            alert = f"Abnormal retry rate detected for {plugin_name}: {recent_avg:.2f} vs historical {hist_avg:.2f}"
            
            if plugin_name not in self.alerts:
                self.alerts[plugin_name] = []
            
            self.alerts[plugin_name].append(alert)
    
    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of retry metrics.
        
        Returns:
            Dictionary with counts, penalties, trends, and alerts
        """
        # Calculate trends for each plugin
        trends = {}
        for plugin, history in self.history.items():
            if len(history) < 2:
                trends[plugin] = "stable"
                continue
            
            # Calculate trend direction based on slope
            retries = [h[0] for h in history]
            
            if len(retries) >= 3:
                # Simple trend detection
                first_half = sum(retries[:len(retries)//2]) / (len(retries)//2)
                second_half = sum(retries[len(retries)//2:]) / (len(retries) - len(retries)//2)
                
                if second_half > first_half * 1.1:
                    trends[plugin] = "increasing"
                elif second_half < first_half * 0.9:
                    trends[plugin] = "decreasing"
                else:
                    trends[plugin] = "stable"
            else:
                trends[plugin] = "insufficient_data"
        
        return {
            'counts': self.counts,
            'penalties': self.penalties,
            'trends': trends,
            'alerts': self.alerts
        }
    
    def get_normalized_penalty(self, plugin_name: str, domain: str = "general") -> float:
        """
        Get domain-normalized penalty score for a plugin.
        
        Args:
            plugin_name: Plugin name
            domain: Domain for normalization
            
        Returns:
            Normalized penalty score (0.0-1.0)
        """
        if plugin_name not in self.penalties:
            return 0.0
        
        raw_penalty = self.penalties[plugin_name]
        
        # Get domain threshold
        domain_threshold = DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS["general"])
        max_penalty = domain_threshold["retry_penalty"]
        
        # Normalize to 0.0-1.0 scale
        return min(1.0, raw_penalty / max_penalty)
    
    def reset(self) -> None:
        """Reset all retry metrics."""
        self.counts = {}
        self.penalties = {}
        self.history = {}
        self.alerts = {}
        self.timestamp_history = {}

class PerformanceAggregator:
    """
    Aggregates performance metrics across multiple runs with domain-specific weighting.
    
    Provides:
    - Cross-benchmark aggregation
    - Domain-specific thresholds and scoring
    - Trend detection and anomaly identification
    - Customizable summary generation
    """
    def __init__(self):
        self.runs: List[Dict[str, Any]] = []
        self.run_timestamps: List[float] = []
        self.domain_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.model_metrics: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_run(self, metrics: Dict[str, Any], domain: str = "general", model: str = "unknown") -> None:
        """
        Add a benchmark run to the aggregator.
        
        Args:
            metrics: Run metrics dictionary
            domain: Domain of the benchmark
            model: Model used in the benchmark
        """
        # Add timestamp
        run_with_meta = {
            "metrics": metrics,
            "domain": domain,
            "model": model,
            "timestamp": time.time()
        }
        
        # Store run
        self.runs.append(run_with_meta)
        self.run_timestamps.append(run_with_meta["timestamp"])
        
        # Update domain metrics
        if domain not in self.domain_metrics:
            self.domain_metrics[domain] = []
        self.domain_metrics[domain].append(run_with_meta)
        
        # Update model metrics
        if model not in self.model_metrics:
            self.model_metrics[model] = []
        self.model_metrics[model].append(run_with_meta)
    
    def get_domain_summary(self, domain: str = None) -> Dict[str, Any]:
        """
        Get performance summary for a specific domain or all domains.
        
        Args:
            domain: Domain to summarize, or None for all domains
            
        Returns:
            Dictionary with domain-specific summary metrics
        """
        domains_to_summarize = [domain] if domain else list(self.domain_metrics.keys())
        result = {}
        
        for d in domains_to_summarize:
            if d not in self.domain_metrics:
                continue
            
            domain_runs = self.domain_metrics[d]
            
            # Extract key metrics
            latencies = []
            factuality_scores = []
            retry_penalties = []
            
            for run in domain_runs:
                metrics = run["metrics"]
                
                # Extract latency
                if "latency" in metrics:
                    for latency in metrics["latency"].values():
                        latencies.append(latency)
                
                # Extract factuality
                if "factuality" in metrics:
                    factuality_scores.append(metrics["factuality"])
                
                # Extract retry penalties
                if "retry_penalties" in metrics:
                    for penalty in metrics["retry_penalties"].values():
                        retry_penalties.append(penalty)
            
            # Calculate statistics
            result[d] = {
                "run_count": len(domain_runs),
                "latency": {
                    "mean": statistics.mean(latencies) if latencies else None,
                    "median": statistics.median(latencies) if latencies else None,
                    "min": min(latencies) if latencies else None,
                    "max": max(latencies) if latencies else None,
                    "stdev": statistics.stdev(latencies) if len(latencies) > 1 else None
                },
                "factuality": {
                    "mean": statistics.mean(factuality_scores) if factuality_scores else None,
                    "median": statistics.median(factuality_scores) if factuality_scores else None,
                    "min": min(factuality_scores) if factuality_scores else None,
                    "max": max(factuality_scores) if factuality_scores else None,
                    "stdev": statistics.stdev(factuality_scores) if len(factuality_scores) > 1 else None
                },
                "retry_penalties": {
                    "mean": statistics.mean(retry_penalties) if retry_penalties else None,
                    "median": statistics.median(retry_penalties) if retry_penalties else None,
                    "min": min(retry_penalties) if retry_penalties else None,
                    "max": max(retry_penalties) if retry_penalties else None,
                    "stdev": statistics.stdev(retry_penalties) if len(retry_penalties) > 1 else None
                }
            }
            
            # Add domain-specific threshold comparisons
            domain_thresholds = DOMAIN_THRESHOLDS.get(d, DOMAIN_THRESHOLDS["general"])
            domain_weights = DOMAIN_WEIGHTS.get(d, DOMAIN_WEIGHTS["general"])
            
            # Calculate factuality success rate
            if factuality_scores:
                success_count = sum(1 for score in factuality_scores if score >= domain_thresholds["factuality"])
                result[d]["factuality"]["success_rate"] = success_count / len(factuality_scores)
            
            # Calculate latency success rate
            if latencies:
                success_count = sum(1 for lat in latencies if lat <= domain_thresholds["latency"] / 1000)  # Convert ms to s
                result[d]["latency"]["success_rate"] = success_count / len(latencies)
            
            # Calculate weighted domain score
            if factuality_scores and latencies and retry_penalties:
                # Normalize metrics
                norm_factuality = statistics.mean(factuality_scores)
                
                # For latency, lower is better, so invert the normalization
                max_latency = domain_thresholds["latency"] / 1000  # Convert ms to s
                norm_latency = 1.0 - min(1.0, statistics.mean(latencies) / max_latency)
                
                # For retry penalties, lower is better
                max_penalty = domain_thresholds["retry_penalty"]
                norm_retry = 1.0 - min(1.0, statistics.mean(retry_penalties) / max_penalty)
                
                # Calculate weighted score
                result[d]["weighted_score"] = (
                    norm_factuality * domain_weights["factuality"] +
                    norm_latency * domain_weights["latency"] +
                    norm_retry * domain_weights["retry"]
                )
        
        return result
    
    def get_model_summary(self, model: str = None) -> Dict[str, Any]:
        """
        Get performance summary for a specific model or all models.
        
        Args:
            model: Model to summarize, or None for all models
            
        Returns:
            Dictionary with model-specific summary metrics
        """
        models_to_summarize = [model] if model else list(self.model_metrics.keys())
        result = {}
        
        for m in models_to_summarize:
            if m not in self.model_metrics:
                continue
            
            model_runs = self.model_metrics[m]
            
            # Extract key metrics
            latencies = []
            factuality_scores = []
            retry_penalties = []
            domains = []
            
            for run in model_runs:
                metrics = run["metrics"]
                domains.append(run["domain"])
                
                # Extract latency
                if "latency" in metrics:
                    for latency in metrics["latency"].values():
                        latencies.append(latency)
                
                # Extract factuality
                if "factuality" in metrics:
                    factuality_scores.append(metrics["factuality"])
                
                # Extract retry penalties
                if "retry_penalties" in metrics:
                    for penalty in metrics["retry_penalties"].values():
                        retry_penalties.append(penalty)
            
            # Calculate statistics
            result[m] = {
                "run_count": len(model_runs),
                "domains": list(set(domains)),
                "latency": {
                    "mean": statistics.mean(latencies) if latencies else None,
                    "median": statistics.median(latencies) if latencies else None,
                    "min": min(latencies) if latencies else None,
                    "max": max(latencies) if latencies else None,
                    "stdev": statistics.stdev(latencies) if len(latencies) > 1 else None
                },
                "factuality": {
                    "mean": statistics.mean(factuality_scores) if factuality_scores else None,
                    "median": statistics.median(factuality_scores) if factuality_scores else None,
                    "min": min(factuality_scores) if factuality_scores else None,
                    "max": max(factuality_scores) if factuality_scores else None,
                    "stdev": statistics.stdev(factuality_scores) if len(factuality_scores) > 1 else None
                },
                "retry_penalties": {
                    "mean": statistics.mean(retry_penalties) if retry_penalties else None,
                    "median": statistics.median(retry_penalties) if retry_penalties else None,
                    "min": min(retry_penalties) if retry_penalties else None,
                    "max": max(retry_penalties) if retry_penalties else None,
                    "stdev": statistics.stdev(retry_penalties) if len(retry_penalties) > 1 else None
                }
            }
            
            # Calculate overall success rate
            if factuality_scores:
                # Use general thresholds for model comparison
                general_threshold = DOMAIN_THRESHOLDS["general"]["factuality"]
                success_count = sum(1 for score in factuality_scores if score >= general_threshold)
                result[m]["factuality"]["success_rate"] = success_count / len(factuality_scores)
        
        return result
    
    def analyze_trends(self, window: int = 5) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            window: Window size for trend analysis
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self.runs) < window:
            return {"status": "insufficient_data", "required_runs": window}
        
        # Group by time windows
        result = {"windows": [], "trend": "stable"}
        
        # Sort runs by timestamp
        sorted_runs = sorted(self.runs, key=lambda x: x["timestamp"])
        
        # Analyze in windows
        for i in range(0, len(sorted_runs), window):
            window_runs = sorted_runs[i:i+window]
            if not window_runs:
                continue
            
            # Calculate average metrics for this window
            window_metrics = {
                "start_time": window_runs[0]["timestamp"],
                "end_time": window_runs[-1]["timestamp"],
                "run_count": len(window_runs),
                "factuality": [],
                "latency": []
            }
            
            for run in window_runs:
                metrics = run["metrics"]
                
                if "factuality" in metrics:
                    window_metrics["factuality"].append(metrics["factuality"])
                
                if "latency" in metrics:
                    for latency in metrics["latency"].values():
                        window_metrics["latency"].append(latency)
            
            # Calculate averages
            if window_metrics["factuality"]:
                window_metrics["avg_factuality"] = statistics.mean(window_metrics["factuality"])
            
            if window_metrics["latency"]:
                window_metrics["avg_latency"] = statistics.mean(window_metrics["latency"])
            
            # Clean up
            del window_metrics["factuality"]
            del window_metrics["latency"]
            
            result["windows"].append(window_metrics)
        
        # Analyze overall trend
        if len(result["windows"]) >= 2:
            first_window = result["windows"][0]
            last_window = result["windows"][-1]
            
            # Compare factuality trend
            if "avg_factuality" in first_window and "avg_factuality" in last_window:
                fact_change = last_window["avg_factuality"] - first_window["avg_factuality"]
                if fact_change > 0.05:
                    result["factuality_trend"] = "improving"
                elif fact_change < -0.05:
                    result["factuality_trend"] = "declining"
                else:
                    result["factuality_trend"] = "stable"
            
            # Compare latency trend
            if "avg_latency" in first_window and "avg_latency" in last_window:
                # For latency, lower is better
                latency_change = first_window["avg_latency"] - last_window["avg_latency"]
                if latency_change > 0.05 * first_window["avg_latency"]:
                    result["latency_trend"] = "improving"
                elif latency_change < -0.05 * first_window["avg_latency"]:
                    result["latency_trend"] = "declining"
                else:
                    result["latency_trend"] = "stable"
            
            # Overall trend determination
            if "factuality_trend" in result and "latency_trend" in result:
                if result["factuality_trend"] == "improving" and result["latency_trend"] != "declining":
                    result["trend"] = "improving"
                elif result["factuality_trend"] == "declining" or result["latency_trend"] == "declining":
                    result["trend"] = "declining"
        
        return result
    
    def compare_runs(self, run_id1: int, run_id2: int) -> Dict[str, Any]:
        """
        Compare two specific benchmark runs.
        
        Args:
            run_id1: Index of first run
            run_id2: Index of second run
            
        Returns:
            Dictionary with comparison results
        """
        if run_id1 < 0 or run_id1 >= len(self.runs) or run_id2 < 0 or run_id2 >= len(self.runs):
            return {"error": "Invalid run indices"}
        
        run1 = self.runs[run_id1]
        run2 = self.runs[run_id2]
        
        # Compare key metrics
        comparison = {
            "run1": {
                "timestamp": run1["timestamp"],
                "domain": run1["domain"],
                "model": run1["model"]
            },
            "run2": {
                "timestamp": run2["timestamp"],
                "domain": run2["domain"],
                "model": run2["model"]
            },
            "metrics_delta": {}
        }
        
        # Compare factuality
        if "factuality" in run1["metrics"] and "factuality" in run2["metrics"]:
            fact1 = run1["metrics"]["factuality"]
            fact2 = run2["metrics"]["factuality"]
            comparison["metrics_delta"]["factuality"] = fact2 - fact1
            comparison["metrics_delta"]["factuality_percent"] = (fact2 / fact1 - 1) * 100 if fact1 else float('inf')
        
        # Compare latency
        if "latency" in run1["metrics"] and "latency" in run2["metrics"]:
            lat1_values = list(run1["metrics"]["latency"].values())
            lat2_values = list(run2["metrics"]["latency"].values())
            
            if lat1_values and lat2_values:
                avg_lat1 = statistics.mean(lat1_values)
                avg_lat2 = statistics.mean(lat2_values)
                
                comparison["metrics_delta"]["latency"] = avg_lat2 - avg_lat1
                comparison["metrics_delta"]["latency_percent"] = (avg_lat2 / avg_lat1 - 1) * 100 if avg_lat1 else float('inf')
        
        return comparison

def write_metrics(metrics: Dict[str, Any], path: str, format: str = 'json') -> str:
    """
    Write metrics to disk in JSON, CSV, or HTML format.
    
    Args:
        metrics: Metrics dictionary to write
        path: File path to write to
        format: Format to use ('json', 'csv', or 'html')
        
    Returns:
        Path to the written file
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    if format == 'json':
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return path
    
    elif format == 'csv':
        # Flatten metrics dictionary for CSV
        flattened_metrics = flatten_dict(metrics)
        
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            
            for key, value in flattened_metrics.items():
                writer.writerow([key, value])
        
        return path
    
    elif format == 'html':
        # Generate HTML report
        html_content = generate_html_report(metrics)
        
        with open(path, 'w') as f:
            f.write(html_content)
        
        return path
    
    else:
        raise ValueError(f"Unsupported format: {format}")

def flatten_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
    """
    Flatten a nested dictionary for CSV export.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        
    Returns:
        Flattened dictionary
    """
    flattened = {}
    
    for k, v in d.items():
        key = f"{parent_key}.{k}" if parent_key else k
        
        if isinstance(v, dict):
            flattened.update(flatten_dict(v, key))
        elif isinstance(v, list):
            # Handle lists by joining with commas
            flattened[key] = ','.join(str(item) for item in v)
        else:
            flattened[key] = v
    
    return flattened

def generate_html_report(metrics: Dict[str, Any]) -> str:
    """
    Generate an HTML report from metrics.
    
    Args:
        metrics: Metrics dictionary
        
    Returns:
        HTML report content
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start HTML document
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ACDEK Performance Metrics Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .section {{ margin-bottom: 20px; padding: 10px; border: 1px solid #eee; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 15px; }}
            th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .good {{ color: green; }}
            .warning {{ color: orange; }}
            .bad {{ color: red; }}
            .info {{ color: blue; }}
            .metric-card {{ display: inline-block; width: 200px; margin: 10px; padding: 15px; 
                          border-radius: 5px; background-color: #f5f5f5; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
            .metric-label {{ font-size: 14px; color: #666; }}
        </style>
    </head>
    <body>
        <h1>ACDEK Performance Metrics Report</h1>
        <p>Generated: {timestamp}</p>
    """
    
    # Add summary section if available
    if "summary" in metrics:
        html += """
        <div class="section">
            <h2>Summary</h2>
            <div class="metric-cards">
        """
        
        for key, value in metrics["summary"].items():
            if isinstance(value, (int, float)):
                # Determine color class based on metric name
                css_class = "info"
                if "success" in key or "factuality" in key:
                    css_class = "good" if value > 0.7 else "warning" if value > 0.5 else "bad"
                elif "latency" in key or "time" in key:
                    css_class = "good" if value < 0.2 else "warning" if value < 0.5 else "bad"
                
                # Format display value
                display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                
                # Add metric card
                html += f"""
                <div class="metric-card">
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value {css_class}">{display_value}</div>
                </div>
                """
        
        html += """
            </div>
        </div>
        """
    
    # Add domain-specific metrics if available
    if "domains" in metrics:
        html += """
        <div class="section">
            <h2>Domain Performance</h2>
            <table>
                <tr>
                    <th>Domain</th>
                    <th>Factuality</th>
                    <th>Latency</th>
                    <th>Success Rate</th>
                </tr>
        """
        
        for domain, domain_metrics in metrics["domains"].items():
            factuality = domain_metrics.get("factuality", {}).get("mean", "N/A")
            latency = domain_metrics.get("latency", {}).get("mean", "N/A")
            success_rate = domain_metrics.get("factuality", {}).get("success_rate", "N/A")
            
            # Determine color classes
            fact_class = "good" if factuality > 0.7 else "warning" if factuality > 0.5 else "bad"
            latency_class = "good" if latency < 0.2 else "warning" if latency < 0.5 else "bad"
            success_class = "good" if success_rate > 0.7 else "warning" if success_rate > 0.5 else "bad"
            
            html += f"""
            <tr>
                <td>{domain}</td>
                <td class="{fact_class}">{factuality:.2f if isinstance(factuality, float) else factuality}</td>
                <td class="{latency_class}">{latency:.2f if isinstance(latency, float) else latency}s</td>
                <td class="{success_class}">{success_rate:.1%f if isinstance(success_rate, float) else success_rate}</td>
            </tr>
            """
        
        html += """
            </table>
        </div>
        """
    
    # Add latency details if available
    if "latency" in metrics:
        html += """
        <div class="section">
            <h2>Latency Metrics</h2>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Duration (s)</th>
                </tr>
        """
        
        for component, duration in metrics["latency"].items():
            # Determine color class based on duration
            css_class = "good" if duration < 0.2 else "warning" if duration < 0.5 else "bad"
            
            html += f"""
            <tr>
                <td>{component}</td>
                <td class="{css_class}">{duration:.3f}</td>
            </tr>
            """
        
        html += """
            </table>
        </div>
        """
    
    # Add retry metrics if available
    if "retries" in metrics:
        html += """
        <div class="section">
            <h2>Retry Metrics</h2>
            <table>
                <tr>
                    <th>Plugin</th>
                    <th>Count</th>
                    <th>Penalty</th>
                    <th>Trend</th>
                </tr>
        """
        
        for plugin, retry_data in metrics["retries"].items():
            count = retry_data.get("count", 0)
            penalty = retry_data.get("penalty", 0.0)
            trend = retry_data.get("trend", "stable")
            
            # Determine color classes
            count_class = "good" if count == 0 else "warning" if count < 3 else "bad"
            penalty_class = "good" if penalty < 0.2 else "warning" if penalty < 0.5 else "bad"
            trend_class = "good" if trend == "decreasing" else "bad" if trend == "increasing" else "info"
            
            html += f"""
            <tr>
                <td>{plugin}</td>
                <td class="{count_class}">{count}</td>
                <td class="{penalty_class}">{penalty:.2f}</td>
                <td class="{trend_class}">{trend}</td>
            </tr>
            """
        
        html += """
            </table>
        </div>
        """
    
    # Close HTML document
    html += """
    </body>
    </html>
    """
    
    return html

# Create a global aggregator instance for convenience
performance_aggregator = PerformanceAggregator()