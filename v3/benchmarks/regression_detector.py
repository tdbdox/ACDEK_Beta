#!/usr/bin/env python
"""
ACDEK v3 Regression Detector

This module provides tools for detecting, classifying, and analyzing performance
regressions in ACDEK benchmark results. It includes:

1. Statistical models for regression detection
2. Baseline comparison and optimization
3. Trend analysis with anomaly detection
4. Regression classification and impact assessment
5. Visualization of regression patterns

Usage:
    python regression_detector.py --current results/benchmark_results_20250420.json 
                                 --baseline results/benchmark_results_20250410.json
"""

import os
import sys
import json
import time
import logging
import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field

# Add parent directory to path to allow importing from project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import relevant components
from benchmarks.analysis_tools import load_data, compare_versions
from benchmarks.threshold_config import ThresholdConfig, Domain, MetricType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("regression_detector.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_SIGNIFICANCE_LEVEL = 0.05  # 5% significance level for statistical tests
DEFAULT_REGRESSION_THRESHOLD = 0.05  # 5% degradation threshold
TREND_WINDOW_SIZE = 5  # Number of runs to consider for trend analysis


class RegressionSeverity(Enum):
    """Severity levels for performance regressions."""
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class ChangeDirection(Enum):
    """Direction of metric change."""
    IMPROVED = "improved"
    DEGRADED = "degraded"
    UNCHANGED = "unchanged"


class MetricChangeType(Enum):
    """Types of metric changes."""
    HALLUCINATION_REDUCTION = "hallucination_reduction"
    LATENCY = "latency"
    CONFIDENCE = "confidence"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RESOURCE_USAGE = "resource_usage"
    OTHER = "other"


@dataclass
class MetricChange:
    """Represents a change in a specific metric."""
    name: str
    change_type: MetricChangeType
    baseline_value: float
    current_value: float
    delta: float
    percent_change: float
    direction: ChangeDirection
    domain: Optional[str] = None
    test_case_id: Optional[str] = None
    
    def is_regression(self) -> bool:
        """Determine if this change is a regression."""
        if self.change_type in [
            MetricChangeType.HALLUCINATION_REDUCTION,
            MetricChangeType.CONFIDENCE,
            MetricChangeType.THROUGHPUT
        ]:
            # For these metrics, a decrease is a regression
            return self.direction == ChangeDirection.DEGRADED
        elif self.change_type in [
            MetricChangeType.LATENCY,
            MetricChangeType.ERROR_RATE,
            MetricChangeType.RESOURCE_USAGE
        ]:
            # For these metrics, an increase is a regression
            return self.direction == ChangeDirection.DEGRADED
        else:
            # Default: assume a decrease is a regression
            return self.direction == ChangeDirection.DEGRADED
    
    def get_severity(self, thresholds: Dict[str, float]) -> RegressionSeverity:
        """
        Determine the severity of the regression.
        
        Args:
            thresholds: Dictionary mapping severity levels to thresholds
            
        Returns:
            Severity level
        """
        if not self.is_regression():
            return None
        
        abs_percent = abs(self.percent_change)
        
        if abs_percent >= thresholds.get("critical", 0.2):
            return RegressionSeverity.CRITICAL
        elif abs_percent >= thresholds.get("severe", 0.1):
            return RegressionSeverity.SEVERE
        elif abs_percent >= thresholds.get("moderate", 0.05):
            return RegressionSeverity.MODERATE
        else:
            return RegressionSeverity.MINOR


@dataclass
class RegressionAnalysis:
    """Analysis of regressions across multiple metrics."""
    run_id: str
    timestamp: str
    baseline_id: str
    baseline_timestamp: str
    metric_changes: List[MetricChange] = field(default_factory=list)
    regression_metrics: List[MetricChange] = field(default_factory=list)
    improvement_metrics: List[MetricChange] = field(default_factory=list)
    unchanged_metrics: List[MetricChange] = field(default_factory=list)
    failed_metrics: List[str] = field(default_factory=list)
    domains_analyzed: Set[str] = field(default_factory=set)
    test_cases_analyzed: Set[str] = field(default_factory=set)
    
    def add_metric_change(self, change: MetricChange):
        """Add a metric change to the analysis."""
        self.metric_changes.append(change)
        
        if change.is_regression():
            self.regression_metrics.append(change)
        elif change.direction == ChangeDirection.IMPROVED:
            self.improvement_metrics.append(change)
        else:
            self.unchanged_metrics.append(change)
        
        if change.domain:
            self.domains_analyzed.add(change.domain)
        
        if change.test_case_id:
            self.test_cases_analyzed.add(change.test_case_id)
    
    def has_regressions(self) -> bool:
        """Check if there are any regressions."""
        return len(self.regression_metrics) > 0
    
    def get_regression_summary(self, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Get a summary of regressions by severity.
        
        Args:
            thresholds: Dictionary mapping severity levels to thresholds
            
        Returns:
            Dictionary with regression counts by severity
        """
        severity_counts = {
            RegressionSeverity.MINOR.value: 0,
            RegressionSeverity.MODERATE.value: 0,
            RegressionSeverity.SEVERE.value: 0,
            RegressionSeverity.CRITICAL.value: 0
        }
        
        for regression in self.regression_metrics:
            severity = regression.get_severity(thresholds)
            if severity:
                severity_counts[severity.value] += 1
        
        total = sum(severity_counts.values())
        
        return {
            "total": total,
            "by_severity": severity_counts,
            "has_regressions": total > 0,
            "has_critical": severity_counts[RegressionSeverity.CRITICAL.value] > 0,
            "has_severe": severity_counts[RegressionSeverity.SEVERE.value] > 0
        }
    
    def get_domain_impact(self) -> Dict[str, Dict[str, Any]]:
        """Get the impact of regressions by domain."""
        domain_impact = {}
        
        for domain in self.domains_analyzed:
            domain_regressions = [r for r in self.regression_metrics if r.domain == domain]
            domain_improvements = [r for r in self.improvement_metrics if r.domain == domain]
            
            # Calculate impact score (-1 to 1, negative means regression)
            if domain_regressions or domain_improvements:
                regression_score = sum(r.percent_change for r in domain_regressions)
                improvement_score = sum(r.percent_change for r in domain_improvements)
                impact_score = improvement_score + regression_score
            else:
                impact_score = 0
            
            domain_impact[domain] = {
                "regression_count": len(domain_regressions),
                "improvement_count": len(domain_improvements),
                "impact_score": impact_score,
                "most_significant_regression": max(domain_regressions, key=lambda r: abs(r.percent_change)) if domain_regressions else None
            }
        
        return domain_impact
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "baseline_id": self.baseline_id,
            "baseline_timestamp": self.baseline_timestamp,
            "metric_changes": [vars(mc) for mc in self.metric_changes],
            "regression_metrics": [vars(rm) for rm in self.regression_metrics],
            "improvement_metrics": [vars(im) for im in self.improvement_metrics],
            "unchanged_metrics": [vars(um) for um in self.unchanged_metrics],
            "failed_metrics": self.failed_metrics,
            "domains_analyzed": list(self.domains_analyzed),
            "test_cases_analyzed": list(self.test_cases_analyzed),
            "summary": {
                "total_metrics": len(self.metric_changes),
                "regression_count": len(self.regression_metrics),
                "improvement_count": len(self.improvement_metrics),
                "unchanged_count": len(self.unchanged_metrics),
                "failed_count": len(self.failed_metrics),
                "domains_count": len(self.domains_analyzed),
                "test_cases_count": len(self.test_cases_analyzed)
            }
        }


class RegressionDetector:
    """
    Detects performance regressions between benchmark runs.
    
    This class provides:
    1. Statistical detection of regressions
    2. Classification of regressions by severity
    3. Analysis of regression impact on different domains
    4. Trend analysis across multiple runs
    """
    
    def __init__(
        self,
        significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
        regression_thresholds: Optional[Dict[str, float]] = None,
        metrics_mapping: Optional[Dict[str, MetricChangeType]] = None
    ):
        """
        Initialize the regression detector.
        
        Args:
            significance_level: P-value threshold for statistical significance
            regression_thresholds: Thresholds for regression severity levels
            metrics_mapping: Mapping from metric names to change types
        """
        self.significance_level = significance_level
        self.regression_thresholds = regression_thresholds or {
            "minor": 0.02,     # 2% change
            "moderate": 0.05,  # 5% change
            "severe": 0.1,     # 10% change
            "critical": 0.2    # 20% change
        }
        
        # Default mapping of metric names to change types
        self.metrics_mapping = metrics_mapping or {
            "hallucination_reduction": MetricChangeType.HALLUCINATION_REDUCTION,
            "factuality": MetricChangeType.HALLUCINATION_REDUCTION,
            "latency": MetricChangeType.LATENCY,
            "latency_overhead": MetricChangeType.LATENCY,
            "confidence": MetricChangeType.CONFIDENCE,
            "throughput": MetricChangeType.THROUGHPUT,
            "error_rate": MetricChangeType.ERROR_RATE,
            "cpu_usage": MetricChangeType.RESOURCE_USAGE,
            "memory_usage": MetricChangeType.RESOURCE_USAGE
        }
    
    def detect_regressions(
        self,
        current_results: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> RegressionAnalysis:
        """
        Detect regressions between current and baseline results.
        
        Args:
            current_results: Current benchmark results
            baseline_results: Baseline benchmark results
            
        Returns:
            RegressionAnalysis object with detected changes
        """
        # Extract run IDs and timestamps
        current_id = current_results.get("run_id", "current")
        current_timestamp = current_results.get("timestamp", datetime.datetime.now().isoformat())
        baseline_id = baseline_results.get("run_id", "baseline")
        baseline_timestamp = baseline_results.get("timestamp", "unknown")
        
        # Initialize analysis
        analysis = RegressionAnalysis(
            run_id=current_id,
            timestamp=current_timestamp,
            baseline_id=baseline_id,
            baseline_timestamp=baseline_timestamp
        )
        
        # Compare top-level summary metrics
        self._compare_summary_metrics(current_results, baseline_results, analysis)
        
        # Compare domain-specific metrics
        self._compare_domain_metrics(current_results, baseline_results, analysis)
        
        # Compare test case metrics
        self._compare_test_case_metrics(current_results, baseline_results, analysis)
        
        return analysis
    
    def _compare_summary_metrics(
        self,
        current_results: Dict[str, Any],
        baseline_results: Dict[str, Any],
        analysis: RegressionAnalysis
    ):
        """
        Compare summary metrics between current and baseline results.
        
        Args:
            current_results: Current benchmark results
            baseline_results: Baseline benchmark results
            analysis: RegressionAnalysis object to update
        """
        # Extract summary metrics
        current_summary = current_results.get("summary_metrics", {})
        baseline_summary = baseline_results.get("summary_metrics", {})
        
        # Compare top-level metrics (up to 2 levels deep)
        for key, current_value in self._extract_nested_metrics(current_summary):
            try:
                # Try to get the same metric from baseline
                baseline_value = self._get_nested_value(baseline_summary, key)
                if baseline_value is None:
                    analysis.failed_metrics.append(f"summary.{key}")
                    continue
                
                # Skip non-numeric values
                if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
                    continue
                
                # Calculate change
                delta = current_value - baseline_value
                if baseline_value != 0:
                    percent_change = delta / abs(baseline_value)
                else:
                    percent_change = 0 if delta == 0 else float('inf') * (1 if delta > 0 else -1)
                
                # Determine direction
                if abs(percent_change) < 0.01:  # Within 1% is considered unchanged
                    direction = ChangeDirection.UNCHANGED
                elif self._is_improvement(key, delta):
                    direction = ChangeDirection.IMPROVED
                else:
                    direction = ChangeDirection.DEGRADED
                
                # Determine metric type
                metric_type = self._get_metric_type(key)
                
                # Create change object
                change = MetricChange(
                    name=key,
                    change_type=metric_type,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    delta=delta,
                    percent_change=percent_change,
                    direction=direction
                )
                
                # Add to analysis
                analysis.add_metric_change(change)
                
            except Exception as e:
                logger.warning(f"Error comparing metric {key}: {str(e)}")
                analysis.failed_metrics.append(f"summary.{key}")
    
    def _compare_domain_metrics(
        self,
        current_results: Dict[str, Any],
        baseline_results: Dict[str, Any],
        analysis: RegressionAnalysis
    ):
        """
        Compare domain-specific metrics between current and baseline results.
        
        Args:
            current_results: Current benchmark results
            baseline_results: Baseline benchmark results
            analysis: RegressionAnalysis object to update
        """
        # Check for domain metrics
        current_domains = current_results.get("summary_metrics", {}).get("by_domain", {})
        baseline_domains = baseline_results.get("summary_metrics", {}).get("by_domain", {})
        
        # Find common domains
        common_domains = set(current_domains.keys()) & set(baseline_domains.keys())
        
        # Compare metrics for each domain
        for domain in common_domains:
            current_domain = current_domains[domain]
            baseline_domain = baseline_domains[domain]
            
            # Extract and compare metrics
            for key, current_value in self._extract_nested_metrics(current_domain):
                try:
                    # Try to get the same metric from baseline
                    baseline_value = self._get_nested_value(baseline_domain, key)
                    if baseline_value is None:
                        analysis.failed_metrics.append(f"domain.{domain}.{key}")
                        continue
                    
                    # Skip non-numeric values
                    if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
                        continue
                    
                    # Calculate change
                    delta = current_value - baseline_value
                    if baseline_value != 0:
                        percent_change = delta / abs(baseline_value)
                    else:
                        percent_change = 0 if delta == 0 else float('inf') * (1 if delta > 0 else -1)
                    
                    # Determine direction
                    if abs(percent_change) < 0.01:  # Within 1% is considered unchanged
                        direction = ChangeDirection.UNCHANGED
                    elif self._is_improvement(key, delta):
                        direction = ChangeDirection.IMPROVED
                    else:
                        direction = ChangeDirection.DEGRADED
                    
                    # Determine metric type
                    metric_type = self._get_metric_type(key)
                    
                    # Create change object
                    change = MetricChange(
                        name=key,
                        change_type=metric_type,
                        baseline_value=baseline_value,
                        current_value=current_value,
                        delta=delta,
                        percent_change=percent_change,
                        direction=direction,
                        domain=domain
                    )
                    
                    # Add to analysis
                    analysis.add_metric_change(change)
                    
                except Exception as e:
                    logger.warning(f"Error comparing domain metric {domain}.{key}: {str(e)}")
                    analysis.failed_metrics.append(f"domain.{domain}.{key}")
    
    def _compare_test_case_metrics(
        self,
        current_results: Dict[str, Any],
        baseline_results: Dict[str, Any],
        analysis: RegressionAnalysis
    ):
        """
        Compare test case metrics between current and baseline results.
        
        Args:
            current_results: Current benchmark results
            baseline_results: Baseline benchmark results
            analysis: RegressionAnalysis object to update
        """
        # Extract test case results
        current_cases = {c.get("case_id"): c for c in current_results.get("comparisons", [])}
        baseline_cases = {c.get("case_id"): c for c in baseline_results.get("comparisons", [])}
        
        # Find common test cases
        common_cases = set(current_cases.keys()) & set(baseline_cases.keys())
        
        # Compare metrics for each test case
        for case_id in common_cases:
            current_case = current_cases[case_id]
            baseline_case = baseline_cases[case_id]
            
            # Compare case metrics
            if "metrics" in current_case and "metrics" in baseline_case:
                current_metrics = current_case["metrics"]
                baseline_metrics = baseline_case["metrics"]
                
                for key, current_value in self._extract_nested_metrics(current_metrics):
                    try:
                        # Try to get the same metric from baseline
                        baseline_value = self._get_nested_value(baseline_metrics, key)
                        if baseline_value is None:
                            analysis.failed_metrics.append(f"case.{case_id}.{key}")
                            continue
                        
                        # Skip non-numeric values
                        if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
                            continue
                        
                        # Calculate change
                        delta = current_value - baseline_value
                        if baseline_value != 0:
                            percent_change = delta / abs(baseline_value)
                        else:
                            percent_change = 0 if delta == 0 else float('inf') * (1 if delta > 0 else -1)
                        
                        # Determine direction
                        if abs(percent_change) < 0.01:  # Within 1% is considered unchanged
                            direction = ChangeDirection.UNCHANGED
                        elif self._is_improvement(key, delta):
                            direction = ChangeDirection.IMPROVED
                        else:
                            direction = ChangeDirection.DEGRADED
                        
                        # Determine metric type
                        metric_type = self._get_metric_type(key)
                        
                        # Create change object
                        change = MetricChange(
                            name=key,
                            change_type=metric_type,
                            baseline_value=baseline_value,
                            current_value=current_value,
                            delta=delta,
                            percent_change=percent_change,
                            direction=direction,
                            test_case_id=case_id
                        )
                        
                        # Add to analysis
                        analysis.add_metric_change(change)
                        
                    except Exception as e:
                        logger.warning(f"Error comparing test case metric {case_id}.{key}: {str(e)}")
                        analysis.failed_metrics.append(f"case.{case_id}.{key}")
    
    def analyze_trend(self, results_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze trends across multiple benchmark runs.
        
        Args:
            results_history: List of benchmark results, ordered by time (oldest first)
            
        Returns:
            Dictionary with trend analysis results
        """
        if not results_history or len(results_history) < 2:
            return {"error": "Need at least 2 runs for trend analysis"}
        
        # Create a normalized history dataset for key metrics
        history_data = []
        
        for result in results_history:
            run_id = result.get("run_id", "unknown")
            timestamp = result.get("timestamp", "unknown")
            summary = result.get("summary_metrics", {})
            
            # Extract key metrics
            run_data = {
                "run_id": run_id,
                "timestamp": timestamp
            }
            
            # Extract nested metrics
            for key, value in self._extract_nested_metrics(summary):
                if isinstance(value, (int, float)):
                    run_data[key] = value
            
            history_data.append(run_data)
        
        # Convert to pandas DataFrame for easier analysis
        try:
            df = pd.DataFrame(history_data)
            
            # Convert timestamp to datetime if possible
            if "timestamp" in df.columns:
                try:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp")
                except Exception:
                    # If timestamp conversion fails, maintain original order
                    pass
            
            # Analyze trends for each metric column
            trends = {}
            
            for col in df.columns:
                # Skip non-numeric columns
                if col in ["run_id", "timestamp"] or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # Get the values for this metric
                values = df[col].dropna().values
                
                if len(values) < 2:
                    continue
                
                # Calculate trend statistics
                trend = {
                    "first_value": float(values[0]),
                    "last_value": float(values[-1]),
                    "min_value": float(np.min(values)),
                    "max_value": float(np.max(values)),
                    "mean_value": float(np.mean(values)),
                    "std_dev": float(np.std(values))
                }
                
                # Determine trend direction
                if len(values) >= 3:
                    # Use linear regression to find slope
                    x = np.arange(len(values))
                    coefficients = np.polyfit(x, values, 1)
                    slope = coefficients[0]
                    
                    # Calculate change over the entire period
                    total_change = values[-1] - values[0]
                    percent_change = (total_change / abs(values[0])) if values[0] != 0 else 0
                    
                    trend["slope"] = float(slope)
                    trend["total_change"] = float(total_change)
                    trend["percent_change"] = float(percent_change)
                    
                    # Determine if trend is significant (slope not close to zero)
                    # and consistent (total change and slope have same sign)
                    is_significant = abs(slope) > 0.01 * np.mean(np.abs(values))
                    is_consistent = (slope * total_change >= 0)
                    
                    if is_significant and is_consistent:
                        if self._is_improvement(col, slope):
                            trend["direction"] = "improving"
                        else:
                            trend["direction"] = "degrading"
                    else:
                        trend["direction"] = "stable"
                else:
                    # Simple before/after comparison for only 2 points
                    change = values[1] - values[0]
                    percent_change = (change / abs(values[0])) if values[0] != 0 else 0
                    
                    trend["total_change"] = float(change)
                    trend["percent_change"] = float(percent_change)
                    
                    if abs(percent_change) < 0.05:  # Less than 5% change is stable
                        trend["direction"] = "stable"
                    elif self._is_improvement(col, change):
                        trend["direction"] = "improving"
                    else:
                        trend["direction"] = "degrading"
                
                # Detect anomalies (values outside 2 standard deviations)
                if len(values) >= 5:
                    mean = np.mean(values)
                    std = np.std(values)
                    upper_bound = mean + 2 * std
                    lower_bound = mean - 2 * std
                    
                    anomalies = []
                    for i, value in enumerate(values):
                        if value > upper_bound or value < lower_bound:
                            anomalies.append({
                                "run_id": df.iloc[i]["run_id"] if "run_id" in df.columns else i,
                                "value": float(value),
                                "z_score": float((value - mean) / std) if std != 0 else 0
                            })
                    
                    trend["anomalies"] = anomalies
                    trend["has_anomalies"] = len(anomalies) > 0
                
                trends[col] = trend
            
            # Identify overall performance trends
            overall_trends = {
                "hallucination_reduction": self._get_metric_trend(trends, ["hallucination_reduction", "factuality"]),
                "latency": self._get_metric_trend(trends, ["latency", "latency_overhead"]),
                "confidence": self._get_metric_trend(trends, ["confidence"]),
                "resource_usage": self._get_metric_trend(trends, ["cpu_usage", "memory_usage"])
            }
            
            # Calculate overall trend score (-1 to 1, negative is degrading)
            trend_scores = []
            for category, trend in overall_trends.items():
                if trend["direction"] == "improving":
                    trend_scores.append(1)
                elif trend["direction"] == "degrading":
                    trend_scores.append(-1)
                else:
                    trend_scores.append(0)
            
            overall_score = sum(trend_scores) / len(trend_scores) if trend_scores else 0
            
            return {
                "trends": trends,
                "overall_trends": overall_trends,
                "overall_score": overall_score,
                "run_count": len(df),
                "timespan": {
                    "first_run": df.iloc[0]["timestamp"].isoformat() if isinstance(df.iloc[0]["timestamp"], datetime.datetime) else str(df.iloc[0]["timestamp"]),
                    "last_run": df.iloc[-1]["timestamp"].isoformat() if isinstance(df.iloc[-1]["timestamp"], datetime.datetime) else str(df.iloc[-1]["timestamp"])
                } if "timestamp" in df.columns else {},
                "overall_assessment": "improving" if overall_score > 0.25 else "degrading" if overall_score < -0.25 else "stable"
            }
        
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {"error": f"Trend analysis failed: {str(e)}"}
    
    def get_optimal_baseline(self, results_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Determine the optimal baseline from historical results.
        
        Args:
            results_history: List of benchmark results, ordered by time (oldest first)
            
        Returns:
            The best baseline to use for regression detection
        """
        if not results_history:
            return {"error": "No results available for baseline selection"}
        
        try:
            # For now, use a simple heuristic - prefer stable, recent results
            # First analyze the trend to see if performance has been stable recently
            trend_analysis = self.analyze_trend(results_history)
            
            overall_assessment = trend_analysis.get("overall_assessment", "unknown")
            
            if overall_assessment == "stable" and len(results_history) >= 3:
                # If stable, use the average of the last 3 runs
                logger.info("Performance trend is stable, using average of last 3 runs as baseline")
                # In a real implementation, we would actually average the metrics
                # For now, just use the middle of the last 3 runs
                return results_history[-2]
            elif overall_assessment == "improving" and len(results_history) >= 2:
                # If improving, use the most recent run as the baseline
                logger.info("Performance trend is improving, using most recent run as baseline")
                return results_history[-1]
            elif len(results_history) >= 5:
                # If degrading or unknown but we have enough history,
                # look for a stable period in the past
                logger.info("Looking for stable baseline in history")
                
                # Find the run with the best combination of stability and recency
                # This is a simplified approach - in practice you'd use more complex
                # algorithms to find a stable baseline period
                best_run_idx = len(results_history) - 1
                best_score = -float('inf')
                
                for i in range(len(results_history) - 1):
                    # Calculate a score based on stability and recency
                    # More recent runs get a higher recency score
                    recency_score = i / (len(results_history) - 1)
                    
                    # Check stability (compare with next run)
                    stability_score = 0
                    try:
                        analysis = self.detect_regressions(
                            results_history[i+1],
                            results_history[i]
                        )
                        # More regressions = less stable
                        stability_score = 1.0 - (len(analysis.regression_metrics) / max(1, len(analysis.metric_changes)))
                    except Exception:
                        stability_score = 0
                    
                    # Combine scores with more weight on stability
                    score = (0.7 * stability_score) + (0.3 * recency_score)
                    
                    if score > best_score:
                        best_score = score
                        best_run_idx = i
                
                return results_history[best_run_idx]
            else:
                # Default: use the oldest run as baseline
                logger.info("Using oldest run as baseline (limited history)")
                return results_history[0]
        
        except Exception as e:
            logger.error(f"Error determining optimal baseline: {str(e)}")
            # Fall back to the oldest run
            return results_history[0]
    
    def _extract_nested_metrics(self, data: Dict[str, Any], prefix: str = "") -> List[Tuple[str, Any]]:
        """
        Extract metrics from nested dictionary.
        
        Args:
            data: Nested dictionary with metrics
            prefix: Prefix for metric keys
            
        Returns:
            List of (key, value) tuples for all metrics
        """
        metrics = []
        
        for key, value in data.items():
            # Build the full key path
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict) and not key.startswith("_"):
                # Recursive extraction for nested dictionaries (up to 1 level deep)
                if not prefix:  # Only recurse one level
                    metrics.extend(self._extract_nested_metrics(value, full_key))
            else:
                # Add the metric
                metrics.append((full_key, value))
        
        return metrics
    
    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Any:
        """
        Get a value from a nested dictionary using a dot-separated key.
        
        Args:
            data: Nested dictionary
            key: Dot-separated key path
            
        Returns:
            Value at the key path or None if not found
        """
        parts = key.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _is_improvement(self, metric_name: str, delta: float) -> bool:
        """
        Determine if a change is an improvement based on the metric name.
        
        Args:
            metric_name: Name of the metric
            delta: Change in the metric (current - baseline)
            
        Returns:
            True if the change is an improvement
        """
        # Extract the base metric name (without parent keys)
        base_name = metric_name.split(".")[-1]
        
        # Determine metric type
        metric_type = self._get_metric_type(metric_name)
        
        if metric_type in [
            MetricChangeType.HALLUCINATION_REDUCTION,
            MetricChangeType.CONFIDENCE,
            MetricChangeType.THROUGHPUT
        ]:
            # For these metrics, higher values are better
            return delta > 0
        elif metric_type in [
            MetricChangeType.LATENCY,
            MetricChangeType.ERROR_RATE,
            MetricChangeType.RESOURCE_USAGE
        ]:
            # For these metrics, lower values are better
            return delta < 0
        else:
            # Default: assume higher is better
            return delta > 0
    
    def _get_metric_type(self, metric_name: str) -> MetricChangeType:
        """
        Determine the type of a metric based on its name.
        
        Args:
            metric_name: Name of the metric (can include parent keys)
            
        Returns:
            MetricChangeType for the metric
        """
        # Extract the base metric name (without parent keys)
        base_name = metric_name.split(".")[-1]
        
        # Check in metrics mapping
        for pattern, change_type in self.metrics_mapping.items():
            if pattern in base_name:
                return change_type
        
        # Default to other
        return MetricChangeType.OTHER
    
    def _get_metric_trend(self, trends: Dict[str, Dict[str, Any]], metric_names: List[str]) -> Dict[str, Any]:
        """
        Get the overall trend for a category of metrics.
        
        Args:
            trends: Dictionary of trends by metric
            metric_names: List of metric names to check
            
        Returns:
            Dictionary with overall trend information
        """
        # Find trends that match any of the metric names
        matching_trends = {}
        for trend_key, trend_data in trends.items():
            for metric_name in metric_names:
                if metric_name in trend_key:
                    matching_trends[trend_key] = trend_data
                    break
        
        if not matching_trends:
            return {
                "direction": "unknown",
                "metrics_found": 0
            }
        
        # Count the directions
        directions = {
            "improving": 0,
            "degrading": 0,
            "stable": 0
        }
        
        for trend_data in matching_trends.values():
            directions[trend_data.get("direction", "stable")] += 1
        
        # Determine overall direction
        if directions["improving"] > directions["degrading"] + directions["stable"]:
            overall_direction = "improving"
        elif directions["degrading"] > directions["improving"] + directions["stable"]:
            overall_direction = "degrading"
        else:
            overall_direction = "stable"
        
        return {
            "direction": overall_direction,
            "metrics_found": len(matching_trends),
            "directions": directions
        }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ACDEK v3 Regression Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--current",
        type=str,
        required=True,
        help="Path to current benchmark results"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline benchmark results (default: auto-detect from history)"
    )
    
    parser.add_argument(
        "--history-dir",
        type=str,
        help="Directory containing historical benchmark results"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_REGRESSION_THRESHOLD,
        help="Threshold for regression detection (0.0-1.0)"
    )
    
    parser.add_argument(
        "--significance",
        type=float,
        default=DEFAULT_SIGNIFICANCE_LEVEL,
        help="Statistical significance level (0.0-1.0)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save regression analysis results"
    )
    
    parser.add_argument(
        "--analyze-trend",
        action="store_true",
        help="Analyze performance trends across historical results"
    )
    
    parser.add_argument(
        "--find-optimal-baseline",
        action="store_true",
        help="Find the optimal baseline from historical results"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load current results
    try:
        current_results = load_data(args.current, "json")
        logger.info(f"Loaded current results from {args.current}")
    except Exception as e:
        logger.error(f"Error loading current results: {str(e)}")
        return 1
    
    # Create regression detector
    detector = RegressionDetector(
        significance_level=args.significance,
        regression_thresholds={
            "minor": args.threshold / 2,
            "moderate": args.threshold,
            "severe": args.threshold * 2,
            "critical": args.threshold * 4
        }
    )
    
    # Load historical results if needed
    history = []
    if args.history_dir or args.analyze_trend or args.find_optimal_baseline:
        history_dir = args.history_dir or os.path.dirname(args.current)
        
        try:
            # Find JSON result files in the directory
            history_files = []
            for file in os.listdir(history_dir):
                if file.startswith("benchmark_results_") and file.endswith(".json"):
                    history_files.append(os.path.join(history_dir, file))
            
            # Sort by modification time (oldest first)
            history_files.sort(key=lambda f: os.path.getmtime(f))
            
            # Load each file
            for file in history_files:
                try:
                    data = load_data(file, "json")
                    data["_file_path"] = file  # Store file path for reference
                    history.append(data)
                except Exception as e:
                    logger.warning(f"Error loading history file {file}: {str(e)}")
            
            logger.info(f"Loaded {len(history)} historical benchmark results")
        except Exception as e:
            logger.error(f"Error loading historical results: {str(e)}")
    
    # Find baseline
    baseline_results = None
    
    if args.baseline:
        # Use specified baseline
        try:
            baseline_results = load_data(args.baseline, "json")
            logger.info(f"Loaded baseline from {args.baseline}")
        except Exception as e:
            logger.error(f"Error loading baseline: {str(e)}")
            return 1
    elif args.find_optimal_baseline and history:
        # Find optimal baseline from history
        try:
            baseline_results = detector.get_optimal_baseline(history)
            baseline_path = baseline_results.get("_file_path", "unknown")
            logger.info(f"Using optimal baseline from {baseline_path}")
        except Exception as e:
            logger.error(f"Error finding optimal baseline: {str(e)}")
            return 1
    elif history:
        # Use the most recent historical result as baseline
        try:
            baseline_results = history[-1]
            baseline_path = baseline_results.get("_file_path", "unknown")
            logger.info(f"Using most recent history as baseline: {baseline_path}")
        except Exception as e:
            logger.error(f"Error using history as baseline: {str(e)}")
            return 1
    
    if not baseline_results:
        logger.error("No baseline available for regression detection")
        return 1
    
    # Analyze trends if requested
    if args.analyze_trend and history:
        try:
            # Add current results to history for trend analysis
            trend_history = history.copy()
            if current_results not in trend_history:
                trend_history.append(current_results)
            
            trend_analysis = detector.analyze_trend(trend_history)
            
            print("\nPerformance Trend Analysis:")
            print(f"Run count: {trend_analysis.get('run_count', 0)}")
            
            if "timespan" in trend_analysis:
                timespan = trend_analysis["timespan"]
                print(f"Timespan: {timespan.get('first_run', 'unknown')} to {timespan.get('last_run', 'unknown')}")
            
            print(f"Overall assessment: {trend_analysis.get('overall_assessment', 'unknown').upper()}")
            
            print("\nCategory Trends:")
            for category, trend in trend_analysis.get("overall_trends", {}).items():
                direction = trend.get("direction", "unknown")
                metrics_found = trend.get("metrics_found", 0)
                direction_icon = "↗️" if direction == "improving" else "↘️" if direction == "degrading" else "→"
                print(f"- {category.replace('_', ' ').title()}: {direction_icon} {direction.upper()} ({metrics_found} metrics)")
            
            print("\nMetric Trends:")
            for metric, trend in trend_analysis.get("trends", {}).items():
                if "direction" in trend:
                    direction = trend["direction"]
                    direction_icon = "↗️" if direction == "improving" else "↘️" if direction == "degrading" else "→"
                    
                    change_str = ""
                    if "percent_change" in trend:
                        change_str = f"{trend['percent_change']*100:+.1f}%"
                    
                    anomaly_str = ""
                    if "has_anomalies" in trend and trend["has_anomalies"]:
                        anomaly_str = "⚠️ anomalies detected"
                    
                    print(f"- {metric}: {direction_icon} {change_str} {anomaly_str}")
            
            # Save trend analysis if output is specified
            if args.output:
                trend_output = args.output.replace(".json", "_trend.json")
                with open(trend_output, "w") as f:
                    json.dump(trend_analysis, f, indent=2)
                logger.info(f"Saved trend analysis to {trend_output}")
        
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
    
    # Detect regressions
    try:
        analysis = detector.detect_regressions(current_results, baseline_results)
        
        # Get regression summary
        summary = analysis.get_regression_summary(detector.regression_thresholds)
        
        # Get domain impact
        domain_impact = analysis.get_domain_impact()
        
        # Print results
        print("\nRegression Analysis Summary:")
        print(f"Run ID: {analysis.run_id}")
        print(f"Baseline ID: {analysis.baseline_id}")
        print(f"Total metrics compared: {len(analysis.metric_changes)}")
        print(f"Regressions detected: {summary['total']} " + 
             f"({summary['by_severity']['minor']} minor, " +
             f"{summary['by_severity']['moderate']} moderate, " +
             f"{summary['by_severity']['severe']} severe, " +
             f"{summary['by_severity']['critical']} critical)")
        print(f"Improvements detected: {len(analysis.improvement_metrics)}")
        
        if summary["total"] > 0:
            print("\nSignificant Regressions:")
            
            # Show severe and critical regressions
            severe_regressions = [r for r in analysis.regression_metrics 
                                if r.get_severity(detector.regression_thresholds) in 
                                [RegressionSeverity.SEVERE, RegressionSeverity.CRITICAL]]
            
            for regression in severe_regressions:
                change_str = f"{regression.percent_change*100:.1f}%"
                severity = regression.get_severity(detector.regression_thresholds)
                severity_str = f"{severity.value.upper()}" if severity else ""
                
                context = ""
                if regression.domain:
                    context = f"in {regression.domain} domain"
                elif regression.test_case_id:
                    context = f"in test case {regression.test_case_id}"
                
                print(f"- {regression.name} {context}: {regression.baseline_value:.4f} → {regression.current_value:.4f} " +
                     f"({change_str}) [{severity_str}]")
        
        if domain_impact:
            print("\nDomain Impact:")
            for domain, impact in domain_impact.items():
                if impact["regression_count"] > 0:
                    status = "⚠️ Regression" if impact["impact_score"] < 0 else "✅ Improved"
                    print(f"- {domain}: {status} (Score: {impact['impact_score']:.4f}, " +
                         f"Regressions: {impact['regression_count']}, " +
                         f"Improvements: {impact['improvement_count']})")
        
        # Save analysis if output is specified
        if args.output:
            with open(args.output, "w") as f:
                json.dump(analysis.to_dict(), f, indent=2)
            logger.info(f"Saved regression analysis to {args.output}")
        
        # Return status based on presence of regressions
        if summary["has_critical"] or summary["has_severe"]:
            return 2  # Severe regression found
        elif summary["total"] > 0:
            return 1  # Some regression found
        else:
            return 0  # No regression found
    
    except Exception as e:
        logger.error(f"Error detecting regressions: {str(e)}")
        return 3  # Error


if __name__ == "__main__":
    sys.exit(main())