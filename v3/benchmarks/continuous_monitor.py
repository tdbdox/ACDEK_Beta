#!/usr/bin/env python
"""
ACDEK v3 Continuous Performance Monitor

This daemon process continuously monitors ACDEK performance, detects regressions,
and sends alerts when performance issues are detected.

The monitor supports:
1. Continuous benchmark execution with configurable intervals
2. Real-time metrics collection and storage
3. Regression detection against baseline performance
4. Configurable alerts via multiple channels (console, email, webhook)
5. Dashboard data export for visualization

Usage:
    python continuous_monitor.py --config monitor_config.json
    python continuous_monitor.py --interval 3600 --baseline baseline.json
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import signal
import queue
import uuid
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable

# Add parent directory to path to allow importing from project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import relevant components
from benchmarks.benchmark_run import (
    generate_test_cases,
    filter_test_cases,
    generate_comparison_report,
    track_resource_usage
)
from benchmarks.hallucination_benchmark import (
    BenchmarkConfig,
    run_benchmark,
    TestCase,
    Domain
)
from observability.performance_profiler import PerformanceProfiler
from benchmarks.threshold_config import ThresholdConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("continuous_monitor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_INTERVAL = 3600  # 1 hour between test runs
DEFAULT_OUTPUT_DIR = Path("./benchmarks/monitor_results")
DEFAULT_METRICS_RETENTION = 30  # days
DEFAULT_ALERT_THRESHOLD = 0.05  # 5% regression triggers an alert
MAX_QUEUE_SIZE = 1000  # Maximum size for metrics queue


class AlertLevel:
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel:
    """Supported alert channels."""
    CONSOLE = "console"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    FILE = "file"


class PerformanceMetrics:
    """Container for performance metrics."""
    
    def __init__(self, run_id: str, timestamp: str):
        """
        Initialize performance metrics.
        
        Args:
            run_id: Unique identifier for the benchmark run
            timestamp: ISO format timestamp
        """
        self.run_id = run_id
        self.timestamp = timestamp
        self.metrics = {}
        self.resource_metrics = {}
        self.domain_metrics = {}
        self.test_case_metrics = {}
        self.regression_metrics = {}
        self.metadata = {}
    
    def add_metric(self, name: str, value: float, domain: Optional[str] = None, case_id: Optional[str] = None):
        """Add a metric value."""
        if domain:
            if domain not in self.domain_metrics:
                self.domain_metrics[domain] = {}
            self.domain_metrics[domain][name] = value
        elif case_id:
            if case_id not in self.test_case_metrics:
                self.test_case_metrics[case_id] = {}
            self.test_case_metrics[case_id][name] = value
        else:
            self.metrics[name] = value
    
    def add_resource_metric(self, name: str, value: float):
        """Add a resource usage metric."""
        self.resource_metrics[name] = value
    
    def add_regression_metric(self, name: str, current: float, baseline: float, delta: float, is_regression: bool):
        """Add a regression comparison metric."""
        self.regression_metrics[name] = {
            "current": current,
            "baseline": baseline,
            "delta": delta,
            "is_regression": is_regression
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "resource_metrics": self.resource_metrics,
            "domain_metrics": self.domain_metrics,
            "test_case_metrics": self.test_case_metrics,
            "regression_metrics": self.regression_metrics,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create PerformanceMetrics from dictionary."""
        metrics = cls(data["run_id"], data["timestamp"])
        metrics.metrics = data.get("metrics", {})
        metrics.resource_metrics = data.get("resource_metrics", {})
        metrics.domain_metrics = data.get("domain_metrics", {})
        metrics.test_case_metrics = data.get("test_case_metrics", {})
        metrics.regression_metrics = data.get("regression_metrics", {})
        metrics.metadata = data.get("metadata", {})
        return metrics


class Alert:
    """Performance alert definition."""
    
    def __init__(
        self,
        level: str,
        message: str,
        metrics: Dict[str, Any],
        source: str,
        timestamp: Optional[str] = None
    ):
        """
        Initialize an alert.
        
        Args:
            level: Alert level (info, warning, error, critical)
            message: Alert message
            metrics: Relevant metrics for the alert
            source: Source of the alert (e.g., "regression_detector")
            timestamp: ISO format timestamp (defaults to now)
        """
        self.level = level
        self.message = message
        self.metrics = metrics
        self.source = source
        self.timestamp = timestamp or datetime.datetime.now().isoformat()
        self.alert_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "alert_id": self.alert_id,
            "level": self.level,
            "message": self.message,
            "metrics": self.metrics,
            "source": self.source,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create Alert from dictionary."""
        alert = cls(
            level=data["level"],
            message=data["message"],
            metrics=data["metrics"],
            source=data["source"],
            timestamp=data["timestamp"]
        )
        alert.alert_id = data.get("alert_id", str(uuid.uuid4()))
        return alert


class MetricsStorage:
    """Storage for performance metrics and alerts."""
    
    def __init__(self, base_dir: str, retention_days: int = DEFAULT_METRICS_RETENTION):
        """
        Initialize metrics storage.
        
        Args:
            base_dir: Base directory for metrics storage
            retention_days: Number of days to retain metrics data
        """
        self.base_dir = Path(base_dir)
        self.metrics_dir = self.base_dir / "metrics"
        self.alerts_dir = self.base_dir / "alerts"
        self.retention_days = retention_days
        
        # Create directories
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize in-memory cache for recent metrics
        self.metrics_cache = {}
        self.alerts_cache = []
        self.max_cache_size = 100
    
    def store_metrics(self, metrics: PerformanceMetrics) -> str:
        """
        Store performance metrics.
        
        Args:
            metrics: Performance metrics to store
            
        Returns:
            Path to the stored metrics file
        """
        # Create date-based directory structure
        date_str = metrics.timestamp.split("T")[0]  # Extract date part from ISO timestamp
        metrics_date_dir = self.metrics_dir / date_str
        metrics_date_dir.mkdir(exist_ok=True)
        
        # Create metrics file
        metrics_file = metrics_date_dir / f"{metrics.run_id}.json"
        
        # Write metrics to file
        with open(metrics_file, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        # Add to cache
        self.metrics_cache[metrics.run_id] = metrics
        
        # Prune cache if needed
        if len(self.metrics_cache) > self.max_cache_size:
            # Remove oldest items
            oldest_keys = sorted(self.metrics_cache.keys(), 
                                key=lambda k: self.metrics_cache[k].timestamp)[:len(self.metrics_cache) - self.max_cache_size]
            for key in oldest_keys:
                del self.metrics_cache[key]
        
        return str(metrics_file)
    
    def store_alert(self, alert: Alert) -> str:
        """
        Store an alert.
        
        Args:
            alert: Alert to store
            
        Returns:
            Path to the stored alert file
        """
        # Create date-based directory structure
        date_str = alert.timestamp.split("T")[0]  # Extract date part from ISO timestamp
        alert_date_dir = self.alerts_dir / date_str
        alert_date_dir.mkdir(exist_ok=True)
        
        # Create alert file
        alert_file = alert_date_dir / f"{alert.alert_id}.json"
        
        # Write alert to file
        with open(alert_file, "w") as f:
            json.dump(alert.to_dict(), f, indent=2)
        
        # Add to cache
        self.alerts_cache.append(alert)
        
        # Prune cache if needed
        if len(self.alerts_cache) > self.max_cache_size:
            self.alerts_cache = self.alerts_cache[-self.max_cache_size:]
        
        return str(alert_file)
    
    def get_metrics(self, run_id: str) -> Optional[PerformanceMetrics]:
        """
        Retrieve metrics by run ID.
        
        Args:
            run_id: Unique identifier for the benchmark run
            
        Returns:
            PerformanceMetrics object or None if not found
        """
        # Check cache first
        if run_id in self.metrics_cache:
            return self.metrics_cache[run_id]
        
        # Search in storage
        for date_dir in self.metrics_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            metrics_file = date_dir / f"{run_id}.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        data = json.load(f)
                    metrics = PerformanceMetrics.from_dict(data)
                    
                    # Add to cache
                    self.metrics_cache[run_id] = metrics
                    
                    return metrics
                except Exception as e:
                    logger.error(f"Error loading metrics file {metrics_file}: {e}")
        
        return None
    
    def get_recent_metrics(self, count: int = 10) -> List[PerformanceMetrics]:
        """
        Get most recent metrics.
        
        Args:
            count: Number of recent metrics to retrieve
            
        Returns:
            List of PerformanceMetrics, most recent first
        """
        all_metrics = []
        
        # First collect all metrics files
        metrics_files = []
        for date_dir in self.metrics_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            for metrics_file in date_dir.iterdir():
                if metrics_file.suffix == ".json":
                    metrics_files.append(metrics_file)
        
        # Sort by modification time (most recent first)
        metrics_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Load the most recent files
        for metrics_file in metrics_files[:count]:
            try:
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                metrics = PerformanceMetrics.from_dict(data)
                all_metrics.append(metrics)
                
                # Add to cache
                self.metrics_cache[metrics.run_id] = metrics
            except Exception as e:
                logger.error(f"Error loading metrics file {metrics_file}: {e}")
        
        return all_metrics
    
    def get_recent_alerts(self, count: int = 10, level: Optional[str] = None) -> List[Alert]:
        """
        Get most recent alerts.
        
        Args:
            count: Number of recent alerts to retrieve
            level: Filter by alert level
            
        Returns:
            List of Alert objects, most recent first
        """
        all_alerts = []
        
        # First collect all alert files
        alert_files = []
        for date_dir in self.alerts_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            for alert_file in date_dir.iterdir():
                if alert_file.suffix == ".json":
                    alert_files.append(alert_file)
        
        # Sort by modification time (most recent first)
        alert_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Load the most recent files
        for alert_file in alert_files:
            try:
                with open(alert_file, "r") as f:
                    data = json.load(f)
                alert = Alert.from_dict(data)
                
                # Filter by level if specified
                if level and alert.level != level:
                    continue
                
                all_alerts.append(alert)
                
                # Stop when we have enough
                if len(all_alerts) >= count:
                    break
            except Exception as e:
                logger.error(f"Error loading alert file {alert_file}: {e}")
        
        return all_alerts
    
    def cleanup_old_data(self):
        """Delete data older than retention period."""
        retention_cutoff = datetime.datetime.now() - datetime.timedelta(days=self.retention_days)
        
        # Clean up metrics
        for date_dir in self.metrics_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            try:
                # Parse date from directory name
                dir_date = datetime.datetime.strptime(date_dir.name, "%Y-%m-%d").date()
                dir_datetime = datetime.datetime.combine(dir_date, datetime.time.min)
                
                # Check if older than retention period
                if dir_datetime < retention_cutoff:
                    logger.info(f"Removing old metrics: {date_dir}")
                    # Remove directory and contents
                    for file in date_dir.iterdir():
                        file.unlink()
                    date_dir.rmdir()
            except Exception as e:
                logger.warning(f"Error cleaning up directory {date_dir}: {e}")
        
        # Clean up alerts
        for date_dir in self.alerts_dir.iterdir():
            if not date_dir.is_dir():
                continue
            
            try:
                # Parse date from directory name
                dir_date = datetime.datetime.strptime(date_dir.name, "%Y-%m-%d").date()
                dir_datetime = datetime.datetime.combine(dir_date, datetime.time.min)
                
                # Check if older than retention period
                if dir_datetime < retention_cutoff:
                    logger.info(f"Removing old alerts: {date_dir}")
                    # Remove directory and contents
                    for file in date_dir.iterdir():
                        file.unlink()
                    date_dir.rmdir()
            except Exception as e:
                logger.warning(f"Error cleaning up directory {date_dir}: {e}")


class AlertManager:
    """Manages alert generation and notification."""
    
    def __init__(
        self,
        storage: MetricsStorage,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize alert manager.
        
        Args:
            storage: Metrics storage instance
            config: Alert configuration dictionary
        """
        self.storage = storage
        self.config = config or {
            "channels": [{"type": AlertChannel.CONSOLE}],
            "thresholds": {
                "regression": {
                    "warning": 0.05,  # 5% regression
                    "error": 0.1,     # 10% regression
                    "critical": 0.2   # 20% regression
                },
                "latency": {
                    "warning": 1.05,  # 5% increase
                    "error": 1.1,     # 10% increase
                    "critical": 1.2   # 20% increase
                }
            },
            "alert_backoff": 3600,  # 1 hour between similar alerts
        }
        
        # Initialize alert tracking
        self.recent_alerts = {}  # Maps alert signature to timestamp
    
    def should_send_alert(self, source: str, metrics: Dict[str, Any]) -> bool:
        """
        Check if an alert should be sent based on backoff settings.
        
        Args:
            source: Alert source
            metrics: Alert metrics
            
        Returns:
            True if alert should be sent
        """
        # Create a signature for this alert
        signature = f"{source}:{json.dumps(metrics, sort_keys=True)}"
        
        # Check if a similar alert was sent recently
        now = time.time()
        if signature in self.recent_alerts:
            last_alert_time = self.recent_alerts[signature]
            backoff_seconds = self.config.get("alert_backoff", 3600)
            
            if now - last_alert_time < backoff_seconds:
                return False
        
        # Update last alert time
        self.recent_alerts[signature] = now
        
        # Clean up old alerts
        old_signatures = []
        for sig, timestamp in self.recent_alerts.items():
            if now - timestamp > self.config.get("alert_backoff", 3600) * 2:
                old_signatures.append(sig)
        
        for sig in old_signatures:
            del self.recent_alerts[sig]
        
        return True
    
    def send_alert(self, alert: Alert):
        """
        Send an alert through configured channels.
        
        Args:
            alert: Alert to send
        """
        # Check alert backoff
        if not self.should_send_alert(alert.source, alert.metrics):
            logger.debug(f"Suppressing similar alert: {alert.message} (level: {alert.level})")
            return
        
        # Store the alert
        self.storage.store_alert(alert)
        
        # Send through each configured channel
        channels = self.config.get("channels", [{"type": AlertChannel.CONSOLE}])
        
        for channel in channels:
            channel_type = channel.get("type", AlertChannel.CONSOLE)
            
            try:
                if channel_type == AlertChannel.CONSOLE:
                    self._send_console_alert(alert, channel)
                elif channel_type == AlertChannel.EMAIL:
                    self._send_email_alert(alert, channel)
                elif channel_type == AlertChannel.WEBHOOK:
                    self._send_webhook_alert(alert, channel)
                elif channel_type == AlertChannel.SLACK:
                    self._send_slack_alert(alert, channel)
                elif channel_type == AlertChannel.FILE:
                    self._send_file_alert(alert, channel)
                else:
                    logger.warning(f"Unknown alert channel type: {channel_type}")
            except Exception as e:
                logger.error(f"Error sending alert through {channel_type} channel: {e}")
    
    def _send_console_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send alert to console."""
        level_prefix = {
            AlertLevel.INFO: "ℹ️ INFO",
            AlertLevel.WARNING: "⚠️ WARNING",
            AlertLevel.ERROR: "❌ ERROR",
            AlertLevel.CRITICAL: "🔥 CRITICAL"
        }.get(alert.level, "INFO")
        
        print(f"\n{level_prefix}: {alert.message}")
        print(f"Time: {alert.timestamp}")
        print(f"Source: {alert.source}")
        print("Metrics:")
        for k, v in alert.metrics.items():
            print(f"  {k}: {v}")
        print("")
    
    def _send_email_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send email alert."""
        # Email alert implementation would go here
        # This would use SMTP or similar to send an email
        recipients = config.get("recipients", [])
        subject = f"ACDEK Monitor {alert.level.upper()}: {alert.message[:50]}"
        
        logger.info(f"Would send email to {recipients} with subject: {subject}")
    
    def _send_webhook_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send webhook alert."""
        # Webhook implementation would go here
        # This would POST to a webhook URL
        webhook_url = config.get("url", "")
        
        logger.info(f"Would send webhook to {webhook_url}")
    
    def _send_slack_alert(self, alert: Alert, config: Dict[str, Any]):
        """Send Slack alert."""
        # Slack implementation would go here
        webhook_url = config.get("webhook_url", "")
        
        logger.info(f"Would send Slack message via webhook {webhook_url}")
    
    def _send_file_alert(self, alert: Alert, config: Dict[str, Any]):
        """Append alert to a file."""
        file_path = config.get("file_path", "alerts.log")
        
        level_prefix = {
            AlertLevel.INFO: "INFO",
            AlertLevel.WARNING: "WARNING",
            AlertLevel.ERROR: "ERROR",
            AlertLevel.CRITICAL: "CRITICAL"
        }.get(alert.level, "INFO")
        
        with open(file_path, "a") as f:
            f.write(f"{alert.timestamp} - {level_prefix}: {alert.message}\n")
            f.write(f"Source: {alert.source}\n")
            f.write(f"Metrics: {json.dumps(alert.metrics)}\n\n")
    
    def create_regression_alert(
        self,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        delta: float,
        domain: Optional[str] = None,
        case_id: Optional[str] = None
    ):
        """
        Create and send a regression alert.
        
        Args:
            metric_name: Name of the metric showing regression
            current_value: Current metric value
            baseline_value: Baseline metric value
            delta: Difference (current - baseline)
            domain: Optional domain name (for domain-specific metrics)
            case_id: Optional test case ID (for case-specific metrics)
        """
        # Determine alert level based on regression magnitude
        thresholds = self.config.get("thresholds", {}).get("regression", {})
        
        abs_delta = abs(delta)
        delta_percent = abs_delta / max(0.0001, abs(baseline_value))
        
        if delta_percent >= thresholds.get("critical", 0.2):
            level = AlertLevel.CRITICAL
        elif delta_percent >= thresholds.get("error", 0.1):
            level = AlertLevel.ERROR
        elif delta_percent >= thresholds.get("warning", 0.05):
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO
        
        # Create message
        if domain:
            context = f"in {domain} domain"
        elif case_id:
            context = f"in test case {case_id}"
        else:
            context = "overall"
        
        message = f"Performance regression detected for {metric_name} {context}: " + \
                  f"{baseline_value:.4f} -> {current_value:.4f} ({delta:+.4f}, {delta_percent:.1%})"
        
        # Create metrics for the alert
        metrics = {
            "metric_name": metric_name,
            "current_value": current_value,
            "baseline_value": baseline_value,
            "delta": delta,
            "delta_percent": delta_percent,
            "domain": domain,
            "case_id": case_id
        }
        
        # Create and send alert
        alert = Alert(
            level=level,
            message=message,
            metrics=metrics,
            source="regression_detector"
        )
        
        self.send_alert(alert)


class RegressionDetector:
    """Detects performance regressions against baseline."""
    
    def __init__(
        self,
        storage: MetricsStorage,
        alert_manager: AlertManager,
        baseline_path: Optional[str] = None,
        threshold: float = DEFAULT_ALERT_THRESHOLD
    ):
        """
        Initialize regression detector.
        
        Args:
            storage: Metrics storage instance
            alert_manager: Alert manager instance
            baseline_path: Path to baseline metrics file
            threshold: Regression threshold (0.0-1.0)
        """
        self.storage = storage
        self.alert_manager = alert_manager
        self.threshold = threshold
        self.baseline = None
        
        # Load baseline if provided
        if baseline_path:
            self.load_baseline(baseline_path)
    
    def load_baseline(self, baseline_path: str):
        """
        Load baseline metrics from file.
        
        Args:
            baseline_path: Path to baseline metrics file
        """
        try:
            if os.path.exists(baseline_path):
                with open(baseline_path, "r") as f:
                    data = json.load(f)
                
                if "run_id" in data and "timestamp" in data:
                    self.baseline = PerformanceMetrics.from_dict(data)
                    logger.info(f"Loaded baseline from {baseline_path} (run_id: {self.baseline.run_id})")
                else:
                    logger.warning(f"Invalid baseline file format: {baseline_path}")
            else:
                logger.warning(f"Baseline file not found: {baseline_path}")
        except Exception as e:
            logger.error(f"Error loading baseline from {baseline_path}: {e}")
    
    def set_baseline_from_metrics(self, metrics: PerformanceMetrics):
        """
        Set baseline from metrics object.
        
        Args:
            metrics: PerformanceMetrics object to use as baseline
        """
        self.baseline = metrics
        logger.info(f"Set new baseline from run_id: {metrics.run_id}")
    
    def save_baseline(self, output_path: str):
        """
        Save current baseline to file.
        
        Args:
            output_path: Path to save baseline metrics
        """
        if not self.baseline:
            logger.warning("No baseline to save")
            return
        
        try:
            with open(output_path, "w") as f:
                json.dump(self.baseline.to_dict(), f, indent=2)
            logger.info(f"Saved baseline to {output_path}")
        except Exception as e:
            logger.error(f"Error saving baseline to {output_path}: {e}")
    
    def detect_regressions(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Detect regressions compared to baseline.
        
        Args:
            metrics: Current metrics to check for regressions
            
        Returns:
            Dictionary with regression information
        """
        if not self.baseline:
            logger.warning("No baseline available for regression detection")
            return {"regressions": [], "has_regressions": False}
        
        regressions = []
        
        # Check overall metrics
        for metric_name, current_value in metrics.metrics.items():
            if metric_name in self.baseline.metrics:
                baseline_value = self.baseline.metrics[metric_name]
                
                # Check if this is a regression
                delta = current_value - baseline_value
                is_regression = self._is_regression(metric_name, current_value, baseline_value)
                
                if is_regression:
                    regression = {
                        "metric": metric_name,
                        "current": current_value,
                        "baseline": baseline_value,
                        "delta": delta,
                        "delta_percent": delta / max(0.0001, abs(baseline_value)),
                        "domain": None,
                        "case_id": None
                    }
                    regressions.append(regression)
                    
                    # Send alert for significant regressions
                    if abs(delta) / max(0.0001, abs(baseline_value)) >= self.threshold:
                        self.alert_manager.create_regression_alert(
                            metric_name=metric_name,
                            current_value=current_value,
                            baseline_value=baseline_value,
                            delta=delta
                        )
                
                # Store regression comparison in metrics
                metrics.add_regression_metric(
                    name=metric_name,
                    current=current_value,
                    baseline=baseline_value,
                    delta=delta,
                    is_regression=is_regression
                )
        
        # Check domain metrics
        for domain, domain_metrics in metrics.domain_metrics.items():
            if domain in self.baseline.domain_metrics:
                baseline_domain = self.baseline.domain_metrics[domain]
                
                for metric_name, current_value in domain_metrics.items():
                    if metric_name in baseline_domain:
                        baseline_value = baseline_domain[metric_name]
                        
                        # Check if this is a regression
                        delta = current_value - baseline_value
                        is_regression = self._is_regression(metric_name, current_value, baseline_value)
                        
                        if is_regression:
                            regression = {
                                "metric": metric_name,
                                "current": current_value,
                                "baseline": baseline_value,
                                "delta": delta,
                                "delta_percent": delta / max(0.0001, abs(baseline_value)),
                                "domain": domain,
                                "case_id": None
                            }
                            regressions.append(regression)
                            
                            # Send alert for significant regressions
                            if abs(delta) / max(0.0001, abs(baseline_value)) >= self.threshold:
                                self.alert_manager.create_regression_alert(
                                    metric_name=metric_name,
                                    current_value=current_value,
                                    baseline_value=baseline_value,
                                    delta=delta,
                                    domain=domain
                                )
        
        # Check case metrics for common test cases
        for case_id, case_metrics in metrics.test_case_metrics.items():
            if case_id in self.baseline.test_case_metrics:
                baseline_case = self.baseline.test_case_metrics[case_id]
                
                for metric_name, current_value in case_metrics.items():
                    if metric_name in baseline_case:
                        baseline_value = baseline_case[metric_name]
                        
                        # Check if this is a regression
                        delta = current_value - baseline_value
                        is_regression = self._is_regression(metric_name, current_value, baseline_value)
                        
                        if is_regression:
                            regression = {
                                "metric": metric_name,
                                "current": current_value,
                                "baseline": baseline_value,
                                "delta": delta,
                                "delta_percent": delta / max(0.0001, abs(baseline_value)),
                                "domain": None,
                                "case_id": case_id
                            }
                            regressions.append(regression)
                            
                            # Send alert for significant regressions
                            if abs(delta) / max(0.0001, abs(baseline_value)) >= self.threshold:
                                self.alert_manager.create_regression_alert(
                                    metric_name=metric_name,
                                    current_value=current_value,
                                    baseline_value=baseline_value,
                                    delta=delta,
                                    case_id=case_id
                                )
        
        # Classify regressions by severity
        severe_regressions = [r for r in regressions 
                            if abs(r["delta_percent"]) >= self.threshold * 2]
        moderate_regressions = [r for r in regressions 
                              if self.threshold <= abs(r["delta_percent"]) < self.threshold * 2]
        minor_regressions = [r for r in regressions 
                           if abs(r["delta_percent"]) < self.threshold]
        
        # Compile results
        result = {
            "regressions": regressions,
            "has_regressions": len(regressions) > 0,
            "severe_regressions": severe_regressions,
            "moderate_regressions": moderate_regressions,
            "minor_regressions": minor_regressions,
            "severe_count": len(severe_regressions),
            "moderate_count": len(moderate_regressions),
            "minor_count": len(minor_regressions),
            "total_count": len(regressions),
            "baseline_run_id": self.baseline.run_id,
            "baseline_timestamp": self.baseline.timestamp
        }
        
        return result
    
    def _is_regression(self, metric_name: str, current: float, baseline: float) -> bool:
        """
        Determine if a metric change is a regression.
        
        Args:
            metric_name: Name of the metric
            current: Current value
            baseline: Baseline value
            
        Returns:
            True if the change is a regression
        """
        # Different metrics have different regression definitions
        if metric_name in ("hallucination_reduction", "factuality", "confidence"):
            # For these metrics, lower values are worse
            return current < baseline
        elif metric_name in ("latency", "overhead", "error_rate"):
            # For these metrics, higher values are worse
            return current > baseline
        else:
            # Default: assume higher is better
            return current < baseline


class ContinuousMonitor:
    """Main monitor class that orchestrates the continuous monitoring process."""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        interval: int = DEFAULT_INTERVAL,
        output_dir: str = str(DEFAULT_OUTPUT_DIR),
        baseline_path: Optional[str] = None,
        alert_threshold: float = DEFAULT_ALERT_THRESHOLD
    ):
        """
        Initialize continuous monitor.
        
        Args:
            config_path: Path to configuration file
            interval: Interval between benchmark runs (seconds)
            output_dir: Directory for output files
            baseline_path: Path to baseline metrics file
            alert_threshold: Regression threshold (0.0-1.0)
        """
        # Load configuration if provided
        self.config = {}
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
        
        # Set parameters from config or arguments
        self.interval = self.config.get("interval", interval)
        self.output_dir = Path(self.config.get("output_dir", output_dir))
        self.baseline_path = self.config.get("baseline_path", baseline_path)
        self.alert_threshold = self.config.get("alert_threshold", alert_threshold)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.storage = MetricsStorage(
            base_dir=str(self.output_dir),
            retention_days=self.config.get("metrics_retention", DEFAULT_METRICS_RETENTION)
        )
        
        self.alert_manager = AlertManager(
            storage=self.storage,
            config=self.config.get("alerts", {})
        )
        
        self.regression_detector = RegressionDetector(
            storage=self.storage,
            alert_manager=self.alert_manager,
            baseline_path=self.baseline_path,
            threshold=self.alert_threshold
        )
        
        # Set up test configuration
        self.benchmark_config = self._create_benchmark_config()
        
        # Control flags
        self.running = False
        self.stop_event = threading.Event()
        
        # Metrics queue for real-time access
        self.metrics_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    
    def _create_benchmark_config(self) -> Dict[str, Any]:
        """Create benchmark configuration from monitor config."""
        benchmark_config = self.config.get("benchmark", {})
        
        # Set defaults if not specified
        if "domains" not in benchmark_config:
            benchmark_config["domains"] = ["medical", "legal", "financial", "technical"]
        
        if "test_count" not in benchmark_config:
            benchmark_config["test_count"] = 5  # Number of test cases per domain
        
        if "include_v2" not in benchmark_config:
            benchmark_config["include_v2"] = True
        
        if "parallel" not in benchmark_config:
            benchmark_config["parallel"] = True
        
        if "max_workers" not in benchmark_config:
            benchmark_config["max_workers"] = 4
        
        return benchmark_config
    
    def _select_test_cases(self) -> List[TestCase]:
        """Select test cases for benchmark run."""
        # Generate all test cases
        all_cases = generate_test_cases()
        logger.info(f"Generated {len(all_cases)} test cases")
        
        # Filter test cases based on configuration
        domains = self.benchmark_config.get("domains", ["medical", "legal", "financial", "technical"])
        test_count = self.benchmark_config.get("test_count", 5)
        
        # Filter by domains
        domain_cases = []
        for domain in domains:
            filtered = filter_test_cases(all_cases, domain=domain)
            # Take up to test_count cases from each domain
            domain_cases.extend(filtered[:test_count])
        
        logger.info(f"Selected {len(domain_cases)} test cases for benchmark run")
        return domain_cases
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run a benchmark and return results."""
        # Select test cases
        test_cases = self._select_test_cases()
        
        # Create benchmark configuration
        include_v2 = self.benchmark_config.get("include_v2", True)
        parallel = self.benchmark_config.get("parallel", True)
        max_workers = self.benchmark_config.get("max_workers", 4)
        
        # Create unique run directory
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"monitor_{timestamp}"
        run_dir = self.output_dir / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create benchmark config
        config = BenchmarkConfig(
            test_cases=test_cases,
            include_v2=include_v2,
            parallel=parallel,
            max_workers=max_workers,
            output_formats=["json"],
            output_dir=str(run_dir)
        )
        
        # Initialize resource tracking
        resource_tracking_stop = threading.Event()
        resource_tracking_path = None
        
        resource_tracking_thread = threading.Thread(
            target=track_resource_usage,
            args=(str(run_dir), 1.0, resource_tracking_stop)
        )
        resource_tracking_thread.daemon = True
        resource_tracking_thread.start()
        
        try:
            # Run benchmark
            logger.info(f"Starting benchmark run {run_id}")
            start_time = time.time()
            results = run_benchmark(config)
            end_time = time.time()
            
            # Stop resource tracking
            resource_tracking_stop.set()
            resource_tracking_thread.join(timeout=5)
            
            # Calculate benchmarking time
            benchmark_time = end_time - start_time
            
            # Extract benchmark metrics
            metrics = PerformanceMetrics(run_id, datetime.datetime.now().isoformat())
            
            # Add overall metrics
            metrics.add_metric("hallucination_reduction", results.summary_metrics["hallucination"]["avg_v3_reduction_rate"])
            metrics.add_metric("latency_overhead", results.summary_metrics["latency"]["avg_v3_overhead_ms"])
            if "confidence" in results.summary_metrics:
                metrics.add_metric("confidence", results.summary_metrics["confidence"].get("avg_v3_confidence", 0))
            
            metrics.add_metric("execution_time", benchmark_time)
            metrics.add_metric("case_count", results.summary_metrics["case_count"])
            metrics.add_metric("success_rate", results.summary_metrics["success_metrics"]["overall_success"])
            
            # Add domain metrics
            if "by_domain" in results.summary_metrics:
                for domain, domain_data in results.summary_metrics["by_domain"].items():
                    metrics.add_metric("hallucination_reduction", domain_data.get("avg_reduction_rate", 0), domain=domain)
                    metrics.add_metric("latency_overhead", domain_data.get("avg_latency_overhead_ms", 0), domain=domain)
            
            # Add resource metrics if available
            try:
                # Look for resource tracking file in run directory
                tracking_files = list(run_dir.glob("resource_tracking_*.json"))
                if tracking_files:
                    resource_tracking_path = str(tracking_files[0])
                    with open(resource_tracking_path, "r") as f:
                        resource_data = json.load(f)
                    
                    if "summary" in resource_data:
                        summary = resource_data["summary"]
                        metrics.add_resource_metric("avg_cpu_percent", summary.get("avg_cpu_percent", 0))
                        metrics.add_resource_metric("max_cpu_percent", summary.get("max_cpu_percent", 0))
                        metrics.add_resource_metric("avg_memory_mb", summary.get("avg_memory_mb", 0))
                        metrics.add_resource_metric("max_memory_mb", summary.get("max_memory_mb", 0))
                        metrics.add_resource_metric("avg_thread_count", summary.get("avg_thread_count", 0))
            except Exception as e:
                logger.warning(f"Error processing resource tracking data: {e}")
            
            # Add metadata
            metrics.metadata["benchmark_config"] = {
                "include_v2": include_v2,
                "parallel": parallel,
                "max_workers": max_workers,
                "test_count": len(test_cases),
                "domains": self.benchmark_config.get("domains", [])
            }
            metrics.metadata["results_path"] = str(run_dir / f"benchmark_results_{timestamp}.json")
            metrics.metadata["resource_path"] = resource_tracking_path
            
            # Store metrics
            metrics_path = self.storage.store_metrics(metrics)
            logger.info(f"Stored metrics at {metrics_path}")
            
            # Put metrics in queue for real-time access
            try:
                if not self.metrics_queue.full():
                    self.metrics_queue.put_nowait(metrics)
            except queue.Full:
                pass
            
            # Run regression detection
            regression_results = self.regression_detector.detect_regressions(metrics)
            
            # If we don't have a baseline yet, set this as baseline
            if not self.regression_detector.baseline:
                self.regression_detector.set_baseline_from_metrics(metrics)
                
                # Save baseline
                baseline_path = self.output_dir / "baseline.json"
                self.regression_detector.save_baseline(str(baseline_path))
            
            # Create summary alert if needed
            if regression_results["has_regressions"]:
                summary_metrics = {
                    "total_regressions": regression_results["total_count"],
                    "severe_regressions": regression_results["severe_count"],
                    "moderate_regressions": regression_results["moderate_count"],
                    "minor_regressions": regression_results["minor_count"],
                    "run_id": run_id
                }
                
                # Only alert for significant regressions
                if regression_results["severe_count"] > 0:
                    level = AlertLevel.ERROR
                elif regression_results["moderate_count"] > 0:
                    level = AlertLevel.WARNING
                else:
                    level = AlertLevel.INFO
                
                summary_alert = Alert(
                    level=level,
                    message=f"Found {regression_results['total_count']} regressions " + \
                           f"({regression_results['severe_count']} severe, " + \
                           f"{regression_results['moderate_count']} moderate, " + \
                           f"{regression_results['minor_count']} minor)",
                    metrics=summary_metrics,
                    source="regression_summary"
                )
                
                self.alert_manager.send_alert(summary_alert)
            
            # Log summary
            logger.info(f"Benchmark run {run_id} completed: " + \
                       f"{metrics.metrics.get('hallucination_reduction', 0):.2%} hallucination reduction, " + \
                       f"{metrics.metrics.get('latency_overhead', 0):.2f}ms latency")
            
            if regression_results["has_regressions"]:
                logger.info(f"Detected {regression_results['total_count']} regressions " + \
                           f"({regression_results['severe_count']} severe, " + \
                           f"{regression_results['moderate_count']} moderate)")
            
            return {
                "run_id": run_id,
                "metrics": metrics.to_dict(),
                "regressions": regression_results
            }
        
        except Exception as e:
            logger.error(f"Error during benchmark run: {e}")
            
            # Stop resource tracking
            resource_tracking_stop.set()
            if resource_tracking_thread.is_alive():
                resource_tracking_thread.join(timeout=5)
            
            # Create error alert
            error_alert = Alert(
                level=AlertLevel.ERROR,
                message=f"Benchmark run failed: {str(e)}",
                metrics={"run_id": run_id, "error": str(e)},
                source="benchmark_runner"
            )
            
            self.alert_manager.send_alert(error_alert)
            
            return {
                "run_id": run_id,
                "error": str(e)
            }
    
    def start(self):
        """Start the continuous monitoring process."""
        if self.running:
            logger.warning("Monitor is already running")
            return
        
        self.running = True
        self.stop_event.clear()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        logger.info(f"Continuous monitoring started with interval {self.interval} seconds")
        
        # Send startup alert
        startup_alert = Alert(
            level=AlertLevel.INFO,
            message=f"Continuous monitoring started with interval {self.interval} seconds",
            metrics={"interval": self.interval},
            source="monitor_startup"
        )
        
        self.alert_manager.send_alert(startup_alert)
        
        return monitor_thread
    
    def stop(self):
        """Stop the continuous monitoring process."""
        if not self.running:
            logger.warning("Monitor is not running")
            return
        
        logger.info("Stopping continuous monitoring...")
        self.stop_event.set()
        self.running = False
        
        # Send shutdown alert
        shutdown_alert = Alert(
            level=AlertLevel.INFO,
            message="Continuous monitoring stopped",
            metrics={},
            source="monitor_shutdown"
        )
        
        self.alert_manager.send_alert(shutdown_alert)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running and not self.stop_event.is_set():
            try:
                # Run a benchmark
                self.run_benchmark()
                
                # Clean up old data
                self.storage.cleanup_old_data()
                
                # Wait for next interval or until stopped
                logger.info(f"Waiting {self.interval} seconds until next benchmark run")
                self.stop_event.wait(self.interval)
            
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                
                # Wait a shorter interval to retry
                retry_interval = min(self.interval, 60)  # Max 1 minute retry
                logger.info(f"Retrying in {retry_interval} seconds")
                self.stop_event.wait(retry_interval)
    
    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the latest metrics."""
        try:
            # Try to get from queue first (most recent)
            if not self.metrics_queue.empty():
                return self.metrics_queue.queue[-1]  # Peek at the most recent
            
            # Fallback to storage
            recent_metrics = self.storage.get_recent_metrics(count=1)
            if recent_metrics:
                return recent_metrics[0]
        except Exception as e:
            logger.error(f"Error getting latest metrics: {e}")
        
        return None
    
    def get_recent_metrics(self, count: int = 10) -> List[PerformanceMetrics]:
        """Get recent metrics."""
        return self.storage.get_recent_metrics(count=count)
    
    def get_recent_alerts(self, count: int = 10, level: Optional[str] = None) -> List[Alert]:
        """Get recent alerts."""
        return self.storage.get_recent_alerts(count=count, level=level)
    
    def run_once(self) -> Dict[str, Any]:
        """Run a single benchmark without starting continuous monitoring."""
        logger.info("Running one-time benchmark")
        return self.run_benchmark()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ACDEK v3 Continuous Performance Monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help="Interval between benchmark runs (seconds)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--baseline",
        type=str,
        help="Path to baseline metrics file"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_ALERT_THRESHOLD,
        help="Regression threshold (0.0-1.0)"
    )
    
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single benchmark without continuous monitoring"
    )
    
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the dashboard server"
    )
    
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8080,
        help="Dashboard server port"
    )
    
    return parser.parse_args()


def handle_signal(signum, frame):
    """Signal handler for graceful shutdown."""
    logger.info(f"Received signal {signum}, shutting down...")
    global monitor
    if monitor:
        monitor.stop()


def main():
    """Main function."""
    args = parse_args()
    
    global monitor
    
    # Create monitor
    monitor = ContinuousMonitor(
        config_path=args.config,
        interval=args.interval,
        output_dir=args.output_dir,
        baseline_path=args.baseline,
        alert_threshold=args.threshold
    )
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    if args.once:
        # Run a single benchmark
        results = monitor.run_once()
        print(f"\nBenchmark results (run_id: {results['run_id']}):")
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            metrics = results["metrics"]
            print(f"Hallucination reduction: {metrics['metrics'].get('hallucination_reduction', 0):.2%}")
            print(f"Latency overhead: {metrics['metrics'].get('latency_overhead', 0):.2f} ms")
            print(f"Test cases: {metrics['metrics'].get('case_count', 0)}")
            
            if "regressions" in results and results["regressions"]["has_regressions"]:
                regressions = results["regressions"]
                print(f"\nDetected {regressions['total_count']} regressions:")
                print(f"- Severe: {regressions['severe_count']}")
                print(f"- Moderate: {regressions['moderate_count']}")
                print(f"- Minor: {regressions['minor_count']}")
    elif args.dashboard:
        # Start the dashboard server
        try:
            from performance_dashboard import run_dashboard
            # Start the monitor
            monitor.start()
            # Run the dashboard (this will block)
            run_dashboard(monitor, port=args.dashboard_port)
        except ImportError:
            logger.error("Dashboard functionality requires performance_dashboard.py module")
            return 1
    else:
        # Start continuous monitoring
        monitor_thread = monitor.start()
        
        # Keep the main thread running
        try:
            while monitor.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
            monitor.stop()
            if monitor_thread.is_alive():
                monitor_thread.join(timeout=5)
    
    return 0


if __name__ == "__main__":
    monitor = None  # Global reference for signal handler
    sys.exit(main())