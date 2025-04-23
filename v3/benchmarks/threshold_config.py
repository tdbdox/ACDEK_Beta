"""
ACDEK v3 Threshold Configuration System

This module provides a comprehensive configuration management system for thresholds
used across the ACDEK benchmarking suite. It handles domain-specific threshold overrides,
validation, optimization, and dynamic adjustment based on benchmark results.

Key features:
1. Central management for all threshold configurations
2. Domain-specific threshold customization
3. Model-specific threshold adjustments
4. Threshold validation and inference
5. Optimization framework for threshold tuning
6. Configuration import/export

Usage:
    # Get default thresholds for a domain
    medical_thresholds = ThresholdConfig.get_domain_thresholds("medical")
    
    # Override a specific threshold
    ThresholdConfig.set_domain_threshold("medical", "factuality", 0.78)
    
    # Export current configuration
    ThresholdConfig.export_configuration("thresholds.yml")
"""

import os
import json
import yaml
import logging
import statistics
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration paths
DEFAULT_CONFIG_DIR = Path("./config")
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "thresholds.yml"


class MetricType(str, Enum):
    """Types of metrics for thresholds."""
    FACTUALITY = "factuality"
    LATENCY = "latency"
    CONFIDENCE = "confidence"
    RETRY_PENALTY = "retry_penalty"
    COVERAGE = "coverage"
    COMBINED = "combined"


class Domain(str, Enum):
    """Supported domains for thresholds."""
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    GENERAL = "general"


@dataclass
class ThresholdSet:
    """Container for a complete set of thresholds for a specific domain/model."""
    domain: str
    model: Optional[str] = None
    factuality: float = 0.7
    latency: float = 200.0  # ms
    confidence: float = 0.5
    retry_penalty: float = 0.8
    coverage: float = 0.6
    combined: float = 0.65
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain,
            "model": self.model,
            "thresholds": {
                "factuality": self.factuality,
                "latency": self.latency,
                "confidence": self.confidence,
                "retry_penalty": self.retry_penalty,
                "coverage": self.coverage,
                "combined": self.combined
            },
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThresholdSet':
        """Create from dictionary."""
        thresholds = data.get("thresholds", {})
        return cls(
            domain=data.get("domain", "general"),
            model=data.get("model"),
            factuality=thresholds.get("factuality", 0.7),
            latency=thresholds.get("latency", 200.0),
            confidence=thresholds.get("confidence", 0.5),
            retry_penalty=thresholds.get("retry_penalty", 0.8),
            coverage=thresholds.get("coverage", 0.6),
            combined=thresholds.get("combined", 0.65),
            metadata=data.get("metadata", {})
        )


@dataclass
class ThresholdOptimizationResult:
    """Result of threshold optimization process."""
    original_threshold: float
    optimized_threshold: float
    metric_type: MetricType
    domain: str
    model: Optional[str] = None
    improvement_score: float = 0.0
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_threshold": self.original_threshold,
            "optimized_threshold": self.optimized_threshold,
            "metric_type": self.metric_type,
            "domain": self.domain,
            "model": self.model,
            "improvement_score": self.improvement_score,
            "confidence_interval": self.confidence_interval,
            "metadata": self.metadata
        }


class ThresholdConfig:
    """
    Central configuration system for ACDEK thresholds.
    
    This class manages threshold configurations, including domain-specific overrides,
    model-specific adjustments, and validation. It provides methods for getting, setting,
    and optimizing thresholds for different metrics across domains and models.
    """
    
    # Default thresholds for each domain
    _default_domain_thresholds = {
        Domain.MEDICAL.value: ThresholdSet(
            domain=Domain.MEDICAL.value,
            factuality=0.75,
            latency=200.0,
            confidence=0.6,
            retry_penalty=0.8,
            coverage=0.7,
            combined=0.7,
            metadata={"critical": True, "require_validation": True}
        ),
        Domain.LEGAL.value: ThresholdSet(
            domain=Domain.LEGAL.value,
            factuality=0.72,
            latency=250.0,
            confidence=0.55,
            retry_penalty=0.9,
            coverage=0.65,
            combined=0.68,
            metadata={"critical": True, "require_validation": True}
        ),
        Domain.FINANCIAL.value: ThresholdSet(
            domain=Domain.FINANCIAL.value,
            factuality=0.68,
            latency=180.0,
            confidence=0.5,
            retry_penalty=0.7,
            coverage=0.6,
            combined=0.65,
            metadata={"critical": True, "require_validation": False}
        ),
        Domain.TECHNICAL.value: ThresholdSet(
            domain=Domain.TECHNICAL.value,
            factuality=0.65,
            latency=150.0,
            confidence=0.45,
            retry_penalty=0.6,
            coverage=0.55,
            combined=0.6,
            metadata={"critical": False, "require_validation": False}
        ),
        Domain.GENERAL.value: ThresholdSet(
            domain=Domain.GENERAL.value,
            factuality=0.7,
            latency=200.0,
            confidence=0.5,
            retry_penalty=0.8,
            coverage=0.6,
            combined=0.65,
            metadata={"critical": False, "require_validation": False}
        )
    }
    
    # Container for domain-specific overrides
    _domain_overrides: Dict[str, ThresholdSet] = {}
    
    # Container for model-specific thresholds
    _model_thresholds: Dict[str, Dict[str, ThresholdSet]] = {}
    
    # Configuration metadata
    _config_metadata: Dict[str, Any] = {
        "last_updated": datetime.datetime.now().isoformat(),
        "version": "1.0.0",
        "description": "ACDEK v3 threshold configuration"
    }
    
    @classmethod
    def get_domain_thresholds(cls, domain: str) -> ThresholdSet:
        """
        Get thresholds for a specific domain.
        
        Args:
            domain: Domain name (medical, legal, financial, technical, general)
            
        Returns:
            ThresholdSet with configured thresholds
        """
        # Check domain overrides first
        if domain in cls._domain_overrides:
            return cls._domain_overrides[domain]
        
        # Check default thresholds
        if domain in cls._default_domain_thresholds:
            return cls._default_domain_thresholds[domain]
        
        # Fall back to general domain
        logger.warning(f"Domain '{domain}' not recognized, using general thresholds")
        return cls._default_domain_thresholds[Domain.GENERAL.value]
    
    @classmethod
    def get_threshold(cls, domain: str, metric_type: Union[str, MetricType], model: Optional[str] = None) -> float:
        """
        Get a specific threshold value.
        
        Args:
            domain: Domain name
            metric_type: Metric type (factuality, latency, etc.)
            model: Optional model name for model-specific thresholds
            
        Returns:
            Threshold value for the specified metric
        """
        # Convert string metric type to enum if needed
        if isinstance(metric_type, str):
            try:
                metric_type = MetricType(metric_type.lower())
            except ValueError:
                logger.warning(f"Metric type '{metric_type}' not recognized, using 'factuality'")
                metric_type = MetricType.FACTUALITY
        
        # Check model-specific thresholds first
        if model and model in cls._model_thresholds and domain in cls._model_thresholds[model]:
            threshold_set = cls._model_thresholds[model][domain]
            return getattr(threshold_set, metric_type.value, cls.get_default_threshold(metric_type))
        
        # Get domain thresholds
        threshold_set = cls.get_domain_thresholds(domain)
        return getattr(threshold_set, metric_type.value, cls.get_default_threshold(metric_type))
    
    @classmethod
    def get_default_threshold(cls, metric_type: Union[str, MetricType]) -> float:
        """
        Get default threshold for a metric.
        
        Args:
            metric_type: Metric type
            
        Returns:
            Default threshold value
        """
        # Convert string metric type to enum if needed
        if isinstance(metric_type, str):
            try:
                metric_type = MetricType(metric_type.lower())
            except ValueError:
                logger.warning(f"Metric type '{metric_type}' not recognized, using 'factuality'")
                metric_type = MetricType.FACTUALITY
        
        # Get general domain thresholds
        general_thresholds = cls._default_domain_thresholds[Domain.GENERAL.value]
        
        # Return appropriate threshold
        return getattr(general_thresholds, metric_type.value, 0.7)
    
    @classmethod
    def set_domain_threshold(cls, domain: str, metric_type: Union[str, MetricType], value: float) -> None:
        """
        Override a threshold for a specific domain.
        
        Args:
            domain: Domain name
            metric_type: Metric type
            value: New threshold value
        """
        # Convert string metric type to enum if needed
        if isinstance(metric_type, str):
            try:
                metric_type = MetricType(metric_type.lower())
            except ValueError:
                logger.warning(f"Metric type '{metric_type}' not recognized, using 'factuality'")
                metric_type = MetricType.FACTUALITY
        
        # Validate value
        if not cls.validate_threshold(metric_type, value):
            logger.warning(f"Invalid threshold value {value} for {metric_type.value}, using default")
            return
        
        # Create override if it doesn't exist
        if domain not in cls._domain_overrides:
            # Start with default thresholds for the domain
            if domain in cls._default_domain_thresholds:
                cls._domain_overrides[domain] = ThresholdSet(
                    domain=domain,
                    factuality=cls._default_domain_thresholds[domain].factuality,
                    latency=cls._default_domain_thresholds[domain].latency,
                    confidence=cls._default_domain_thresholds[domain].confidence,
                    retry_penalty=cls._default_domain_thresholds[domain].retry_penalty,
                    coverage=cls._default_domain_thresholds[domain].coverage,
                    combined=cls._default_domain_thresholds[domain].combined,
                    metadata=dict(cls._default_domain_thresholds[domain].metadata)
                )
            else:
                # Create new threshold set with general defaults
                general = cls._default_domain_thresholds[Domain.GENERAL.value]
                cls._domain_overrides[domain] = ThresholdSet(
                    domain=domain,
                    factuality=general.factuality,
                    latency=general.latency,
                    confidence=general.confidence,
                    retry_penalty=general.retry_penalty,
                    coverage=general.coverage,
                    combined=general.combined
                )
        
        # Set the specific threshold
        setattr(cls._domain_overrides[domain], metric_type.value, value)
        
        # Update metadata
        cls._domain_overrides[domain].metadata["last_updated"] = datetime.datetime.now().isoformat()
        cls._config_metadata["last_updated"] = datetime.datetime.now().isoformat()
    
    @classmethod
    def set_model_threshold(cls, model: str, domain: str, metric_type: Union[str, MetricType], value: float) -> None:
        """
        Set a model-specific threshold.
        
        Args:
            model: Model name
            domain: Domain name
            metric_type: Metric type
            value: Threshold value
        """
        # Convert string metric type to enum if needed
        if isinstance(metric_type, str):
            try:
                metric_type = MetricType(metric_type.lower())
            except ValueError:
                logger.warning(f"Metric type '{metric_type}' not recognized, using 'factuality'")
                metric_type = MetricType.FACTUALITY
        
        # Validate value
        if not cls.validate_threshold(metric_type, value):
            logger.warning(f"Invalid threshold value {value} for {metric_type.value}, using default")
            return
        
        # Create model entry if it doesn't exist
        if model not in cls._model_thresholds:
            cls._model_thresholds[model] = {}
        
        # Create domain entry if it doesn't exist
        if domain not in cls._model_thresholds[model]:
            # Start with domain thresholds
            domain_thresholds = cls.get_domain_thresholds(domain)
            cls._model_thresholds[model][domain] = ThresholdSet(
                domain=domain,
                model=model,
                factuality=domain_thresholds.factuality,
                latency=domain_thresholds.latency,
                confidence=domain_thresholds.confidence,
                retry_penalty=domain_thresholds.retry_penalty,
                coverage=domain_thresholds.coverage,
                combined=domain_thresholds.combined,
                metadata={"derived_from": domain}
            )
        
        # Set the specific threshold
        setattr(cls._model_thresholds[model][domain], metric_type.value, value)
        
        # Update metadata
        cls._model_thresholds[model][domain].metadata["last_updated"] = datetime.datetime.now().isoformat()
        cls._config_metadata["last_updated"] = datetime.datetime.now().isoformat()
    
    @classmethod
    def validate_threshold(cls, metric_type: Union[str, MetricType], value: float) -> bool:
        """
        Validate a threshold value.
        
        Args:
            metric_type: Metric type
            value: Threshold value to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Convert string metric type to enum if needed
        if isinstance(metric_type, str):
            try:
                metric_type = MetricType(metric_type.lower())
            except ValueError:
                return False
        
        # Validate based on metric type
        if metric_type == MetricType.FACTUALITY:
            return 0.0 <= value <= 1.0
        elif metric_type == MetricType.CONFIDENCE:
            return 0.0 <= value <= 1.0
        elif metric_type == MetricType.COVERAGE:
            return 0.0 <= value <= 1.0
        elif metric_type == MetricType.COMBINED:
            return 0.0 <= value <= 1.0
        elif metric_type == MetricType.LATENCY:
            return value > 0.0  # Latency must be positive
        elif metric_type == MetricType.RETRY_PENALTY:
            return 0.0 <= value <= 1.0
        else:
            return False
    
    @classmethod
    def reset_to_defaults(cls) -> None:
        """Reset all configurations to defaults."""
        cls._domain_overrides = {}
        cls._model_thresholds = {}
        cls._config_metadata["last_updated"] = datetime.datetime.now().isoformat()
    
    @classmethod
    def export_configuration(cls, filepath: Union[str, Path] = DEFAULT_CONFIG_FILE) -> bool:
        """
        Export the current configuration to a file.
        
        Args:
            filepath: Path to save the configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare configuration
            config = {
                "metadata": cls._config_metadata,
                "domains": {
                    k: v.to_dict() for k, v in cls._default_domain_thresholds.items()
                },
                "overrides": {
                    k: v.to_dict() for k, v in cls._domain_overrides.items()
                },
                "models": {
                    model: {
                        domain: threshold_set.to_dict() 
                        for domain, threshold_set in domains.items()
                    }
                    for model, domains in cls._model_thresholds.items()
                }
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                if filepath.suffix in ('.yaml', '.yml'):
                    yaml.dump(config, f, sort_keys=False)
                else:
                    json.dump(config, f, indent=2)
            
            logger.info(f"Configuration exported to {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
            return False
    
    @classmethod
    def import_configuration(cls, filepath: Union[str, Path] = DEFAULT_CONFIG_FILE) -> bool:
        """
        Import configuration from a file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = Path(filepath)
            
            # Load configuration
            with open(filepath, 'r') as f:
                if filepath.suffix in ('.yaml', '.yml'):
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            # Reset current configuration
            cls.reset_to_defaults()
            
            # Import metadata
            if "metadata" in config:
                cls._config_metadata.update(config["metadata"])
            
            # Import domain overrides
            if "overrides" in config:
                for domain, data in config["overrides"].items():
                    cls._domain_overrides[domain] = ThresholdSet.from_dict(data)
            
            # Import model thresholds
            if "models" in config:
                for model, domains in config["models"].items():
                    cls._model_thresholds[model] = {
                        domain: ThresholdSet.from_dict(data)
                        for domain, data in domains.items()
                    }
            
            # Optionally import default domains if present
            if "domains" in config:
                for domain, data in config["domains"].items():
                    # Only override default domains that already exist
                    if domain in cls._default_domain_thresholds:
                        cls._default_domain_thresholds[domain] = ThresholdSet.from_dict(data)
            
            logger.info(f"Configuration imported from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return False
    
    @classmethod
    def get_configuration_summary(cls) -> Dict[str, Any]:
        """
        Get a summary of the current configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            "metadata": cls._config_metadata,
            "domains": list(cls._default_domain_thresholds.keys()),
            "overridden_domains": list(cls._domain_overrides.keys()),
            "models": list(cls._model_thresholds.keys()),
            "domain_count": len(cls._default_domain_thresholds),
            "override_count": len(cls._domain_overrides),
            "model_count": len(cls._model_thresholds)
        }
    
    @classmethod
    def infer_domain_thresholds(cls, benchmark_data: Dict[str, Any]) -> Dict[str, ThresholdSet]:
        """
        Infer appropriate thresholds from benchmark data.
        
        Args:
            benchmark_data: Benchmark data to analyze
            
        Returns:
            Dictionary mapping domains to inferred ThresholdSets
        """
        inferred_thresholds = {}
        domain_metrics = {}
        
        # Extract metrics by domain
        results = benchmark_data.get("results", [])
        for result in results:
            domain = result.get("domain", "general")
            
            if domain not in domain_metrics:
                domain_metrics[domain] = {
                    "factuality": [],
                    "latency": [],
                    "confidence": []
                }
            
            # Extract metrics
            if "factuality_score" in result:
                domain_metrics[domain]["factuality"].append(result["factuality_score"])
            elif "acdek_factuality_score" in result:
                domain_metrics[domain]["factuality"].append(result["acdek_factuality_score"])
            
            if "execution_time" in result:
                domain_metrics[domain]["latency"].append(result["execution_time"])
            
            if "acdek_confidence" in result:
                domain_metrics[domain]["confidence"].append(result["acdek_confidence"])
        
        # Calculate thresholds for each domain
        for domain, metrics in domain_metrics.items():
            # Skip domains with insufficient data
            if (len(metrics["factuality"]) < 3 or 
                len(metrics["latency"]) < 3 or 
                len(metrics["confidence"]) < 3):
                continue
            
            # Calculate factuality threshold (25th percentile)
            factuality_values = sorted(metrics["factuality"])
            idx = max(0, int(len(factuality_values) * 0.25) - 1)
            factuality_threshold = factuality_values[idx]
            
            # Calculate latency threshold (75th percentile)
            latency_values = sorted(metrics["latency"])
            idx = min(len(latency_values) - 1, int(len(latency_values) * 0.75))
            latency_threshold = latency_values[idx]
            
            # Calculate confidence threshold (median)
            confidence_values = sorted(metrics["confidence"])
            confidence_threshold = statistics.median(confidence_values)
            
            # Create threshold set
            inferred_thresholds[domain] = ThresholdSet(
                domain=domain,
                factuality=factuality_threshold,
                latency=latency_threshold,
                confidence=confidence_threshold,
                metadata={
                    "inferred": True,
                    "sample_size": len(metrics["factuality"]),
                    "timestamp": datetime.datetime.now().isoformat()
                }
            )
        
        return inferred_thresholds
    
    @classmethod
    def optimize_threshold(
        cls,
        domain: str,
        metric_type: Union[str, MetricType],
        benchmark_data: Dict[str, Any],
        search_range: Optional[Tuple[float, float]] = None,
        steps: int = 10,
        model: Optional[str] = None
    ) -> ThresholdOptimizationResult:
        """
        Optimize threshold based on benchmark data.
        
        Args:
            domain: Domain to optimize for
            metric_type: Metric type to optimize
            benchmark_data: Benchmark data to use for optimization
            search_range: Optional range for threshold search
            steps: Number of search steps
            model: Optional model for model-specific optimization
            
        Returns:
            ThresholdOptimizationResult with optimization results
        """
        # Convert string metric type to enum if needed
        if isinstance(metric_type, str):
            try:
                metric_type = MetricType(metric_type.lower())
            except ValueError:
                logger.warning(f"Metric type '{metric_type}' not recognized, using 'factuality'")
                metric_type = MetricType.FACTUALITY
        
        # Get current threshold
        current_threshold = cls.get_threshold(domain, metric_type, model)
        
        # Default search range if not provided
        if search_range is None:
            if metric_type == MetricType.LATENCY:
                # For latency, search within 50-200% of current
                search_range = (current_threshold * 0.5, current_threshold * 2.0)
            else:
                # For other metrics, search within +/- 30%
                min_val = max(0.0, current_threshold - 0.3)
                max_val = min(1.0, current_threshold + 0.3)
                search_range = (min_val, max_val)
        
        # Generate threshold candidates
        step_size = (search_range[1] - search_range[0]) / steps
        candidates = [search_range[0] + i * step_size for i in range(steps + 1)]
        
        # Evaluate each candidate
        best_threshold = current_threshold
        best_score = 0.0
        test_results = []
        
        for threshold in candidates:
            # Calculate score based on the metric
            score = cls._evaluate_threshold(
                domain=domain,
                metric_type=metric_type,
                threshold=threshold,
                benchmark_data=benchmark_data,
                model=model
            )
            
            test_results.append({
                "threshold": threshold,
                "score": score
            })
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Calculate confidence interval
        threshold_scores = [(result["threshold"], result["score"]) for result in test_results]
        # Sort by score descending
        threshold_scores.sort(key=lambda x: x[1], reverse=True)
        # Get top 20% of scores
        top_indices = max(1, int(len(threshold_scores) * 0.2))
        top_thresholds = [t for t, _ in threshold_scores[:top_indices]]
        
        # Calculate confidence interval
        if len(top_thresholds) > 1:
            min_threshold = min(top_thresholds)
            max_threshold = max(top_thresholds)
            confidence_interval = (min_threshold, max_threshold)
        else:
            # Default confidence interval
            confidence_interval = (best_threshold * 0.95, best_threshold * 1.05)
        
        # Create result
        result = ThresholdOptimizationResult(
            original_threshold=current_threshold,
            optimized_threshold=best_threshold,
            metric_type=metric_type,
            domain=domain,
            model=model,
            improvement_score=best_score - cls._evaluate_threshold(
                domain=domain,
                metric_type=metric_type,
                threshold=current_threshold,
                benchmark_data=benchmark_data,
                model=model
            ),
            test_results=test_results,
            confidence_interval=confidence_interval,
            metadata={
                "timestamp": datetime.datetime.now().isoformat(),
                "search_range": search_range,
                "steps": steps
            }
        )
        
        return result
    
    @classmethod
    def _evaluate_threshold(
        cls,
        domain: str,
        metric_type: MetricType,
        threshold: float,
        benchmark_data: Dict[str, Any],
        model: Optional[str] = None
    ) -> float:
        """
        Evaluate a threshold on benchmark data.
        
        Args:
            domain: Domain to evaluate for
            metric_type: Metric type to evaluate
            threshold: Threshold value to evaluate
            benchmark_data: Benchmark data to use for evaluation
            model: Optional model name
            
        Returns:
            Score for the threshold (higher is better)
        """
        # Extract matching results
        results = []
        for result in benchmark_data.get("results", []):
            result_domain = result.get("domain", "general")
            result_model = result.get("model", "unknown")
            
            # Filter by domain and optionally by model
            if result_domain == domain and (model is None or result_model == model):
                results.append(result)
        
        if not results:
            return 0.0  # No data to evaluate
        
        # Evaluate based on metric type
        if metric_type == MetricType.FACTUALITY:
            return cls._evaluate_factuality_threshold(threshold, results)
        elif metric_type == MetricType.LATENCY:
            return cls._evaluate_latency_threshold(threshold, results)
        elif metric_type == MetricType.CONFIDENCE:
            return cls._evaluate_confidence_threshold(threshold, results)
        else:
            # Default generic evaluation
            return cls._evaluate_generic_threshold(metric_type.value, threshold, results)
    
    @classmethod
    def _evaluate_factuality_threshold(cls, threshold: float, results: List[Dict[str, Any]]) -> float:
        """Evaluate factuality threshold."""
        # Key metrics to optimize for:
        # 1. Minimize false positives (responses below threshold that should be above)
        # 2. Minimize false negatives (responses above threshold that should be below)
        # 3. Maximize clarity of decision boundary
        
        scores = []
        
        for result in results:
            # Get factuality score - try different possible keys
            factuality = result.get("acdek_factuality_score", 
                         result.get("factuality_score", 
                         result.get("factuality", 0.0)))
            
            # Get hallucination flag - true if it contains hallucinations
            is_hallucination = result.get("acdek_hallucination_flag",
                               result.get("hallucination_flag", 
                               result.get("has_hallucination", False)))
            
            # Calculate score contribution
            if is_hallucination:
                # For hallucinations, we want factuality below threshold
                score = 1.0 if factuality < threshold else 0.0
                # Partial credit for near-misses
                if factuality >= threshold:
                    # Linear penalty based on distance from threshold
                    penalty = min(1.0, (factuality - threshold) / 0.2)
                    score = max(0.0, 1.0 - penalty)
            else:
                # For non-hallucinations, we want factuality above threshold
                score = 1.0 if factuality >= threshold else 0.0
                # Partial credit for near-misses
                if factuality < threshold:
                    # Linear penalty based on distance from threshold
                    penalty = min(1.0, (threshold - factuality) / 0.2)
                    score = max(0.0, 1.0 - penalty)
            
            scores.append(score)
        
        # Return average score
        return statistics.mean(scores) if scores else 0.0
    
    @classmethod
    def _evaluate_latency_threshold(cls, threshold: float, results: List[Dict[str, Any]]) -> float:
        """Evaluate latency threshold."""
        # For latency, we want to maximize the number of responses below the threshold
        # while maintaining high factuality
        
        scores = []
        
        for result in results:
            # Get latency - try different possible keys
            latency = result.get("execution_time",
                      result.get("latency", 
                      result.get("response_time", 0.0)))
            
            # Get factuality score
            factuality = result.get("acdek_factuality_score", 
                         result.get("factuality_score", 
                         result.get("factuality", 0.0)))
            
            # Calculate score contribution
            if latency <= threshold:
                # Below threshold is good
                score = 1.0
            else:
                # Above threshold gets penalty based on how far above
                overage = latency / threshold - 1.0
                score = max(0.0, 1.0 - min(1.0, overage))
            
            # Weight by factuality to ensure we don't optimize for speed at expense of quality
            score *= max(0.5, factuality)
            
            scores.append(score)
        
        # Return average score
        return statistics.mean(scores) if scores else 0.0
    
    @classmethod
    def _evaluate_confidence_threshold(cls, threshold: float, results: List[Dict[str, Any]]) -> float:
        """Evaluate confidence threshold."""
        # For confidence, we want to:
        # 1. Maximize correlation between confidence and factuality
        # 2. Ensure confidence threshold correctly separates good/bad responses
        
        scores = []
        
        for result in results:
            # Get confidence score
            confidence = result.get("acdek_confidence", 
                          result.get("confidence_score", 
                          result.get("confidence", 0.0)))
            
            # Get factuality score
            factuality = result.get("acdek_factuality_score", 
                         result.get("factuality_score", 
                         result.get("factuality", 0.0)))
            
            # Calculate score contribution based on alignment
            alignment = 1.0 - abs(confidence - factuality)
            
            # Calculate threshold effectiveness
            if confidence >= threshold and factuality >= 0.6:
                # True positive (high confidence, high factuality)
                threshold_score = 1.0
            elif confidence < threshold and factuality < 0.6:
                # True negative (low confidence, low factuality)
                threshold_score = 1.0
            elif confidence >= threshold and factuality < 0.6:
                # False positive (high confidence, low factuality)
                threshold_score = 0.0
            else:
                # False negative (low confidence, high factuality)
                threshold_score = 0.5  # Less bad than false positive
            
            # Combine scores (60% alignment, 40% threshold effectiveness)
            score = 0.6 * alignment + 0.4 * threshold_score
            scores.append(score)
        
        # Return average score
        return statistics.mean(scores) if scores else 0.0
    
    @classmethod
    def _evaluate_generic_threshold(cls, metric_name: str, threshold: float, results: List[Dict[str, Any]]) -> float:
        """Evaluate a generic threshold."""
        # Generic evaluation simply counts how many results exceed the threshold
        
        scores = []
        
        for result in results:
            # Try to find the metric value
            metric_value = result.get(f"acdek_{metric_name}", 
                           result.get(f"{metric_name}_score", 
                           result.get(metric_name, 0.0)))
            
            # Calculate score (1.0 if above threshold, partial credit for near misses)
            if metric_value >= threshold:
                score = 1.0
            else:
                # Partial credit based on how close to threshold
                proximity = metric_value / threshold
                score = max(0.0, proximity)
            
            scores.append(score)
        
        # Return average score
        return statistics.mean(scores) if scores else 0.0
    
    @classmethod
    def apply_optimized_threshold(cls, optimization_result: ThresholdOptimizationResult) -> None:
        """
        Apply an optimized threshold configuration.
        
        Args:
            optimization_result: Result from threshold optimization
        """
        # Extract key information
        domain = optimization_result.domain
        metric_type = optimization_result.metric_type
        optimized_threshold = optimization_result.optimized_threshold
        model = optimization_result.model
        
        # Apply the optimized threshold
        if model:
            cls.set_model_threshold(model, domain, metric_type, optimized_threshold)
        else:
            cls.set_domain_threshold(domain, metric_type, optimized_threshold)
        
        # Add optimization metadata
        metadata = {
            "optimized": True,
            "original_threshold": optimization_result.original_threshold,
            "improvement_score": optimization_result.improvement_score,
            "confidence_interval": optimization_result.confidence_interval,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Update threshold metadata
        if model:
            cls._model_thresholds[model][domain].metadata.update(metadata)
        else:
            cls._domain_overrides[domain].metadata.update(metadata)


def main():
    """Command-line interface for threshold configuration."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="ACDEK v3 Threshold Configuration Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Get threshold command
    get_parser = subparsers.add_parser("get", help="Get threshold values")
    get_parser.add_argument("domain", help="Domain name")
    get_parser.add_argument("--metric", help="Specific metric to get", default=None)
    get_parser.add_argument("--model", help="Optional model name", default=None)
    
    # Set threshold command
    set_parser = subparsers.add_parser("set", help="Set threshold values")
    set_parser.add_argument("domain", help="Domain name")
    set_parser.add_argument("metric", help="Metric name to set")
    set_parser.add_argument("value", type=float, help="Threshold value")
    set_parser.add_argument("--model", help="Optional model name", default=None)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export configuration")
    export_parser.add_argument("--output", help="Output file path", default=str(DEFAULT_CONFIG_FILE))
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import configuration")
    import_parser.add_argument("input", help="Input file path")
    
    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize thresholds")
    optimize_parser.add_argument("domain", help="Domain to optimize")
    optimize_parser.add_argument("metric", help="Metric to optimize")
    optimize_parser.add_argument("benchmark_file", help="Benchmark data file path")
    optimize_parser.add_argument("--model", help="Optional model name", default=None)
    optimize_parser.add_argument("--steps", type=int, help="Number of optimization steps", default=10)
    optimize_parser.add_argument("--apply", action="store_true", help="Apply optimized threshold")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Get configuration summary")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset configuration to defaults")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Execute command
    if args.command == "get":
        if args.metric:
            value = ThresholdConfig.get_threshold(args.domain, args.metric, args.model)
            print(f"{args.domain} / {args.metric}: {value}")
        else:
            thresholds = ThresholdConfig.get_domain_thresholds(args.domain)
            print(f"Thresholds for {args.domain}:")
            for metric in MetricType:
                value = getattr(thresholds, metric.value)
                print(f"  {metric.value}: {value}")
    
    elif args.command == "set":
        if args.model:
            ThresholdConfig.set_model_threshold(args.model, args.domain, args.metric, args.value)
            print(f"Set {args.domain} / {args.metric} threshold to {args.value} for model {args.model}")
        else:
            ThresholdConfig.set_domain_threshold(args.domain, args.metric, args.value)
            print(f"Set {args.domain} / {args.metric} threshold to {args.value}")
    
    elif args.command == "export":
        success = ThresholdConfig.export_configuration(args.output)
        if success:
            print(f"Configuration exported to {args.output}")
        else:
            print("Failed to export configuration")
            return 1
    
    elif args.command == "import":
        success = ThresholdConfig.import_configuration(args.input)
        if success:
            print(f"Configuration imported from {args.input}")
        else:
            print("Failed to import configuration")
            return 1
    
    elif args.command == "optimize":
        try:
            # Load benchmark data
            with open(args.benchmark_file, 'r') as f:
                benchmark_data = json.load(f)
            
            # Optimize threshold
            result = ThresholdConfig.optimize_threshold(
                domain=args.domain,
                metric_type=args.metric,
                benchmark_data=benchmark_data,
                steps=args.steps,
                model=args.model
            )
            
            # Print results
            print(f"Threshold Optimization Results for {args.domain} / {args.metric}:")
            print(f"  Original threshold: {result.original_threshold}")
            print(f"  Optimized threshold: {result.optimized_threshold}")
            print(f"  Improvement score: {result.improvement_score}")
            print(f"  Confidence interval: {result.confidence_interval}")
            
            # Apply if requested
            if args.apply:
                ThresholdConfig.apply_optimized_threshold(result)
                print("Optimized threshold applied to configuration")
        
        except Exception as e:
            print(f"Error during optimization: {e}")
            return 1
    
    elif args.command == "summary":
        summary = ThresholdConfig.get_configuration_summary()
        print("Threshold Configuration Summary:")
        print(f"  Domains: {', '.join(summary['domains'])}")
        print(f"  Overridden domains: {', '.join(summary['overridden_domains'])}")
        print(f"  Models: {', '.join(summary['models'])}")
        print(f"  Total domain count: {summary['domain_count']}")
        print(f"  Override count: {summary['override_count']}")
        print(f"  Model count: {summary['model_count']}")
        print(f"  Last updated: {summary['metadata'].get('last_updated', 'unknown')}")
        print(f"  Version: {summary['metadata'].get('version', 'unknown')}")
    
    elif args.command == "reset":
        ThresholdConfig.reset_to_defaults()
        print("Configuration reset to defaults")
    
    else:
        parser.print_help()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())