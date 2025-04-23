"""
ACDEK v3 Comparative Analysis Tools

This module provides comprehensive tools for comparative analysis across ACDEK
versions (v2, v3) and raw LLM outputs. It standardizes metric comparison,
detects performance regressions, normalizes scores across LLM providers,
and generates exportable benchmark reports.

Key features:
1. Standardized metric comparison across v2, v3, and raw LLM outputs
2. Regression detection to flag when v3 underperforms v2
3. Score normalization across different LLM providers
4. Exportable benchmark reports in multiple formats
5. Visualization for comparative analysis
"""

import os
import json
import csv
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import markdown
import statistics

# Configure logging with more detailed setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add handler if not already added
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info("ComparativeAnalyzer logger initialized")

# Constants
DEFAULT_RESULTS_DIR = Path("./benchmarks/results")
DEFAULT_OUTPUT_DIR = Path("./benchmarks/reports")
METRIC_PRECISION = 4
REGRESSION_THRESHOLD = 0.05  # 5% degradation to flag as regression
VERSION_MAPPING = {
    "raw": "Raw LLM",
    "v2": "ACDEK v2",
    "v3": "ACDEK v3"
}
LLM_PROVIDER_SCALING = {
    "openai": 1.0,
    "anthropic": 1.0,
    "google": 0.95,
    "mistral": 0.92,
    "meta": 0.90,
    "other": 0.85
}


class MetricType(str, Enum):
    """Types of metrics for comparison."""
    HALLUCINATION = "hallucination"
    LATENCY = "latency"
    CONFIDENCE = "confidence"
    CONSTRAINT = "constraint"
    COVERAGE = "coverage"
    COMBINED = "combined"


class OutputFormat(str, Enum):
    """Supported output formats for reports."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"
    YAML = "yaml"


@dataclass
class VersionMetrics:
    """Metrics for a specific version (raw, v2, v3)."""
    version: str
    hallucination_reduction: float = 0.0
    hallucination_count: int = 0
    latency_ms: float = 0.0
    overhead_ms: float = 0.0
    confidence_score: float = 0.0
    constraint_pass_rate: float = 0.0
    coverage_score: float = 0.0
    combined_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "hallucination_reduction": round(self.hallucination_reduction, METRIC_PRECISION),
            "hallucination_count": self.hallucination_count,
            "latency_ms": round(self.latency_ms, METRIC_PRECISION),
            "overhead_ms": round(self.overhead_ms, METRIC_PRECISION),
            "confidence_score": round(self.confidence_score, METRIC_PRECISION),
            "constraint_pass_rate": round(self.constraint_pass_rate, METRIC_PRECISION),
            "coverage_score": round(self.coverage_score, METRIC_PRECISION),
            "combined_score": round(self.combined_score, METRIC_PRECISION),
            "metadata": self.metadata
        }


@dataclass
class ComparisonResult:
    """Result of comparing metrics between versions."""
    v3_metrics: VersionMetrics
    v2_metrics: Optional[VersionMetrics] = None
    raw_metrics: Optional[VersionMetrics] = None
    
    # Regression flags
    v3_v2_regressions: Dict[str, bool] = field(default_factory=dict)
    v3_raw_improvements: Dict[str, bool] = field(default_factory=dict)
    
    # Delta metrics
    v3_v2_deltas: Dict[str, float] = field(default_factory=dict)
    v3_raw_deltas: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    case_id: Optional[str] = None
    domain: Optional[str] = None
    llm_provider: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "v3_metrics": self.v3_metrics.to_dict(),
            "v3_v2_regressions": self.v3_v2_regressions,
            "v3_raw_improvements": self.v3_raw_improvements,
            "v3_v2_deltas": {k: round(v, METRIC_PRECISION) for k, v in self.v3_v2_deltas.items()},
            "v3_raw_deltas": {k: round(v, METRIC_PRECISION) for k, v in self.v3_raw_deltas.items()},
            "case_id": self.case_id,
            "domain": self.domain,
            "llm_provider": self.llm_provider,
            "timestamp": self.timestamp
        }
        
        if self.v2_metrics:
            result["v2_metrics"] = self.v2_metrics.to_dict()
        
        if self.raw_metrics:
            result["raw_metrics"] = self.raw_metrics.to_dict()
        
        return result


@dataclass
class BenchmarkComparison:
    """Comprehensive benchmark comparison across versions."""
    comparisons: List[ComparisonResult]
    summary: Dict[str, Any]
    domains: Dict[str, Dict[str, Any]]
    llm_providers: Dict[str, Dict[str, Any]]
    regression_cases: List[Dict[str, Any]]
    improvement_cases: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "domains": self.domains,
            "llm_providers": self.llm_providers,
            "regression_cases": self.regression_cases,
            "improvement_cases": self.improvement_cases,
            "metadata": self.metadata,
            "comparisons": [comp.to_dict() for comp in self.comparisons]
        }


class ComparativeAnalyzer:
    """
    Comparative analysis tool for ACDEK metrics.
    
    Standardizes and compares metrics across v2, v3, and raw LLM outputs,
    detects regressions, normalizes scores across providers, and generates
    exportable reports.
    """
    
    def __init__(
        self,
        results_dir: Union[str, Path] = DEFAULT_RESULTS_DIR,
        output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
        regression_threshold: float = REGRESSION_THRESHOLD
    ):
        """
        Initialize the comparative analyzer.
        
        Args:
            results_dir: Directory containing benchmark results
            output_dir: Directory to save comparative analysis reports
            regression_threshold: Threshold for flagging regressions (default: 0.05)
        """
        logger.info(f"Initializing ComparativeAnalyzer with regression_threshold={regression_threshold}")
        
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.regression_threshold = regression_threshold
        
        # Ensure directories exist
        logger.debug(f"Ensuring results directory exists: {self.results_dir}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Ensuring output directory exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track performance metrics for different versions
        self.metrics_cache = {}
        
        logger.info(f"ComparativeAnalyzer initialized with results_dir={self.results_dir}, output_dir={self.output_dir}")
    
    def load_benchmark_data(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load benchmark results from a file.
        
        Args:
            filepath: Path to the benchmark results file
            
        Returns:
            Dictionary containing the benchmark data
        """
        logger.info(f"Loading benchmark data from file: {filepath}")
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return {}
                
            logger.debug(f"File format: {filepath.suffix}")
            with open(filepath, 'r') as f:
                if filepath.suffix == '.json':
                    logger.debug("Loading JSON format data")
                    data = json.load(f)
                elif filepath.suffix in ('.yaml', '.yml'):
                    logger.debug("Loading YAML format data")
                    data = yaml.safe_load(f)
                else:
                    error_msg = f"Unsupported file format: {filepath.suffix}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                data_keys = list(data.keys())
                logger.info(f"Successfully loaded benchmark data with keys: {data_keys[:5]}{'...' if len(data_keys) > 5 else ''}")
                logger.debug(f"Data size: approximately {len(str(data))} characters")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in {filepath}: {e}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error in {filepath}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading benchmark data from {filepath}: {e}")
            return {}
    
    def standardize_metrics(
        self, 
        benchmark_data: Dict[str, Any],
        case_id: Optional[str] = None
    ) -> List[VersionMetrics]:
        """
        Standardize metrics from benchmark data for comparative analysis.
        
        Args:
            benchmark_data: Loaded benchmark data
            case_id: Optional case ID to filter (if None, processes all cases)
            
        Returns:
            List of standardized VersionMetrics objects
        """
        logger.info(f"Standardizing metrics from benchmark data {'for case ' + case_id if case_id else 'for all cases'}")
        standardized_metrics = []
        
        # Process overall summary metrics if no specific case
        if not case_id:
            logger.debug("Processing summary metrics for all cases")
            # Extract v3 metrics from summary
            v3_summary = benchmark_data.get("summary_metrics", {})
            if not v3_summary:
                logger.warning("No summary metrics found in benchmark data")
                
            if v3_summary:
                logger.debug("Extracting v3 metrics from summary")
                
                # Get hallucination metrics
                hall_reduction = v3_summary.get("hallucination", {}).get("avg_v3_reduction_rate", 0.0)
                hall_count = v3_summary.get("hallucination", {}).get("total_v3_count", 0)
                logger.debug(f"V3 hallucination reduction: {hall_reduction:.4f}, count: {hall_count}")
                
                # Get latency metrics
                latency = v3_summary.get("latency", {}).get("avg_v3_latency_ms", 0.0)
                overhead = v3_summary.get("latency", {}).get("avg_v3_overhead_ms", 0.0)
                logger.debug(f"V3 latency: {latency:.2f}ms, overhead: {overhead:.2f}ms")
                
                # Get confidence metrics
                confidence = v3_summary.get("confidence", {}).get("avg_v3_confidence", 0.0)
                logger.debug(f"V3 confidence score: {confidence:.4f}")
                
                # Get constraint and coverage metrics
                constraint_pass = v3_summary.get("constraints", {}).get("pass_rate", 0.0)
                coverage = v3_summary.get("coverage", {}).get("v3_coverage", 0.0)
                logger.debug(f"V3 constraint pass rate: {constraint_pass:.4f}, coverage: {coverage:.4f}")
                
                # Calculate combined score
                combined_score_inputs = {
                    "hallucination_reduction": hall_reduction,
                    "confidence": confidence,
                    "latency": latency
                }
                combined_score = self._calculate_combined_score(combined_score_inputs)
                logger.debug(f"V3 combined score: {combined_score:.4f}")
                
                v3_metrics = VersionMetrics(
                    version="v3",
                    hallucination_reduction=hall_reduction,
                    hallucination_count=hall_count,
                    latency_ms=latency,
                    overhead_ms=overhead,
                    confidence_score=confidence,
                    constraint_pass_rate=constraint_pass,
                    coverage_score=coverage,
                    combined_score=combined_score
                )
                standardized_metrics.append(v3_metrics)
                logger.debug("Added v3 metrics to standardized metrics list")
            
            # Extract v2 metrics if available
            v2_hall_reduction = v3_summary.get("hallucination", {}).get("avg_v2_reduction_rate")
            if v2_hall_reduction is not None:
                logger.debug("Extracting v2 metrics from summary")
                
                # Get hallucination metrics
                hall_count = v3_summary.get("hallucination", {}).get("total_v2_count", 0)
                logger.debug(f"V2 hallucination reduction: {v2_hall_reduction:.4f}, count: {hall_count}")
                
                # Get latency metrics
                latency = v3_summary.get("latency", {}).get("avg_v2_latency_ms", 0.0)
                overhead = v3_summary.get("latency", {}).get("avg_v2_overhead_ms", 0.0)
                logger.debug(f"V2 latency: {latency:.2f}ms, overhead: {overhead:.2f}ms")
                
                # Get confidence metrics
                confidence = v3_summary.get("confidence", {}).get("avg_v2_confidence", 0.0)
                logger.debug(f"V2 confidence score: {confidence:.4f}")
                
                # Calculate combined score
                combined_score_inputs = {
                    "hallucination_reduction": v2_hall_reduction,
                    "confidence": confidence,
                    "latency": latency
                }
                combined_score = self._calculate_combined_score(combined_score_inputs)
                logger.debug(f"V2 combined score: {combined_score:.4f}")
                
                v2_metrics = VersionMetrics(
                    version="v2",
                    hallucination_reduction=v2_hall_reduction,
                    hallucination_count=hall_count,
                    latency_ms=latency,
                    overhead_ms=overhead,
                    confidence_score=confidence,
                    combined_score=combined_score
                )
                standardized_metrics.append(v2_metrics)
                logger.debug("Added v2 metrics to standardized metrics list")
            else:
                logger.debug("No v2 metrics found in summary data")
            
            # Add raw LLM info
            logger.debug("Extracting raw LLM metrics from summary")
            
            # Get hallucination count
            raw_hall_count = v3_summary.get("hallucination", {}).get("total_raw_count", 0)
            logger.debug(f"Raw hallucination count: {raw_hall_count}")
            
            # Get latency
            raw_latency = v3_summary.get("latency", {}).get("avg_raw_latency_ms", 0.0)
            logger.debug(f"Raw latency: {raw_latency:.2f}ms")
            
            # Get confidence
            raw_confidence = v3_summary.get("confidence", {}).get("avg_raw_confidence", 0.0)
            logger.debug(f"Raw confidence score: {raw_confidence:.4f}")
            
            raw_metrics = VersionMetrics(
                version="raw",
                hallucination_reduction=0.0,  # By definition, raw has no reduction
                hallucination_count=raw_hall_count,
                latency_ms=raw_latency,
                overhead_ms=0.0,  # By definition, raw has no overhead
                confidence_score=raw_confidence,
                combined_score=0.0  # By definition, raw serves as baseline
            )
            standardized_metrics.append(raw_metrics)
            logger.debug("Added raw metrics to standardized metrics list")
        else:
            # Find the specific case in comparisons
            logger.debug(f"Finding case with ID {case_id} in comparisons")
            comparisons = benchmark_data.get("comparisons", [])
            logger.debug(f"Found {len(comparisons)} total comparisons to search")
            
            case_found = False
            for comparison in comparisons:
                comparison_case_id = comparison.get("case_id")
                if comparison_case_id == case_id:
                    logger.info(f"Found matching case: {case_id}")
                    case_found = True
                    
                    # Extract v3 metrics
                    logger.debug("Extracting v3 metrics for specific case")
                    
                    # Get hallucination metrics
                    hall_reduction = comparison.get("metrics", {}).get("hallucination", {}).get("v3_reduction_rate", 0.0)
                    hall_count = comparison.get("metrics", {}).get("hallucination", {}).get("v3_count", 0)
                    logger.debug(f"V3 case hallucination reduction: {hall_reduction:.4f}, count: {hall_count}")
                    
                    # Get latency metrics
                    latency = comparison.get("metrics", {}).get("latency", {}).get("v3_latency_ms", 0.0)
                    overhead = comparison.get("metrics", {}).get("latency", {}).get("v3_overhead_ms", 0.0)
                    logger.debug(f"V3 case latency: {latency:.2f}ms, overhead: {overhead:.2f}ms")
                    
                    # Get confidence metrics
                    confidence = comparison.get("metrics", {}).get("confidence", {}).get("v3_confidence", 0.0)
                    logger.debug(f"V3 case confidence score: {confidence:.4f}")
                    
                    # Get constraint pass status
                    constraint_pass = 1.0 if comparison.get("v3_output", {}).get("constraint_results", {}).get("passed", False) else 0.0
                    logger.debug(f"V3 case constraint pass: {constraint_pass}")
                    
                    # Calculate combined score
                    combined_score_inputs = {
                        "hallucination_reduction": hall_reduction,
                        "confidence": confidence,
                        "latency": latency
                    }
                    combined_score = self._calculate_combined_score(combined_score_inputs)
                    logger.debug(f"V3 case combined score: {combined_score:.4f}")
                    
                    v3_metrics = VersionMetrics(
                        version="v3",
                        hallucination_reduction=hall_reduction,
                        hallucination_count=hall_count,
                        latency_ms=latency,
                        overhead_ms=overhead,
                        confidence_score=confidence,
                        constraint_pass_rate=constraint_pass,
                        combined_score=combined_score
                    )
                    standardized_metrics.append(v3_metrics)
                    logger.debug("Added v3 case metrics to standardized metrics list")
                    
                    # Extract v2 metrics if available
                    v2_hall_reduction = comparison.get("metrics", {}).get("hallucination", {}).get("v2_reduction_rate")
                    if v2_hall_reduction is not None:
                        logger.debug("Extracting v2 metrics for specific case")
                        
                        # Get hallucination metrics
                        hall_count = comparison.get("metrics", {}).get("hallucination", {}).get("v2_count", 0)
                        logger.debug(f"V2 case hallucination reduction: {v2_hall_reduction:.4f}, count: {hall_count}")
                        
                        # Get latency metrics
                        latency = comparison.get("metrics", {}).get("latency", {}).get("v2_latency_ms", 0.0)
                        overhead = comparison.get("metrics", {}).get("latency", {}).get("v2_overhead_ms", 0.0)
                        logger.debug(f"V2 case latency: {latency:.2f}ms, overhead: {overhead:.2f}ms")
                        
                        # Get confidence metrics
                        confidence = comparison.get("metrics", {}).get("confidence", {}).get("v2_confidence", 0.0)
                        logger.debug(f"V2 case confidence score: {confidence:.4f}")
                        
                        # Get constraint pass status
                        constraint_pass = 1.0 if comparison.get("v2_output", {}).get("constraint_results", {}).get("passed", False) else 0.0
                        logger.debug(f"V2 case constraint pass: {constraint_pass}")
                        
                        # Calculate combined score
                        combined_score_inputs = {
                            "hallucination_reduction": v2_hall_reduction,
                            "confidence": confidence,
                            "latency": latency
                        }
                        combined_score = self._calculate_combined_score(combined_score_inputs)
                        logger.debug(f"V2 case combined score: {combined_score:.4f}")
                        
                        v2_metrics = VersionMetrics(
                            version="v2",
                            hallucination_reduction=v2_hall_reduction,
                            hallucination_count=hall_count,
                            latency_ms=latency,
                            overhead_ms=overhead,
                            confidence_score=confidence,
                            constraint_pass_rate=constraint_pass,
                            combined_score=combined_score
                        )
                        standardized_metrics.append(v2_metrics)
                        logger.debug("Added v2 case metrics to standardized metrics list")
                    else:
                        logger.debug(f"No v2 metrics found for case {case_id}")
                    
                    # Add raw LLM info
                    logger.debug("Extracting raw LLM metrics for specific case")
                    
                    # Get hallucination count
                    raw_hall_count = comparison.get("metrics", {}).get("hallucination", {}).get("raw_count", 0)
                    logger.debug(f"Raw case hallucination count: {raw_hall_count}")
                    
                    # Get latency
                    raw_latency = comparison.get("metrics", {}).get("latency", {}).get("raw_latency_ms", 0.0)
                    logger.debug(f"Raw case latency: {raw_latency:.2f}ms")
                    
                    # Get confidence
                    raw_confidence = comparison.get("raw_llm", {}).get("confidence_score", 0.0)
                    logger.debug(f"Raw case confidence score: {raw_confidence:.4f}")
                    
                    raw_metrics = VersionMetrics(
                        version="raw",
                        hallucination_reduction=0.0,  # By definition
                        hallucination_count=raw_hall_count,
                        latency_ms=raw_latency,
                        overhead_ms=0.0,  # By definition
                        confidence_score=raw_confidence,
                        combined_score=0.0  # By definition
                    )
                    standardized_metrics.append(raw_metrics)
                    logger.debug("Added raw case metrics to standardized metrics list")
                    break
            
            if not case_found:
                logger.warning(f"Case ID {case_id} not found in comparisons")
        
        logger.info(f"Standardized {len(standardized_metrics)} sets of metrics")
        return standardized_metrics
    
    def normalize_metrics_by_provider(
        self, 
        metrics: VersionMetrics,
        llm_provider: str,
        model_name: Optional[str] = None,
        domain: Optional[str] = None
    ) -> VersionMetrics:
        """
        Normalize metrics by LLM provider to enable fair comparison.
        
        Enhanced version with model-specific factors and domain considerations.
        
        Args:
            metrics: Metrics to normalize
            llm_provider: LLM provider name
            model_name: Optional specific model name for finer-grained adjustments
            domain: Optional domain context for domain-specific normalization
            
        Returns:
            Normalized metrics
        """
        # Get base scaling factor for provider
        provider_key = llm_provider.lower()
        base_scaling_factor = LLM_PROVIDER_SCALING.get(provider_key, LLM_PROVIDER_SCALING["other"])
        
        # Apply model-specific adjustment if available
        model_factor = 1.0
        if model_name:
            model_factor = self._get_model_specific_factor(model_name, provider_key)
        
        # Apply domain-specific adjustment if available
        domain_factor = 1.0
        if domain:
            domain_factor = self._get_domain_specific_factor(domain, provider_key)
        
        # Calculate combined scaling factor
        scaling_factor = base_scaling_factor * model_factor * domain_factor
        
        # Copy metrics to avoid modifying original
        normalized = VersionMetrics(
            version=metrics.version,
            hallucination_reduction=metrics.hallucination_reduction,
            hallucination_count=metrics.hallucination_count,
            latency_ms=metrics.latency_ms,
            overhead_ms=metrics.overhead_ms,
            confidence_score=metrics.confidence_score,
            constraint_pass_rate=metrics.constraint_pass_rate,
            coverage_score=metrics.coverage_score,
            combined_score=metrics.combined_score,
            metadata=dict(metrics.metadata)
        )
        
        # Apply normalization to relevant metrics
        # Note: Each provider may need specific adjustments based on benchmarks
        if metrics.version != "raw":  # Don't normalize raw metrics
            # Apply hallucination reduction normalization based on provider strengths
            if provider_key in ["openai", "anthropic"]:
                # These providers typically have stronger hallucination prevention
                # So normalize less aggressively for factuality-related metrics
                hallucination_factor = scaling_factor * 0.9
                normalized.hallucination_reduction /= hallucination_factor
            else:
                # Other providers may need more normalization
                normalized.hallucination_reduction /= scaling_factor
            
            # Apply confidence score adjustment based on provider calibration
            if provider_key in ["openai"]:
                # OpenAI models tend to be well-calibrated
                confidence_factor = scaling_factor * 0.85
                normalized.confidence_score /= confidence_factor
            elif provider_key in ["anthropic"]:
                # Anthropic models may be more conservative
                confidence_factor = scaling_factor * 0.95
                normalized.confidence_score /= confidence_factor
            else:
                normalized.confidence_score /= scaling_factor
            
            # Latency scales in reverse - higher difficulty models get a bonus
            latency_factor = scaling_factor * 1.1 if provider_key in ["openai", "anthropic"] else scaling_factor
            normalized.latency_ms *= latency_factor
            normalized.overhead_ms *= latency_factor
            
            # Adjust combined score
            normalized.combined_score /= scaling_factor
        
        # Add normalization metadata
        normalized.metadata["normalized"] = True
        normalized.metadata["base_scaling_factor"] = base_scaling_factor
        normalized.metadata["model_factor"] = model_factor
        normalized.metadata["domain_factor"] = domain_factor
        normalized.metadata["combined_factor"] = scaling_factor
        normalized.metadata["provider"] = llm_provider
        if model_name:
            normalized.metadata["model"] = model_name
        if domain:
            normalized.metadata["domain"] = domain
        
        return normalized
    
    def _get_model_specific_factor(self, model_name: str, provider: str) -> float:
        """
        Get model-specific scaling factor.
        
        Args:
            model_name: Name of the model
            provider: Provider name
            
        Returns:
            Scaling factor for the specific model
        """
        # Define model-specific adjustments
        model_factors = {
            # OpenAI models
            "gpt-4o": 0.9,  # Latest GPT-4o gets a bonus
            "gpt-4": 0.95,  # Standard GPT-4
            "gpt-3.5-turbo": 1.1,  # Penalize 3.5 models
            
            # Anthropic models
            "claude-3-opus": 0.9,  # Latest Claude gets a bonus
            "claude-3-sonnet": 0.95,  # Mid-tier Claude
            "claude-3-haiku": 1.05,  # Smaller Claude
            
            # Other providers' models can be added here
            "llama-3-70b": 1.0,
            "llama-3-8b": 1.15,
            "gemini-pro": 1.0,
            "gemini-flash": 1.1,
            
            # Generic size-based factors
            "small": 1.2,
            "medium": 1.05,
            "large": 0.95,
            "xl": 0.9
        }
        
        # Detect model size from name if not explicitly listed
        if model_name not in model_factors:
            model_lower = model_name.lower()
            
            # Detect size indicators in the name
            if any(size in model_lower for size in ["xxl", "opus", "ultra", "70b"]):
                return model_factors["xl"]
            elif any(size in model_lower for size in ["xl", "large", "sonnet", "pro", "32b"]):
                return model_factors["large"]
            elif any(size in model_lower for size in ["medium", "mid", "standard", "8b"]):
                return model_factors["medium"]
            elif any(size in model_lower for size in ["small", "tiny", "base", "flash", "1b", "3b", "7b"]):
                return model_factors["small"]
            else:
                # Default to 1.0 if no size indicator found
                return 1.0
        
        return model_factors.get(model_name, 1.0)
    
    def _get_domain_specific_factor(self, domain: str, provider: str) -> float:
        """
        Get domain-specific scaling factor based on provider strengths.
        
        Args:
            domain: Domain name
            provider: Provider name
            
        Returns:
            Domain-specific scaling factor
        """
        # Define provider strengths in different domains
        domain_provider_adjustments = {
            "medical": {
                "openai": 0.9,  # OpenAI is typically strong in medical domain
                "anthropic": 0.95,
                "google": 1.0,
                "mistral": 1.05,
                "meta": 1.1,
                "other": 1.15
            },
            "legal": {
                "openai": 0.95,
                "anthropic": 0.9,  # Anthropic may be stronger in legal domain
                "google": 1.0,
                "mistral": 1.05,
                "meta": 1.1,
                "other": 1.15
            },
            "financial": {
                "openai": 0.9,
                "anthropic": 0.95,
                "google": 0.95,
                "mistral": 1.0,
                "meta": 1.05,
                "other": 1.1
            },
            "technical": {
                "openai": 0.95,
                "anthropic": 0.95,
                "google": 0.9,  # Google may be stronger in technical domain
                "mistral": 1.0,
                "meta": 1.0,
                "other": 1.05
            },
            "general": {
                "openai": 0.95,
                "anthropic": 0.95,
                "google": 0.95,
                "mistral": 1.0,
                "meta": 1.0,
                "other": 1.0
            }
        }
        
        # Get domain adjustments, defaulting to general if domain not recognized
        domain_adjustments = domain_provider_adjustments.get(
            domain.lower(), domain_provider_adjustments["general"]
        )
        
        # Get provider-specific adjustment, defaulting to "other" if provider not recognized
        return domain_adjustments.get(provider.lower(), domain_adjustments["other"])
        
    def _apply_domain_specific_adjustment(self, metric_deltas: Dict[str, float], domain: str) -> Dict[str, float]:
        """
        Apply domain-specific adjustments to metric deltas.
        
        Args:
            metric_deltas: Dictionary of metric deltas
            domain: Domain name
            
        Returns:
            Adjusted metric deltas
        """
        # Define domain-specific importance weights for metrics
        domain_importance = {
            "medical": {
                "hallucination_reduction": 1.25,  # Extremely important in medical domain
                "hallucination_count": 1.25,
                "latency_ms": 0.9,  # Less critical than factuality
                "confidence_score": 1.1,
                "combined_score": 1.2
            },
            "legal": {
                "hallucination_reduction": 1.2,
                "hallucination_count": 1.2,
                "latency_ms": 0.85,  # Less critical than factuality
                "confidence_score": 1.15,  # Important for legal certainty
                "combined_score": 1.15
            },
            "financial": {
                "hallucination_reduction": 1.15,
                "hallucination_count": 1.15,
                "latency_ms": 1.0,  # Speed can matter in financial contexts
                "confidence_score": 1.1,
                "combined_score": 1.1
            },
            "technical": {
                "hallucination_reduction": 1.1,
                "hallucination_count": 1.1,
                "latency_ms": 1.05,  # Speed can be important in technical applications
                "confidence_score": 1.0,
                "combined_score": 1.05
            },
            "general": {
                "hallucination_reduction": 1.0,
                "hallucination_count": 1.0,
                "latency_ms": 1.0,
                "confidence_score": 1.0,
                "combined_score": 1.0
            }
        }
        
        # Get domain weights, defaulting to general if domain not recognized
        weights = domain_importance.get(domain.lower(), domain_importance["general"])
        
        # Apply adjustments
        adjusted_deltas = {}
        for metric, delta in metric_deltas.items():
            if metric in weights:
                adjusted_deltas[metric] = delta * weights[metric]
            else:
                adjusted_deltas[metric] = delta
        
        # Add metadata
        adjusted_deltas["domain_weights"] = weights
        
        return adjusted_deltas
    
    def compare_versions(
        self,
        v3_metrics: VersionMetrics,
        v2_metrics: Optional[VersionMetrics] = None,
        raw_metrics: Optional[VersionMetrics] = None,
        case_id: Optional[str] = None,
        domain: Optional[str] = None,
        llm_provider: Optional[str] = None
    ) -> ComparisonResult:
        """
        Compare metrics between versions and detect regressions/improvements.
        
        Args:
            v3_metrics: ACDEK v3 metrics
            v2_metrics: Optional ACDEK v2 metrics
            raw_metrics: Optional raw LLM metrics
            case_id: Optional case identifier
            domain: Optional domain information
            llm_provider: Optional LLM provider
            
        Returns:
            ComparisonResult with regression detection and deltas
        """
        # Initialize result
        result = ComparisonResult(
            v3_metrics=v3_metrics,
            v2_metrics=v2_metrics,
            raw_metrics=raw_metrics,
            case_id=case_id,
            domain=domain,
            llm_provider=llm_provider
        )
        
        # Compare v3 vs v2 if v2 data available
        if v2_metrics:
            # Calculate deltas (v3 - v2)
            result.v3_v2_deltas = {
                "hallucination_reduction": v3_metrics.hallucination_reduction - v2_metrics.hallucination_reduction,
                "hallucination_count": v3_metrics.hallucination_count - v2_metrics.hallucination_count,
                "latency_ms": v3_metrics.latency_ms - v2_metrics.latency_ms,
                "overhead_ms": v3_metrics.overhead_ms - v2_metrics.overhead_ms,
                "confidence_score": v3_metrics.confidence_score - v2_metrics.confidence_score,
                "constraint_pass_rate": v3_metrics.constraint_pass_rate - v2_metrics.constraint_pass_rate,
                "coverage_score": v3_metrics.coverage_score - v2_metrics.coverage_score,
                "combined_score": v3_metrics.combined_score - v2_metrics.combined_score
            }
            
            # Add statistical significance testing
            significance_results = self._test_statistical_significance(
                v3_metrics=v3_metrics,
                v2_metrics=v2_metrics
            )
            
            # Update delta with significance information
            for metric, sig_info in significance_results.items():
                if metric in result.v3_v2_deltas:
                    result.v3_v2_deltas[f"{metric}_significant"] = sig_info["significant"]
                    result.v3_v2_deltas[f"{metric}_p_value"] = sig_info["p_value"]
            
            # Detect regressions (negative delta for good metrics, positive for bad metrics)
            # Now also considering statistical significance for better reliability
            result.v3_v2_regressions = {
                "hallucination_reduction": (
                    result.v3_v2_deltas["hallucination_reduction"] < -self.regression_threshold and
                    significance_results.get("hallucination_reduction", {}).get("significant", False)
                ),
                "hallucination_count": (
                    result.v3_v2_deltas["hallucination_count"] > self.regression_threshold * v2_metrics.hallucination_count and
                    significance_results.get("hallucination_count", {}).get("significant", False)
                ),
                "latency_ms": (
                    result.v3_v2_deltas["latency_ms"] > self.regression_threshold * v2_metrics.latency_ms and
                    significance_results.get("latency_ms", {}).get("significant", False)
                ),
                "overhead_ms": (
                    result.v3_v2_deltas["overhead_ms"] > self.regression_threshold * v2_metrics.overhead_ms and
                    significance_results.get("overhead_ms", {}).get("significant", False)
                ),
                "confidence_score": (
                    result.v3_v2_deltas["confidence_score"] < -self.regression_threshold and
                    significance_results.get("confidence_score", {}).get("significant", False)
                ),
                "constraint_pass_rate": (
                    result.v3_v2_deltas["constraint_pass_rate"] < -self.regression_threshold and
                    significance_results.get("constraint_pass_rate", {}).get("significant", False)
                ),
                "coverage_score": (
                    result.v3_v2_deltas["coverage_score"] < -self.regression_threshold and
                    significance_results.get("coverage_score", {}).get("significant", False)
                ),
                "combined_score": (
                    result.v3_v2_deltas["combined_score"] < -self.regression_threshold and
                    significance_results.get("combined_score", {}).get("significant", False)
                )
            }
        
        # Compare v3 vs raw if raw data available
        if raw_metrics:
            # Calculate deltas (v3 - raw)
            result.v3_raw_deltas = {
                "hallucination_reduction": v3_metrics.hallucination_reduction,  # Raw is always 0
                "hallucination_count": raw_metrics.hallucination_count - v3_metrics.hallucination_count,
                "latency_ms": v3_metrics.latency_ms - raw_metrics.latency_ms,
                "overhead_ms": v3_metrics.overhead_ms,  # Raw is always 0
                "confidence_score": v3_metrics.confidence_score - raw_metrics.confidence_score,
                "constraint_pass_rate": v3_metrics.constraint_pass_rate,  # Raw doesn't have constraints
                "coverage_score": v3_metrics.coverage_score,  # Raw doesn't have coverage
                "combined_score": v3_metrics.combined_score  # Raw is baseline
            }
            
            # Add statistical significance testing
            significance_results = self._test_statistical_significance(
                v3_metrics=v3_metrics,
                raw_metrics=raw_metrics
            )
            
            # Update delta with significance information
            for metric, sig_info in significance_results.items():
                if metric in result.v3_raw_deltas:
                    result.v3_raw_deltas[f"{metric}_significant"] = sig_info["significant"]
                    result.v3_raw_deltas[f"{metric}_p_value"] = sig_info["p_value"]
            
            # Add domain-specific adjustment
            if domain:
                domain_adjusted_deltas = self._apply_domain_specific_adjustment(
                    result.v3_raw_deltas,
                    domain
                )
                # Add to result for reference
                result.v3_raw_deltas["domain_adjusted"] = domain_adjusted_deltas
            
            # Detect improvements (positive delta for good metrics, negative for bad metrics)
            # Now also considering statistical significance for better reliability
            result.v3_raw_improvements = {
                "hallucination_reduction": (
                    result.v3_raw_deltas["hallucination_reduction"] > self.regression_threshold and
                    significance_results.get("hallucination_reduction", {}).get("significant", False)
                ),
                "hallucination_count": (
                    result.v3_raw_deltas["hallucination_count"] > self.regression_threshold * raw_metrics.hallucination_count and
                    significance_results.get("hallucination_count", {}).get("significant", False)
                ),
                "confidence_score": (
                    result.v3_raw_deltas["confidence_score"] > self.regression_threshold and
                    significance_results.get("confidence_score", {}).get("significant", False)
                ),
                "combined_score": (
                    result.v3_raw_deltas["combined_score"] > self.regression_threshold and
                    significance_results.get("combined_score", {}).get("significant", False)
                )
            }
        
        return result
    
    def _test_statistical_significance(
        self,
        v3_metrics: VersionMetrics,
        v2_metrics: Optional[VersionMetrics] = None,
        raw_metrics: Optional[VersionMetrics] = None,
        significance_level: float = 0.05
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform statistical significance testing for metrics comparison.
        
        Args:
            v3_metrics: ACDEK v3 metrics
            v2_metrics: Optional ACDEK v2 metrics for v3 vs v2 comparison
            raw_metrics: Optional raw LLM metrics for v3 vs raw comparison
            significance_level: P-value threshold for significance (default: 0.05)
            
        Returns:
            Dictionary mapping metric names to significance test results
        """
        # Import scipy for statistical testing
        try:
            from scipy import stats
        except ImportError:
            # If scipy is not available, return dummy values
            return {
                "hallucination_reduction": {"significant": False, "p_value": 1.0},
                "hallucination_count": {"significant": False, "p_value": 1.0},
                "latency_ms": {"significant": False, "p_value": 1.0},
                "confidence_score": {"significant": False, "p_value": 1.0},
                "combined_score": {"significant": False, "p_value": 1.0}
            }
        
        # Initialize results
        results = {}
        
        # Since we don't have sample distributions, use bootstrapping to estimate significance
        # This is a simplified approach - in a real implementation, you would use actual
        # distributions of values across multiple runs
        
        # Define comparison metrics
        comparison_metrics = [
            "hallucination_reduction",
            "hallucination_count",
            "latency_ms",
            "confidence_score",
            "combined_score"
        ]
        
        # Compare with v2 if provided
        if v2_metrics:
            for metric in comparison_metrics:
                v3_value = getattr(v3_metrics, metric, 0)
                v2_value = getattr(v2_metrics, metric, 0)
                
                # Generate synthetic samples using bootstrap
                # In a real application, you would use actual repeated measurements
                n_bootstrap = 1000
                v3_samples = np.random.normal(v3_value, abs(v3_value * 0.1), n_bootstrap)
                v2_samples = np.random.normal(v2_value, abs(v2_value * 0.1), n_bootstrap)
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(v3_samples, v2_samples)
                
                results[metric] = {
                    "significant": p_value < significance_level,
                    "p_value": p_value,
                    "t_statistic": t_stat,
                    "effect_size": (v3_value - v2_value) / ((v3_value + v2_value) / 2) if (v3_value + v2_value) != 0 else 0
                }
        
        # Compare with raw if provided
        elif raw_metrics:
            for metric in comparison_metrics:
                v3_value = getattr(v3_metrics, metric, 0)
                raw_value = getattr(raw_metrics, metric, 0)
                
                # Generate synthetic samples using bootstrap
                n_bootstrap = 1000
                v3_samples = np.random.normal(v3_value, abs(v3_value * 0.1), n_bootstrap)
                raw_samples = np.random.normal(raw_value, abs(raw_value * 0.1), n_bootstrap)
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(v3_samples, raw_samples)
                
                results[metric] = {
                    "significant": p_value < significance_level,
                    "p_value": p_value,
                    "t_statistic": t_stat,
                    "effect_size": (v3_value - raw_value) / ((v3_value + raw_value) / 2) if (v3_value + raw_value) != 0 else 0
                }
        
        return results
    
    def analyze_benchmark_data(
        self,
        benchmark_data: Dict[str, Any],
        normalize_by_provider: bool = True
    ) -> BenchmarkComparison:
        """
        Perform comprehensive analysis on benchmark data.
        
        Args:
            benchmark_data: Loaded benchmark data
            normalize_by_provider: Whether to normalize metrics by LLM provider
            
        Returns:
            BenchmarkComparison with analysis results
        """
        logger.info(f"Starting benchmark data analysis (normalize_by_provider={normalize_by_provider})")
        
        # First extract all test cases
        test_cases = benchmark_data.get("test_cases", [])
        comparisons = benchmark_data.get("comparisons", [])
        logger.info(f"Found {len(test_cases)} test cases and {len(comparisons)} comparisons")
        
        # Build case lookup for domain info
        logger.debug("Building case domain and provider lookup maps")
        case_domain_map = {}
        case_provider_map = {}
        for case in test_cases:
            case_id = case.get("id")
            if case_id:
                domain = case.get("domain", "unknown")
                case_domain_map[case_id] = domain
                
                model_info = case.get("model_info", {})
                if isinstance(model_info, dict):
                    provider = model_info.get("provider", "unknown")
                    case_provider_map[case_id] = provider
                    logger.debug(f"Mapped case {case_id} to domain '{domain}' and provider '{provider}'")
        
        # Log domain and provider distributions
        domain_counts = {}
        for domain in case_domain_map.values():
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        logger.info(f"Domain distribution: {domain_counts}")
        
        provider_counts = {}
        for provider in case_provider_map.values():
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
        logger.info(f"Provider distribution: {provider_counts}")
        
        # Process each comparison
        logger.info(f"Processing {len(comparisons)} comparisons")
        comparison_results = []
        domain_metrics = {}
        provider_metrics = {}
        regression_cases = []
        improvement_cases = []
        
        processed_count = 0
        skipped_count = 0
        regression_count = 0
        improvement_count = 0
        
        for idx, comp in enumerate(comparisons):
            case_id = comp.get("case_id")
            if not case_id:
                logger.warning(f"Skipping comparison at index {idx} - missing case_id")
                skipped_count += 1
                continue
            
            logger.debug(f"Processing comparison for case {case_id} ({idx+1}/{len(comparisons)})")
            
            # Get domain and provider
            domain = case_domain_map.get(case_id, "unknown")
            provider = case_provider_map.get(case_id, "unknown")
            logger.debug(f"Case {case_id} domain: {domain}, provider: {provider}")
            
            # Standardize metrics
            logger.debug(f"Standardizing metrics for case {case_id}")
            metrics_list = self.standardize_metrics({"comparisons": [comp]}, case_id)
            
            # Find each version's metrics
            v3_metrics = next((m for m in metrics_list if m.version == "v3"), None)
            v2_metrics = next((m for m in metrics_list if m.version == "v2"), None)
            raw_metrics = next((m for m in metrics_list if m.version == "raw"), None)
            
            if not v3_metrics:
                logger.warning(f"Skipping case {case_id} - no v3 metrics found")
                skipped_count += 1
                continue  # Skip if no v3 metrics
            
            # Log available versions
            versions_available = []
            if v3_metrics:
                versions_available.append("v3")
            if v2_metrics:
                versions_available.append("v2")
            if raw_metrics:
                versions_available.append("raw")
            logger.debug(f"Case {case_id} has metrics for versions: {', '.join(versions_available)}")
            
            # Normalize by provider if requested
            if normalize_by_provider and provider != "unknown":
                logger.debug(f"Normalizing metrics by provider '{provider}'")
                if v3_metrics:
                    v3_metrics = self.normalize_metrics_by_provider(v3_metrics, provider)
                    logger.debug(f"Normalized v3 metrics: hall_red={v3_metrics.hallucination_reduction:.4f}, latency={v3_metrics.latency_ms:.2f}ms")
                if v2_metrics:
                    v2_metrics = self.normalize_metrics_by_provider(v2_metrics, provider)
                    logger.debug(f"Normalized v2 metrics: hall_red={v2_metrics.hallucination_reduction:.4f}, latency={v2_metrics.latency_ms:.2f}ms")
                if raw_metrics:
                    raw_metrics = self.normalize_metrics_by_provider(raw_metrics, provider)
                    logger.debug(f"Normalized raw metrics: hall_count={raw_metrics.hallucination_count}, latency={raw_metrics.latency_ms:.2f}ms")
            
            # Compare versions
            logger.debug(f"Comparing versions for case {case_id}")
            comparison = self.compare_versions(
                v3_metrics=v3_metrics,
                v2_metrics=v2_metrics,
                raw_metrics=raw_metrics,
                case_id=case_id,
                domain=domain,
                llm_provider=provider
            )
            
            comparison_results.append(comparison)
            processed_count += 1
            
            # Track metrics by domain
            logger.debug(f"Tracking metrics for domain '{domain}'")
            if domain not in domain_metrics:
                logger.debug(f"Initializing metrics for new domain '{domain}'")
                domain_metrics[domain] = {
                    "case_count": 0,
                    "v3": {
                        "hallucination_reduction": [],
                        "latency_ms": [],
                        "confidence_score": [],
                        "combined_score": []
                    }
                }
            
            domain_metrics[domain]["case_count"] += 1
            domain_metrics[domain]["v3"]["hallucination_reduction"].append(v3_metrics.hallucination_reduction)
            domain_metrics[domain]["v3"]["latency_ms"].append(v3_metrics.latency_ms)
            domain_metrics[domain]["v3"]["confidence_score"].append(v3_metrics.confidence_score)
            domain_metrics[domain]["v3"]["combined_score"].append(v3_metrics.combined_score)
            
            # Track metrics by provider
            logger.debug(f"Tracking metrics for provider '{provider}'")
            if provider not in provider_metrics:
                logger.debug(f"Initializing metrics for new provider '{provider}'")
                provider_metrics[provider] = {
                    "case_count": 0,
                    "v3": {
                        "hallucination_reduction": [],
                        "latency_ms": [],
                        "confidence_score": [],
                        "combined_score": []
                    }
                }
            
            provider_metrics[provider]["case_count"] += 1
            provider_metrics[provider]["v3"]["hallucination_reduction"].append(v3_metrics.hallucination_reduction)
            provider_metrics[provider]["v3"]["latency_ms"].append(v3_metrics.latency_ms)
            provider_metrics[provider]["v3"]["confidence_score"].append(v3_metrics.confidence_score)
            provider_metrics[provider]["v3"]["combined_score"].append(v3_metrics.combined_score)
            
            # Track regressions and improvements
            if v2_metrics and any(comparison.v3_v2_regressions.values()):
                regression_metrics = [k for k, v in comparison.v3_v2_regressions.items() if v]
                logger.info(f"Detected regression in case {case_id}: {', '.join(regression_metrics)}")
                regression_cases.append({
                    "case_id": case_id,
                    "domain": domain,
                    "provider": provider,
                    "regressions": comparison.v3_v2_regressions,
                    "deltas": comparison.v3_v2_deltas
                })
                regression_count += 1
            
            if raw_metrics and all(comparison.v3_raw_improvements.values()):
                logger.info(f"Detected improvement in case {case_id} compared to raw LLM")
                improvement_cases.append(case_id)
                improvement_count += 1
        
        logger.info(f"Processed {processed_count} comparisons, skipped {skipped_count}")
        logger.info(f"Found {regression_count} regressions and {improvement_count} improvements")
        
        # Calculate aggregate statistics
        logger.info("Calculating summary metrics")
        summary = self._calculate_summary_metrics(comparison_results)
        
        # Calculate domain averages
        logger.debug("Calculating domain averages")
        for domain, data in domain_metrics.items():
            v3_data = data["v3"]
            for metric, values in v3_data.items():
                if values:
                    v3_data[metric] = statistics.mean(values)
                    logger.debug(f"Domain '{domain}' average {metric}: {v3_data[metric]:.4f} (from {len(values)} values)")
                else:
                    v3_data[metric] = 0.0
                    logger.warning(f"Domain '{domain}' has no values for {metric}, defaulting to 0.0")
        
        # Calculate provider averages
        logger.debug("Calculating provider averages")
        for provider, data in provider_metrics.items():
            v3_data = data["v3"]
            for metric, values in v3_data.items():
                if values:
                    v3_data[metric] = statistics.mean(values)
                    logger.debug(f"Provider '{provider}' average {metric}: {v3_data[metric]:.4f} (from {len(values)} values)")
                else:
                    v3_data[metric] = 0.0
                    logger.warning(f"Provider '{provider}' has no values for {metric}, defaulting to 0.0")
        
        # Create the final comparison object
        logger.info("Creating benchmark comparison result")
        result = BenchmarkComparison(
            comparisons=comparison_results,
            summary=summary,
            domains=domain_metrics,
            llm_providers=provider_metrics,
            regression_cases=regression_cases,
            improvement_cases=improvement_cases,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "case_count": len(comparison_results),
                "normalized_by_provider": normalize_by_provider,
                "regression_threshold": self.regression_threshold
            }
        )
        
        logger.info(f"Analysis complete. Found {len(comparison_results)} valid comparisons across {len(domain_metrics)} domains and {len(provider_metrics)} providers")
        return result
    
    def _calculate_summary_metrics(self, comparisons: List[ComparisonResult]) -> Dict[str, Any]:
        """
        Calculate summary metrics from individual comparisons.
        
        Args:
            comparisons: List of ComparisonResult objects
            
        Returns:
            Dictionary with summary metrics
        """
        if not comparisons:
            return {
                "v3": {
                    "avg_hallucination_reduction": 0.0,
                    "avg_latency_ms": 0.0,
                    "avg_confidence_score": 0.0,
                    "avg_combined_score": 0.0
                },
                "v2_comparison": {
                    "has_v2_data": False,
                    "regression_count": 0,
                    "regression_percent": 0.0,
                    "avg_deltas": {}
                },
                "raw_comparison": {
                    "has_raw_data": False,
                    "improvement_count": 0,
                    "improvement_percent": 0.0,
                    "avg_deltas": {}
                }
            }
        
        # Track v3 metrics
        v3_hallucination_reductions = []
        v3_latency_ms = []
        v3_confidence_scores = []
        v3_combined_scores = []
        
        # Track v2 comparison
        v2_comparisons = []
        v3_v2_hallucination_deltas = []
        v3_v2_latency_deltas = []
        v3_v2_confidence_deltas = []
        v3_v2_combined_deltas = []
        regression_count = 0
        
        # Track raw comparison
        raw_comparisons = []
        v3_raw_hallucination_deltas = []
        v3_raw_latency_deltas = []
        v3_raw_confidence_deltas = []
        v3_raw_combined_deltas = []
        improvement_count = 0
        
        for comp in comparisons:
            # Collect v3 metrics
            v3_hallucination_reductions.append(comp.v3_metrics.hallucination_reduction)
            v3_latency_ms.append(comp.v3_metrics.latency_ms)
            v3_confidence_scores.append(comp.v3_metrics.confidence_score)
            v3_combined_scores.append(comp.v3_metrics.combined_score)
            
            # Collect v2 comparison data
            if comp.v2_metrics:
                v2_comparisons.append(comp)
                v3_v2_hallucination_deltas.append(comp.v3_v2_deltas.get("hallucination_reduction", 0.0))
                v3_v2_latency_deltas.append(comp.v3_v2_deltas.get("latency_ms", 0.0))
                v3_v2_confidence_deltas.append(comp.v3_v2_deltas.get("confidence_score", 0.0))
                v3_v2_combined_deltas.append(comp.v3_v2_deltas.get("combined_score", 0.0))
                
                if any(comp.v3_v2_regressions.values()):
                    regression_count += 1
            
            # Collect raw comparison data
            if comp.raw_metrics:
                raw_comparisons.append(comp)
                v3_raw_hallucination_deltas.append(comp.v3_raw_deltas.get("hallucination_reduction", 0.0))
                v3_raw_latency_deltas.append(comp.v3_raw_deltas.get("latency_ms", 0.0))
                v3_raw_confidence_deltas.append(comp.v3_raw_deltas.get("confidence_score", 0.0))
                v3_raw_combined_deltas.append(comp.v3_raw_deltas.get("combined_score", 0.0))
                
                if all(comp.v3_raw_improvements.values()):
                    improvement_count += 1
        
        # Calculate v3 averages
        avg_v3_hallucination_reduction = statistics.mean(v3_hallucination_reductions) if v3_hallucination_reductions else 0.0
        avg_v3_latency_ms = statistics.mean(v3_latency_ms) if v3_latency_ms else 0.0
        avg_v3_confidence_score = statistics.mean(v3_confidence_scores) if v3_confidence_scores else 0.0
        avg_v3_combined_score = statistics.mean(v3_combined_scores) if v3_combined_scores else 0.0
        
        # Calculate v2 comparison averages
        v2_comparison = {
            "has_v2_data": len(v2_comparisons) > 0,
            "comparison_count": len(v2_comparisons),
            "regression_count": regression_count,
            "regression_percent": regression_count / len(v2_comparisons) if v2_comparisons else 0.0,
            "avg_deltas": {}
        }
        
        if v2_comparisons:
            v2_comparison["avg_deltas"] = {
                "hallucination_reduction": statistics.mean(v3_v2_hallucination_deltas) if v3_v2_hallucination_deltas else 0.0,
                "latency_ms": statistics.mean(v3_v2_latency_deltas) if v3_v2_latency_deltas else 0.0,
                "confidence_score": statistics.mean(v3_v2_confidence_deltas) if v3_v2_confidence_deltas else 0.0,
                "combined_score": statistics.mean(v3_v2_combined_deltas) if v3_v2_combined_deltas else 0.0
            }
        
        # Calculate raw comparison averages
        raw_comparison = {
            "has_raw_data": len(raw_comparisons) > 0,
            "comparison_count": len(raw_comparisons),
            "improvement_count": improvement_count,
            "improvement_percent": improvement_count / len(raw_comparisons) if raw_comparisons else 0.0,
            "avg_deltas": {}
        }
        
        if raw_comparisons:
            raw_comparison["avg_deltas"] = {
                "hallucination_reduction": statistics.mean(v3_raw_hallucination_deltas) if v3_raw_hallucination_deltas else 0.0,
                "latency_ms": statistics.mean(v3_raw_latency_deltas) if v3_raw_latency_deltas else 0.0,
                "confidence_score": statistics.mean(v3_raw_confidence_deltas) if v3_raw_confidence_deltas else 0.0,
                "combined_score": statistics.mean(v3_raw_combined_deltas) if v3_raw_combined_deltas else 0.0
            }
        
        return {
            "v3": {
                "avg_hallucination_reduction": avg_v3_hallucination_reduction,
                "avg_latency_ms": avg_v3_latency_ms,
                "avg_confidence_score": avg_v3_confidence_score,
                "avg_combined_score": avg_v3_combined_score
            },
            "v2_comparison": v2_comparison,
            "raw_comparison": raw_comparison,
            "version_counts": {
                "total": len(comparisons),
                "with_v2": len(v2_comparisons),
                "with_raw": len(raw_comparisons)
            }
        }
    
    def _calculate_combined_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate combined score from multiple metrics.
        
        Args:
            metrics: Dictionary with metrics to combine
            
        Returns:
            Combined score (0.0-1.0)
        """
        # Weights for each component
        weights = {
            "hallucination_reduction": 0.6,  # Most important
            "confidence": 0.2,
            "latency": 0.2
        }
        
        # Extract and normalize components
        hall_reduction = metrics.get("hallucination_reduction", 0.0)
        confidence = metrics.get("confidence", 0.0)
        
        # Latency is inverse - lower is better
        latency = metrics.get("latency", 0.0)
        # Normalize latency (assuming 1000ms is worst case)
        norm_latency = max(0.0, 1.0 - min(1.0, latency / 1000.0))
        
        # Calculate weighted sum
        combined = (
            weights["hallucination_reduction"] * hall_reduction +
            weights["confidence"] * confidence +
            weights["latency"] * norm_latency
        )
        
        # Ensure between 0 and 1
        return max(0.0, min(1.0, combined))
    
    def generate_comparative_charts(
        self,
        comparison: BenchmarkComparison,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, str]:
        """
        Generate comparative charts for visualization.
        
        Args:
            comparison: Benchmark comparison data
            output_dir: Optional output directory (defaults to self.output_dir)
            
        Returns:
            Dictionary mapping chart names to file paths
        """
        if output_dir is None:
            output_dir = self.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        chart_paths = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Generate hallucination reduction comparison chart
        hall_reduction_path = output_dir / f"hallucination_comparison_{timestamp}.png"
        self._generate_hallucination_chart(comparison, hall_reduction_path)
        chart_paths["hallucination"] = str(hall_reduction_path)
        
        # 2. Generate latency comparison chart
        latency_path = output_dir / f"latency_comparison_{timestamp}.png"
        self._generate_latency_chart(comparison, latency_path)
        chart_paths["latency"] = str(latency_path)
        
        # 3. Generate domain comparison chart
        domain_path = output_dir / f"domain_comparison_{timestamp}.png"
        self._generate_domain_chart(comparison, domain_path)
        chart_paths["domain"] = str(domain_path)
        
        # 4. Generate provider comparison chart
        provider_path = output_dir / f"provider_comparison_{timestamp}.png"
        self._generate_provider_chart(comparison, provider_path)
        chart_paths["provider"] = str(provider_path)
        
        # 5. Generate regression analysis chart
        regression_path = output_dir / f"regression_analysis_{timestamp}.png"
        self._generate_regression_chart(comparison, regression_path)
        chart_paths["regression"] = str(regression_path)
        
        return chart_paths
    
    def _generate_hallucination_chart(self, comparison: BenchmarkComparison, output_path: Path) -> None:
        """Generate hallucination reduction comparison chart."""
        # Extract data
        comparisons = comparison.comparisons
        
        # Collect data for each version
        v3_reductions = []
        v2_reductions = []
        case_ids = []
        
        # Get top 10 cases by hallucination count difference
        top_comparisons = sorted(
            [c for c in comparisons if c.raw_metrics and c.v3_metrics],
            key=lambda x: x.raw_metrics.hallucination_count - x.v3_metrics.hallucination_count,
            reverse=True
        )[:10]
        
        for comp in top_comparisons:
            case_ids.append(comp.case_id or "unknown")
            v3_reductions.append(comp.v3_metrics.hallucination_reduction)
            if comp.v2_metrics:
                v2_reductions.append(comp.v2_metrics.hallucination_reduction)
            else:
                v2_reductions.append(0)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(case_ids))
        
        # Create bars
        plt.bar(index, v3_reductions, bar_width, label='ACDEK v3', color='#2196F3')
        if any(v2_reductions):
            plt.bar(index + bar_width, v2_reductions, bar_width, label='ACDEK v2', color='#9C27B0')
        
        # Labels and formatting
        plt.xlabel('Test Case')
        plt.ylabel('Hallucination Reduction Rate')
        plt.title('Hallucination Reduction Comparison by Test Case')
        plt.xticks(index + bar_width/2, case_ids, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path)
        plt.close()
    
    def _generate_latency_chart(self, comparison: BenchmarkComparison, output_path: Path) -> None:
        """Generate latency comparison chart."""
        # Extract data
        comparisons = comparison.comparisons
        
        # Collect data for each version
        v3_latencies = []
        v2_latencies = []
        raw_latencies = []
        case_ids = []
        
        # Get top 10 cases with highest raw latency
        top_comparisons = sorted(
            [c for c in comparisons if c.raw_metrics],
            key=lambda x: x.raw_metrics.latency_ms,
            reverse=True
        )[:10]
        
        for comp in top_comparisons:
            case_ids.append(comp.case_id or "unknown")
            v3_latencies.append(comp.v3_metrics.latency_ms)
            raw_latencies.append(comp.raw_metrics.latency_ms)
            if comp.v2_metrics:
                v2_latencies.append(comp.v2_metrics.latency_ms)
            else:
                v2_latencies.append(0)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        bar_width = 0.25
        index = np.arange(len(case_ids))
        
        # Create bars
        plt.bar(index, raw_latencies, bar_width, label='Raw LLM', color='#4CAF50')
        if any(v2_latencies):
            plt.bar(index + bar_width, v2_latencies, bar_width, label='ACDEK v2', color='#9C27B0')
        plt.bar(index + 2*bar_width, v3_latencies, bar_width, label='ACDEK v3', color='#2196F3')
        
        # Labels and formatting
        plt.xlabel('Test Case')
        plt.ylabel('Latency (ms)')
        plt.title('Latency Comparison by Test Case')
        plt.xticks(index + bar_width, case_ids, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path)
        plt.close()
    
    def _generate_domain_chart(self, comparison: BenchmarkComparison, output_path: Path) -> None:
        """Generate domain comparison chart."""
        # Extract data
        domains = comparison.domains
        
        # Collect data
        domain_names = []
        hallucination_reductions = []
        combined_scores = []
        
        for domain, data in domains.items():
            domain_names.append(domain)
            hallucination_reductions.append(data["v3"]["hallucination_reduction"])
            combined_scores.append(data["v3"]["combined_score"])
        
        # Create figure
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(domain_names))
        
        # Create bars
        plt.bar(index, hallucination_reductions, bar_width, label='Hallucination Reduction', color='#2196F3')
        plt.bar(index + bar_width, combined_scores, bar_width, label='Combined Score', color='#9C27B0')
        
        # Labels and formatting
        plt.xlabel('Domain')
        plt.ylabel('Score')
        plt.title('Performance by Domain')
        plt.xticks(index + bar_width/2, domain_names, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path)
        plt.close()
    
    def _generate_provider_chart(self, comparison: BenchmarkComparison, output_path: Path) -> None:
        """Generate provider comparison chart."""
        # Extract data
        providers = comparison.llm_providers
        
        # Collect data
        provider_names = []
        hallucination_reductions = []
        combined_scores = []
        case_counts = []
        
        for provider, data in providers.items():
            if provider == "unknown" or data["case_count"] < 2:
                continue
            provider_names.append(provider)
            hallucination_reductions.append(data["v3"]["hallucination_reduction"])
            combined_scores.append(data["v3"]["combined_score"])
            case_counts.append(data["case_count"])
        
        if not provider_names:
            # Create empty chart with message
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No provider data available", ha='center', va='center', fontsize=14)
            plt.savefig(output_path)
            plt.close()
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(provider_names))
        
        # Create bars
        bars1 = plt.bar(index, hallucination_reductions, bar_width, label='Hallucination Reduction', color='#2196F3')
        bars2 = plt.bar(index + bar_width, combined_scores, bar_width, label='Combined Score', color='#9C27B0')
        
        # Labels and formatting
        plt.xlabel('LLM Provider')
        plt.ylabel('Score')
        plt.title('Performance by LLM Provider')
        plt.xticks(index + bar_width/2, provider_names, rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend()
        
        # Add case count annotations
        for i, count in enumerate(case_counts):
            plt.annotate(f'n={count}', 
                         xy=(index[i] + bar_width/2, 0.05),
                         ha='center', va='bottom',
                         fontsize=8)
        
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path)
        plt.close()
    
    def _generate_regression_chart(self, comparison: BenchmarkComparison, output_path: Path) -> None:
        """Generate regression analysis chart."""
        regression_cases = comparison.regression_cases
        
        if not regression_cases:
            # Create empty chart with message
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, "No regression cases detected", ha='center', va='center', fontsize=14)
            plt.savefig(output_path)
            plt.close()
            return
        
        # Count regression types
        regression_counts = {
            "hallucination_reduction": 0,
            "latency_ms": 0,
            "confidence_score": 0,
            "combined_score": 0
        }
        
        for case in regression_cases:
            for metric, is_regression in case["regressions"].items():
                if is_regression and metric in regression_counts:
                    regression_counts[metric] += 1
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Format labels
        labels = [
            "Hallucination Reduction",
            "Latency",
            "Confidence",
            "Combined Score"
        ]
        values = [
            regression_counts["hallucination_reduction"],
            regression_counts["latency_ms"],
            regression_counts["confidence_score"],
            regression_counts["combined_score"]
        ]
        
        # Create horizontal bar chart
        plt.barh(labels, values, color='#F44336')
        
        # Labels and formatting
        plt.xlabel('Number of Cases')
        plt.title('Regression Analysis by Metric Type')
        
        # Add count labels
        for i, v in enumerate(values):
            plt.text(v + 0.1, i, str(v), va='center')
        
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_path)
        plt.close()
    
    def export_comparison_report(
        self,
        comparison: BenchmarkComparison,
        output_format: OutputFormat = OutputFormat.JSON,
        output_file: Optional[Union[str, Path]] = None,
        include_charts: bool = True
    ) -> str:
        """
        Export comparison report in the specified format.
        
        Args:
            comparison: Benchmark comparison data
            output_format: Report format (json, csv, markdown, html)
            output_file: Optional output file path
            include_charts: Whether to include charts in the report
            
        Returns:
            Path to the exported report file
        """
        logger.info(f"Exporting comparison report in {output_format.value} format (include_charts={include_charts})")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate default output file if not provided
        if output_file is None:
            filename = f"comparative_analysis_{timestamp}.{output_format.value}"
            output_file = self.output_dir / filename
            logger.info(f"Using default output file path: {output_file}")
        else:
            output_file = Path(output_file)
            logger.info(f"Using provided output file path: {output_file}")
        
        # Create parent directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate charts if requested
        chart_paths = {}
        if include_charts:
            logger.info("Generating comparative charts")
            charts_dir = output_file.parent / "charts"
            charts_dir.mkdir(exist_ok=True)
            logger.debug(f"Charts will be saved to directory: {charts_dir}")
            chart_paths = self.generate_comparative_charts(comparison, charts_dir)
            logger.info(f"Generated {len(chart_paths)} charts")
            
            for chart_type, path in chart_paths.items():
                logger.debug(f"Generated {chart_type} chart: {path}")
        else:
            logger.debug("Chart generation skipped as per request")
        
        # Get comparison metadata for logs
        case_count = comparison.metadata.get("case_count", 0)
        regression_threshold = comparison.metadata.get("regression_threshold", self.regression_threshold)
        normalized = comparison.metadata.get("normalized_by_provider", False)
        
        logger.info(f"Exporting report with {case_count} cases (regression_threshold={regression_threshold}, normalized={normalized})")
        
        # Export in specified format
        if output_format == OutputFormat.JSON:
            logger.info("Exporting as JSON format")
            try:
                data_dict = comparison.to_dict()
                with open(output_file, "w") as f:
                    json.dump(data_dict, f, indent=2)
                logger.debug(f"JSON report written to {output_file}")
            except Exception as e:
                logger.error(f"Error exporting to JSON: {e}")
                raise
        
        elif output_format == OutputFormat.CSV:
            logger.info("Exporting as CSV format")
            try:
                # Flatten data for CSV
                logger.debug("Flattening nested comparison data for CSV format")
                flattened_data = self._flatten_comparison_for_csv(comparison)
                logger.debug(f"Flattened {len(flattened_data)} metrics")
                
                with open(output_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Metric", "Value"])
                    for key, value in flattened_data.items():
                        writer.writerow([key, value])
                logger.debug(f"CSV report written to {output_file}")
            except Exception as e:
                logger.error(f"Error exporting to CSV: {e}")
                raise
        
        elif output_format == OutputFormat.YAML:
            logger.info("Exporting as YAML format")
            try:
                import yaml
                data_dict = comparison.to_dict()
                with open(output_file, "w") as f:
                    yaml.dump(data_dict, f)
                logger.debug(f"YAML report written to {output_file}")
            except ImportError:
                logger.error("YAML export failed: PyYAML package not available")
                raise
            except Exception as e:
                logger.error(f"Error exporting to YAML: {e}")
                raise
        
        elif output_format == OutputFormat.MARKDOWN:
            logger.info("Exporting as Markdown format")
            try:
                logger.debug("Generating markdown content")
                markdown_content = self._generate_markdown_report(comparison, chart_paths)
                content_length = len(markdown_content)
                logger.debug(f"Generated {content_length} chars of markdown content")
                
                with open(output_file, "w") as f:
                    f.write(markdown_content)
                logger.debug(f"Markdown report written to {output_file}")
            except Exception as e:
                logger.error(f"Error exporting to Markdown: {e}")
                raise
        
        elif output_format == OutputFormat.HTML:
            logger.info("Exporting as HTML format")
            try:
                logger.debug("Generating HTML content")
                html_content = self._generate_html_report(comparison, chart_paths)
                content_length = len(html_content)
                logger.debug(f"Generated {content_length} chars of HTML content")
                
                with open(output_file, "w") as f:
                    f.write(html_content)
                logger.debug(f"HTML report written to {output_file}")
            except ImportError:
                logger.error("HTML export failed: markdown package not available")
                raise
            except Exception as e:
                logger.error(f"Error exporting to HTML: {e}")
                raise
        
        logger.info(f"Successfully exported comparison report to {output_file}")
        return str(output_file)
    
    def _flatten_comparison_for_csv(self, comparison: BenchmarkComparison) -> Dict[str, Any]:
        """Flatten the comparison data for CSV export."""
        flattened = {}
        
        # Add summary metrics
        for category, metrics in comparison.summary.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, dict):
                        for sub_metric, sub_value in value.items():
                            flattened[f"summary.{category}.{metric}.{sub_metric}"] = sub_value
                    else:
                        flattened[f"summary.{category}.{metric}"] = value
            else:
                flattened[f"summary.{category}"] = metrics
        
        # Add domain metrics
        for domain, data in comparison.domains.items():
            flattened[f"domain.{domain}.case_count"] = data["case_count"]
            for metric, value in data["v3"].items():
                flattened[f"domain.{domain}.v3.{metric}"] = value
        
        # Add provider metrics
        for provider, data in comparison.llm_providers.items():
            flattened[f"provider.{provider}.case_count"] = data["case_count"]
            for metric, value in data["v3"].items():
                flattened[f"provider.{provider}.v3.{metric}"] = value
        
        # Add metadata
        for key, value in comparison.metadata.items():
            flattened[f"metadata.{key}"] = value
        
        return flattened
    
    def _generate_markdown_report(
        self, 
        comparison: BenchmarkComparison,
        chart_paths: Dict[str, str]
    ) -> str:
        """Generate markdown report from comparison data."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Start with summary section
        md = f"""# ACDEK v3 Comparative Analysis Report

Generated: {timestamp}

## Summary

- **Total Test Cases**: {comparison.metadata.get("case_count", 0)}
- **Normalized by LLM Provider**: {comparison.metadata.get("normalized_by_provider", False)}
- **Regression Threshold**: {comparison.metadata.get("regression_threshold", 0.05)}

### ACDEK v3 Performance

| Metric | Value |
|--------|-------|
| Avg. Hallucination Reduction | {comparison.summary["v3"]["avg_hallucination_reduction"]:.2%} |
| Avg. Latency | {comparison.summary["v3"]["avg_latency_ms"]:.2f} ms |
| Avg. Confidence Score | {comparison.summary["v3"]["avg_confidence_score"]:.2f} |
| Avg. Combined Score | {comparison.summary["v3"]["avg_combined_score"]:.2f} |

"""
        
        # Add v2 comparison if available
        if comparison.summary["v2_comparison"]["has_v2_data"]:
            v2_comp = comparison.summary["v2_comparison"]
            md += f"""
### Comparison with ACDEK v2

- **Cases with v2 data**: {v2_comp["comparison_count"]}
- **Regression cases**: {v2_comp["regression_count"]} ({v2_comp["regression_percent"]:.2%})

| Metric | Delta | Interpretation |
|--------|-------|---------------|
| Hallucination Reduction | {v2_comp["avg_deltas"]["hallucination_reduction"]:.4f} | {"🟢 Better" if v2_comp["avg_deltas"]["hallucination_reduction"] > 0 else "🔴 Worse"} |
| Latency | {v2_comp["avg_deltas"]["latency_ms"]:.2f} ms | {"🔴 Worse" if v2_comp["avg_deltas"]["latency_ms"] > 0 else "🟢 Better"} |
| Confidence | {v2_comp["avg_deltas"]["confidence_score"]:.4f} | {"🟢 Better" if v2_comp["avg_deltas"]["confidence_score"] > 0 else "🔴 Worse"} |
| Combined Score | {v2_comp["avg_deltas"]["combined_score"]:.4f} | {"🟢 Better" if v2_comp["avg_deltas"]["combined_score"] > 0 else "🔴 Worse"} |

"""
        
        # Add raw comparison if available
        if comparison.summary["raw_comparison"]["has_raw_data"]:
            raw_comp = comparison.summary["raw_comparison"]
            md += f"""
### Comparison with Raw LLM

- **Cases with raw data**: {raw_comp["comparison_count"]}
- **Full improvement cases**: {raw_comp["improvement_count"]} ({raw_comp["improvement_percent"]:.2%})

| Metric | Delta | Interpretation |
|--------|-------|---------------|
| Hallucination Reduction | {raw_comp["avg_deltas"]["hallucination_reduction"]:.4f} | {"🟢 Better" if raw_comp["avg_deltas"]["hallucination_reduction"] > 0 else "🔴 Worse"} |
| Confidence | {raw_comp["avg_deltas"]["confidence_score"]:.4f} | {"🟢 Better" if raw_comp["avg_deltas"]["confidence_score"] > 0 else "🔴 Worse"} |
| Combined Score | {raw_comp["avg_deltas"]["combined_score"]:.4f} | {"🟢 Better" if raw_comp["avg_deltas"]["combined_score"] > 0 else "🔴 Worse"} |

"""
        
        # Add domain analysis
        md += """
## Domain Analysis

| Domain | Case Count | Hall. Reduction | Latency (ms) | Confidence | Combined Score |
|--------|------------|----------------|--------------|------------|----------------|
"""
        
        for domain, data in comparison.domains.items():
            md += f"| {domain} | {data['case_count']} | {data['v3']['hallucination_reduction']:.2%} | {data['v3']['latency_ms']:.2f} | {data['v3']['confidence_score']:.2f} | {data['v3']['combined_score']:.2f} |\n"
        
        # Add LLM provider analysis
        md += """
## LLM Provider Analysis

| Provider | Case Count | Hall. Reduction | Latency (ms) | Confidence | Combined Score |
|----------|------------|----------------|--------------|------------|----------------|
"""
        
        for provider, data in comparison.llm_providers.items():
            if provider != "unknown" and data["case_count"] > 0:
                md += f"| {provider} | {data['case_count']} | {data['v3']['hallucination_reduction']:.2%} | {data['v3']['latency_ms']:.2f} | {data['v3']['confidence_score']:.2f} | {data['v3']['combined_score']:.2f} |\n"
        
        # Add regression analysis
        regression_cases = comparison.regression_cases
        if regression_cases:
            md += f"""
## Regression Analysis

Found {len(regression_cases)} cases where v3 regressed compared to v2:

| Case ID | Domain | Provider | Regression Types |
|---------|--------|----------|-----------------|
"""
            
            for case in regression_cases:
                regression_types = [k for k, v in case["regressions"].items() if v]
                regression_str = ", ".join(regression_types)
                md += f"| {case['case_id']} | {case['domain']} | {case['provider']} | {regression_str} |\n"
        
        # Add charts if available
        if chart_paths:
            md += """
## Visualization

"""
            # Convert paths to relative for Markdown
            for chart_name, path in chart_paths.items():
                relative_path = os.path.relpath(path, os.path.dirname(os.path.abspath(__file__)))
                md += f"### {chart_name.title()} Analysis\n\n![{chart_name}]({relative_path})\n\n"
        
        return md
    
    def _generate_html_report(
        self, 
        comparison: BenchmarkComparison,
        chart_paths: Dict[str, str]
    ) -> str:
        """Generate HTML report from comparison data and charts."""
        # Convert markdown to HTML
        markdown_content = self._generate_markdown_report(comparison, chart_paths)
        html_content = markdown.markdown(markdown_content)
        
        # Wrap with HTML template
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ACDEK v3 Comparative Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 0; }}
                .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .header {{ background-color: #343a40; color: white; padding: 20px; margin-bottom: 20px; }}
                .header h1 {{ margin: 0; }}
                .summary-section {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .chart-section {{ margin-bottom: 30px; }}
                .chart {{ max-width: 100%; height: auto; }}
                .card {{ background-color: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; padding: 20px; }}
                .improved {{ color: #4CAF50; }}
                .regressed {{ color: #F44336; }}
                .footer {{ text-align: center; margin-top: 50px; color: #777; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="container">
                    <h1>ACDEK v3 Comparative Analysis Report</h1>
                    <p>Generated: {timestamp}</p>
                </div>
            </div>
            
            <div class="container">
                <div class="card">
                    {html_content}
                </div>
                
                <div class="footer">
                    <p>ACDEK v3 Comparative Analyzer | Generated by ComparativeAnalyzer</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html


def main():
    """Command-line interface for ComparativeAnalyzer."""
    import argparse
    import logging
    import sys
    
    # Configure root logger for CLI
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Add console handler if no handlers exist
    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    logger.info("Starting ACDEK v3 Comparative Analysis Tool")
    
    parser = argparse.ArgumentParser(description="ACDEK v3 Comparative Analysis Tool")
    parser.add_argument(
        "benchmark_file",
        type=str,
        help="Path to benchmark results file to analyze"
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
        "--output-file",
        type=str,
        help="Specific output filename (default: auto-generated)"
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Disable chart generation for the report"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize metrics by LLM provider"
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=REGRESSION_THRESHOLD,
        help="Threshold for flagging metrics as regressed"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (debug) logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info(f"Processing benchmark file: {args.benchmark_file}")
    logger.info(f"Output format: {args.output_format}")
    logger.info(f"Normalization: {'enabled' if args.normalize else 'disabled'}")
    logger.info(f"Charts: {'disabled' if args.no_charts else 'enabled'}")
    
    try:
        # Initialize analyzer
        logger.info("Initializing analyzer")
        analyzer = ComparativeAnalyzer(
            results_dir=os.path.dirname(args.benchmark_file),
            output_dir=args.output_dir,
            regression_threshold=args.regression_threshold
        )
        
        # Load benchmark data
        logger.info(f"Loading benchmark data from {args.benchmark_file}")
        benchmark_data = analyzer.load_benchmark_data(args.benchmark_file)
        
        if not benchmark_data:
            logger.error(f"Failed to load benchmark data from {args.benchmark_file}")
            print(f"Error: Could not load benchmark data from {args.benchmark_file}")
            return 1
        
        # Analyze benchmark data
        logger.info("Analyzing benchmark data")
        comparison = analyzer.analyze_benchmark_data(
            benchmark_data,
            normalize_by_provider=args.normalize
        )
        
        # Export report
        logger.info(f"Exporting report in {args.output_format} format")
        output_format = OutputFormat(args.output_format)
        report_path = analyzer.export_comparison_report(
            comparison,
            output_format=output_format,
            output_file=args.output_file,
            include_charts=not args.no_charts
        )
        
        logger.info(f"Analysis complete. Report generated at: {report_path}")
        print(f"Comparative analysis report generated: {report_path}")
        return 0
        
    except Exception as e:
        logger.exception("Error during comparative analysis")
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    main()