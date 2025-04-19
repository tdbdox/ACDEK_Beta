"""
Hallucination Metrics Connector

This module provides integration between the ACDEK core enrichment components
(drift detector, confidence scorer) and the hallucination metrics system.
It simplifies the process of collecting hallucination metrics by providing
unified access to the underlying components.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union

# Import core components
from enrichment.drift_detector import SemanticDriftDetector
from enrichment.confidence_scorer import VectorConfidenceScorer
from enrichment.hallucination_metrics import (
    HallucinationMetricsCollector,
    ReferenceSource,
    DriftDetectionResult,
    ConfidenceAnalysisResult,
    HallucinationMetricsResult,
    DriftSeverity,
    ConfidenceCorrelation
)

# Configure logging
logger = logging.getLogger(__name__)


class HallucinationConnector:
    """
    Connector class that bridges between the core ACDEK components and
    the hallucination metrics collection system. This class simplifies
    the integration and usage of these components together.
    """
    
    def __init__(
        self,
        drift_detector: Optional[SemanticDriftDetector] = None,
        confidence_scorer: Optional[VectorConfidenceScorer] = None,
        metrics_collector: Optional[HallucinationMetricsCollector] = None,
        enable_persistence: bool = True
    ):
        """
        Initialize the hallucination connector.
        
        Args:
            drift_detector: Optional existing drift detector instance
            confidence_scorer: Optional existing confidence scorer instance
            metrics_collector: Optional existing metrics collector instance
            enable_persistence: Whether to enable persistence of metrics
        """
        # Use provided instances or create new ones
        self.drift_detector = drift_detector or SemanticDriftDetector()
        self.confidence_scorer = confidence_scorer or VectorConfidenceScorer()
        self.metrics_collector = metrics_collector or HallucinationMetricsCollector(
            store_results=enable_persistence
        )
    
    def analyze_response(
        self,
        prompt: str,
        response: str,
        reference_sources: List[Union[str, Dict[str, Any], ReferenceSource]],
        reported_confidence: float = 0.0,
        constraint_results: Optional[Dict[str, Any]] = None,
        domain_context: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HallucinationMetricsResult:
        """
        Analyze a generated response for hallucinations using all components.
        
        Args:
            prompt: Original prompt
            response: Generated response to analyze
            reference_sources: Reference sources for factuality checking
            reported_confidence: Model's self-reported confidence (0.0-1.0)
            constraint_results: Optional constraint validation results
            domain_context: Optional domain context for drift detection
            model_info: Optional model information
            metadata: Optional additional metadata
            
        Returns:
            Complete hallucination metrics result
        """
        # Convert reference sources to proper format if needed
        if reference_sources and not isinstance(reference_sources[0], ReferenceSource):
            references = []
            for i, source in enumerate(reference_sources):
                if isinstance(source, str):
                    references.append(ReferenceSource(
                        id=f"ref_{i+1}",
                        content=source,
                        source_type="text"
                    ))
                elif isinstance(source, dict):
                    references.append(ReferenceSource(
                        id=source.get("id", f"ref_{i+1}"),
                        content=source.get("content", ""),
                        source_type=source.get("source_type", "text"),
                        metadata=source.get("metadata", {})
                    ))
                else:
                    raise ValueError(f"Unsupported reference source type: {type(source)}")
            reference_sources = references
        
        # 1. Run drift detection
        drift_input = {
            "prompt": prompt,
            "response": response,  # Additional field for our connector
            "text": response       # Field expected by drift detector
        }
        
        if domain_context:
            drift_input["domain_context"] = domain_context
        
        drift_result = self.drift_detector.process(drift_input)
        
        # Convert to hallucination metrics format
        drift_detection_result = self.drift_detector.get_hallucination_drift_result(drift_result)
        
        # 2. Check factuality with confidence scoring
        # This would integrate with the existing metrics collector
        
        # 3. Collect and return complete metrics
        metrics_result = self.metrics_collector.collect_metrics(
            prompt=prompt,
            response=response,
            reference_sources=reference_sources,
            reported_confidence=reported_confidence,
            constraint_results=constraint_results,
            domain_context=domain_context,
            model_info=model_info,
            metadata=metadata
        )
        
        return metrics_result
    
    def update_confidence_with_factuality(
        self,
        confidence_score: float,
        factuality_score: float
    ) -> Dict[str, Any]:
        """
        Update confidence score with factuality correlation analysis.
        
        Args:
            confidence_score: Model's reported confidence score
            factuality_score: Measured factuality score
            
        Returns:
            Dictionary with confidence correlation analysis
        """
        return self.confidence_scorer.analyze_confidence_correlation(
            confidence_score, factuality_score
        )
    
    def detect_semantic_drift(
        self,
        prompt: str,
        response: str,
        domain_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect semantic drift in a response relative to the prompt and domain context.
        
        Args:
            prompt: Original prompt
            response: Generated response
            domain_context: Optional additional domain context
            
        Returns:
            Dictionary with drift detection results
        """
        drift_input = {
            "prompt": prompt,
            "response": response,
            "text": response
        }
        
        if domain_context:
            drift_input["domain_context"] = domain_context
        
        drift_result = self.drift_detector.process(drift_input)
        return self.drift_detector.get_hallucination_drift_result(drift_result)
    
    def get_correlation_statistics(self) -> Dict[str, Any]:
        """
        Get confidence-factuality correlation statistics.
        
        Returns:
            Dictionary with correlation statistics
        """
        return self.confidence_scorer.get_correlation_stats()
    
    def export_metrics_report(
        self,
        format: str = "json",
        file_path: Optional[str] = None
    ) -> str:
        """
        Export collected metrics in a specified format.
        
        Args:
            format: Export format ("json", "csv", or "html")
            file_path: Optional output file path
            
        Returns:
            Path to the exported file
        """
        return self.metrics_collector.export_metrics_report(format, file_path)


# Create default instance for easy import
default_connector = HallucinationConnector()