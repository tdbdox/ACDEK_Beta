"""
Hallucination Metrics Interface

Defines the interface for hallucination metrics collection components.
These components are responsible for measuring, logging, and analyzing
hallucination metrics for LLM outputs.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Union


class HallucinationMetricsInterface(ABC):
    """
    Interface for components that collect and analyze hallucination metrics.
    
    Components implementing this interface are responsible for:
    1. Measuring factuality against reference sources
    2. Detecting semantic drift from context
    3. Correlating confidence with factuality
    4. Tracking and logging hallucination data
    5. Providing structured outputs for benchmarking and dashboards
    """
    
    @abstractmethod
    def measure_factuality(
        self,
        response: str,
        reference_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Measure factuality of a response against reference sources.
        
        Args:
            response: Generated response to measure
            reference_sources: List of reference source objects
            
        Returns:
            Dictionary with factuality metrics
        """
        pass
    
    @abstractmethod
    def detect_semantic_drift(
        self,
        prompt: str,
        response: str,
        domain_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect semantic drift in a response from the prompt context.
        
        Args:
            prompt: Original prompt
            response: Generated response
            domain_context: Optional domain context for additional validation
            
        Returns:
            Dictionary with drift detection metrics
        """
        pass
    
    @abstractmethod
    def analyze_confidence_correlation(
        self,
        reported_confidence: float,
        measured_factuality: float,
        constraint_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze correlation between reported confidence and measured factuality.
        
        Args:
            reported_confidence: Model's self-reported confidence (0.0-1.0)
            measured_factuality: Measured factuality score (0.0-1.0)
            constraint_results: Optional constraint validation results
            
        Returns:
            Dictionary with confidence correlation metrics
        """
        pass
    
    @abstractmethod
    def collect_metrics(
        self,
        prompt: str,
        response: str,
        reference_sources: List[Dict[str, Any]],
        reported_confidence: float = 0.0,
        constraint_results: Optional[Dict[str, Any]] = None,
        domain_context: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Collect comprehensive hallucination metrics for a response.
        
        Args:
            prompt: Original prompt
            response: Generated response
            reference_sources: List of reference sources for factuality checking
            reported_confidence: Model's self-reported confidence (0.0-1.0)
            constraint_results: Optional constraint validation results
            domain_context: Optional domain context for drift detection
            model_info: Optional model information
            metadata: Optional additional metadata
            
        Returns:
            Dictionary with comprehensive hallucination metrics
        """
        pass
    
    @abstractmethod
    def export_metrics_report(
        self,
        format: str = "json",
        file_path: Optional[str] = None,
        include_raw_data: bool = False
    ) -> str:
        """
        Export metrics report in the specified format.
        
        Args:
            format: Report format ("json", "csv", or "html")
            file_path: Optional file path, if None a default path is generated
            include_raw_data: Whether to include raw metrics data
            
        Returns:
            Path to the exported report
        """
        pass
    
    @abstractmethod
    def get_metrics_stats(self) -> Dict[str, Any]:
        """
        Get statistics of collected metrics.
        
        Returns:
            Dictionary with metrics statistics
        """
        pass