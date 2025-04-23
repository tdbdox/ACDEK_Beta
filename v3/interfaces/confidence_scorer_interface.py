from abc import ABC, abstractmethod
from typing import Dict, Any, TypeVar, Generic
from dataclasses import dataclass, field
import time

T_Output = TypeVar('T_Output')

@dataclass
class ConfidenceScoreBase:
    """Base class for confidence score results"""
    aggregate_score: float
    factors: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __float__(self):
        """Allow direct conversion to float."""
        return self.aggregate_score
    
    def is_reliable(self, threshold: float = 0.6) -> bool:
        """Check if the score is above the reliability threshold."""
        return self.aggregate_score >= threshold


class ConfidenceScorerInterface(Generic[T_Output], ABC):
    """Interface for confidence scoring components."""
    
    @abstractmethod
    def score(self, data: Dict[str, Any]) -> T_Output:
        """
        Score the confidence of input data based on multiple factors.
        
        Args:
            data: Input data for scoring
            
        Returns:
            Confidence score with detailed breakdown
        """
        pass
    
    @abstractmethod
    def evaluate_with_threshold(self, data: Dict[str, Any], threshold: float = 0.6) -> bool:
        """
        Evaluate if input data meets a confidence threshold.
        
        Args:
            data: Input data for scoring
            threshold: Confidence threshold
            
        Returns:
            Boolean indicating if data meets threshold
        """
        pass