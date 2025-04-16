import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import time
from collections import deque

from registry.registry_types import PluginType
from registry.plugin_registry import register_plugin
from interfaces.vector_alignment_interface import VectorAlignmentInterface

logger = logging.getLogger(__name__)


@dataclass
class DriftStats:
    """Statistics about detected drift."""
    drift_score: float
    mean_distance: float
    max_distance: float
    drift_detected: bool
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "drift_score": self.drift_score,
            "mean_distance": self.mean_distance,
            "max_distance": self.max_distance,
            "drift_detected": self.drift_detected,
            "details": self.details
        }


@register_plugin(
    plugin_type=PluginType.ENRICHMENT,
    name="semantic_drift_detector",
    version="1.0.0",
    description="Detects semantic drift in data using vector embeddings",
    interfaces=["EnrichmentInterface"],
    is_default=True
)
class SemanticDriftDetector:
    """
    Detects semantic drift in data using vector embeddings.
    
    This component monitors for changes in the semantic meaning of data
    over time, allowing for early detection of data drift that could
    affect model performance.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.3,
        reference_vectors: Optional[Dict[str, np.ndarray]] = None,
        vector_provider: Optional[VectorAlignmentInterface] = None
    ):
        """
        Initialize the drift detector.
        
        Args:
            window_size: Number of recent data points to consider
            drift_threshold: Threshold for flagging significant drift
            reference_vectors: Dictionary of reference vectors for baseline comparison
            vector_provider: Optional vector alignment component to use
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_vectors = reference_vectors or {}
        self.vector_provider = vector_provider
        
        # Sliding windows for tracking recent vectors by category
        self.recent_vectors: Dict[str, deque] = {}
        
        # Track history of drift scores
        self.drift_history: deque = deque(maxlen=window_size)
        
        logger.info(f"Initialized semantic drift detector (window_size={window_size}, threshold={drift_threshold})")
    
    def process(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> DriftStats:
        """
        Process input data to detect semantic drift.
        
        Args:
            input_data: Dictionary containing data and embeddings
            context: Optional context information
            
        Returns:
            Drift statistics
        """
        try:
            # Extract text content from input data
            text_data = self._extract_text_from_input(input_data)
            category = self._get_category(input_data, context)
            
            # Get embeddings (either directly or compute them)
            embeddings = self._get_embeddings(text_data, input_data)
            
            # Update reference vectors if needed
            if category not in self.reference_vectors and embeddings:
                first_key = next(iter(embeddings.keys()))
                self.reference_vectors[category] = embeddings[first_key]
                logger.info(f"Set reference vector for category '{category}'")
            
            # Update recent vectors
            if category not in self.recent_vectors:
                self.recent_vectors[category] = deque(maxlen=self.window_size)
            
            # Add all embeddings to the sliding window
            for embedding in embeddings.values():
                self.recent_vectors[category].append(embedding)
            
            # Calculate drift statistics
            return self._calculate_drift(category)
            
        except Exception as e:
            logger.error(f"Error in drift detection: {str(e)}")
            
            # Return default values on error
            return DriftStats(
                drift_score=0.0,
                mean_distance=0.0,
                max_distance=0.0,
                drift_detected=False,
                details={"error": str(e)}
            )
    
    def _extract_text_from_input(self, input_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract text content from input data structure."""
        text_data = {}
        
        # Try to extract from ar_enrichment slot if available
        if "ar_enrichment" in input_data and isinstance(input_data["ar_enrichment"], dict):
            ar_data = input_data["ar_enrichment"]
            
            # Extract from slots if available
            if "slots" in ar_data and isinstance(ar_data["slots"], dict):
                for key, slot in ar_data["slots"].items():
                    if isinstance(slot, dict) and "value" in slot:
                        text_data[key] = str(slot["value"])
                    else:
                        text_data[key] = str(slot)
            else:
                # Try to extract from flat dictionary
                for key, value in ar_data.items():
                    text_data[key] = str(value)
        
        # Fallback to original input if ar_enrichment not found
        if not text_data:
            for key, value in input_data.items():
                if isinstance(value, (str, int, float, bool)):
                    text_data[key] = str(value)
                elif isinstance(value, dict) and "value" in value:
                    text_data[key] = str(value["value"])
        
        return text_data
    
    def _get_category(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]]) -> str:
        """Determine the category for drift tracking."""
        # Try to get from context
        if context and "category" in context:
            return str(context["category"])
        
        # Try to get from input
        if "category" in input_data:
            return str(input_data["category"])
        
        # Try to get from query
        if "q_enriched" in input_data:
            q_data = input_data["q_enriched"]
            if isinstance(q_data, dict) and "intent" in q_data:
                return str(q_data["intent"])
        
        # Default category
        return "default"
    
    def _get_embeddings(self, text_data: Dict[str, str], input_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Get embeddings for text data, either from input or by computing them."""
        embeddings = {}
        
        # Check if embeddings already exist in ar_enrichment
        if "ar_enrichment" in input_data and isinstance(input_data["ar_enrichment"], dict):
            ar_data = input_data["ar_enrichment"]
            
            if "slots" in ar_data and isinstance(ar_data["slots"], dict):
                for key, slot in ar_data["slots"].items():
                    if isinstance(slot, dict) and "embedding" in slot and slot["embedding"] is not None:
                        embeddings[key] = slot["embedding"]
        
        # Compute embeddings for any missing items
        missing_keys = [k for k in text_data.keys() if k not in embeddings]
        
        if missing_keys and self.vector_provider:
            for key in missing_keys:
                embeddings[key] = self.vector_provider.get_embedding(text_data[key])
        
        return embeddings
    
    def _calculate_drift(self, category: str) -> DriftStats:
        """Calculate drift statistics for a category."""
        # If no reference or recent vectors, can't calculate drift
        if category not in self.reference_vectors or category not in self.recent_vectors:
            return DriftStats(
                drift_score=0.0,
                mean_distance=0.0,
                max_distance=0.0,
                drift_detected=False,
                details={"status": "insufficient_data"}
            )
        
        reference = self.reference_vectors[category]
        recent = list(self.recent_vectors[category])
        
        if not recent:
            return DriftStats(
                drift_score=0.0,
                mean_distance=0.0,
                max_distance=0.0,
                drift_detected=False,
                details={"status": "no_recent_data"}
            )
        
        # Calculate distances from reference
        distances = []
        
        for vector in recent:
            if self.vector_provider:
                # Use vector provider for distance calculation
                similarity = self.vector_provider.calculate_similarity(reference, vector)
                distance = 1.0 - similarity
            else:
                # Direct calculation
                similarity = np.dot(reference, vector) / (np.linalg.norm(reference) * np.linalg.norm(vector) + 1e-8)
                distance = 1.0 - similarity
            
            distances.append(distance)
        
        # Calculate drift statistics
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        
        # Weighted average of mean and max distance
        drift_score = (0.7 * mean_distance) + (0.3 * max_distance)
        drift_detected = drift_score > self.drift_threshold
        
        # Add to history
        self.drift_history.append(drift_score)
        
        # Calculate trend
        trend = 0.0
        if len(self.drift_history) > 1:
            recent_avg = np.mean(list(self.drift_history)[-min(5, len(self.drift_history)):])
            older_avg = np.mean(list(self.drift_history)[:-min(5, len(self.drift_history))] or [0])
            trend = recent_avg - older_avg
        
        return DriftStats(
            drift_score=drift_score,
            mean_distance=mean_distance,
            max_distance=max_distance,
            drift_detected=drift_detected,
            details={
                "trend": trend,
                "sample_size": len(recent),
                "history_size": len(self.drift_history),
                "category": category
            }
        )
    
    def update_reference(self, category: str, vector: np.ndarray) -> None:
        """
        Update the reference vector for a category.
        
        Args:
            category: Category to update
            vector: New reference vector
        """
        self.reference_vectors[category] = vector
        logger.info(f"Updated reference vector for category '{category}'")
    
    def reset(self, category: Optional[str] = None) -> None:
        """
        Reset drift detection state.
        
        Args:
            category: Specific category to reset, or all if None
        """
        if category:
            if category in self.reference_vectors:
                del self.reference_vectors[category]
            if category in self.recent_vectors:
                self.recent_vectors[category].clear()
            logger.info(f"Reset drift detector for category '{category}'")
        else:
            self.reference_vectors.clear()
            self.recent_vectors.clear()
            self.drift_history.clear()
            logger.info("Reset drift detector completely")
    
    def get_drift_history(self) -> List[float]:
        """
        Get history of drift scores.
        
        Returns:
            List of historical drift scores
        """
        return list(self.drift_history)