import logging
import numpy as np
import os
import pickle
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta

from registry.registry_types import PluginType
from registry.plugin_registry import register_plugin
from interfaces.vector_alignment_interface import VectorAlignmentInterface
from interfaces.enrichment_interface import EnrichmentInterface

logger = logging.getLogger(__name__)


@dataclass
class DriftSnapshot:
    """Point-in-time snapshot of vector and metadata."""
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __init__(self, vector: np.ndarray, metadata: Dict[str, Any] = None, timestamp: float = None):
        self.vector = vector
        self.metadata = metadata or {}
        self.timestamp = timestamp or time.time()


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


@dataclass
class CategoryStats:
    """Detailed statistics for a category."""
    category: str
    sample_count: int
    drift_score: float
    drift_detected: bool
    trend: float
    last_update: float
    stability_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "category": self.category,
            "sample_count": self.sample_count,
            "drift_score": self.drift_score,
            "drift_detected": self.drift_detected,
            "trend": self.trend,
            "last_update": self.last_update,
            "stability_score": self.stability_score
        }


@register_plugin(
    plugin_type=PluginType.ENRICHMENT,
    name="semantic_drift_detector",
    version="1.0.0",
    description="Detects semantic drift in data using vector embeddings",
    interfaces=["EnrichmentInterface"],
    is_default=True
)
class SemanticDriftDetector(EnrichmentInterface[Dict[str, Any], DriftStats]):
    """
    Enhanced detector for semantic drift in vector embeddings.
    
    This component monitors for changes in the semantic meaning of data
    over time, allowing for early detection of data drift that could
    affect model performance. It provides detailed historical tracking
    and analysis of semantic shifts.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        drift_threshold: float = 0.3,
        reference_vectors: Optional[Dict[str, np.ndarray]] = None,
        vector_provider: Optional[VectorAlignmentInterface] = None,
        history_length: int = 1000,
        snapshot_interval: int = 10,
        persistence_path: Optional[str] = None,
        auto_reference_update: bool = False,
        auto_reference_update_interval: int = 1000
    ):
        """
        Initialize the enhanced drift detector.
        
        Args:
            window_size: Number of recent data points to consider
            drift_threshold: Threshold for flagging significant drift
            reference_vectors: Dictionary of reference vectors for baseline comparison
            vector_provider: Optional vector alignment component to use
            history_length: Maximum number of historical snapshots to maintain
            snapshot_interval: Take vector snapshots every N data points
            persistence_path: Optional path for disk persistence
            auto_reference_update: Whether to automatically update reference vectors
            auto_reference_update_interval: Interval for automatic reference updates
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_vectors = reference_vectors or {}
        self.reference_timestamps: Dict[str, float] = {}
        self.vector_provider = vector_provider
        self.history_length = history_length
        self.snapshot_interval = snapshot_interval
        self.persistence_path = persistence_path
        self.auto_reference_update = auto_reference_update
        self.auto_reference_update_interval = auto_reference_update_interval
        
        # Sliding windows for tracking recent vectors by category
        self.recent_vectors: Dict[str, deque] = {}
        
        # Track history of drift scores by category
        self.drift_history: Dict[str, deque] = {}
        
        # Vector history snapshots by category
        self.vector_history: Dict[str, List[DriftSnapshot]] = {}
        
        # Counter for triggering snapshots
        self.snapshot_counters: Dict[str, int] = {}
        
        # Statistics by category
        self.category_stats: Dict[str, CategoryStats] = {}
        
        # Novelty detection - keep track of previously unseen patterns
        self.novelty_vectors: Dict[str, List[np.ndarray]] = {}
        self.stability_scores: Dict[str, float] = {}
        
        # Load from persistence if available
        if persistence_path:
            self._load()
        
        logger.info(f"Initialized enhanced semantic drift detector (window_size={window_size}, threshold={drift_threshold})")
    
    def enrich(
        self, 
        input_data: Dict[str, Any], 
        context: Optional[Dict[str, Any]] = None
    ) -> DriftStats:
        """
        Enrich input data with drift detection.
        
        Args:
            input_data: Dictionary containing data and embeddings
            context: Optional context information
            
        Returns:
            Drift statistics
        """
        return self.process(input_data, context)
    
    def get_confidence(self, enriched_data: DriftStats) -> float:
        """
        Extract the overall confidence score from enriched data.
        
        Args:
            enriched_data: The enriched data to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        # Lower confidence if drift is detected
        if enriched_data.drift_detected:
            return max(0.1, 1.0 - enriched_data.drift_score)
        
        return min(1.0, 1.0 - (enriched_data.drift_score / 2))
    
    def apply_constraints(
        self, 
        enriched_data: DriftStats, 
        constraints: Dict[str, Any]
    ) -> DriftStats:
        """
        Apply additional constraints to already enriched drift data.
        
        Args:
            enriched_data: The previously enriched data
            constraints: Dictionary of constraints to apply
            
        Returns:
            Enriched data with constraints applied
        """
        if not constraints:
            return enriched_data
            
        # Get custom drift threshold if provided
        if "drift_threshold" in constraints:
            try:
                threshold = float(constraints["drift_threshold"])
                # Recalculate drift_detected with new threshold
                drift_detected = enriched_data.drift_score > threshold
                
                # Create new DriftStats with updated detection flag
                return DriftStats(
                    drift_score=enriched_data.drift_score,
                    mean_distance=enriched_data.mean_distance,
                    max_distance=enriched_data.max_distance,
                    drift_detected=drift_detected,
                    details={
                        **enriched_data.details,
                        "custom_threshold": threshold
                    }
                )
            except (ValueError, TypeError):
                pass
                
        return enriched_data
    
    def supports_input_type(self, input_type: str) -> bool:
        """
        Check if this component supports a given input type.
        
        Args:
            input_type: String identifier of input type
            
        Returns:
            Boolean indicating support
        """
        return input_type in ["data", "embeddings", "mixed"]
    
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
            
            # If no embeddings, return default stats
            if not embeddings:
                return DriftStats(
                    drift_score=0.0,
                    mean_distance=0.0,
                    max_distance=0.0,
                    drift_detected=False,
                    details={"status": "no_embeddings"}
                )
            
            # Initialize structures for this category if needed
            self._init_category_structures(category)
            
            # Update reference vectors if needed
            if category not in self.reference_vectors:
                self._set_reference_vector(category, embeddings)
            
            # Process embeddings and update sliding windows
            self._process_embeddings(category, embeddings, input_data)
            
            # Check if we should update reference vector automatically
            if (self.auto_reference_update and 
                category in self.reference_timestamps and
                len(self.recent_vectors[category]) >= self.auto_reference_update_interval):
                time_since_update = time.time() - self.reference_timestamps[category]
                # Update reference if it's been more than a day
                if time_since_update > 86400:  # 24 hours
                    self._update_reference_from_recent(category)
            
            # Calculate drift statistics
            drift_stats = self._calculate_drift(category)
            
            # Update and persist state if needed
            if self.persistence_path and random.random() < 0.05:  # 5% chance to persist
                self._save()
            
            return drift_stats
            
        except Exception as e:
            logger.error(f"Error in drift detection: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return default values on error
            return DriftStats(
                drift_score=0.0,
                mean_distance=0.0,
                max_distance=0.0,
                drift_detected=False,
                details={"error": str(e)}
            )
    
    def _init_category_structures(self, category: str) -> None:
        """Initialize data structures for a category if needed."""
        if category not in self.recent_vectors:
            self.recent_vectors[category] = deque(maxlen=self.window_size)
        
        if category not in self.drift_history:
            self.drift_history[category] = deque(maxlen=self.history_length)
        
        if category not in self.vector_history:
            self.vector_history[category] = []
        
        if category not in self.snapshot_counters:
            self.snapshot_counters[category] = 0
        
        if category not in self.novelty_vectors:
            self.novelty_vectors[category] = []
        
        if category not in self.stability_scores:
            self.stability_scores[category] = 1.0
    
    def _set_reference_vector(self, category: str, embeddings: Dict[str, np.ndarray]) -> None:
        """Set initial reference vector for a category."""
        first_key = next(iter(embeddings.keys()))
        self.reference_vectors[category] = embeddings[first_key]
        self.reference_timestamps[category] = time.time()
        logger.info(f"Set reference vector for category '{category}'")
    
    def _process_embeddings(
        self, 
        category: str, 
        embeddings: Dict[str, np.ndarray],
        input_data: Dict[str, Any]
    ) -> None:
        """Process new embeddings and update state."""
        # Get additional metadata from input if available
        metadata = {}
        if "metadata" in input_data:
            metadata = input_data["metadata"]
        elif "context" in input_data:
            metadata = input_data["context"]
        
        # Process each embedding
        for key, embedding in embeddings.items():
            # Add to recent vectors sliding window
            self.recent_vectors[category].append(embedding)
            
            # Check if we should take a snapshot
            self.snapshot_counters[category] += 1
            if self.snapshot_counters[category] >= self.snapshot_interval:
                self.snapshot_counters[category] = 0
                
                # Create a snapshot with metadata
                snapshot = DriftSnapshot(
                    vector=embedding.copy(),  # Ensure we have a copy
                    metadata={
                        "key": key,
                        "timestamp": time.time(),
                        **metadata
                    }
                )
                
                # Add to history (limiting size)
                self.vector_history[category].append(snapshot)
                if len(self.vector_history[category]) > self.history_length:
                    self.vector_history[category] = self.vector_history[category][-self.history_length:]
            
            # Check for novelty
            self._check_for_novelty(category, embedding)
    
    def _check_for_novelty(self, category: str, embedding: np.ndarray) -> None:
        """
        Check if an embedding represents a novel pattern.
        
        A novel pattern is one that's significantly different from both
        the reference and previously seen novel patterns.
        """
        # Calculate similarity to reference
        reference = self.reference_vectors[category]
        if self.vector_provider:
            similarity = self.vector_provider.calculate_similarity(reference, embedding)
        else:
            similarity = np.dot(reference, embedding) / (np.linalg.norm(reference) * np.linalg.norm(embedding) + 1e-8)
        
        # If significantly different from reference, check if it's a new pattern
        if similarity < 0.5:  # Threshold for potential novelty
            is_novel = True
            
            # Check against existing novel patterns
            for novel_vector in self.novelty_vectors[category]:
                if self.vector_provider:
                    novel_similarity = self.vector_provider.calculate_similarity(novel_vector, embedding)
                else:
                    novel_similarity = np.dot(novel_vector, embedding) / (np.linalg.norm(novel_vector) * np.linalg.norm(embedding) + 1e-8)
                
                # If similar to an existing novel pattern, not truly novel
                if novel_similarity > 0.7:  # Threshold for similarity to existing novelty
                    is_novel = False
                    break
            
            # If genuinely novel, add to tracking
            if is_novel:
                self.novelty_vectors[category].append(embedding.copy())
                # Limit number of tracked novel patterns
                if len(self.novelty_vectors[category]) > 20:
                    self.novelty_vectors[category] = self.novelty_vectors[category][-20:]
                
                # Update stability score - more novelty means less stability
                self.stability_scores[category] = max(0.0, self.stability_scores[category] - 0.05)
                logger.info(f"Detected novel pattern for category '{category}', stability now {self.stability_scores[category]:.2f}")
        else:
            # Gradually increase stability for consistent patterns
            self.stability_scores[category] = min(1.0, self.stability_scores[category] + 0.01)
    
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
        
        # Check if embeddings are directly provided
        if "embeddings" in input_data and isinstance(input_data["embeddings"], dict):
            for key, value in input_data["embeddings"].items():
                if isinstance(value, np.ndarray):
                    embeddings[key] = value
        
        # Compute embeddings for any missing items
        missing_keys = [k for k in text_data.keys() if k not in embeddings]
        
        if missing_keys and self.vector_provider:
            for key in missing_keys:
                embeddings[key] = self.vector_provider.get_embedding(text_data[key])
        
        return embeddings
    
    def _calculate_drift(self, category: str) -> DriftStats:
        """Calculate enhanced drift statistics for a category."""
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
        self.drift_history[category].append(drift_score)
        
        # Calculate trend
        trend = 0.0
        if len(self.drift_history[category]) > 1:
            recent_avg = np.mean(list(self.drift_history[category])[-min(5, len(self.drift_history[category])):])
            older_avg = np.mean(list(self.drift_history[category])[:-min(5, len(self.drift_history[category]))] or [0])
            trend = recent_avg - older_avg
        
        # Calculate advanced metrics
        
        # Temporal analysis - measure consistency over time
        temporal_stability = self._calculate_temporal_stability(category)
        
        # Volatility - measure how much the drift score changes
        volatility = self._calculate_volatility(category)
        
        # Create or update category stats
        self.category_stats[category] = CategoryStats(
            category=category,
            sample_count=len(self.recent_vectors[category]),
            drift_score=drift_score,
            drift_detected=drift_detected,
            trend=trend,
            last_update=time.time(),
            stability_score=self.stability_scores[category]
        )
        
        return DriftStats(
            drift_score=drift_score,
            mean_distance=mean_distance,
            max_distance=max_distance,
            drift_detected=drift_detected,
            details={
                "trend": trend,
                "sample_count": len(recent),
                "history_size": len(self.drift_history[category]),
                "category": category,
                "stability": self.stability_scores[category],
                "novelty_count": len(self.novelty_vectors[category]),
                "temporal_stability": temporal_stability,
                "volatility": volatility,
                "reference_age": time.time() - self.reference_timestamps.get(category, time.time())
            }
        )
    
    def _calculate_temporal_stability(self, category: str) -> float:
        """
        Calculate temporal stability of vectors over time.
        
        Returns a score from 0.0 to 1.0, where 1.0 is completely stable.
        """
        if len(self.vector_history[category]) < 5:
            return 1.0  # Not enough history
        
        # Get snapshots from different time periods
        histories = self.vector_history[category]
        
        # Calculate average pairwise similarities between vectors from different time periods
        similarities = []
        
        for i in range(len(histories) - 1):
            for j in range(i + 1, len(histories)):
                vec1 = histories[i].vector
                vec2 = histories[j].vector
                
                if self.vector_provider:
                    similarity = self.vector_provider.calculate_similarity(vec1, vec2)
                else:
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8)
                
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        # Average similarity as stability score
        return np.mean(similarities)
    
    def _calculate_volatility(self, category: str) -> float:
        """
        Calculate volatility of drift scores.
        
        Returns a value from 0.0 to 1.0, where 0.0 is no volatility.
        """
        if len(self.drift_history[category]) < 5:
            return 0.0  # Not enough history
        
        # Calculate standard deviation of recent drift scores
        recent_scores = list(self.drift_history[category])[-min(20, len(self.drift_history[category])):]
        
        if not recent_scores:
            return 0.0
        
        # Standard deviation as volatility measure, normalized to 0-1 range
        std_dev = np.std(recent_scores)
        return min(1.0, std_dev * 5.0)  # Scale to reasonable range
    
    def _update_reference_from_recent(self, category: str) -> None:
        """Update reference vector using recent data."""
        if category not in self.recent_vectors or len(self.recent_vectors[category]) < 10:
            return
        
        # Use mean of recent vectors as new reference
        recent = list(self.recent_vectors[category])
        new_reference = np.mean(recent, axis=0)
        
        # Normalize
        new_reference = new_reference / (np.linalg.norm(new_reference) + 1e-8)
        
        # Update reference
        self.reference_vectors[category] = new_reference
        self.reference_timestamps[category] = time.time()
        
        logger.info(f"Auto-updated reference vector for category '{category}' from {len(recent)} samples")
    
    def update_reference(self, category: str, vector: np.ndarray) -> None:
        """
        Update the reference vector for a category.
        
        Args:
            category: Category to update
            vector: New reference vector
        """
        self.reference_vectors[category] = vector
        self.reference_timestamps[category] = time.time()
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
            if category in self.reference_timestamps:
                del self.reference_timestamps[category]
            if category in self.recent_vectors:
                self.recent_vectors[category].clear()
            if category in self.drift_history:
                self.drift_history[category].clear()
            if category in self.vector_history:
                self.vector_history[category].clear()
            if category in self.snapshot_counters:
                self.snapshot_counters[category] = 0
            if category in self.novelty_vectors:
                self.novelty_vectors[category].clear()
            if category in self.stability_scores:
                self.stability_scores[category] = 1.0
            if category in self.category_stats:
                del self.category_stats[category]
                
            logger.info(f"Reset drift detector for category '{category}'")
        else:
            self.reference_vectors.clear()
            self.reference_timestamps.clear()
            self.recent_vectors.clear()
            self.drift_history.clear()
            self.vector_history.clear()
            self.snapshot_counters.clear()
            self.novelty_vectors.clear()
            self.stability_scores.clear()
            self.category_stats.clear()
            
            logger.info("Reset drift detector completely")
    
    def get_drift_history(self, category: str = None) -> Union[Dict[str, List[float]], List[float]]:
        """
        Get history of drift scores.
        
        Args:
            category: Optional category to get history for
            
        Returns:
            Dictionary mapping categories to drift histories, or a single list if category specified
        """
        if category:
            if category in self.drift_history:
                return list(self.drift_history[category])
            return []
        
        return {cat: list(history) for cat, history in self.drift_history.items()}
    
    def get_category_stats(self, category: Optional[str] = None) -> Union[Dict[str, CategoryStats], Optional[CategoryStats]]:
        """
        Get statistics for categories.
        
        Args:
            category: Optional specific category to get stats for
            
        Returns:
            Dictionary of category stats or a single CategoryStats object
        """
        if category:
            return self.category_stats.get(category)
        
        return self.category_stats
    
    def get_vector_snapshots(
        self, 
        category: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[DriftSnapshot]:
        """
        Get historical vector snapshots for a category.
        
        Args:
            category: Category to get snapshots for
            start_time: Optional start time (epoch seconds)
            end_time: Optional end time (epoch seconds)
            limit: Maximum number of snapshots to return
            
        Returns:
            List of snapshots sorted by timestamp
        """
        if category not in self.vector_history:
            return []
        
        snapshots = self.vector_history[category]
        
        # Apply time filters if specified
        if start_time is not None or end_time is not None:
            filtered = []
            for snapshot in snapshots:
                if start_time is not None and snapshot.timestamp < start_time:
                    continue
                if end_time is not None and snapshot.timestamp > end_time:
                    continue
                filtered.append(snapshot)
            snapshots = filtered
        
        # Sort by timestamp and limit results
        return sorted(snapshots, key=lambda s: s.timestamp)[-limit:]
    
    def compare_vectors(
        self,
        vector_a: np.ndarray,
        vector_b: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compare two vectors and provide similarity analysis.
        
        Args:
            vector_a: First vector
            vector_b: Second vector
            
        Returns:
            Dictionary with comparison results
        """
        if self.vector_provider:
            similarity = self.vector_provider.calculate_similarity(vector_a, vector_b)
        else:
            similarity = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b) + 1e-8)
        
        distance = 1.0 - similarity
        
        # Calculate additional measures
        cosine_distance = distance
        euclidean_distance = np.linalg.norm(vector_a - vector_b)
        
        return {
            "similarity": similarity,
            "distance": distance,
            "cosine_distance": cosine_distance,
            "euclidean_distance": euclidean_distance,
            "drift_detected": distance > self.drift_threshold
        }
    
    def _save(self) -> bool:
        """Save state to disk."""
        if not self.persistence_path:
            return False
            
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            
            # Convert deques to lists for serialization
            data = {
                "reference_vectors": {k: v.tolist() for k, v in self.reference_vectors.items()},
                "reference_timestamps": self.reference_timestamps,
                "drift_history": {k: list(v) for k, v in self.drift_history.items()},
                "snapshot_counters": self.snapshot_counters,
                "stability_scores": self.stability_scores,
                "category_stats": {k: v.to_dict() for k, v in self.category_stats.items()},
                "config": {
                    "window_size": self.window_size,
                    "drift_threshold": self.drift_threshold,
                    "history_length": self.history_length,
                    "snapshot_interval": self.snapshot_interval,
                    "auto_reference_update": self.auto_reference_update,
                    "auto_reference_update_interval": self.auto_reference_update_interval
                },
                "timestamp": time.time()
            }
            
            # Save vector history separately - too large for the main file
            vector_history_path = f"{self.persistence_path}_vectors"
            with open(vector_history_path, 'wb') as f:
                # Convert to serializable format
                serializable_history = {}
                for category, snapshots in self.vector_history.items():
                    serializable_history[category] = [
                        {
                            "vector": snapshot.vector.tolist(),
                            "metadata": snapshot.metadata,
                            "timestamp": snapshot.timestamp
                        }
                        for snapshot in snapshots
                    ]
                pickle.dump(serializable_history, f)
            
            # Save main data
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"Saved drift detector state to {self.persistence_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save drift detector state: {str(e)}")
            return False
    
    def _load(self) -> bool:
        """Load state from disk."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return False
            
        try:
            # Load main data
            with open(self.persistence_path, 'rb') as f:
                data = pickle.load(f)
            
            # Load configuration
            if "config" in data:
                config = data["config"]
                self.window_size = config.get("window_size", self.window_size)
                self.drift_threshold = config.get("drift_threshold", self.drift_threshold)
                self.history_length = config.get("history_length", self.history_length)
                self.snapshot_interval = config.get("snapshot_interval", self.snapshot_interval)
                self.auto_reference_update = config.get("auto_reference_update", self.auto_reference_update)
                self.auto_reference_update_interval = config.get("auto_reference_update_interval", self.auto_reference_update_interval)
            
            # Load reference vectors
            if "reference_vectors" in data:
                self.reference_vectors = {
                    k: np.array(v) for k, v in data["reference_vectors"].items()
                }
            
            # Load reference timestamps
            if "reference_timestamps" in data:
                self.reference_timestamps = data["reference_timestamps"]
            
            # Load drift history
            if "drift_history" in data:
                self.drift_history = {
                    k: deque(v, maxlen=self.history_length) 
                    for k, v in data["drift_history"].items()
                }
            
            # Load snapshot counters
            if "snapshot_counters" in data:
                self.snapshot_counters = data["snapshot_counters"]
            
            # Load stability scores
            if "stability_scores" in data:
                self.stability_scores = data["stability_scores"]
            
            # Load category stats
            if "category_stats" in data:
                self.category_stats = {
                    k: CategoryStats(**v) for k, v in data["category_stats"].items()
                }
            
            # Try to load vector history
            vector_history_path = f"{self.persistence_path}_vectors"
            if os.path.exists(vector_history_path):
                try:
                    with open(vector_history_path, 'rb') as f:
                        serialized_history = pickle.load(f)
                        
                    # Convert back to DriftSnapshot objects
                    for category, snapshots in serialized_history.items():
                        self.vector_history[category] = [
                            DriftSnapshot(
                                vector=np.array(snapshot["vector"]),
                                metadata=snapshot["metadata"],
                                timestamp=snapshot["timestamp"]
                            )
                            for snapshot in snapshots
                        ]
                except Exception as e:
                    logger.error(f"Failed to load vector history: {str(e)}")
                    # Initialize empty history
                    self.vector_history = {}
            
            # Initialize sliding windows
            for category in self.reference_vectors.keys():
                if category not in self.recent_vectors:
                    self.recent_vectors[category] = deque(maxlen=self.window_size)
                if category not in self.novelty_vectors:
                    self.novelty_vectors[category] = []
            
            logger.info(f"Loaded drift detector state from {self.persistence_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load drift detector state: {str(e)}")
            return False


# Import random here to avoid circular imports
import random