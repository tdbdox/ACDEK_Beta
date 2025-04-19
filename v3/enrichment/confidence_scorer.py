import logging
import time
import math  # for signal decay calculations
import numpy as np
import traceback
import json
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable

# Registry imports
from registry.registry_types import PluginType
from registry.plugin_registry import register_plugin
from interfaces.enrichment_interface import EnrichmentInterface

# Vector utilities
from storage.vector_store.embed_utils import cosine_similarity

# For hallucination metrics integration
try:
    from enrichment.hallucination_metrics import ConfidenceCorrelation
except ImportError:
    # Define locally if module not available
    class ConfidenceCorrelation(Enum):
        """Types of confidence correlation mismatches."""
        ACCURATE = "accurate"                  # Confidence aligns with factuality
        OVERCONFIDENT = "overconfident"        # High confidence but low factuality
        UNDERCONFIDENT = "underconfident"      # Low confidence but high factuality
        CONFIDENTLY_WRONG = "confidently_wrong"  # Very high confidence but very low factuality
        CONFIDENTLY_RIGHT = "confidently_right"  # Very high confidence and very high factuality

# Configure logging
logger = logging.getLogger(__name__)

class ConfidenceFactorType(str, Enum):
    """Enumeration of factors that influence confidence scores."""
    VECTOR_QUALITY = "vector_quality"          # Quality of the vector itself (norm, dimensionality)
    EMBEDDING_MODEL = "embedding_model"        # Which embedding model was used
    FALLBACK_STATUS = "fallback_status"        # Whether fallbacks were used during embedding
    SEMANTIC_CONTEXT = "semantic_context"      # Contextual relevance to a reference point
    SEMANTIC_DRIFT = "semantic_drift"          # Amount of drift from historical norms
    QUERY_RELEVANCE = "query_relevance"        # Relevance to the original query
    MODALITY_MATCH = "modality_match"          # Whether content modality matches expected type
    RECENCY = "recency"                        # How recently the vector was generated
    SOURCE_RELIABILITY = "source_reliability"  # Reliability of the content source
    CUSTOM = "custom"                          # Custom factor defined by user


@dataclass
class ConfidenceFactor:
    """Definition of a factor that affects confidence scoring."""
    factor_type: ConfidenceFactorType
    weight: float = 1.0
    min_value: float = 0.0
    max_value: float = 1.0
    enabled: bool = True
    description: str = ""
    custom_scorer: Optional[Callable[[Dict[str, Any]], float]] = None


@dataclass
class ConfidenceConfig:
    """Configuration for the confidence scoring system."""
    factors: List[ConfidenceFactor] = field(default_factory=list)
    default_threshold: float = 0.6
    min_aggregate_score: float = 0.0
    max_aggregate_score: float = 1.0
    normalization_method: str = "weighted_average"  # weighted_average, min, max, product
    detailed_output: bool = False
    persistence_enabled: bool = True
    cache_dir: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        'cache'
    ))
    decay_rate: float = 0.001  # rate for exponential confidence decay


@dataclass
class ConfidenceScore:
    """Detailed confidence score with factor breakdown."""
    aggregate_score: float
    factors: Dict[ConfidenceFactorType, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __float__(self):
        """Allow direct conversion to float."""
        return self.aggregate_score
    
    def is_reliable(self, threshold: Optional[float] = None) -> bool:
        """Check if the score is above the reliability threshold."""
        if threshold is None:
            threshold = self.metadata.get("threshold", 0.6)
        return self.aggregate_score >= threshold
    
    def get_weakest_factors(self, limit: int = 3) -> List[Tuple[ConfidenceFactorType, float]]:
        """Get the factors with the lowest scores."""
        return sorted(self.factors.items(), key=lambda x: x[1])[:limit]
    
    def get_strongest_factors(self, limit: int = 3) -> List[Tuple[ConfidenceFactorType, float]]:
        """Get the factors with the highest scores."""
        return sorted(self.factors.items(), key=lambda x: x[1], reverse=True)[:limit]


def _create_default_config() -> ConfidenceConfig:
    """Create a default configuration with common confidence factors."""
    return ConfidenceConfig(
        factors=[
            ConfidenceFactor(
                factor_type=ConfidenceFactorType.VECTOR_QUALITY,
                weight=1.0,
                description="Quality of the vector itself based on norm and dimensionality"
            ),
            ConfidenceFactor(
                factor_type=ConfidenceFactorType.EMBEDDING_MODEL,
                weight=0.8,
                description="Confidence based on the embedding model used"
            ),
            ConfidenceFactor(
                factor_type=ConfidenceFactorType.FALLBACK_STATUS,
                weight=0.7,
                description="Adjustment based on whether fallbacks were used"
            ),
            ConfidenceFactor(
                factor_type=ConfidenceFactorType.SEMANTIC_CONTEXT,
                weight=1.0,
                description="Contextual relevance to a reference point"
            ),
            ConfidenceFactor(
                factor_type=ConfidenceFactorType.SEMANTIC_DRIFT,
                weight=0.9,
                description="Amount of drift from historical norms"
            ),
            ConfidenceFactor(
                factor_type=ConfidenceFactorType.QUERY_RELEVANCE,
                weight=1.2,
                description="Relevance to the original query"
            ),
            ConfidenceFactor(
                factor_type=ConfidenceFactorType.MODALITY_MATCH,
                weight=0.9,
                description="Whether content modality matches expected type"
            ),
            ConfidenceFactor(
                factor_type=ConfidenceFactorType.RECENCY,
                weight=0.6,
                description="How recently the vector was generated"
            ),
            ConfidenceFactor(
                factor_type=ConfidenceFactorType.SOURCE_RELIABILITY,
                weight=0.8,
                description="Reliability of the content source"
            ),
        ],
        default_threshold=0.6
    )


@register_plugin(
    plugin_type=PluginType.ENRICHMENT,
    name="vector_confidence_scorer",
    version="1.0.0",
    description="Configurable confidence scoring system for vector embeddings",
    interfaces=["EnrichmentInterface"],
    is_default=True
)
class VectorConfidenceScorer(EnrichmentInterface[Dict[str, Any], ConfidenceScore]):
    """
    Configurable confidence scoring system for vector embeddings.
    
    This component evaluates the reliability of vector embeddings based on
    multiple configurable factors like vector quality, embedding model,
    semantic drift, and contextual relevance.
    """
    
    def __init__(self, config: Optional[ConfidenceConfig] = None):
        """
        Initialize the confidence scoring system.
        
        Args:
            config: Optional configuration for the scorer
        """
        self.config = config or _create_default_config()
        
        # Create cache directory if persistence is enabled
        if self.config.persistence_enabled:
            os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Process factors for fast lookup
        self.factor_weights = {
            factor.factor_type: factor.weight
            for factor in self.config.factors
            if factor.enabled
        }
        
        # Total weight for normalization
        self.total_weight = sum(self.factor_weights.values())
        
        # Factor lookup by type
        self.factors_by_type = {
            factor.factor_type: factor
            for factor in self.config.factors
        }
        
        # Track historical scores for drift detection
        self.historical_scores: Dict[str, List[ConfidenceScore]] = {}
        
        logger.info(f"Initialized VectorConfidenceScorer with {len(self.config.factors)} factors")
    
    def _score_vector_quality(self, data: Dict[str, Any]) -> float:
        """
        Score vector quality based on norm and dimensionality.
        
        Args:
            data: Context data including the vector
            
        Returns:
            Quality score between 0 and 1
        """
        try:
            vector = data.get("vector")
            if vector is None or not isinstance(vector, np.ndarray):
                return 0.0
            
            # Check for NaN values
            if np.isnan(vector).any():
                return 0.1  # Severe penalty for NaN values
            
            # Vector norm (should be close to 1.0 for normalized vectors)
            norm = float(np.linalg.norm(vector))
            norm_score = max(0.0, 1.0 - abs(1.0 - norm))
            
            # Dimension check
            dimension = vector.shape[0]
            expected_dimension = data.get("expected_dimension", 0)
            
            # If expected dimension is provided, check against it
            dim_score = 1.0
            if expected_dimension > 0:
                dim_score = 1.0 if dimension == expected_dimension else max(0.3, 0.3 + (0.7 * dimension / expected_dimension))
            
            # Zero vector check
            zero_ratio = np.count_nonzero(vector == 0) / vector.size
            zero_score = 1.0 - min(1.0, zero_ratio * 2)  # Penalize if more than 50% zeros
            
            # Combine scores
            quality_score = 0.4 * norm_score + 0.3 * dim_score + 0.3 * zero_score
            return min(1.0, max(0.0, quality_score))
            
        except Exception as e:
            logger.warning(f"Error calculating vector quality score: {str(e)}")
            return 0.5  # Default to neutral score on error
    
    def _score_embedding_model(self, data: Dict[str, Any]) -> float:
        """
        Score based on the embedding model used.
        
        Args:
            data: Context data including model information
            
        Returns:
            Model confidence score between 0 and 1
        """
        try:
            # Get model information
            model_name = data.get("model_name", "unknown")
            expected_model = data.get("expected_model", "")
            preferred_models = data.get("preferred_models", [])
            
            # Default scores for different conditions
            if expected_model and model_name == expected_model:
                return 1.0  # Exact match with expected model
            
            if preferred_models and model_name in preferred_models:
                return 0.9  # One of the preferred models
            
            # Known good models (hardcoded for now, could be configurable)
            strong_models = [
                "all-mpnet-base-v2", "all-MiniLM-L12-v2", "paraphrase-multilingual-mpnet-base-v2",
                "all-MiniLM-L6-v2", "distiluse-base-multilingual-cased-v1"
            ]
            
            decent_models = [
                "paraphrase-MiniLM-L3-v2", "paraphrase-MiniLM-L6-v2", 
                "distilbert-base-nli-mean-tokens"
            ]
            
            # Score based on model quality
            if model_name in strong_models:
                return 0.85
            elif model_name in decent_models:
                return 0.7
            elif "base" in model_name.lower():
                return 0.65
            elif "distil" in model_name.lower() or "mini" in model_name.lower():
                return 0.55
            else:
                return 0.5  # Unknown model, neutral score
                
        except Exception as e:
            logger.warning(f"Error calculating embedding model score: {str(e)}")
            return 0.5  # Default to neutral score on error
    
    def _score_fallback_status(self, data: Dict[str, Any]) -> float:
        """
        Score based on whether fallbacks were used during embedding.
        
        Args:
            data: Context data including fallback information
            
        Returns:
            Fallback confidence score between 0 and 1
        """
        try:
            # Check if embedding used fallback
            used_fallback = data.get("used_fallback", False)
            fallback_tier = data.get("fallback_tier", 0)
            attempts = data.get("attempts", 1)
            
            if not used_fallback:
                return 1.0  # No fallback used, full confidence
            
            # Adjust based on fallback tier and attempts
            base_score = max(0.3, 1.0 - (fallback_tier * 0.2))
            attempt_penalty = min(0.2, (attempts - 1) * 0.05)
            
            return max(0.1, base_score - attempt_penalty)
            
        except Exception as e:
            logger.warning(f"Error calculating fallback status score: {str(e)}")
            return 0.5  # Default to neutral score on error
    
    def _score_semantic_context(self, data: Dict[str, Any]) -> float:
        """
        Score based on semantic context relative to reference points.
        
        Args:
            data: Context data including the vector and reference points
            
        Returns:
            Semantic context score between 0 and 1
        """
        try:
            vector = data.get("vector")
            if vector is None or not isinstance(vector, np.ndarray):
                return 0.5
            
            # Reference vectors for context comparison
            reference_vectors = data.get("reference_vectors", [])
            if not reference_vectors:
                return 0.7  # No references, slightly reduced confidence
            
            # Calculate similarity to each reference
            similarities = []
            for ref_vector in reference_vectors:
                if isinstance(ref_vector, np.ndarray) and ref_vector.shape == vector.shape:
                    sim = cosine_similarity(vector, ref_vector)
                    similarities.append(sim)
            
            if not similarities:
                return 0.5  # No valid similarities calculated
            
            # Use the highest similarity as the score
            # Alternatively, could use average or a weighted approach
            return max(similarities)
            
        except Exception as e:
            logger.warning(f"Error calculating semantic context score: {str(e)}")
            return 0.5  # Default to neutral score on error
    
    def _score_semantic_drift(self, data: Dict[str, Any]) -> float:
        """
        Score based on semantic drift from historical norms.
        
        Args:
            data: Context data including the vector and category
            
        Returns:
            Drift confidence score between 0 and 1
        """
        try:
            vector = data.get("vector")
            category = data.get("category", "default")
            drift_stats = data.get("drift_stats", {})
            
            if vector is None or not isinstance(vector, np.ndarray):
                return 0.5
            
            # If explicit drift measurements are provided, use them
            if drift_stats:
                drift_score = drift_stats.get("drift_score", 0.0)
                return max(0.0, 1.0 - drift_score)
            
            # If we have historical data for this category, calculate drift
            if category in self.historical_scores and self.historical_scores[category]:
                # Get recent historical vectors (up to 10)
                recent_scores = self.historical_scores[category][-10:]
                recent_vectors = [
                    score.metadata.get("vector", None) 
                    for score in recent_scores
                    if "vector" in score.metadata
                ]
                
                # Calculate average similarity to recent vectors
                similarities = []
                for hist_vector in recent_vectors:
                    if isinstance(hist_vector, np.ndarray) and hist_vector.shape == vector.shape:
                        sim = cosine_similarity(vector, hist_vector)
                        similarities.append(sim)
                
                if similarities:
                    avg_similarity = sum(similarities) / len(similarities)
                    # High similarity = low drift = high confidence
                    return avg_similarity
            
            # No historical data or drift stats
            return 0.7  # Slightly reduced confidence due to lack of history
            
        except Exception as e:
            logger.warning(f"Error calculating semantic drift score: {str(e)}")
            return 0.5  # Default to neutral score on error
    
    def _score_query_relevance(self, data: Dict[str, Any]) -> float:
        """
        Score based on relevance to the original query.
        
        Args:
            data: Context data including the vector and query vector
            
        Returns:
            Query relevance score between 0 and 1
        """
        try:
            vector = data.get("vector")
            query_vector = data.get("query_vector")
            
            if vector is None or query_vector is None:
                return 0.5
            
            if not isinstance(vector, np.ndarray) or not isinstance(query_vector, np.ndarray):
                return 0.5
            
            # Calculate direct similarity to query
            similarity = cosine_similarity(vector, query_vector)
            
            # Apply a threshold curve to emphasize higher similarities
            # For similarities < 0.5, reduce confidence more sharply
            if similarity < 0.5:
                return similarity * 0.7
            else:
                # Scale 0.5-1.0 range to 0.35-1.0
                return 0.35 + (similarity - 0.5) * 1.3
            
        except Exception as e:
            logger.warning(f"Error calculating query relevance score: {str(e)}")
            return 0.5  # Default to neutral score on error
    
    def _score_modality_match(self, data: Dict[str, Any]) -> float:
        """
        Score based on whether content modality matches expected type.
        
        Args:
            data: Context data including modality information
            
        Returns:
            Modality match score between 0 and 1
        """
        try:
            # Get modality information
            content_modality = data.get("content_modality", "unknown")
            expected_modality = data.get("expected_modality", "")
            
            if not expected_modality:
                return 0.8  # No expected modality specified
            
            if content_modality == expected_modality:
                return 1.0  # Perfect match
            
            # Check for partial matches or compatibility
            content_modality = content_modality.lower()
            expected_modality = expected_modality.lower()
            
            # Compatible modalities
            text_compatible = ["text", "string", "document", "article"]
            structured_compatible = ["structured", "json", "dict", "table", "dataframe"]
            
            if content_modality in text_compatible and expected_modality in text_compatible:
                return 0.9  # Compatible text modalities
            
            if content_modality in structured_compatible and expected_modality in structured_compatible:
                return 0.8  # Compatible structured data modalities
            
            # For mixed modality content, give partial credit if expected is one of the components
            if content_modality == "mixed":
                if any(m in expected_modality for m in text_compatible + structured_compatible):
                    return 0.6
            
            # Significant mismatch
            return 0.3
            
        except Exception as e:
            logger.warning(f"Error calculating modality match score: {str(e)}")
            return 0.5  # Default to neutral score on error
    
    def _score_recency(self, data: Dict[str, Any]) -> float:
        """
        Score based on how recently the vector was generated.
        
        Args:
            data: Context data including timestamp information
            
        Returns:
            Recency score between 0 and 1
        """
        try:
            # Get timestamp information
            generation_time = data.get("generation_time", 0)
            if isinstance(generation_time, str):
                try:
                    # Try to parse ISO format timestamp
                    from datetime import datetime
                    dt = datetime.fromisoformat(generation_time)
                    generation_time = dt.timestamp()
                except (ValueError, TypeError):
                    generation_time = 0
            
            current_time = time.time()
            
            # Invalid or missing timestamp
            if not generation_time:
                return 0.5
            
            # Calculate age in days
            age_seconds = max(0, current_time - generation_time)
            age_days = age_seconds / (24 * 3600)
            
            # Score based on age:
            # - Less than 1 day: 1.0
            # - 1-7 days: 0.9 to 0.7
            # - 7-30 days: 0.7 to 0.5
            # - 30-90 days: 0.5 to 0.3
            # - 90+ days: 0.3 to 0.1
            if age_days < 1:
                return 1.0
            elif age_days < 7:
                return 0.9 - (age_days - 1) * 0.2 / 6
            elif age_days < 30:
                return 0.7 - (age_days - 7) * 0.2 / 23
            elif age_days < 90:
                return 0.5 - (age_days - 30) * 0.2 / 60
            else:
                return max(0.1, 0.3 - (age_days - 90) * 0.2 / 90)
            
        except Exception as e:
            logger.warning(f"Error calculating recency score: {str(e)}")
            return 0.5  # Default to neutral score on error
    
    def _score_source_reliability(self, data: Dict[str, Any]) -> float:
        """
        Score based on reliability of the content source.
        
        Args:
            data: Context data including source information
            
        Returns:
            Source reliability score between 0 and 1
        """
        try:
            # Get source information
            source = data.get("source", "unknown")
            trusted_sources = data.get("trusted_sources", [])
            untrusted_sources = data.get("untrusted_sources", [])
            
            # Check against trusted and untrusted lists
            if trusted_sources and source in trusted_sources:
                return 1.0
            
            if untrusted_sources and source in untrusted_sources:
                return 0.2
            
            # No specific information, use reasonable defaults
            if source == "unknown":
                return 0.5
            
            # Heuristics for common sources (could be made configurable)
            if "database" in source.lower():
                return 0.9
            elif "api" in source.lower():
                return 0.8
            elif "cache" in source.lower():
                return 0.7
            elif "user" in source.lower():
                return 0.7
            elif "file" in source.lower():
                return 0.6
            elif "external" in source.lower():
                return 0.5
            elif "web" in source.lower() or "http" in source.lower():
                return 0.4
            
            # Default for unrecognized sources
            return 0.6
            
        except Exception as e:
            logger.warning(f"Error calculating source reliability score: {str(e)}")
            return 0.5  # Default to neutral score on error
    
    def _calculate_custom_score(self, factor: ConfidenceFactor, data: Dict[str, Any]) -> float:
        """
        Calculate score using a custom scoring function.
        
        Args:
            factor: The custom confidence factor with scoring function
            data: Context data for scoring
            
        Returns:
            Custom score between 0 and 1
        """
        try:
            if factor.custom_scorer is None:
                return 0.5
                
            score = factor.custom_scorer(data)
            return min(factor.max_value, max(factor.min_value, score))
            
        except Exception as e:
            logger.warning(f"Error calculating custom score: {str(e)}")
            return 0.5  # Default to neutral score on error
    
    def _normalize_scores(self, factor_scores: Dict[ConfidenceFactorType, float]) -> float:
        """
        Normalize individual factor scores into an aggregate score.
        
        Args:
            factor_scores: Dictionary of factors and their scores
            
        Returns:
            Normalized aggregate score
        """
        if not factor_scores:
            return 0.5
        
        method = self.config.normalization_method.lower()
        
        # Calculate weighted average (default method)
        if method == "weighted_average":
            weighted_sum = 0.0
            total_weight = 0.0
            
            for factor_type, score in factor_scores.items():
                weight = self.factor_weights.get(factor_type, 1.0)
                weighted_sum += score * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.5
                
            return weighted_sum / total_weight
        
        # Minimum of all scores
        elif method == "min":
            return min(factor_scores.values())
        
        # Maximum of all scores
        elif method == "max":
            return max(factor_scores.values())
        
        # Product of all scores
        elif method == "product":
            if not factor_scores:
                return 0.5
                
            product = 1.0
            for score in factor_scores.values():
                product *= score
                
            # Take nth root to keep in similar range
            return product ** (1.0 / len(factor_scores))
        
        # Special method combining weighted average and minimum
        elif method == "hybrid":
            weighted_avg = 0.0
            total_weight = 0.0
            min_score = min(factor_scores.values())
            
            for factor_type, score in factor_scores.items():
                weight = self.factor_weights.get(factor_type, 1.0)
                weighted_avg += score * weight
                total_weight += weight
            
            if total_weight == 0:
                weighted_avg = sum(factor_scores.values()) / len(factor_scores)
            else:
                weighted_avg = weighted_avg / total_weight
            
            # Blend weighted average with minimum score
            # This ensures that very low scores in any factor have significant impact
            return 0.7 * weighted_avg + 0.3 * min_score
        
        # Default to simple average if method not recognized
        return sum(factor_scores.values()) / len(factor_scores)
    
    def score(self, data: Dict[str, Any]) -> ConfidenceScore:
        """
        Score the confidence of a vector embedding based on multiple factors.
        
        Args:
            data: Context data for scoring
            
        Returns:
            Confidence score with detailed breakdown
        """
        try:
            # Collect scores for all enabled factors
            factor_scores = {}
            
            # Scoring functions for standard factors
            scoring_functions = {
                ConfidenceFactorType.VECTOR_QUALITY: self._score_vector_quality,
                ConfidenceFactorType.EMBEDDING_MODEL: self._score_embedding_model,
                ConfidenceFactorType.FALLBACK_STATUS: self._score_fallback_status,
                ConfidenceFactorType.SEMANTIC_CONTEXT: self._score_semantic_context,
                ConfidenceFactorType.SEMANTIC_DRIFT: self._score_semantic_drift,
                ConfidenceFactorType.QUERY_RELEVANCE: self._score_query_relevance,
                ConfidenceFactorType.MODALITY_MATCH: self._score_modality_match,
                ConfidenceFactorType.RECENCY: self._score_recency,
                ConfidenceFactorType.SOURCE_RELIABILITY: self._score_source_reliability,
            }
            
            # Score each enabled factor
            for factor in self.config.factors:
                if not factor.enabled:
                    continue
                    
                if factor.factor_type == ConfidenceFactorType.CUSTOM:
                    # Handle custom factors
                    if factor.custom_scorer:
                        factor_scores[factor.factor_type] = self._calculate_custom_score(factor, data)
                else:
                    # Handle standard factors
                    if factor.factor_type in scoring_functions:
                        raw_score = scoring_functions[factor.factor_type](data)
                        # Apply factor min/max bounds
                        score = min(factor.max_value, max(factor.min_value, raw_score))
                        factor_scores[factor.factor_type] = score
            
            # Calculate aggregate score
            aggregate_score = self._normalize_scores(factor_scores)
            
            # Apply global min/max bounds
            aggregate_score = min(
                self.config.max_aggregate_score, 
                max(self.config.min_aggregate_score, aggregate_score)
            )
            
            # Create metadata
            metadata = {
                "timestamp": time.time(),
                "threshold": data.get("threshold", self.config.default_threshold),
                "normalization_method": self.config.normalization_method,
            }
            
            # Include vector in metadata if detailed output is enabled
            if self.config.detailed_output and "vector" in data:
                metadata["vector"] = data["vector"]
            
            # Include category for historical tracking if provided
            category = data.get("category", "default")
            metadata["category"] = category
            
            # Check if factuality score is provided for correlation analysis
            measured_factuality = data.get("measured_factuality")
            if measured_factuality is not None:
                # Analyze confidence-factuality correlation
                correlation_result = self.analyze_confidence_correlation(
                    aggregate_score, measured_factuality
                )
                metadata["confidence_correlation"] = correlation_result
                
                # Track correlation statistics
                self._track_correlation_stats(
                    correlation_result["correlation_type"],
                    aggregate_score,
                    measured_factuality,
                    correlation_result["correlation_score"]
                )
            
            # Create confidence score object
            score = ConfidenceScore(
                aggregate_score=aggregate_score,
                factors=factor_scores,
                metadata=metadata
            )
            
            # Record score in historical data if enabled
            if self.config.persistence_enabled:
                if category not in self.historical_scores:
                    self.historical_scores[category] = []
                
                # Limit history size to avoid memory issues
                if len(self.historical_scores[category]) >= 1000:
                    self.historical_scores[category] = self.historical_scores[category][-900:]
                
                self.historical_scores[category].append(score)
                
                # Periodically save to disk if enabled
                if len(self.historical_scores[category]) % 100 == 0:
                    self._save_historical_data(category)
            
            # Uncertainty quantification
            try:
                uncertainty = 1.0 - score.aggregate_score
                score.metadata['uncertainty'] = uncertainty
            except Exception:
                score.metadata['uncertainty'] = None
            # Signal confidence decay based on input timestamp
            input_ts = data.get('timestamp')
            if input_ts:
                age = time.time() - input_ts
                decay_factor = math.exp(-self.config.decay_rate * age)
                score.aggregate_score *= decay_factor
                score.factors['signal_decay'] = decay_factor
                score.metadata['decay_factor'] = decay_factor
            # Correlation with constraint strength if provided
            constraint_strength = data.get('constraint_strength')
            if constraint_strength is not None:
                score.aggregate_score *= constraint_strength
                score.factors['constraint_correlation'] = constraint_strength
                score.metadata['constraint_strength'] = constraint_strength
            return score
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Return a fallback score on error
            return ConfidenceScore(
                aggregate_score=0.5,
                factors={},
                metadata={"error": str(e), "fallback": True}
            )
    
    def _track_correlation_stats(
        self,
        correlation_type: str,
        confidence: float,
        factuality: float,
        correlation_score: float
    ) -> None:
        """
        Track statistics for confidence-factuality correlation.
        This helps analyze how well confidence scores predict factuality.
        
        Args:
            correlation_type: Type of correlation pattern observed
            confidence: Model's confidence score
            factuality: Measured factuality score
            correlation_score: Correlation quality score
        """
        # Initialize correlation stats storage if not exists
        if not hasattr(self, "correlation_stats"):
            self.correlation_stats = {
                "accurate": {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_factuality": 0.0,
                    "avg_correlation": 0.0
                },
                "overconfident": {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_factuality": 0.0,
                    "avg_correlation": 0.0
                },
                "underconfident": {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_factuality": 0.0,
                    "avg_correlation": 0.0
                },
                "confidently_wrong": {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_factuality": 0.0,
                    "avg_correlation": 0.0
                },
                "confidently_right": {
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_factuality": 0.0,
                    "avg_correlation": 0.0
                },
                "total_count": 0,
                "overall_avg_correlation": 0.0
            }
        
        # Skip if invalid correlation type
        if correlation_type not in self.correlation_stats or correlation_type == "total_count":
            return
        
        # Update overall stats
        self.correlation_stats["total_count"] += 1
        total = self.correlation_stats["total_count"]
        self.correlation_stats["overall_avg_correlation"] = (
            (self.correlation_stats["overall_avg_correlation"] * (total - 1) + correlation_score) / total
        )
        
        # Update specific correlation type stats
        type_stats = self.correlation_stats[correlation_type]
        type_stats["count"] += 1
        count = type_stats["count"]
        
        # Update rolling averages
        type_stats["avg_confidence"] = (type_stats["avg_confidence"] * (count - 1) + confidence) / count
        type_stats["avg_factuality"] = (type_stats["avg_factuality"] * (count - 1) + factuality) / count
        type_stats["avg_correlation"] = (type_stats["avg_correlation"] * (count - 1) + correlation_score) / count
    
    def get_correlation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about confidence-factuality correlation.
        Useful for hallucination metrics analysis.
        
        Returns:
            Dictionary of correlation statistics
        """
        if not hasattr(self, "correlation_stats"):
            return {
                "total_count": 0,
                "overall_avg_correlation": 0.0,
                "correlation_types": {
                    "accurate": 0,
                    "overconfident": 0,
                    "underconfident": 0,
                    "confidently_wrong": 0,
                    "confidently_right": 0
                }
            }
        
        # Create a simplified version with just counts for each type
        simplified = {
            "total_count": self.correlation_stats["total_count"],
            "overall_avg_correlation": self.correlation_stats["overall_avg_correlation"],
            "correlation_types": {
                k: v["count"] for k, v in self.correlation_stats.items() 
                if k not in ["total_count", "overall_avg_correlation"]
            }
        }
        
        return simplified
    
    def process(self, input_data: Dict[str, Any]) -> ConfidenceScore:
        """
        Process input data to generate a confidence score.
        
        Args:
            input_data: Input data containing vector and context information
            
        Returns:
            ConfidenceScore with detailed breakdown
        """
        return self.score(input_data)
    
    def _save_historical_data(self, category: str) -> None:
        """
        Save historical scoring data to disk.
        
        Args:
            category: The category to save data for
        """
        if not self.config.persistence_enabled or category not in self.historical_scores:
            return
            
        try:
            # Create category subdirectory
            category_dir = os.path.join(self.config.cache_dir, "confidence_history")
            os.makedirs(category_dir, exist_ok=True)
            
            # Create filename with normalized category name
            safe_category = category.replace(" ", "_").replace("/", "_").replace("\\", "_")
            filename = os.path.join(category_dir, f"{safe_category}_history.json")
            
            # Serialize data (excluding vectors to save space)
            serialized_data = []
            for score in self.historical_scores[category][-100:]:  # Only save most recent 100
                score_dict = {
                    "aggregate_score": score.aggregate_score,
                    "factors": {k.value if isinstance(k, Enum) else k: v for k, v in score.factors.items()},
                    "timestamp": score.timestamp,
                    "metadata": {
                        k: v for k, v in score.metadata.items() 
                        if k != "vector" and not isinstance(v, np.ndarray)
                    }
                }
                serialized_data.append(score_dict)
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(serialized_data, f, indent=2)
                
            logger.debug(f"Saved historical confidence data for category '{category}'")
            
        except Exception as e:
            logger.warning(f"Failed to save historical confidence data: {str(e)}")
    
    def _load_historical_data(self, category: str) -> None:
        """
        Load historical scoring data from disk.
        
        Args:
            category: The category to load data for
        """
        if not self.config.persistence_enabled:
            return
            
        try:
            # Check for category file
            category_dir = os.path.join(self.config.cache_dir, "confidence_history")
            safe_category = category.replace(" ", "_").replace("/", "_").replace("\\", "_")
            filename = os.path.join(category_dir, f"{safe_category}_history.json")
            
            if not os.path.exists(filename):
                return
                
            # Load data
            with open(filename, 'r') as f:
                serialized_data = json.load(f)
            
            # Deserialize into ConfidenceScore objects
            scores = []
            for data in serialized_data:
                # Convert string keys back to enum values
                factors = {}
                for k, v in data.get("factors", {}).items():
                    try:
                        factor_type = ConfidenceFactorType(k)
                        factors[factor_type] = v
                    except ValueError:
                        factors[k] = v
                
                score = ConfidenceScore(
                    aggregate_score=data.get("aggregate_score", 0.5),
                    factors=factors,
                    metadata=data.get("metadata", {}),
                    timestamp=data.get("timestamp", time.time())
                )
                scores.append(score)
            
            self.historical_scores[category] = scores
            logger.debug(f"Loaded {len(scores)} historical confidence scores for category '{category}'")
            
        except Exception as e:
            logger.warning(f"Failed to load historical confidence data: {str(e)}")
    
    def get_category_stats(self, category: str) -> Dict[str, Any]:
        """
        Get statistics for a specific category.
        
        Args:
            category: Category to get statistics for
            
        Returns:
            Dictionary of statistics
        """
        if category not in self.historical_scores:
            self._load_historical_data(category)
            
        if category not in self.historical_scores or not self.historical_scores[category]:
            return {
                "category": category,
                "count": 0,
                "avg_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "recent_avg": 0.0,
                "trend": 0.0
            }
        
        scores = self.historical_scores[category]
        aggregate_scores = [score.aggregate_score for score in scores]
        
        # Calculate basic statistics
        avg_score = sum(aggregate_scores) / len(aggregate_scores)
        min_score = min(aggregate_scores)
        max_score = max(aggregate_scores)
        
        # Calculate recent average (last 10% or at least 5 entries)
        recent_count = max(5, len(scores) // 10)
        recent_scores = aggregate_scores[-recent_count:]
        recent_avg = sum(recent_scores) / len(recent_scores)
        
        # Calculate trend (positive = improving, negative = declining)
        trend = recent_avg - avg_score
        
        return {
            "category": category,
            "count": len(scores),
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "recent_avg": recent_avg,
            "trend": trend,
            "last_updated": scores[-1].timestamp if scores else 0
        }
    
    def analyze_confidence_correlation(
        self,
        reported_confidence: float,
        measured_factuality: float,
        correlation_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Analyze the correlation between reported confidence and measured factuality.
        This method supports integration with hallucination metrics.
        
        Args:
            reported_confidence: Model's self-reported confidence (0.0-1.0)
            measured_factuality: Measured factuality score (0.0-1.0)
            correlation_threshold: Threshold for correlation quality (default 0.7)
            
        Returns:
            Dictionary with confidence correlation analysis results
        """
        # Calculate correlation score (1.0 = perfect correlation)
        correlation_score = 1.0 - abs(reported_confidence - measured_factuality)
        
        # Determine correlation type
        if correlation_score >= correlation_threshold:
            # Good correlation
            if reported_confidence >= 0.8 and measured_factuality >= 0.8:
                correlation_type = ConfidenceCorrelation.CONFIDENTLY_RIGHT
            else:
                correlation_type = ConfidenceCorrelation.ACCURATE
        else:
            # Poor correlation
            if reported_confidence > measured_factuality:
                # Overconfident case
                if reported_confidence >= 0.8 and measured_factuality < 0.5:
                    correlation_type = ConfidenceCorrelation.CONFIDENTLY_WRONG
                else:
                    correlation_type = ConfidenceCorrelation.OVERCONFIDENT
            else:
                # Underconfident case
                correlation_type = ConfidenceCorrelation.UNDERCONFIDENT
        
        # Return formatted results compatible with hallucination metrics
        return {
            "correlation_type": correlation_type.value,
            "reported_confidence": reported_confidence,
            "measured_factuality": measured_factuality,
            "correlation_score": correlation_score,
            "is_reliable": correlation_score >= correlation_threshold
        }
    
    def evaluate_with_threshold(
        self, 
        data: Dict[str, Any], 
        threshold: Optional[float] = None
    ) -> Tuple[bool, ConfidenceScore]:
        """
        Score and evaluate against a confidence threshold.
        
        Args:
            data: Input data for scoring
            threshold: Optional threshold, defaults to config value
            
        Returns:
            Tuple of (passes_threshold, confidence_score)
        """
        # Get the confidence score
        score = self.score(data)
        
        # Use provided threshold or config default
        effective_threshold = threshold if threshold is not None else self.config.default_threshold
        
        # Evaluate against threshold
        passes = score.aggregate_score >= effective_threshold
        
        return (passes, score)