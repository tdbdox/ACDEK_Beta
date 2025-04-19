"""
Hallucination Metrics Collection System

This module provides comprehensive hallucination detection, scoring, and monitoring 
capabilities for ACDEK v3. It implements:

1. Vector-based factuality scoring between generated outputs and reference content
2. Semantic boundary detection for detecting contextual drift
3. Confidence correlation metrics between model self-estimation and validation results
4. Structured logging for visualization and benchmark integration

The system serves as the core metrics layer for all hallucination reduction claims
in ACDEK v3 and provides standardized formats for benchmarking and visualization.
"""

import logging
import time
import json
import os
import numpy as np
import re
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import uuid
import traceback
import threading
from collections import defaultdict

# Import ACDEK components for scoring
from storage.vector_store.embed_utils import (
    get_embedding, 
    cosine_similarity, 
    semantic_search, 
    batch_embed
)
from interfaces.constraint_interface import (
    ConstraintSeverity,
    ConstraintCategory,
    ConstraintValidationResult
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_SIMILARITY_THRESHOLD = 0.75
DEFAULT_DRIFT_THRESHOLD = 0.65
VECTOR_DIM = 384  # Default embedding dimension
CONFIDENCE_CORRELATION_THRESHOLD = 0.7
RESULTS_DIR = Path("./enrichment/hallucination_results")
METRICS_VERSION = "1.0.0"

# Create results directory
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class FactualityGrade(Enum):
    """Factuality grade for a generated response."""
    HIGH = "high"               # Highly factual (90%+ score)
    MODERATE = "moderate"       # Moderately factual (70-90% score)
    LOW = "low"                 # Low factuality (50-70% score)
    UNFACTUAL = "unfactual"     # Not factual (<50% score)


class DriftSeverity(Enum):
    """Severity level for semantic drift."""
    NONE = "none"               # No semantic drift detected
    MINOR = "minor"             # Minor semantic drift
    MODERATE = "moderate"       # Moderate semantic drift
    SEVERE = "severe"           # Severe semantic drift


class ConfidenceCorrelation(Enum):
    """Types of confidence correlation mismatches."""
    ACCURATE = "accurate"                  # Confidence aligns with factuality
    OVERCONFIDENT = "overconfident"        # High confidence but low factuality
    UNDERCONFIDENT = "underconfident"      # Low confidence but high factuality
    CONFIDENTLY_WRONG = "confidently_wrong"  # Very high confidence but very low factuality
    CONFIDENTLY_RIGHT = "confidently_right"  # Very high confidence and very high factuality


class HallucinationCategory(Enum):
    """Categories of hallucinations for detailed analysis."""
    FACTUAL_ERROR = "factual_error"         # Incorrect facts
    FACTUAL_FABRICATION = "fabrication"     # Made-up facts
    LOGICAL_INCONSISTENCY = "logical"       # Logical errors or contradictions
    TEMPORAL_ERROR = "temporal"             # Time-related errors
    NUMERICAL_ERROR = "numerical"           # Incorrect numbers or statistics
    ENTITY_ERROR = "entity"                 # Incorrect entities, people, places
    CONTEXT_DRIFT = "context_drift"         # Drifting from original context
    CONSTRAINT_VIOLATION = "constraint"     # Violates specific constraints
    CONFIDENCE_MISMATCH = "confidence"      # Confidence doesn't match factuality


@dataclass
class ReferenceSource:
    """Reference source information for factuality checking."""
    id: str
    content: str
    source_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding the embedding."""
        result = {
            "id": self.id,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "source_type": self.source_type,
            "metadata": self.metadata,
        }
        return result


@dataclass
class FactualityEvidence:
    """Evidence for factuality assessment."""
    source_id: str
    similarity_score: float
    text_segment: str
    evidence_type: str = "semantic_similarity"
    aligned_reference: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source_id": self.source_id,
            "similarity_score": self.similarity_score,
            "text_segment": self.text_segment,
            "evidence_type": self.evidence_type,
            "aligned_reference": self.aligned_reference
        }


@dataclass
class DriftDetectionResult:
    """Result of semantic drift detection."""
    severity: DriftSeverity
    boundary_position: Optional[int] = None
    drift_score: float = 0.0
    original_context_similarity: float = 1.0
    drifted_segments: List[Tuple[int, int, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "severity": self.severity.value,
            "boundary_position": self.boundary_position,
            "drift_score": self.drift_score,
            "original_context_similarity": self.original_context_similarity,
            "drifted_segments": [
                {"start": start, "end": end, "score": score}
                for start, end, score in self.drifted_segments
            ]
        }


@dataclass
class ConfidenceAnalysisResult:
    """Result of confidence correlation analysis."""
    correlation_type: ConfidenceCorrelation
    reported_confidence: float
    measured_factuality: float
    correlation_score: float
    is_reliable: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "correlation_type": self.correlation_type.value,
            "reported_confidence": self.reported_confidence,
            "measured_factuality": self.measured_factuality,
            "correlation_score": self.correlation_score,
            "is_reliable": self.is_reliable
        }


@dataclass
class HallucinationIssue:
    """Detected hallucination with detailed information."""
    id: str
    category: HallucinationCategory
    text_span: Tuple[int, int]
    severity: float  # 0.0-1.0
    description: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "category": self.category.value,
            "text_span": self.text_span,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence
        }


@dataclass
class HallucinationMetricsResult:
    """Complete hallucination metrics for a generated response."""
    # Basic identification
    id: str
    timestamp: str
    prompt: str
    response: str
    model_info: Dict[str, Any]
    
    # Core metrics
    factuality_score: float
    factuality_grade: FactualityGrade
    semantic_drift: DriftDetectionResult
    confidence_analysis: ConfidenceAnalysisResult
    
    # Details
    detected_issues: List[HallucinationIssue] = field(default_factory=list)
    constraint_results: Dict[str, Any] = field(default_factory=dict)
    evidence_sources: List[ReferenceSource] = field(default_factory=list)
    segment_scores: Dict[str, float] = field(default_factory=dict)
    
    # Performance and metadata
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self, include_full_content: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Args:
            include_full_content: Whether to include full prompt and response
        """
        result = {
            "id": self.id,
            "timestamp": self.timestamp,
            "model_info": self.model_info,
            "factuality_score": self.factuality_score,
            "factuality_grade": self.factuality_grade.value,
            "semantic_drift": self.semantic_drift.to_dict(),
            "confidence_analysis": self.confidence_analysis.to_dict(),
            "detected_issues": [issue.to_dict() for issue in self.detected_issues],
            "constraint_results": self.constraint_results,
            "evidence_sources": [source.to_dict() for source in self.evidence_sources],
            "segment_scores": self.segment_scores,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata
        }
        
        if include_full_content:
            result["prompt"] = self.prompt
            result["response"] = self.response
        else:
            # Include truncated content
            result["prompt"] = self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt
            result["response"] = self.response[:100] + "..." if len(self.response) > 100 else self.response
        
        return result
    
    def save_to_file(self, file_path: Optional[str] = None) -> str:
        """
        Save results to a JSON file.
        
        Args:
            file_path: Optional file path, if None a default path is generated
            
        Returns:
            Path to the saved file
        """
        if not file_path:
            # Generate a meaningful filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hallucination_metrics_{self.id}_{timestamp}.json"
            file_path = os.path.join(RESULTS_DIR, filename)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            # Include full content when saving to file
            json.dump(self.to_dict(include_full_content=True), f, indent=2)
            
        return file_path


class TextSegmenter:
    """Utility to segment text for more granular analysis."""
    
    @staticmethod
    def segment_by_sentences(text: str, max_segments: int = 10) -> List[str]:
        """
        Segment text into sentences.
        
        Args:
            text: Text to segment
            max_segments: Maximum number of segments to return
            
        Returns:
            List of sentence segments
        """
        if not text:
            return []
        
        # Simple regex-based sentence splitting
        # In a production system this would use a more robust NLP library
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Group sentences into reasonable segments
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            # If adding this sentence would make the segment too long, start a new one
            if current_segment and len(current_segment) + len(sentence) > 300:
                segments.append(current_segment)
                current_segment = sentence
            else:
                if current_segment:
                    current_segment += " " + sentence
                else:
                    current_segment = sentence
        
        # Add the last segment if not empty
        if current_segment:
            segments.append(current_segment)
        
        # Limit to max_segments by combining smaller segments
        if segments and len(segments) > max_segments:
            # Calculate how many segments to combine
            group_size = (len(segments) + max_segments - 1) // max_segments
            
            grouped_segments = []
            for i in range(0, len(segments), group_size):
                group = " ".join(segments[i:i+group_size])
                grouped_segments.append(group)
            
            return grouped_segments[:max_segments]
        
        return segments
    
    @staticmethod
    def segment_by_paragraphs(text: str, max_segments: int = 5) -> List[str]:
        """
        Segment text into paragraphs.
        
        Args:
            text: Text to segment
            max_segments: Maximum number of segments to return
            
        Returns:
            List of paragraph segments
        """
        if not text:
            return []
        
        # Split by double newlines
        paragraphs = text.split("\n\n")
        
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Handle single paragraph with no newlines by falling back to sentence segmentation
        if len(paragraphs) <= 1 and len(text) > 500:
            return TextSegmenter.segment_by_sentences(text, max_segments)
        
        # Limit to max_segments by combining smaller paragraphs
        if paragraphs and len(paragraphs) > max_segments:
            # Calculate how many paragraphs to combine
            group_size = (len(paragraphs) + max_segments - 1) // max_segments
            
            grouped_paragraphs = []
            for i in range(0, len(paragraphs), group_size):
                group = "\n\n".join(paragraphs[i:i+group_size])
                grouped_paragraphs.append(group)
            
            return grouped_paragraphs[:max_segments]
        
        return paragraphs
    
    @staticmethod
    def get_text_segments(
        text: str, 
        segmentation_type: str = "auto",
        max_segments: int = 10
    ) -> List[str]:
        """
        Get text segments using the appropriate segmentation method.
        
        Args:
            text: Text to segment
            segmentation_type: Type of segmentation to use ("sentences", "paragraphs", or "auto")
            max_segments: Maximum number of segments to return
            
        Returns:
            List of text segments
        """
        if not text:
            return []
        
        if segmentation_type == "sentences":
            return TextSegmenter.segment_by_sentences(text, max_segments)
        elif segmentation_type == "paragraphs":
            return TextSegmenter.segment_by_paragraphs(text, max_segments)
        else:  # "auto"
            # Choose segmentation method based on text structure
            if "\n\n" in text and text.count("\n\n") >= 2:
                # Text has multiple paragraphs
                return TextSegmenter.segment_by_paragraphs(text, max_segments)
            else:
                # Text is a single block, use sentence segmentation
                return TextSegmenter.segment_by_sentences(text, max_segments)


class FactualityScorer:
    """
    Vector-based factuality scoring system.
    
    Assesses the semantic similarity and factual alignment between generated
    responses and reference materials. Uses embedding comparisons to quantify
    how well the response content aligns with known facts.
    """
    
    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        segmentation_type: str = "auto",
        max_segments: int = 10,
        model_name: Optional[str] = None
    ):
        """
        Initialize the factuality scorer.
        
        Args:
            similarity_threshold: Threshold for considering content factual
            segmentation_type: Type of text segmentation ("sentences", "paragraphs", or "auto")
            max_segments: Maximum segments to divide text into for granular scoring
            model_name: Optional embedding model name
        """
        self.similarity_threshold = similarity_threshold
        self.segmentation_type = segmentation_type
        self.max_segments = max_segments
        self.model_name = model_name
        self.stats = {
            "total_processed": 0,
            "high_factuality": 0,
            "moderate_factuality": 0,
            "low_factuality": 0,
            "unfactual": 0
        }
    
    def score_factuality(
        self,
        response: str,
        reference_sources: List[ReferenceSource],
        return_evidence: bool = True
    ) -> Tuple[float, Dict[str, float], Optional[List[FactualityEvidence]]]:
        """
        Score the factuality of a response against reference sources.
        
        Args:
            response: Generated response to score
            reference_sources: List of reference sources to check against
            return_evidence: Whether to return detailed evidence
            
        Returns:
            Tuple of (overall_score, segment_scores, evidence)
        """
        if not response or not reference_sources:
            return 0.0, {}, [] if return_evidence else None
        
        start_time = time.time()
        
        # Segment the response for more granular analysis
        segments = TextSegmenter.get_text_segments(
            response, 
            segmentation_type=self.segmentation_type,
            max_segments=self.max_segments
        )
        
        if not segments:
            segments = [response]
        
        # Get embeddings for all segments
        segment_embeddings = {}
        for i, segment in enumerate(segments):
            try:
                embedding = get_embedding(segment, normalize=True, model_name=self.model_name)
                segment_embeddings[f"segment_{i}"] = (segment, embedding)
            except Exception as e:
                logger.error(f"Error embedding segment {i}: {str(e)}")
                # Skip this segment
                continue
        
        # Prepare reference embeddings if not already present
        reference_embeddings = {}
        for source in reference_sources:
            if source.embedding is None:
                try:
                    source.embedding = get_embedding(
                        source.content, normalize=True, model_name=self.model_name
                    )
                except Exception as e:
                    logger.error(f"Error embedding reference source {source.id}: {str(e)}")
                    # Skip this reference
                    continue
            
            reference_embeddings[source.id] = source.embedding
        
        # Calculate segment scores and collect evidence
        segment_scores = {}
        evidence_list = [] if return_evidence else None
        
        for segment_id, (segment_text, segment_embedding) in segment_embeddings.items():
            best_score = 0.0
            best_source_id = None
            
            # Compare with each reference source
            for source_id, reference_embedding in reference_embeddings.items():
                similarity = cosine_similarity(segment_embedding, reference_embedding)
                
                if similarity > best_score:
                    best_score = similarity
                    best_source_id = source_id
            
            # Store segment score
            segment_scores[segment_id] = best_score
            
            # Add evidence if requested
            if return_evidence and best_source_id:
                # Find the matching reference source
                source = next((s for s in reference_sources if s.id == best_source_id), None)
                
                if source:
                    # Find most similar segment in the reference content
                    ref_segments = TextSegmenter.get_text_segments(
                        source.content, segmentation_type="sentences", max_segments=20
                    )
                    
                    best_ref_segment = ""
                    best_ref_score = 0.0
                    
                    # Get embedding for segment
                    for ref_segment in ref_segments:
                        try:
                            ref_segment_embedding = get_embedding(
                                ref_segment, normalize=True, model_name=self.model_name
                            )
                            ref_similarity = cosine_similarity(segment_embedding, ref_segment_embedding)
                            
                            if ref_similarity > best_ref_score:
                                best_ref_score = ref_similarity
                                best_ref_segment = ref_segment
                        except Exception:
                            continue
                    
                    evidence_list.append(FactualityEvidence(
                        source_id=best_source_id,
                        similarity_score=best_score,
                        text_segment=segment_text,
                        evidence_type="semantic_similarity",
                        aligned_reference=best_ref_segment if best_ref_score > 0.7 else None
                    ))
        
        # Calculate overall score as weighted average of segment scores
        # Weight by segment length to account for longer important sections
        if segment_scores:
            total_length = sum(len(segment_embeddings[seg_id][0]) for seg_id in segment_scores)
            overall_score = sum(
                score * len(segment_embeddings[seg_id][0]) / total_length
                for seg_id, score in segment_scores.items()
            )
        else:
            overall_score = 0.0
        
        # Update stats
        self.stats["total_processed"] += 1
        
        if overall_score >= 0.9:
            self.stats["high_factuality"] += 1
        elif overall_score >= 0.7:
            self.stats["moderate_factuality"] += 1
        elif overall_score >= 0.5:
            self.stats["low_factuality"] += 1
        else:
            self.stats["unfactual"] += 1
        
        processing_time = time.time() - start_time
        logger.debug(f"Factuality scoring completed in {processing_time:.2f}s with score {overall_score:.2f}")
        
        return overall_score, segment_scores, evidence_list
    
    def get_factuality_grade(self, score: float) -> FactualityGrade:
        """
        Get the factuality grade for a score.
        
        Args:
            score: Factuality score (0.0-1.0)
            
        Returns:
            FactualityGrade enum value
        """
        if score >= 0.9:
            return FactualityGrade.HIGH
        elif score >= 0.7:
            return FactualityGrade.MODERATE
        elif score >= 0.5:
            return FactualityGrade.LOW
        else:
            return FactualityGrade.UNFACTUAL
    
    def detect_hallucination_issues(
        self,
        response: str,
        segment_scores: Dict[str, float],
        segment_map: Dict[str, Tuple[str, np.ndarray]]
    ) -> List[HallucinationIssue]:
        """
        Detect specific hallucination issues in the response.
        
        Args:
            response: Generated response
            segment_scores: Scores for each segment
            segment_map: Mapping of segment IDs to (text, embedding) tuples
            
        Returns:
            List of detected hallucination issues
        """
        issues = []
        
        # Find low-scoring segments that indicate hallucinations
        for segment_id, score in segment_scores.items():
            if score < self.similarity_threshold:
                # This segment doesn't align well with references
                segment_text = segment_map[segment_id][0]
                
                # Estimate the position of this segment in the full response
                # This is an approximation - a proper implementation would track exact spans
                start_pos = response.find(segment_text)
                if start_pos >= 0:
                    end_pos = start_pos + len(segment_text)
                else:
                    # If exact match not found, use approximation
                    words = segment_text.split()[:3]
                    segment_start = ' '.join(words)
                    start_pos = response.find(segment_start)
                    if start_pos >= 0:
                        end_pos = min(start_pos + len(segment_text), len(response))
                    else:
                        # Default to beginning if we can't locate
                        start_pos = 0
                        end_pos = min(len(segment_text), len(response))
                
                # Determine hallucination category based on content patterns
                category = HallucinationCategory.FACTUAL_ERROR  # Default category
                
                # Check for specific patterns that indicate different types
                if re.search(r'\b(in \d{4}|years? ago|century)\b', segment_text):
                    category = HallucinationCategory.TEMPORAL_ERROR
                elif re.search(r'\b\d+%|\d+\.\d+|\$\d+\b', segment_text):
                    category = HallucinationType.NUMERICAL_ERROR
                elif re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', segment_text):
                    category = HallucinationType.ENTITY_ERROR
                elif "but" in segment_text or "however" in segment_text or "although" in segment_text:
                    category = HallucinationType.LOGICAL_INCONSISTENCY
                
                # Create the issue
                issue = HallucinationIssue(
                    id=f"issue_{len(issues)}_{uuid.uuid4().hex[:8]}",
                    category=category,
                    text_span=(start_pos, end_pos),
                    severity=1.0 - score,  # Invert score for severity
                    description=f"Low factuality content ({score:.2f}): {segment_text[:100]}...",
                    evidence=[{
                        "type": "low_reference_similarity",
                        "score": score,
                        "threshold": self.similarity_threshold
                    }]
                )
                
                issues.append(issue)
        
        return issues


class DriftDetector:
    """
    Semantic boundary detection system.
    
    Detects when generated content drifts from the expected domain or context
    based on semantic similarity measurements and boundary identification.
    """
    
    def __init__(
        self,
        drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
        window_size: int = 3,
        model_name: Optional[str] = None
    ):
        """
        Initialize the drift detector.
        
        Args:
            drift_threshold: Threshold for considering content drifted
            window_size: Size of the sliding window for drift detection
            model_name: Optional embedding model name
        """
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.model_name = model_name
        self.stats = {
            "total_processed": 0,
            "drifts_detected": 0,
            "drift_severity": {
                "none": 0,
                "minor": 0,
                "moderate": 0,
                "severe": 0
            }
        }
    
    def detect_drift(
        self,
        prompt: str,
        response: str,
        domain_context: Optional[str] = None
    ) -> DriftDetectionResult:
        """
        Detect semantic drift in the response from the prompt context.
        
        Args:
            prompt: Original prompt
            response: Generated response
            domain_context: Optional additional domain context to check against
            
        Returns:
            DriftDetectionResult with detection details
        """
        if not prompt or not response:
            return DriftDetectionResult(
                severity=DriftSeverity.NONE,
                drift_score=0.0,
                original_context_similarity=1.0
            )
        
        start_time = time.time()
        
        # Get embeddings for prompt and domain context
        try:
            prompt_embedding = get_embedding(prompt, normalize=True, model_name=self.model_name)
            
            context_embedding = None
            if domain_context:
                context_embedding = get_embedding(
                    domain_context, normalize=True, model_name=self.model_name
                )
        except Exception as e:
            logger.error(f"Error embedding prompt or context: {str(e)}")
            return DriftDetectionResult(
                severity=DriftSeverity.NONE,
                drift_score=0.0,
                original_context_similarity=1.0
            )
        
        # Segment the response
        segments = TextSegmenter.get_text_segments(
            response, segmentation_type="sentences", max_segments=20
        )
        
        if not segments:
            return DriftDetectionResult(
                severity=DriftSeverity.NONE,
                drift_score=0.0,
                original_context_similarity=1.0
            )
        
        # Get embeddings for response segments
        segment_embeddings = []
        
        for segment in segments:
            try:
                embedding = get_embedding(segment, normalize=True, model_name=self.model_name)
                segment_embeddings.append(embedding)
            except Exception:
                # Create a zero vector as fallback
                segment_embeddings.append(np.zeros(VECTOR_DIM))
        
        # Calculate similarity to prompt for each segment
        similarities = []
        
        for embedding in segment_embeddings:
            similarity = cosine_similarity(prompt_embedding, embedding)
            similarities.append(similarity)
        
        # If domain context provided, also check similarity to it
        domain_similarities = []
        
        if context_embedding is not None:
            for embedding in segment_embeddings:
                similarity = cosine_similarity(context_embedding, embedding)
                domain_similarities.append(similarity)
            
            # Combine with prompt similarities (weighted average)
            # We give more weight to prompt similarity since it's more specific
            similarities = [
                0.7 * sim_p + 0.3 * sim_d
                for sim_p, sim_d in zip(similarities, domain_similarities)
            ]
        
        # Detect drift using sliding window
        # We look for a significant and sustained drop in similarity
        drift_position = None
        max_drift_score = 0.0
        drifted_segments = []
        
        # Use the sliding window to detect sustained drift
        if len(similarities) > self.window_size:
            for i in range(len(similarities) - self.window_size + 1):
                window = similarities[i:i+self.window_size]
                avg_similarity = sum(window) / len(window)
                
                # Check if this window shows drift
                if avg_similarity < self.drift_threshold:
                    drift_score = 1.0 - avg_similarity
                    
                    # If this is the strongest drift, update the position
                    if drift_score > max_drift_score:
                        max_drift_score = drift_score
                        drift_position = i
                        
                    # Add to drifted segments
                    # Calculate the character position approximately
                    start_char = sum(len(segments[j]) for j in range(i))
                    end_char = start_char + sum(len(segments[j]) for j in range(i, i+self.window_size))
                    
                    drifted_segments.append((
                        start_char,
                        end_char,
                        drift_score
                    ))
        
        # Determine drift severity
        if max_drift_score < 0.2:
            severity = DriftSeverity.NONE
        elif max_drift_score < 0.4:
            severity = DriftSeverity.MINOR
        elif max_drift_score < 0.6:
            severity = DriftSeverity.MODERATE
        else:
            severity = DriftSeverity.SEVERE
        
        # Calculate overall context similarity
        # This is the average similarity across all segments
        original_context_similarity = sum(similarities) / len(similarities) if similarities else 1.0
        
        # Update stats
        self.stats["total_processed"] += 1
        
        if severity != DriftSeverity.NONE:
            self.stats["drifts_detected"] += 1
            self.stats["drift_severity"][severity.value] += 1
        else:
            self.stats["drift_severity"]["none"] += 1
        
        processing_time = time.time() - start_time
        logger.debug(
            f"Drift detection completed in {processing_time:.2f}s with score {max_drift_score:.2f}"
        )
        
        # Create and return result
        return DriftDetectionResult(
            severity=severity,
            boundary_position=drift_position,
            drift_score=max_drift_score,
            original_context_similarity=original_context_similarity,
            drifted_segments=drifted_segments
        )


class ConfidenceCorrelator:
    """
    Confidence correlation metrics system.
    
    Evaluates how well the model's self-estimated confidence aligns with
    validation results from constraint checking and vector scoring. This enables
    accurate auditing of overconfident hallucinations or underconfident truths.
    """
    
    def __init__(
        self,
        correlation_threshold: float = CONFIDENCE_CORRELATION_THRESHOLD
    ):
        """
        Initialize the confidence correlator.
        
        Args:
            correlation_threshold: Threshold for considering confidence aligned with factuality
        """
        self.correlation_threshold = correlation_threshold
        self.stats = {
            "total_processed": 0,
            "correlation_types": {
                "accurate": 0,
                "overconfident": 0,
                "underconfident": 0,
                "confidently_wrong": 0,
                "confidently_right": 0
            }
        }
    
    def analyze_confidence(
        self,
        reported_confidence: float,
        measured_factuality: float,
        constraint_results: Optional[Dict[str, Any]] = None
    ) -> ConfidenceAnalysisResult:
        """
        Analyze the correlation between reported confidence and measured factuality.
        
        Args:
            reported_confidence: Model's self-reported confidence (0.0-1.0)
            measured_factuality: Measured factuality score (0.0-1.0)
            constraint_results: Optional constraint validation results
            
        Returns:
            ConfidenceAnalysisResult with correlation analysis
        """
        # Calculate correlation score
        # Perfect correlation = 1.0 (confidence matches factuality)
        # Worst correlation = 0.0 (completely misaligned)
        correlation_score = 1.0 - abs(reported_confidence - measured_factuality)
        
        # Adjust based on constraint results if available
        if constraint_results:
            # If constraints failed but confidence is high, reduce correlation
            constraints_passed = constraint_results.get("passed", True)
            if not constraints_passed:
                violation_count = constraint_results.get("violation_count", 0)
                if violation_count > 0:
                    # Scale penalty by the number of violations
                    penalty = min(0.5, 0.1 * violation_count)
                    correlation_score = max(0.0, correlation_score - penalty)
        
        # Determine correlation type
        if correlation_score >= self.correlation_threshold:
            # Good correlation
            if reported_confidence >= 0.8 and measured_factuality >= 0.8:
                correlation_type = ConfidenceCorrelation.CONFIDENTLY_RIGHT
            else:
                correlation_type = ConfidenceCorrelation.ACCURATE
        else:
            # Poor correlation
            if reported_confidence > measured_factuality:
                # Overconfident
                if reported_confidence >= 0.8 and measured_factuality < 0.5:
                    correlation_type = ConfidenceCorrelation.CONFIDENTLY_WRONG
                else:
                    correlation_type = ConfidenceCorrelation.OVERCONFIDENT
            else:
                # Underconfident
                correlation_type = ConfidenceCorrelation.UNDERCONFIDENT
        
        # Determine if the confidence is reliable
        is_reliable = correlation_score >= self.correlation_threshold
        
        # Update stats
        self.stats["total_processed"] += 1
        self.stats["correlation_types"][correlation_type.value] += 1
        
        # Create and return result
        return ConfidenceAnalysisResult(
            correlation_type=correlation_type,
            reported_confidence=reported_confidence,
            measured_factuality=measured_factuality,
            correlation_score=correlation_score,
            is_reliable=is_reliable
        )


class HallucinationMetricsCollector:
    """
    Comprehensive hallucination metrics collection system.
    
    This class orchestrates the factuality scoring, drift detection, and
    confidence correlation components to provide a complete hallucination
    metrics report for a generated response.
    """
    
    def __init__(
        self,
        factuality_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
        correlation_threshold: float = CONFIDENCE_CORRELATION_THRESHOLD,
        model_name: Optional[str] = None,
        store_results: bool = True,
        results_dir: Optional[str] = None
    ):
        """
        Initialize the hallucination metrics collector.
        
        Args:
            factuality_threshold: Threshold for factuality scoring
            drift_threshold: Threshold for drift detection
            correlation_threshold: Threshold for confidence correlation
            model_name: Optional embedding model name
            store_results: Whether to store results to disk
            results_dir: Optional directory to store results (default: RESULTS_DIR)
        """
        self.factuality_scorer = FactualityScorer(
            similarity_threshold=factuality_threshold,
            model_name=model_name
        )
        
        self.drift_detector = DriftDetector(
            drift_threshold=drift_threshold,
            model_name=model_name
        )
        
        self.confidence_correlator = ConfidenceCorrelator(
            correlation_threshold=correlation_threshold
        )
        
        self.store_results = store_results
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        
        # Store component parameters
        self.config = {
            "factuality_threshold": factuality_threshold,
            "drift_threshold": drift_threshold,
            "correlation_threshold": correlation_threshold,
            "model_name": model_name,
            "store_results": store_results,
            "metrics_version": METRICS_VERSION
        }
        
        # Create results directory
        if self.store_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Track metrics collection stats
        self.stats = {
            "total_processed": 0,
            "processing_time_ms": 0.0,
            "factuality_grades": {
                "high": 0,
                "moderate": 0, 
                "low": 0,
                "unfactual": 0
            },
            "drift_detections": 0,
            "confidence_mismatches": 0
        }
        
        # Thread lock for thread safety
        self._lock = threading.RLock()
    
    def collect_metrics(
        self,
        prompt: str,
        response: str,
        reference_sources: List[ReferenceSource],
        reported_confidence: float = 0.0,
        constraint_results: Optional[Dict[str, Any]] = None,
        domain_context: Optional[str] = None,
        model_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HallucinationMetricsResult:
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
            HallucinationMetricsResult with comprehensive metrics
        """
        start_time = time.time()
        result_id = str(uuid.uuid4())
        
        # Default model info if not provided
        if not model_info:
            model_info = {
                "name": "unknown",
                "version": "unknown",
                "type": "unknown"
            }
        
        # Default metadata if not provided
        meta = metadata or {}
        meta.update({
            "collection_time": datetime.now().isoformat(),
            "collector_version": METRICS_VERSION
        })
        
        # 1. Perform factuality scoring
        # First get embeddings for all segments of the response
        segments = TextSegmenter.get_text_segments(
            response, segmentation_type="auto", max_segments=10
        )
        
        segment_map = {}
        for i, segment in enumerate(segments):
            try:
                embedding = get_embedding(segment, normalize=True, model_name=self.factuality_scorer.model_name)
                segment_map[f"segment_{i}"] = (segment, embedding)
            except Exception as e:
                logger.error(f"Error embedding segment {i}: {str(e)}")
                # Skip this segment for now
                continue
        
        # Score factuality
        factuality_score, segment_scores, evidence = self.factuality_scorer.score_factuality(
            response, reference_sources, return_evidence=True
        )
        
        # 2. Detect semantic drift
        drift_result = self.drift_detector.detect_drift(
            prompt, response, domain_context
        )
        
        # 3. Analyze confidence correlation
        confidence_result = self.confidence_correlator.analyze_confidence(
            reported_confidence, factuality_score, constraint_results
        )
        
        # 4. Detect specific hallucination issues
        detected_issues = self.factuality_scorer.detect_hallucination_issues(
            response, segment_scores, segment_map
        )
        
        # Add any drift issues
        if drift_result.severity != DriftSeverity.NONE:
            # If we have specific drifted segments
            for start, end, score in drift_result.drifted_segments:
                drift_issue = HallucinationIssue(
                    id=f"drift_{len(detected_issues)}_{uuid.uuid4().hex[:8]}",
                    category=HallucinationCategory.CONTEXT_DRIFT,
                    text_span=(start, end),
                    severity=score,
                    description="Content drifted from original context",
                    evidence=[{
                        "type": "semantic_drift",
                        "drift_score": score,
                        "original_context_similarity": drift_result.original_context_similarity
                    }]
                )
                detected_issues.append(drift_issue)
        
        # Add confidence issues if substantial mismatch
        if not confidence_result.is_reliable:
            confidence_issue = HallucinationIssue(
                id=f"confidence_{len(detected_issues)}_{uuid.uuid4().hex[:8]}",
                category=HallucinationCategory.CONFIDENCE_MISMATCH,
                text_span=(0, min(200, len(response))),  # Mark beginning of response
                severity=1.0 - confidence_result.correlation_score,
                description=f"Confidence mismatch: reported {reported_confidence:.2f} vs. measured {factuality_score:.2f}",
                evidence=[{
                    "type": "confidence_mismatch",
                    "correlation_score": confidence_result.correlation_score,
                    "correlation_type": confidence_result.correlation_type.value
                }]
            )
            detected_issues.append(confidence_issue)
        
        # Add constraint violation issues
        if constraint_results and not constraint_results.get("passed", True):
            violations = constraint_results.get("violations", [])
            
            for i, violation in enumerate(violations):
                constraint_issue = HallucinationIssue(
                    id=f"constraint_{len(detected_issues)}_{uuid.uuid4().hex[:8]}",
                    category=HallucinationCategory.CONSTRAINT_VIOLATION,
                    text_span=(
                        violation.get("span_start", 0),
                        violation.get("span_end", min(200, len(response)))
                    ),
                    severity=violation.get("severity", 0.5),
                    description=f"Constraint violation: {violation.get('message', 'Unknown constraint')}",
                    evidence=[{
                        "type": "constraint_violation",
                        "constraint_id": violation.get("constraint_id", f"unknown_{i}"),
                        "severity": violation.get("severity", 0.5)
                    }]
                )
                detected_issues.append(constraint_issue)
        
        # Calculate total processing time
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Get factuality grade
        factuality_grade = self.factuality_scorer.get_factuality_grade(factuality_score)
        
        # Create the complete results object
        result = HallucinationMetricsResult(
            id=result_id,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            response=response,
            model_info=model_info,
            factuality_score=factuality_score,
            factuality_grade=factuality_grade,
            semantic_drift=drift_result,
            confidence_analysis=confidence_result,
            detected_issues=detected_issues,
            constraint_results=constraint_results or {},
            evidence_sources=reference_sources,
            segment_scores=segment_scores,
            processing_time_ms=processing_time_ms,
            metadata=meta
        )
        
        # Update statistics
        with self._lock:
            self.stats["total_processed"] += 1
            self.stats["processing_time_ms"] += processing_time_ms
            self.stats["factuality_grades"][factuality_grade.value] += 1
            
            if drift_result.severity != DriftSeverity.NONE:
                self.stats["drift_detections"] += 1
                
            if not confidence_result.is_reliable:
                self.stats["confidence_mismatches"] += 1
        
        # Store results if enabled
        if self.store_results:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"hallucination_metrics_{result_id}_{timestamp}.json"
                file_path = self.results_dir / filename
                
                result.save_to_file(str(file_path))
                logger.info(f"Saved hallucination metrics to {file_path}")
            except Exception as e:
                logger.error(f"Error saving hallucination metrics: {str(e)}")
        
        return result
    
    def get_metrics_dashboard_data(self) -> Dict[str, Any]:
        """
        Get dashboard data from collected metrics.
        
        Returns:
            Dictionary with dashboard data
        """
        # Combine stats from all components
        dashboard_data = {
            "collector_stats": self.stats,
            "factuality_stats": self.factuality_scorer.stats,
            "drift_stats": self.drift_detector.stats,
            "confidence_stats": self.confidence_correlator.stats,
            "config": self.config,
            "timestamp": datetime.now().isoformat()
        }
        
        return dashboard_data
    
    def export_metrics_report(
        self,
        format: str = "json",
        file_path: Optional[str] = None
    ) -> str:
        """
        Export metrics report in the specified format.
        
        Args:
            format: Report format ("json", "csv", or "html")
            file_path: Optional file path, if None a default path is generated
            
        Returns:
            Path to the exported report
        """
        # Generate dashboard data
        dashboard_data = self.get_metrics_dashboard_data()
        
        # Generate default file path if not provided
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hallucination_metrics_report_{timestamp}.{format}"
            file_path = os.path.join(self.results_dir, filename)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Export in specified format
        if format == "json":
            with open(file_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
        elif format == "csv":
            # Flatten the data structure for CSV
            flat_data = {}
            
            # Helper function to flatten nested dict
            def flatten_dict(d, prefix=''):
                for k, v in d.items():
                    if isinstance(v, dict):
                        flatten_dict(v, f"{prefix}{k}.")
                    else:
                        flat_data[f"{prefix}{k}"] = v
            
            # Flatten the dashboard data
            flatten_dict(dashboard_data)
            
            # Write to CSV
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for k, v in sorted(flat_data.items()):
                    writer.writerow([k, v])
        elif format == "html":
            # Simple HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Hallucination Metrics Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .section {{ margin-bottom: 20px; padding: 10px; border: 1px solid #eee; border-radius: 5px; }}
                    .metric {{ margin: 5px 0; }}
                    .label {{ font-weight: bold; }}
                    .value {{ color: #666; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <h1>ACDEK v3 Hallucination Metrics Report</h1>
                <p>Generated: {dashboard_data['timestamp']}</p>
                
                <div class="section">
                    <h2>Summary</h2>
                    <div class="metric">
                        <span class="label">Total Processed:</span>
                        <span class="value">{dashboard_data['collector_stats']['total_processed']}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Avg Processing Time:</span>
                        <span class="value">{dashboard_data['collector_stats']['processing_time_ms'] / max(1, dashboard_data['collector_stats']['total_processed']):.2f} ms</span>
                    </div>
                    <div class="metric">
                        <span class="label">High Factuality:</span>
                        <span class="value">{dashboard_data['collector_stats']['factuality_grades']['high']}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Moderate Factuality:</span>
                        <span class="value">{dashboard_data['collector_stats']['factuality_grades']['moderate']}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Low Factuality:</span>
                        <span class="value">{dashboard_data['collector_stats']['factuality_grades']['low']}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Unfactual:</span>
                        <span class="value">{dashboard_data['collector_stats']['factuality_grades']['unfactual']}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Drift Detections:</span>
                        <span class="value">{dashboard_data['collector_stats']['drift_detections']}</span>
                    </div>
                    <div class="metric">
                        <span class="label">Confidence Mismatches:</span>
                        <span class="value">{dashboard_data['collector_stats']['confidence_mismatches']}</span>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Factuality Statistics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Processed</td>
                            <td>{dashboard_data['factuality_stats']['total_processed']}</td>
                        </tr>
                        <tr>
                            <td>High Factuality</td>
                            <td>{dashboard_data['factuality_stats']['high_factuality']}</td>
                        </tr>
                        <tr>
                            <td>Moderate Factuality</td>
                            <td>{dashboard_data['factuality_stats']['moderate_factuality']}</td>
                        </tr>
                        <tr>
                            <td>Low Factuality</td>
                            <td>{dashboard_data['factuality_stats']['low_factuality']}</td>
                        </tr>
                        <tr>
                            <td>Unfactual</td>
                            <td>{dashboard_data['factuality_stats']['unfactual']}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Drift Statistics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Processed</td>
                            <td>{dashboard_data['drift_stats']['total_processed']}</td>
                        </tr>
                        <tr>
                            <td>Drifts Detected</td>
                            <td>{dashboard_data['drift_stats']['drifts_detected']}</td>
                        </tr>
                        <tr>
                            <td>None Severity</td>
                            <td>{dashboard_data['drift_stats']['drift_severity']['none']}</td>
                        </tr>
                        <tr>
                            <td>Minor Severity</td>
                            <td>{dashboard_data['drift_stats']['drift_severity']['minor']}</td>
                        </tr>
                        <tr>
                            <td>Moderate Severity</td>
                            <td>{dashboard_data['drift_stats']['drift_severity']['moderate']}</td>
                        </tr>
                        <tr>
                            <td>Severe Severity</td>
                            <td>{dashboard_data['drift_stats']['drift_severity']['severe']}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Confidence Statistics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Processed</td>
                            <td>{dashboard_data['confidence_stats']['total_processed']}</td>
                        </tr>
                        <tr>
                            <td>Accurate Correlation</td>
                            <td>{dashboard_data['confidence_stats']['correlation_types']['accurate']}</td>
                        </tr>
                        <tr>
                            <td>Overconfident</td>
                            <td>{dashboard_data['confidence_stats']['correlation_types']['overconfident']}</td>
                        </tr>
                        <tr>
                            <td>Underconfident</td>
                            <td>{dashboard_data['confidence_stats']['correlation_types']['underconfident']}</td>
                        </tr>
                        <tr>
                            <td>Confidently Wrong</td>
                            <td>{dashboard_data['confidence_stats']['correlation_types']['confidently_wrong']}</td>
                        </tr>
                        <tr>
                            <td>Confidently Right</td>
                            <td>{dashboard_data['confidence_stats']['correlation_types']['confidently_right']}</td>
                        </tr>
                    </table>
                </div>
                
                <div class="section">
                    <h2>Configuration</h2>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
            """
            
            # Add config parameters
            for key, value in dashboard_data['config'].items():
                html_content += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            </body>
            </html>
            """
            
            with open(file_path, 'w') as f:
                f.write(html_content)
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        return file_path


# Create module-level instances for easy import
DEFAULT_METRICS_COLLECTOR = HallucinationMetricsCollector()

# Helper function for easy metrics collection
def collect_hallucination_metrics(
    prompt: str,
    response: str,
    reference_content: Union[str, List[str], List[Dict[str, Any]]],
    reported_confidence: float = 0.0,
    constraint_results: Optional[Dict[str, Any]] = None,
    model_info: Optional[Dict[str, Any]] = None
) -> HallucinationMetricsResult:
    """
    Collect hallucination metrics for a response using the default collector.
    
    Args:
        prompt: Original prompt
        response: Generated response
        reference_content: Reference content for factuality checking (string, list of strings, or list of dicts)
        reported_confidence: Model's self-reported confidence (0.0-1.0)
        constraint_results: Optional constraint validation results
        model_info: Optional model information
        
    Returns:
        HallucinationMetricsResult with metrics
    """
    # Convert reference content to ReferenceSource objects
    reference_sources = []
    
    if isinstance(reference_content, str):
        # Single string
        reference_sources.append(ReferenceSource(
            id="ref_1",
            content=reference_content,
            source_type="text"
        ))
    elif isinstance(reference_content, list):
        # List of strings or dicts
        for i, ref in enumerate(reference_content):
            if isinstance(ref, str):
                reference_sources.append(ReferenceSource(
                    id=f"ref_{i+1}",
                    content=ref,
                    source_type="text"
                ))
            elif isinstance(ref, dict):
                reference_sources.append(ReferenceSource(
                    id=ref.get("id", f"ref_{i+1}"),
                    content=ref.get("content", ""),
                    source_type=ref.get("source_type", "text"),
                    metadata=ref.get("metadata", {})
                ))
    
    # Use the default collector to get metrics
    return DEFAULT_METRICS_COLLECTOR.collect_metrics(
        prompt=prompt,
        response=response,
        reference_sources=reference_sources,
        reported_confidence=reported_confidence,
        constraint_results=constraint_results,
        model_info=model_info
    )