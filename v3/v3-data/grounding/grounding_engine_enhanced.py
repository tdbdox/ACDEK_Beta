#!/usr/bin/env python3
"""
Enhanced Grounding Engine for ACDEK v3 (Phase 3)

This module provides improved hallucination detection through:
- Enhanced semantic chunking
- Improved contradiction detection
- Adaptive thresholds
- Better reference data handling
- Performance optimizations
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import sys
import importlib.util

# Configure logging
logger = logging.getLogger(__name__)

# Add parent path to enable imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

try:
    from storage.vector_store.embed_utils import get_embedding, cosine_similarity, semantic_search
except ImportError:
    from v3.storage.vector_store.embed_utils import get_embedding, cosine_similarity, semantic_search

# Enhanced domain-specific thresholds (adjusted based on Phase 3 evaluation results)
FACTUALITY_THRESHOLD: float = 0.65  # Adjusted global threshold for better precision/recall balance

# Updated domain-specific thresholds (reduced to minimize false positives)
DOMAIN_THRESHOLDS = {
    'medical': 0.68,   # Adjusted from 0.72 to 0.68 to reduce false positives
    'legal': 0.65,     # Adjusted from 0.70 to 0.65 for better balance
    'financial': 0.60  # Adjusted from 0.65 to 0.60 to reduce false positives
}

# Domain aliases for mapping domain values to reference filenames
DOMAIN_ALIAS = {
    'financial': 'finance',
    'medical': 'medical',
    'legal': 'legal'
}

# Cache for reference data to avoid repeated file loading
_reference_cache: Dict[str, List[Dict[str, Any]]] = {}

# Import enhanced modules with graceful fallbacks
try:
    from semantic_chunking_enhanced import (
        SemanticChunkerEnhanced,
        chunk_reference_data_enhanced,
        retrieve_reference_chunks_enhanced,
        score_factuality_with_chunks_enhanced
    )
    ENHANCED_CHUNKING_AVAILABLE = True
    logger.info("Enhanced semantic chunking module loaded successfully")
except ImportError:
    logger.warning("Enhanced semantic chunking not available, trying standard version")
    try:
        from semantic_chunking import (
            SemanticChunker,
            chunk_reference_data,
            retrieve_reference_chunks,
            score_factuality_with_chunks
        )
        ENHANCED_CHUNKING_AVAILABLE = False
        logger.info("Standard semantic chunking module loaded successfully")
    except ImportError:
        logger.warning("Semantic chunking not available")
        SemanticChunker = None
        chunk_reference_data = None
        retrieve_reference_chunks = None
        score_factuality_with_chunks = None
        ENHANCED_CHUNKING_AVAILABLE = False
        SEMANTIC_CHUNKING_AVAILABLE = False

# Import enhanced contradiction detection
try:
    from contradiction_detection_enhanced import (
        detect_contradictions_enhanced
    )
    ENHANCED_CONTRADICTION_AVAILABLE = True
    logger.info("Enhanced contradiction detection module loaded successfully")
except ImportError:
    logger.warning("Enhanced contradiction detection not available, trying standard version")
    try:
        from contradiction_detection import detect_contradictions
        ENHANCED_CONTRADICTION_AVAILABLE = False
        logger.info("Standard contradiction detection module loaded successfully")
    except ImportError:
        logger.warning("Contradiction detection not available")
        detect_contradictions = None
        ENHANCED_CONTRADICTION_AVAILABLE = False
        CONTRADICTION_DETECTION_AVAILABLE = False

def _load_references(domain: str, enhanced: bool = True) -> List[Dict[str, Any]]:
    """
    Load reference entries for the given domain from JSON data files with caching.
    
    Args:
        domain: Domain name to load references for
        enhanced: Whether to look for enhanced reference data
        
    Returns:
        List of reference entries
        
    Raises:
        FileNotFoundError: If no reference data is found for the domain
    """
    domain_key = DOMAIN_ALIAS.get(domain, domain)
    cache_key = f"{domain_key}_{'enhanced' if enhanced else 'standard'}"
    
    # Check cache first
    if cache_key in _reference_cache:
        return _reference_cache[cache_key]
    
    # First try to load enhanced reference data if requested
    if enhanced:
        enhanced_data_dir = Path(os.path.dirname(__file__)).parent / 'data' / f'enhanced_{domain_key}'
        enhanced_file_path = enhanced_data_dir / f'{domain_key}_enhanced.json'
        
        try:
            if enhanced_file_path.exists():
                with open(enhanced_file_path, 'r') as f:
                    enhanced_data = json.load(f)
                
                # Convert enhanced format to standard format if needed
                if "concepts" in enhanced_data:
                    references = []
                    for concept in enhanced_data["concepts"]:
                        # Create a standard reference entry
                        entry = {
                            "id": concept.get("id", "unknown"),
                            "text": concept.get("definition", ""),
                            "source": concept.get("source", ""),
                            "term": concept.get("term", ""),
                            "examples": concept.get("examples", []),
                            "related_concepts": concept.get("related_concepts", [])
                        }
                        references.append(entry)
                    
                    logger.info(f"Loaded {len(references)} enhanced reference entries for domain '{domain}'")
                    _reference_cache[cache_key] = references
                    return references
        except Exception as e:
            logger.warning(f"Error loading enhanced reference data for domain '{domain}': {str(e)}")
    
    # Fall back to standard reference data
    # Reference data directory path using pathlib for improved path handling
    data_dir = Path(os.path.dirname(__file__)).parent / 'data'
    file_path = data_dir / f"{domain_key}.json"
    
    if not file_path.exists():
        available_files = [f.name for f in data_dir.glob('*.json') if f.is_file()]
        raise FileNotFoundError(
            f"No reference data found for domain '{domain}' (expected at '{file_path}'). "
            f"Available domains: {available_files}"
        )
    
    try:
        with open(file_path, 'r') as f:
            references = json.load(f)
            
        # Validate reference format
        if not isinstance(references, list):
            raise ValueError(f"Reference data for domain '{domain}' is not a list")
            
        # Store in cache for future calls
        _reference_cache[cache_key] = references
        logger.info(f"Loaded {len(references)} standard reference entries for domain '{domain}'")
        return references
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing reference data for domain '{domain}': {str(e)}")
        raise ValueError(f"Invalid JSON in reference data for domain '{domain}'") from e

def retrieve_reference_enhanced(prompt: str, domain: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Enhanced vector-based retrieval from the appropriate reference dataset.
    
    Args:
        prompt: The query prompt to match against references
        domain: Domain name to retrieve references from
        top_k: Number of top matches to consider
        
    Returns:
        Dictionary containing best matching reference with id, text, and similarity
        
    Raises:
        ValueError: If no references are available for the domain or 
                  if no reference similarity exceeds minimum threshold
    """
    # Try to load enhanced reference data first
    try:
        references = _load_references(domain, enhanced=True)
    except FileNotFoundError:
        # Fall back to standard reference data
        references = _load_references(domain, enhanced=False)
    
    if not references:
        raise ValueError(f"No references available for domain '{domain}'")
    
    # Extract reference texts for semantic search
    reference_texts = []
    for entry in references:
        # For enhanced format, include term in the text for better matching
        if "term" in entry:
            reference_texts.append(f"{entry.get('term', '')}: {entry.get('text', '')}")
        else:
            reference_texts.append(entry.get('text', ''))
    
    # Perform semantic search to find best matches
    search_results = semantic_search(
        query=prompt,
        candidates=reference_texts,
        threshold=0.4,  # Lower threshold to ensure we get results
        max_results=top_k
    )
    
    # No matches found
    if not search_results:
        logger.warning(f"No matching references found for domain '{domain}'. Using first reference as fallback.")
        return {
            'id': references[0].get('id', 'unknown'),
            'text': references[0].get('text', ''),
            'similarity': 0.0,
            'is_fallback': True
        }
    
    # Get the best match (highest similarity)
    best_match = search_results[0]
    best_index, _, best_score, _ = best_match
    
    # Get all matches for potentially providing more context
    all_matches = [
        {
            'id': references[idx].get('id', 'unknown'),
            'text': references[idx].get('text', ''),
            'similarity': score,
            'term': references[idx].get('term', '') if 'term' in references[idx] else None
        }
        for idx, _, score, _ in search_results[:min(top_k, len(search_results))]
    ]
    
    # Get the full entry from the reference data
    best_entry = references[best_index].copy()
    best_entry_result = {
        'id': best_entry.get('id', 'unknown'),
        'text': best_entry.get('text', ''),
        'similarity': best_score,
        'is_fallback': False,
        'all_matches': all_matches
    }
    
    # Add enhanced fields if available
    for field in ['term', 'examples', 'related_concepts']:
        if field in best_entry:
            best_entry_result[field] = best_entry[field]
    
    logger.debug(f"Retrieved reference for domain '{domain}' with similarity {best_score:.4f}")
    return best_entry_result

def contains_contradiction(text: str, reference: str) -> Tuple[bool, float, str]:
    """
    Basic contradiction detection (fallback if enhanced module not available).
    
    Args:
        text: The response text to check
        reference: The reference text to compare against
        
    Returns:
        Tuple of (contradiction_found, confidence_score, reason)
    """
    # Convert to lowercase for comparison
    text_lower = text.lower()
    ref_lower = reference.lower()
    
    # Strong contradiction patterns with high confidence
    domain_specific_contradictions = [
        # Financial domain contradictions
        {
            "pattern": ("asset" in ref_lower and "liability" in text_lower and "type of" in text_lower),
            "confidence": 0.9,
            "reason": "Response incorrectly states an asset is a type of liability"
        },
        {
            "pattern": ("equity" in ref_lower and "nothing to do with" in text_lower),
            "confidence": 0.9,
            "reason": "Response incorrectly states equity has nothing to do with referenced concept"
        },
        {
            "pattern": ("income statement" in ref_lower and "only tracks cash flow" in text_lower),
            "confidence": 0.85,
            "reason": "Response incorrectly states income statement only tracks cash flow"
        },
        
        # Medical domain contradictions
        {
            "pattern": ("diabetes" in ref_lower and "cured" in text_lower and "completely" in text_lower),
            "confidence": 0.95,
            "reason": "Response incorrectly claims diabetes can be completely cured"
        },
        
        # Legal domain contradictions
        {
            "pattern": ("contract" in ref_lower and "never requires" in text_lower),
            "confidence": 0.85,
            "reason": "Response incorrectly claims contracts never require specific elements"
        }
    ]
    
    # Check domain-specific strong contradictions first
    for pattern in domain_specific_contradictions:
        if pattern["pattern"]:
            return True, pattern["confidence"], pattern["reason"]
    
    # List of strong contradiction marker pairs that require more careful checking
    strong_contradiction_markers = [
        # Format: (marker1, marker2, context_words, confidence)
        ("never", "always", ["requires", "needed", "necessary"], 0.8),
        ("none", "all", ["cases", "instances", "situations"], 0.8),
        ("impossible", "guaranteed", ["results", "outcomes"], 0.85),
        ("no evidence", "proven", ["scientific", "medical", "studies"], 0.85)
    ]
    
    # Check for strong contradictions with context requirements
    for marker1, marker2, context_words, confidence in strong_contradiction_markers:
        if marker1 in text_lower and marker2 in ref_lower:
            # Check if context words are present near the markers
            for context in context_words:
                if context in text_lower and context in ref_lower:
                    return True, confidence, f"Strong contradiction: '{marker1}' vs '{marker2}' in '{context}' context"
                    
        if marker2 in text_lower and marker1 in ref_lower:
            # Check the reverse case
            for context in context_words:
                if context in text_lower and context in ref_lower:
                    return True, confidence, f"Strong contradiction: '{marker2}' vs '{marker1}' in '{context}' context"
    
    # List of common words that should not trigger contradiction detection when used in different contexts
    common_words_to_ignore = ["is", "are", "not", "be", "can", "may", "will", "should"]
    
    # Weaker contradiction patterns - now ignored unless in same sentence with same subject
    weak_contradiction_markers = [
        ("not", "is", "are"),
        ("never", "always"),
        ("none", "all"),
        ("zero", "all"),
        ("impossible", "possible"),
        ("guaranteed", "may"),
        ("100%", "potential")
    ]
    
    # Improved context-aware contradiction checking
    for marker_pair in weak_contradiction_markers:
        for marker1 in marker_pair:
            # Skip if marker is in the ignore list (too common, causes false positives)
            if marker1 in common_words_to_ignore:
                continue
                
            if marker1 in ref_lower:
                for marker2 in marker_pair:
                    # Skip common words that cause false positives
                    if marker2 in common_words_to_ignore:
                        continue
                        
                    if marker2 != marker1 and marker2 in text_lower:
                        # Found potential weak contradiction - check if markers are in very similar contexts
                        ref_sentences = ref_lower.split(".")
                        text_sentences = text_lower.split(".")
                        
                        for r_sent in ref_sentences:
                            if marker1 in r_sent:
                                r_keywords = [w for w in r_sent.split() if len(w) > 5]
                                
                                for t_sent in text_sentences:
                                    if marker2 in t_sent:
                                        # Check if they share significant keywords (suggesting same topic)
                                        shared_keywords = [w for w in r_keywords if w in t_sent.split()]
                                        
                                        if len(shared_keywords) >= 2:
                                            return True, 0.7, f"Context-specific contradiction: '{marker1}' vs '{marker2}'"
    
    # If we got here, no contradictions were found
    return False, 0.0, "No contradiction detected"

def score_factuality_enhanced(response: str, reference: str, domain: str = None, use_enhanced: bool = True) -> Dict[str, Any]:
    """
    Enhanced factuality scoring with sophisticated processing.
    
    Args:
        response: Generated text to evaluate
        reference: Reference text to compare against
        domain: Domain for domain-specific processing
        use_enhanced: Whether to use enhanced modules if available
        
    Returns:
        Dictionary with factuality score and metadata
    """
    # Start timing
    start_time = time.time()
    
    # Handle edge cases
    if not response or not reference:
        logger.warning("Empty input for factuality scoring")
        return {
            "score": 0.0,
            "has_contradiction": False,
            "contradiction_confidence": 0.0,
            "contradiction_reason": "Empty input",
            "vector_similarity": 0.0,
            "processing_time_ms": 0
        }
    
    # Initialize result dictionary
    result = {
        "score": 0.0,
        "has_contradiction": False,
        "contradiction_confidence": 0.0,
        "contradiction_reason": "",
        "vector_similarity": 0.0
    }
    
    # Step 1: Check for contradictions using best available method
    if use_enhanced and ENHANCED_CONTRADICTION_AVAILABLE and domain:
        try:
            # Use enhanced contradiction detection
            contradiction_result = detect_contradictions_enhanced(response, reference, domain)
            has_contradiction = contradiction_result["has_contradiction"]
            contradiction_confidence = contradiction_result["contradiction_confidence"]
            contradiction_reason = contradiction_result["contradiction_reason"]
            
            result["has_contradiction"] = has_contradiction
            result["contradiction_confidence"] = contradiction_confidence
            result["contradiction_reason"] = contradiction_reason
            
            if has_contradiction:
                logger.info(f"Enhanced contradiction detection found contradiction: {contradiction_reason}")
        except Exception as e:
            logger.warning(f"Error in enhanced contradiction detection: {e}. Falling back to standard detection.")
            
            # Try standard detection if available
            if 'detect_contradictions' in globals() and detect_contradictions:
                try:
                    contradiction_result = detect_contradictions(response, reference, domain)
                    result["has_contradiction"] = contradiction_result["has_contradiction"]
                    result["contradiction_confidence"] = contradiction_result["contradiction_confidence"]
                    result["contradiction_reason"] = contradiction_result["contradiction_reason"]
                except Exception as e2:
                    logger.warning(f"Error in standard contradiction detection: {e2}. Falling back to basic detection.")
                    # Fallback to basic contradiction detection
                    has_contradiction, contradiction_confidence, contradiction_reason = contains_contradiction(response, reference)
                    result["has_contradiction"] = has_contradiction
                    result["contradiction_confidence"] = contradiction_confidence
                    result["contradiction_reason"] = contradiction_reason
            else:
                # Fallback to basic contradiction detection
                has_contradiction, contradiction_confidence, contradiction_reason = contains_contradiction(response, reference)
                result["has_contradiction"] = has_contradiction
                result["contradiction_confidence"] = contradiction_confidence
                result["contradiction_reason"] = contradiction_reason
    
    elif 'detect_contradictions' in globals() and detect_contradictions and domain:
        # Try standard contradiction detection
        try:
            contradiction_result = detect_contradictions(response, reference, domain)
            result["has_contradiction"] = contradiction_result["has_contradiction"]
            result["contradiction_confidence"] = contradiction_result["contradiction_confidence"]
            result["contradiction_reason"] = contradiction_result["contradiction_reason"]
        except Exception as e:
            logger.warning(f"Error in standard contradiction detection: {e}. Falling back to basic detection.")
            # Use basic contradiction detection
            has_contradiction, contradiction_confidence, contradiction_reason = contains_contradiction(response, reference)
            result["has_contradiction"] = has_contradiction
            result["contradiction_confidence"] = contradiction_confidence
            result["contradiction_reason"] = contradiction_reason
    else:
        # Use basic contradiction detection
        has_contradiction, contradiction_confidence, contradiction_reason = contains_contradiction(response, reference)
        result["has_contradiction"] = has_contradiction
        result["contradiction_confidence"] = contradiction_confidence
        result["contradiction_reason"] = contradiction_reason
    
    # Step 2: Calculate vector similarity (with best available chunking if enabled)
    try:
        if use_enhanced and ENHANCED_CHUNKING_AVAILABLE:
            # Use enhanced semantic chunking
            chunker = SemanticChunkerEnhanced()
            ref_chunks = chunker.chunk_text(reference, domain=domain)
            
            # Compute factuality with enhanced chunks
            chunk_result = score_factuality_with_chunks_enhanced(response, ref_chunks, domain=domain)
            
            # Extract scores
            vector_similarity = chunk_result.get("score", 0.0)
            result["vector_similarity"] = vector_similarity
            result["chunk_scores"] = chunk_result.get("chunk_scores", [])
            result["chunks_used"] = chunk_result.get("chunks_used", 0)
            
        elif 'SemanticChunker' in globals() and SemanticChunker:
            # Use standard semantic chunking
            chunker = SemanticChunker()
            ref_chunks = chunker.chunk_text(reference)
            
            # Compute factuality with standard chunks
            chunk_result = score_factuality_with_chunks(response, ref_chunks)
            
            # Extract scores
            vector_similarity = chunk_result.get("score", 0.0)
            result["vector_similarity"] = vector_similarity
            result["chunk_scores"] = chunk_result.get("chunk_scores", [])
            result["chunks_used"] = chunk_result.get("chunks_used", 0)
        
        else:
            # Standard vector similarity calculation
            resp_emb = get_embedding(response)
            ref_emb = get_embedding(reference)
            vector_similarity = cosine_similarity(resp_emb, ref_emb)
            result["vector_similarity"] = vector_similarity
        
        # Step 3: Calculate final score based on vector similarity and contradiction detection
        if result["has_contradiction"]:
            # Apply a proportional penalty based on contradiction confidence
            # More confident contradictions receive stronger penalties
            contradiction_penalty = result["contradiction_confidence"] * 0.6
            
            # Apply graduated penalty - more similar texts with contradictions still get some credit
            # This prevents the "cliff effect" where good responses with minor issues get severely punished
            final_score = max(0.3, vector_similarity * (1 - contradiction_penalty))
        else:
            final_score = vector_similarity
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, final_score))
        result["score"] = final_score
        
        # Add processing time
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        
        return result
        
    except Exception as e:
        logger.error(f"Error in factuality scoring: {str(e)}")
        result["error"] = str(e)
        # Add processing time even for errors
        result["processing_time_ms"] = (time.time() - start_time) * 1000
        return result

def build_grounding_context_enhanced(prompt: str, domain: str) -> Dict[str, Any]:
    """
    Build a comprehensive enhanced grounding context for the prompt and domain.
    
    Args:
        prompt: The query prompt to build context for
        domain: Domain name for the context
        
    Returns:
        Dictionary with grounding context including relevant references
    """
    try:
        # Retrieve top reference with enhanced retrieval
        ref_entry = retrieve_reference_enhanced(prompt, domain)
        
        # Prepare context with additional metadata if available
        context = {
            'domain': domain,
            'primary_reference': {
                'id': ref_entry['id'],
                'text': ref_entry['text'],
                'similarity': ref_entry['similarity']
            },
            'additional_references': ref_entry.get('all_matches', []),
            'timestamp': time.time(),
            'threshold': get_domain_threshold(domain)
        }
        
        # Add enhanced fields if available
        for field in ['term', 'examples', 'related_concepts']:
            if field in ref_entry:
                context['primary_reference'][field] = ref_entry[field]
        
        return context
        
    except Exception as e:
        logger.error(f"Error building grounding context: {str(e)}")
        return {
            'domain': domain,
            'error': str(e),
            'primary_reference': None,
            'additional_references': [],
            'threshold': get_domain_threshold(domain),
            'timestamp': time.time()
        }

def get_domain_threshold(domain: str) -> float:
    """
    Get the factuality threshold for a specific domain.
    
    Args:
        domain: Domain name
        
    Returns:
        Appropriate threshold value
    """
    return DOMAIN_THRESHOLDS.get(domain.lower(), FACTUALITY_THRESHOLD)

def adaptive_threshold(prompt: str, domain: str, response_length: int) -> float:
    """
    Compute an adaptive threshold based on prompt complexity and response length.
    
    Args:
        prompt: Query prompt
        domain: Domain
        response_length: Length of the response text
        
    Returns:
        Adjusted threshold
    """
    # Get base threshold for domain
    base_threshold = get_domain_threshold(domain)
    
    # Adjust based on prompt complexity
    prompt_words = len(prompt.split())
    if prompt_words < 5:  # Very simple prompt
        complexity_adjustment = -0.02  # Lower threshold (more permissive)
    elif prompt_words > 20:  # Complex prompt
        complexity_adjustment = 0.02  # Higher threshold (more strict)
    else:
        complexity_adjustment = 0.0
    
    # Adjust based on response length
    if response_length < 100:  # Very short response
        length_adjustment = -0.01  # Lower threshold (more permissive)
    elif response_length > 1000:  # Very long response
        length_adjustment = 0.02  # Higher threshold (more strict)
    else:
        length_adjustment = 0.0
    
    # Combine adjustments (with limits)
    adjusted_threshold = base_threshold + complexity_adjustment + length_adjustment
    
    # Ensure threshold is within reasonable bounds
    adjusted_threshold = max(0.55, min(0.85, adjusted_threshold))
    
    return adjusted_threshold

def grounded_validation_pipeline_enhanced(
    prompt: str, 
    response: str, 
    domain: str,
    include_debug: bool = False,
    use_enhanced: bool = True,
    use_adaptive_threshold: bool = True
) -> Dict[str, Any]:
    """
    Enhanced grounding and validation pipeline with sophisticated processing.
    
    Returns a dict with:
      - reference_text: Retrieved reference content
      - factuality_score: Embedding similarity score
      - constraint_pass: Boolean from constraint validation
      - hallucination_flag: True if response is hallucinated
      - debug_metadata: Additional diagnostic information
      
    Args:
        prompt: Original query prompt
        response: Generated response to validate
        domain: Domain name for validation
        include_debug: Whether to include debug metadata in result
        use_enhanced: Whether to use enhanced modules if available
        use_adaptive_threshold: Whether to use adaptive thresholds
        
    Returns:
        Dictionary with validation results
    """
    result = {}
    debug_metadata = {}
    start_time = time.time()
    
    try:
        # 1) Retrieve reference with enhanced retrieval if available
        if use_enhanced:
            ref_entry = retrieve_reference_enhanced(prompt, domain)
        else:
            ref_entry = retrieve_reference(prompt, domain)
            
        reference_text = ref_entry['text']
        reference_id = ref_entry['id']
        similarity = ref_entry['similarity']
        
        # Store reference info
        debug_metadata['reference_similarity'] = similarity
        debug_metadata['reference_id'] = reference_id
        result['reference_text'] = reference_text
        
        # Add enhanced reference fields if available
        if 'term' in ref_entry:
            result['reference_term'] = ref_entry['term']
        if 'examples' in ref_entry:
            result['reference_examples'] = ref_entry['examples']
        
        # 2) Score factuality with enhanced scoring
        factuality_result = score_factuality_enhanced(
            response, 
            reference_text, 
            domain=domain,
            use_enhanced=use_enhanced
        )
        factuality_score = factuality_result["score"]
        result['factuality_score'] = factuality_score
        
        # Add factuality details to debug metadata
        debug_metadata['factuality_details'] = {
            'vector_similarity': factuality_result["vector_similarity"],
            'has_contradiction': factuality_result["has_contradiction"],
            'contradiction_confidence': factuality_result["contradiction_confidence"],
            'contradiction_reason': factuality_result["contradiction_reason"],
            'processing_time_ms': factuality_result.get("processing_time_ms", 0)
        }
        
        # Add chunk-specific details if available
        if "chunk_scores" in factuality_result:
            debug_metadata['factuality_details']['chunk_scores'] = factuality_result["chunk_scores"]
            debug_metadata['factuality_details']['chunks_used'] = factuality_result["chunks_used"]
        
        # 3) Constraint validation
        constraint_pass = True
        constraint_results = {}
        
        try:
            from enrichment.constraint_verifier import ConstraintVerifier
            
            verifier = ConstraintVerifier()
            validation_data = {
                'content': response,
                'context': prompt,
                'reference': reference_text,
                'domain': domain
            }
            
            validation_result = verifier.validate(validation_data)
            constraint_pass = validation_result.passed
            constraint_results = validation_result.to_dict()
            debug_metadata['constraint_results'] = constraint_results
            
        except ImportError:
            # Handle case when constraint verifier is not available
            logger.warning("ConstraintVerifier not available, skipping constraint validation")
            constraint_results = {'error': 'ConstraintVerifier not available'}
            debug_metadata['constraint_results'] = {'error': 'ConstraintVerifier module not found'}
            
        except Exception as e:
            # Handle constraint validation errors
            logger.error(f"Error in constraint validation: {str(e)}")
            constraint_pass = False
            constraint_results = {'error': str(e)}
            debug_metadata['constraint_results'] = {'error': str(e)}
        
        result['constraint_pass'] = constraint_pass
        
        # 4) Determine threshold - basic or adaptive
        if use_adaptive_threshold:
            threshold = adaptive_threshold(prompt, domain, len(response))
            debug_metadata['threshold_type'] = 'adaptive'
        else:
            threshold = get_domain_threshold(domain)
            debug_metadata['threshold_type'] = 'static'
            
        debug_metadata['threshold'] = threshold
        
        # 5) Determine hallucination flag - factuality must exceed threshold AND pass constraints
        hallucination_flag = (factuality_score < threshold) or (not constraint_pass) or factuality_result["has_contradiction"]
        result['hallucination_flag'] = hallucination_flag
        
        # 6) Calculate confidence score with enhanced weighting
        # Base confidence on factuality score
        confidence = factuality_score
        
        # Adjust confidence based on various factors
        if factuality_result["has_contradiction"]:
            # Reduce confidence based on contradiction confidence
            contradiction_penalty = factuality_result["contradiction_confidence"] * 0.25
            confidence = max(0.0, confidence - contradiction_penalty)
        
        if constraint_pass:
            # Boost confidence if constraints pass
            confidence = min(1.0, confidence * 1.05)
        
        # Apply adaptive confidence adjustment based on similarity to reference
        ref_similarity = ref_entry['similarity']
        if ref_similarity < 0.5:
            # Lower confidence when reference match is poor
            confidence = confidence * 0.9
        
        # Ensure valid range
        confidence = max(0.0, min(1.0, confidence))
        result['confidence_score'] = confidence
        
        # 7) Add execution time
        debug_metadata['execution_time_ms'] = (time.time() - start_time) * 1000
        
        # 8) Add debug metadata if requested
        if include_debug:
            result['debug_metadata'] = debug_metadata
            
        return result
        
    except Exception as e:
        # Handle critical failures
        logger.error(f"Critical error in grounding pipeline: {str(e)}")
        
        # Provide a valid fallback response
        result = {
            'reference_text': "Error retrieving reference.",
            'factuality_score': 0.0,
            'constraint_pass': False,
            'hallucination_flag': True,
            'confidence_score': 0.0,
            'error': str(e)
        }
        
        if include_debug:
            debug_metadata['error'] = str(e)
            debug_metadata['traceback'] = "Exception in grounding pipeline"
            debug_metadata['execution_time_ms'] = (time.time() - start_time) * 1000
            result['debug_metadata'] = debug_metadata
            
        return result

def get_available_domains(include_enhanced: bool = True) -> Dict[str, List[str]]:
    """
    Get list of available domains for grounding with extended info.
    
    Args:
        include_enhanced: Whether to include enhanced domains
        
    Returns:
        Dictionary mapping domain types to lists of available domains
    """
    try:
        domains = {
            'standard': [],
            'enhanced': []
        }
        
        # Reference data directory path
        data_dir = Path(os.path.dirname(__file__)).parent / 'data'
        
        # Get all standard JSON files
        standard_domains = [f.stem for f in data_dir.glob('*.json') if f.is_file()]
        domains['standard'].extend(standard_domains)
        
        # Add domains from alias mapping
        for alias, domain in DOMAIN_ALIAS.items():
            if domain in standard_domains and alias not in domains['standard']:
                domains['standard'].append(alias)
        
        # Get enhanced domains if requested
        if include_enhanced:
            enhanced_dirs = [d for d in data_dir.glob('enhanced_*') if d.is_dir()]
            for enhanced_dir in enhanced_dirs:
                domain_name = enhanced_dir.name.replace('enhanced_', '')
                if domain_name not in domains['enhanced']:
                    domains['enhanced'].append(domain_name)
                
                # Add aliased versions
                for alias, domain in DOMAIN_ALIAS.items():
                    if domain == domain_name and alias not in domains['enhanced']:
                        domains['enhanced'].append(alias)
        
        # Sort the lists
        domains['standard'] = sorted(domains['standard'])
        domains['enhanced'] = sorted(domains['enhanced'])
        
        return domains
        
    except Exception as e:
        logger.error(f"Error getting available domains: {str(e)}")
        return {
            'standard': list(DOMAIN_ALIAS.keys()),
            'enhanced': []
        }

def clear_reference_cache() -> None:
    """Clear the reference data cache."""
    global _reference_cache
    _reference_cache = {}
    logger.info("Reference cache cleared")

def check_hallucination(factuality_score: float, domain: str, use_adaptive: bool = False, 
                      prompt: Optional[str] = None, response_length: int = 0) -> bool:
    """
    Check if a response is considered a hallucination with enhanced thresholds.
    
    Args:
        factuality_score: The factuality score (0.0 to 1.0)
        domain: The domain for domain-specific thresholds
        use_adaptive: Whether to use adaptive thresholds
        prompt: Query prompt (needed for adaptive thresholds)
        response_length: Length of response (needed for adaptive thresholds)
        
    Returns:
        True if the response is considered a hallucination, False otherwise
    """
    # Use appropriate threshold based on settings
    if use_adaptive and prompt and response_length > 0:
        threshold = adaptive_threshold(prompt, domain, response_length)
    else:
        threshold = get_domain_threshold(domain.lower())
    
    # Response is a hallucination if score is below threshold
    return factuality_score < threshold

# Test function
def test_enhanced_grounding(prompt: str, response: str, domain: str):
    """
    Test the enhanced grounding functionality.
    
    Args:
        prompt: Test prompt
        response: Test response
        domain: Domain to test
    """
    print(f"\n--- Testing Enhanced Grounding for {domain} Domain ---")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    # Run the pipeline with enhanced settings
    result = grounded_validation_pipeline_enhanced(
        prompt=prompt,
        response=response,
        domain=domain,
        include_debug=True,
        use_enhanced=True,
        use_adaptive_threshold=True
    )
    
    # Print results
    print("\nResults:")
    print(f"  Reference: {result['reference_text'][:100]}...")
    print(f"  Factuality Score: {result['factuality_score']:.4f}")
    print(f"  Hallucination Flag: {result['hallucination_flag']}")
    print(f"  Confidence Score: {result['confidence_score']:.4f}")
    
    # Print debug info
    debug = result.get('debug_metadata', {})
    print("\nDebug Information:")
    print(f"  Threshold: {debug.get('threshold', 'N/A')} ({debug.get('threshold_type', 'unknown')})")
    print(f"  Reference Similarity: {debug.get('reference_similarity', 'N/A')}")
    print(f"  Reference ID: {debug.get('reference_id', 'N/A')}")
    print(f"  Execution Time: {debug.get('execution_time_ms', 0):.2f} ms")
    
    # Print factuality details
    factuality = debug.get('factuality_details', {})
    print("\nFactuality Details:")
    print(f"  Vector Similarity: {factuality.get('vector_similarity', 'N/A')}")
    print(f"  Has Contradiction: {factuality.get('has_contradiction', 'N/A')}")
    
    if factuality.get('has_contradiction', False):
        print(f"  Contradiction Confidence: {factuality.get('contradiction_confidence', 'N/A')}")
        print(f"  Contradiction Reason: {factuality.get('contradiction_reason', 'N/A')}")
    
    if 'chunks_used' in factuality:
        print(f"  Chunks Used: {factuality.get('chunks_used', 0)}")
    
    # Print constraint results if available
    constraints = debug.get('constraint_results', {})
    if constraints and 'error' not in constraints:
        print("\nConstraint Results:")
        print(f"  Passed: {result.get('constraint_pass', 'N/A')}")
        print(f"  Details: {constraints}")

def compare_grounding_versions(prompt: str, response: str, domain: str):
    """
    Compare different grounding engine configurations.
    
    Args:
        prompt: Test prompt
        response: Test response
        domain: Domain to test
    """
    print(f"\n--- Comparing Grounding Engine Versions ({domain}) ---")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    configurations = [
        {
            "name": "Basic",
            "enhanced": False,
            "adaptive": False
        },
        {
            "name": "Enhanced (no adaptive)",
            "enhanced": True,
            "adaptive": False
        },
        {
            "name": "Full Enhanced",
            "enhanced": True,
            "adaptive": True
        }
    ]
    
    results = []
    
    for config in configurations:
        # Run the pipeline with this configuration
        start_time = time.time()
        result = grounded_validation_pipeline_enhanced(
            prompt=prompt,
            response=response,
            domain=domain,
            include_debug=True,
            use_enhanced=config["enhanced"],
            use_adaptive_threshold=config["adaptive"]
        )
        processing_time = (time.time() - start_time) * 1000
        
        # Add result with configuration
        result["config"] = config["name"]
        result["processing_time"] = processing_time
        results.append(result)
    
    # Print comparison
    print("\n--- Results Comparison ---")
    print(f"{'Configuration':<20} {'Factuality':<12} {'Hallucination':<12} {'Confidence':<12} {'Time (ms)':<10}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['config']:<20} {result['factuality_score']:<12.4f} {str(result['hallucination_flag']):<12} {result['confidence_score']:<12.4f} {result['processing_time']:<10.2f}")
    
    # Print reason for hallucination determination
    print("\n--- Hallucination Determination Reasons ---")
    for result in results:
        debug = result.get('debug_metadata', {})
        factuality = debug.get('factuality_details', {})
        
        reason = []
        if result['factuality_score'] < debug.get('threshold', 0.0):
            reason.append(f"Score {result['factuality_score']:.4f} below threshold {debug.get('threshold', 0.0):.4f}")
        
        if factuality.get('has_contradiction', False):
            reason.append(f"Contradiction: {factuality.get('contradiction_reason', 'unspecified')}")
        
        if not result.get('constraint_pass', True):
            reason.append("Failed constraint validation")
        
        print(f"{result['config']}: {'Hallucination' if result['hallucination_flag'] else 'Not hallucination'} - {', '.join(reason) if reason else 'No issues'}")

if __name__ == "__main__":
    # Run tests when executed directly
    test_prompt = "What is the role of the Federal Reserve in controlling inflation?"
    test_response = "The Federal Reserve controls inflation by adjusting interest rates. When inflation is too high, they raise rates to make borrowing more expensive, which slows economic activity. When inflation is too low, they lower rates to stimulate economic growth."
    
    compare_grounding_versions(test_prompt, test_response, "financial")
    
    test_prompt = "Can type 2 diabetes be cured?"
    test_response = "Yes, type 2 diabetes can be completely cured through diet and exercise alone in most cases, without medication."
    
    compare_grounding_versions(test_prompt, test_response, "medical")