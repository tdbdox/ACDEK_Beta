import os
import json
import logging
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

# Import semantic chunking and contradiction detection modules if available
try:
    # First try local imports
    from semantic_chunking import SemanticChunker, chunk_reference_data, retrieve_reference_chunks, score_factuality_with_chunks
    from contradiction_detection import detect_contradictions
    SEMANTIC_CHUNKING_AVAILABLE = True
    CONTRADICTION_DETECTION_AVAILABLE = True
except ImportError:
    # Then try import by path
    SEMANTIC_CHUNKING_AVAILABLE = False
    CONTRADICTION_DETECTION_AVAILABLE = False
    
    # Try to import semantic chunking
    module_path = Path(__file__).parent / "semantic_chunking.py"
    if module_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("semantic_chunking", module_path)
            semantic_chunking = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(semantic_chunking)
            SemanticChunker = semantic_chunking.SemanticChunker
            chunk_reference_data = semantic_chunking.chunk_reference_data
            retrieve_reference_chunks = semantic_chunking.retrieve_reference_chunks
            score_factuality_with_chunks = semantic_chunking.score_factuality_with_chunks
            SEMANTIC_CHUNKING_AVAILABLE = True
        except Exception as e:
            logger.warning(f"Error importing semantic chunking module: {e}")
    
    # Try to import contradiction detection
    module_path = Path(__file__).parent / "contradiction_detection.py"
    if module_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("contradiction_detection", module_path)
            contradiction_detection = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(contradiction_detection)
            detect_contradictions = contradiction_detection.detect_contradictions
            CONTRADICTION_DETECTION_AVAILABLE = True
        except Exception as e:
            logger.warning(f"Error importing contradiction detection module: {e}")

# Threshold for factuality scoring: below this is considered hallucination
FACTUALITY_THRESHOLD: float = 0.70  # Adjusted from 0.75 to 0.70 to improve hallucination detection

# Domain-specific thresholds (overrides global threshold)
DOMAIN_THRESHOLDS = {
    'medical': 0.75,  # Medical domain requires higher factuality
    'legal': 0.72,    # Legal domain has specific threshold - maintained as performance is adequate
    'financial': 0.68  # Financial domain adjusted from 0.64 to 0.68 for better balance
}

# Domain aliases for mapping domain values to reference filenames
DOMAIN_ALIAS = {
    'financial': 'finance',
    'medical': 'medical',
    'legal': 'legal'
}

# Cache for reference data to avoid repeated file loading
_reference_cache: Dict[str, List[Dict[str, Any]]] = {}


def _load_references(domain: str) -> List[Dict[str, Any]]:
    """
    Load reference entries for the given domain from JSON data files with caching.
    
    Args:
        domain: Domain name to load references for
        
    Returns:
        List of reference entries
        
    Raises:
        FileNotFoundError: If no reference data is found for the domain
    """
    domain_key = DOMAIN_ALIAS.get(domain, domain)
    
    # Check cache first
    if domain_key in _reference_cache:
        return _reference_cache[domain_key]
    
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
        _reference_cache[domain_key] = references
        logger.info(f"Loaded {len(references)} reference entries for domain '{domain}'")
        return references
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing reference data for domain '{domain}': {str(e)}")
        raise ValueError(f"Invalid JSON in reference data for domain '{domain}'") from e


def retrieve_reference(prompt: str, domain: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Perform vector-based retrieval from the appropriate reference dataset.
    Returns the best matching reference entry with id, text, and similarity.
    
    Args:
        prompt: The query prompt to match against references
        domain: Domain name to retrieve references from
        top_k: Number of top matches to consider (default: 3)
        
    Returns:
        Dictionary containing best matching reference with id, text, and similarity
        
    Raises:
        ValueError: If no references are available for the domain or 
                   if no reference similarity exceeds minimum threshold
    """
    references = _load_references(domain)
    
    if not references:
        raise ValueError(f"No references available for domain '{domain}'")
    
    # Extract reference texts for semantic search
    reference_texts = [entry.get('text', '') for entry in references]
    
    # Perform semantic search to find best matches
    search_results = semantic_search(
        query=prompt,
        candidates=reference_texts,
        threshold=0.5,  # Lower threshold to ensure we get results
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
    
    # Get the full entry from the reference data
    best_entry = {
        'id': references[best_index].get('id', 'unknown'),
        'text': references[best_index].get('text', ''),
        'similarity': best_score,
        'is_fallback': False,
        'all_matches': [
            {
                'id': references[idx].get('id', 'unknown'),
                'text': references[idx].get('text', ''),
                'similarity': score
            }
            for idx, _, score, _ in search_results[:min(3, len(search_results))]
        ]
    }
    
    logger.debug(f"Retrieved reference for domain '{domain}' with similarity {best_score:.4f}")
    return best_entry


def contains_contradiction(text: str, reference: str) -> Tuple[bool, float, str]:
    """
    Check if text contains direct contradictions to the reference.
    
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

def score_factuality(response: str, reference: str, domain: str = None, use_chunks: bool = True) -> Dict[str, Any]:
    """
    Score factuality by computing embedding similarity between response and reference text.
    Returns a dictionary with factuality score and metadata.
    
    With semantic chunking enabled, the function will use a chunk-based approach for
    more precise matching. With contradiction detection enabled, it will perform
    more sophisticated contradiction analysis.
    
    Args:
        response: Generated text to evaluate
        reference: Reference text to compare against
        domain: Domain for domain-specific processing (defaults to None)
        use_chunks: Whether to use semantic chunking (if available)
        
    Returns:
        Dictionary with factuality score and metadata
    """
    # Handle edge cases
    if not response or not reference:
        logger.warning("Empty input for factuality scoring")
        return {
            "score": 0.0,
            "has_contradiction": False,
            "contradiction_confidence": 0.0,
            "contradiction_reason": "Empty input",
            "vector_similarity": 0.0
        }
    
    # Initialize result dictionary
    result = {
        "score": 0.0,
        "has_contradiction": False,
        "contradiction_confidence": 0.0,
        "contradiction_reason": "",
        "vector_similarity": 0.0
    }
    
    # Step 1: Check for contradictions using enhanced detection if available
    if CONTRADICTION_DETECTION_AVAILABLE and domain:
        try:
            # Use enhanced contradiction detection
            contradiction_result = detect_contradictions(response, reference, domain)
            has_contradiction = contradiction_result["has_contradiction"]
            contradiction_confidence = contradiction_result["contradiction_confidence"]
            contradiction_reason = contradiction_result["contradiction_reason"]
            
            result["has_contradiction"] = has_contradiction
            result["contradiction_confidence"] = contradiction_confidence
            result["contradiction_reason"] = contradiction_reason
            
            if has_contradiction:
                logger.info(f"Enhanced contradiction detection found contradiction: {contradiction_reason}")
        except Exception as e:
            logger.warning(f"Error in enhanced contradiction detection: {e}. Falling back to basic detection.")
            # Fallback to basic contradiction detection
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
    
    # Step 2: Calculate vector similarity (with chunking if available)
    try:
        if SEMANTIC_CHUNKING_AVAILABLE and use_chunks:
            # Use semantic chunking for more precise matching
            chunker = SemanticChunker()
            ref_chunks = chunker.chunk_text(reference)
            
            # Compute factuality with chunks
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
        if has_contradiction:
            # Apply a proportional penalty based on contradiction confidence
            # More confident contradictions receive stronger penalties
            contradiction_penalty = contradiction_confidence * 0.6
            
            # Apply graduated penalty - more similar texts with contradictions still get some credit
            # This prevents the "cliff effect" where good responses with minor issues get severely punished
            final_score = max(0.3, vector_similarity * (1 - contradiction_penalty))
        else:
            final_score = vector_similarity
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, final_score))
        result["score"] = final_score
        
        return result
        
    except Exception as e:
        logger.error(f"Error in factuality scoring: {str(e)}")
        result["error"] = str(e)
        return result


def build_grounding_context(prompt: str, domain: str) -> Dict[str, Any]:
    """
    Build a comprehensive grounding context for the prompt and domain.
    
    Args:
        prompt: The query prompt to build context for
        domain: Domain name for the context
        
    Returns:
        Dictionary with grounding context including relevant references
    """
    try:
        # Retrieve top reference
        ref_entry = retrieve_reference(prompt, domain)
        
        # Prepare context
        context = {
            'domain': domain,
            'primary_reference': {
                'id': ref_entry['id'],
                'text': ref_entry['text'],
                'similarity': ref_entry['similarity']
            },
            'additional_references': ref_entry.get('all_matches', []),
            'timestamp': None,  # Will be filled by the caller if needed
            'threshold': FACTUALITY_THRESHOLD
        }
        
        return context
        
    except Exception as e:
        logger.error(f"Error building grounding context: {str(e)}")
        return {
            'domain': domain,
            'error': str(e),
            'primary_reference': None,
            'additional_references': [],
            'threshold': FACTUALITY_THRESHOLD
        }


def grounded_validation_pipeline(
    prompt: str, 
    response: str, 
    domain: str,
    include_debug: bool = False,
    use_chunks: bool = True,
    use_enhanced_contradiction: bool = True
) -> Dict[str, Any]:
    """
    Full grounding and validation pipeline using RAG and constraint validation.
    
    Enhanced with semantic chunking and sophisticated contradiction detection.
    
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
        use_chunks: Whether to use semantic chunking (if available)
        use_enhanced_contradiction: Whether to use enhanced contradiction detection (if available)
        
    Returns:
        Dictionary with validation results
    """
    result = {}
    debug_metadata = {}
    start_time = None
    
    try:
        import time
        start_time = time.time()
        
        # 1) Retrieve reference - with chunking if available and enabled
        if SEMANTIC_CHUNKING_AVAILABLE and use_chunks:
            # Log that we're using semantic chunking
            logger.info("Using semantic chunking for reference retrieval")
            debug_metadata['using_semantic_chunking'] = True
            
            # Load reference data
            try:
                references = _load_references(domain)
                
                # Chunk reference data
                chunker = SemanticChunker()
                chunked_references = []
                for ref in references:
                    ref_chunks = chunker.chunk_text(ref.get('text', ''))
                    for chunk in ref_chunks:
                        chunked_ref = {
                            'id': f"{ref.get('id', 'unknown')}_{chunk['id']}",
                            'text': chunk['text'],
                            'parent_id': ref.get('id', 'unknown'),
                            'source': ref.get('source', '')
                        }
                        chunked_references.append(chunked_ref)
                
                # Retrieve most relevant chunks
                chunks = retrieve_reference_chunks(prompt, domain, chunked_references, top_k=3)
                
                if chunks:
                    # Use most relevant chunk as primary reference
                    best_chunk = chunks[0]
                    reference_text = best_chunk.get('text', '')
                    reference_id = best_chunk.get('id', 'unknown')
                    similarity = best_chunk.get('similarity', 0.0)
                    
                    # Store all chunks for debug
                    debug_metadata['reference_chunks'] = chunks
                    debug_metadata['reference_chunks_count'] = len(chunks)
                else:
                    # Fallback to standard retrieval if no chunks were found
                    logger.warning("No chunks found, falling back to standard retrieval")
                    ref_entry = retrieve_reference(prompt, domain)
                    reference_text = ref_entry['text']
                    reference_id = ref_entry['id']
                    similarity = ref_entry['similarity']
            except Exception as e:
                logger.error(f"Error in chunked retrieval: {e}. Falling back to standard retrieval.")
                # Fallback to standard retrieval
                ref_entry = retrieve_reference(prompt, domain)
                reference_text = ref_entry['text']
                reference_id = ref_entry['id']
                similarity = ref_entry['similarity']
                debug_metadata['chunking_error'] = str(e)
        else:
            # Standard reference retrieval
            ref_entry = retrieve_reference(prompt, domain)
            reference_text = ref_entry['text']
            reference_id = ref_entry['id']
            similarity = ref_entry['similarity']
        
        # Store reference info
        debug_metadata['reference_similarity'] = similarity
        debug_metadata['reference_id'] = reference_id
        result['reference_text'] = reference_text
        
        # 2) Score factuality with enhanced scoring
        factuality_result = score_factuality(
            response, 
            reference_text, 
            domain=domain if use_enhanced_contradiction and CONTRADICTION_DETECTION_AVAILABLE else None,
            use_chunks=use_chunks and SEMANTIC_CHUNKING_AVAILABLE
        )
        factuality_score = factuality_result["score"]
        result['factuality_score'] = factuality_score
        
        # Add factuality details to debug metadata
        debug_metadata['factuality_details'] = {
            'vector_similarity': factuality_result["vector_similarity"],
            'has_contradiction': factuality_result["has_contradiction"],
            'contradiction_confidence': factuality_result["contradiction_confidence"],
            'contradiction_reason': factuality_result["contradiction_reason"]
        }
        
        # Add chunk-specific details if available
        if "chunk_scores" in factuality_result:
            debug_metadata['factuality_details']['chunk_scores'] = factuality_result["chunk_scores"]
            debug_metadata['factuality_details']['chunks_used'] = factuality_result["chunks_used"]
        
        # Get domain-specific threshold
        threshold = DOMAIN_THRESHOLDS.get(domain, FACTUALITY_THRESHOLD)
        
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
        
        # Get domain-specific threshold or fall back to global threshold
        threshold = DOMAIN_THRESHOLDS.get(domain, FACTUALITY_THRESHOLD)
        debug_metadata['threshold'] = threshold
        
        # 4) Determine hallucination flag - factuality must exceed threshold AND pass constraints
        hallucination_flag = (factuality_score < threshold) or (not constraint_pass)
        result['hallucination_flag'] = hallucination_flag
        
        # 5) Calculate confidence score with enhanced weighting
        # Base confidence on factuality score
        confidence = factuality_score
        
        # Adjust confidence based on various factors
        if factuality_result["has_contradiction"]:
            # Reduce confidence based on contradiction confidence
            contradiction_penalty = factuality_result["contradiction_confidence"] * 0.2
            confidence = max(0.0, confidence - contradiction_penalty)
        
        if constraint_pass:
            # Boost confidence if constraints pass
            confidence = min(1.0, confidence * 1.05)
        
        # Ensure valid range
        confidence = max(0.0, min(1.0, confidence))
        result['confidence_score'] = confidence
        
        # 6) Add execution time if applicable
        if start_time:
            debug_metadata['execution_time_ms'] = (time.time() - start_time) * 1000
        
        # 7) Add debug metadata if requested
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
            'confidence_score': 0.0
        }
        
        if include_debug:
            debug_metadata['error'] = str(e)
            debug_metadata['traceback'] = "Exception in grounding pipeline"
            if start_time:
                debug_metadata['execution_time_ms'] = (time.time() - start_time) * 1000
            result['debug_metadata'] = debug_metadata
            
        return result


def get_available_domains() -> List[str]:
    """
    Get list of available domains for grounding.
    
    Returns:
        List of available domain names
    """
    try:
        # Reference data directory path
        data_dir = Path(os.path.dirname(__file__)).parent / 'data'
        
        # Get all JSON files
        domain_files = [f.stem for f in data_dir.glob('*.json') if f.is_file()]
        
        # Add domains from alias mapping
        domains = set(domain_files)
        for alias, domain in DOMAIN_ALIAS.items():
            if domain in domains:
                domains.add(alias)
                
        return sorted(list(domains))
        
    except Exception as e:
        logger.error(f"Error getting available domains: {str(e)}")
        return list(DOMAIN_ALIAS.keys())


def clear_reference_cache() -> None:
    """Clear the reference data cache."""
    global _reference_cache
    _reference_cache = {}
    logger.info("Reference cache cleared")


def check_hallucination(factuality_score: float, domain: str) -> bool:
    """
    Check if a response is considered a hallucination based on factuality score and domain-specific thresholds.
    
    Args:
        factuality_score: The factuality score (0.0 to 1.0)
        domain: The domain for domain-specific thresholds
        
    Returns:
        True if the response is considered a hallucination, False otherwise
    """
    # Get domain-specific threshold or fall back to global threshold
    threshold = DOMAIN_THRESHOLDS.get(domain.lower(), FACTUALITY_THRESHOLD)
    
    # Response is a hallucination if score is below threshold
    return factuality_score < threshold