# embed_utils.py

from sentence_transformers import SentenceTransformer
import numpy as np
import functools
import logging
import os
from typing import Optional, Union, Dict, List, Tuple, Any
import time
import hashlib
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global cache to avoid redundant model loads
_model = None
_model_name = None
_embedding_cache: Dict[str, np.ndarray] = {}

# Constants
MAX_CACHE_SIZE = 10000  # Maximum number of embeddings to keep in memory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)


def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load the sentence transformer model with caching.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Loaded SentenceTransformer model
    """
    global _model, _model_name
    
    if _model is None or _model_name != model_name:
        try:
            logger.info(f"Loading embedding model: {model_name}")
            start_time = time.time()
            _model = SentenceTransformer(model_name)
            _model_name = model_name
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Attempt to load cached embeddings
            _load_cache()
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            # Fallback to default model if specified model fails
            if model_name != DEFAULT_MODEL:
                logger.info(f"Falling back to default model: {DEFAULT_MODEL}")
                return load_model(DEFAULT_MODEL)
            else:
                raise
    
    return _model


def _get_cache_path() -> str:
    """Get the path to the embedding cache file"""
    return os.path.join(CACHE_DIR, f"embedding_cache_{_model_name}.pkl")


def _save_cache() -> None:
    """Save the current embedding cache to disk"""
    if not _embedding_cache:
        return
    
    try:
        cache_path = _get_cache_path()
        with open(cache_path, 'wb') as f:
            pickle.dump(_embedding_cache, f)
        logger.info(f"Saved {len(_embedding_cache)} embeddings to cache")
    except Exception as e:
        logger.warning(f"Failed to save embedding cache: {str(e)}")


def _load_cache() -> None:
    """Load the embedding cache from disk"""
    global _embedding_cache
    
    try:
        cache_path = _get_cache_path()
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                _embedding_cache = pickle.load(f)
            logger.info(f"Loaded {len(_embedding_cache)} embeddings from cache")
    except Exception as e:
        logger.warning(f"Failed to load embedding cache: {str(e)}")
        _embedding_cache = {}


def _compute_text_hash(text: str) -> str:
    """Compute a hash for the input text"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def get_embedding(text: str, normalize: bool = True) -> np.ndarray:
    """
    Returns the embedding vector for a given input string.
    Uses in-memory and disk caching for performance.
    
    Args:
        text: Text to embed
        normalize: Whether to normalize the embedding vectors
        
    Returns:
        Numpy array containing the embedding vector
    """
    if not text:
        logger.warning("Empty text provided for embedding")
        # Return zero vector of appropriate dimension
        model = load_model()
        dim = model.get_sentence_embedding_dimension()
        return np.zeros(dim)
    
    # Compute hash for cache lookup
    text_hash = _compute_text_hash(text)
    
    # Check in-memory cache first
    if text_hash in _embedding_cache:
        return _embedding_cache[text_hash]
    
    try:
        # Not in cache, compute embedding
        model = load_model()
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=normalize)
        
        # Cache management: if cache too large, remove oldest entries
        if len(_embedding_cache) >= MAX_CACHE_SIZE:
            # Remove 10% of the cache
            remove_count = MAX_CACHE_SIZE // 10
            keys_to_remove = list(_embedding_cache.keys())[:remove_count]
            for k in keys_to_remove:
                _embedding_cache.pop(k, None)
        
        # Add to cache
        _embedding_cache[text_hash] = embedding
        
        # Periodically save cache to disk (every 100 new embeddings)
        if len(_embedding_cache) % 100 == 0:
            _save_cache()
        
        return embedding
    
    except Exception as e:
        logger.error(f"Error generating embedding for text: {str(e)}")
        # Return zero vector as fallback
        model = load_model()
        dim = model.get_sentence_embedding_dimension()
        return np.zeros(dim)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors with error handling.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    try:
        # Handle zero vectors
        if np.all(a == 0) or np.all(b == 0):
            return 0.0
            
        # Calculate cosine similarity with numerical stability
        similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        
        # Ensure result is in valid range
        return max(0.0, min(1.0, similarity))
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return 0.0


def semantic_search(query: str, candidates: List[str], threshold: float = 0.6) -> List[Tuple[int, str, float]]:
    """
    Find semantically similar items from a list of candidates.
    
    Args:
        query: The search query
        candidates: List of text items to search through
        threshold: Minimum similarity score to include in results
        
    Returns:
        List of tuples containing (index, text, score) sorted by similarity score
    """
    if not query or not candidates:
        return []
        
    try:
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Get embeddings for all candidates
        results = []
        
        for i, text in enumerate(candidates):
            # Skip empty candidates
            if not text:
                continue
                
            # Get embedding and calculate similarity
            candidate_embedding = get_embedding(text)
            similarity = cosine_similarity(query_embedding, candidate_embedding)
            
            # Add to results if above threshold
            if similarity >= threshold:
                results.append((i, text, similarity))
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []


def batch_embed(texts: List[str], normalize: bool = True) -> Dict[str, np.ndarray]:
    """
    Create embeddings for multiple texts efficiently.
    
    Args:
        texts: List of texts to embed
        normalize: Whether to normalize the embedding vectors
        
    Returns:
        Dictionary mapping each text to its embedding
    """
    if not texts:
        return {}
        
    try:
        # Filter out empty texts and duplicates
        unique_texts = list(set(filter(None, texts)))
        
        # Check which texts are already cached
        cached_embeddings = {}
        texts_to_embed = []
        
        for text in unique_texts:
            text_hash = _compute_text_hash(text)
            if text_hash in _embedding_cache:
                cached_embeddings[text] = _embedding_cache[text_hash]
            else:
                texts_to_embed.append(text)
        
        # If we have any uncached texts, compute their embeddings
        if texts_to_embed:
            model = load_model()
            new_embeddings = model.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=normalize)
            
            # Cache the new embeddings
            for i, text in enumerate(texts_to_embed):
                text_hash = _compute_text_hash(text)
                _embedding_cache[text_hash] = new_embeddings[i]
                cached_embeddings[text] = new_embeddings[i]
                
            # Save cache if we added a significant number of embeddings
            if len(texts_to_embed) > 10:
                _save_cache()
        
        # Create the final result dictionary
        return {text: cached_embeddings.get(text) for text in texts if text in cached_embeddings}
        
    except Exception as e:
        logger.error(f"Error in batch_embed: {str(e)}")
        return {}
