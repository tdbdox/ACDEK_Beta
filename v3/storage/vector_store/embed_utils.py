import logging
import functools
import os
import time
import pickle
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

# Registry imports
from registry.registry_types import PluginType
from registry.plugin_registry import register_plugin
from interfaces.vector_alignment_interface import VectorAlignmentInterface

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MAX_CACHE_SIZE = 10000  # Maximum number of embeddings to keep in memory
MAX_BATCH_SIZE = 64     # Maximum batch size for embedding generation
CACHE_TTL = 86400       # Time-to-live for cache entries in seconds (24 hours)
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # Default embedding model

# Ensure directories
def ensure_directory(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)

# Default cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
ensure_directory(CACHE_DIR)


class EmbeddingCache:
    """
    Manages embedding vectors with time-based expiration and disk persistence.
    """
    
    def __init__(self, cache_dir: str = CACHE_DIR, max_size: int = MAX_CACHE_SIZE, ttl: int = CACHE_TTL):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory for disk persistence
            max_size: Maximum number of items in memory cache
            ttl: Time-to-live for cache entries in seconds
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl = ttl
        
        # In-memory cache: key -> (embedding, timestamp)
        self.cache: Dict[str, Tuple[np.ndarray, float]] = {}
        
        # Access tracking for LRU eviction
        self.access_times: Dict[str, float] = {}
        
        # Disk cache path
        self.cache_path = os.path.join(cache_dir, "embedding_cache.pkl")
        
        # Load cache from disk if available
        self._load_cache()
        
        logger.info(f"Initialized embedding cache (max_size={max_size}, ttl={ttl}s)")
    
    def _compute_key(self, text: str, model_name: str) -> str:
        """Compute a cache key for text and model."""
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        model_hash = hashlib.md5(model_name.encode('utf-8')).hexdigest()[:8]
        return f"{text_hash}_{model_hash}"
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache if available and not expired.
        
        Args:
            text: Text to get embedding for
            model_name: Name of embedding model
            
        Returns:
            Embedding vector if in cache and valid, None otherwise
        """
        key = self._compute_key(text, model_name)
        
        # Check if in memory cache
        if key in self.cache:
            embedding, timestamp = self.cache[key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl:
                # Expired, remove from cache
                logger.debug(f"Cache entry expired: {key}")
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return None
            
            # Update access time for LRU tracking
            self.access_times[key] = time.time()
            
            return embedding
        
        return None
    
    def put(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Text the embedding is for
            model_name: Name of embedding model
            embedding: Embedding vector to store
        """
        key = self._compute_key(text, model_name)
        
        # Check cache size and evict if needed
        if len(self.cache) >= self.max_size:
            self._evict_entries()
        
        # Store in cache with current timestamp
        self.cache[key] = (embedding, time.time())
        self.access_times[key] = time.time()
        
        # Schedule periodic saving
        if len(self.cache) % 100 == 0:
            self._save_cache()
    
    def _evict_entries(self) -> None:
        """Evict least recently used entries or expired entries to make space."""
        # First try to evict expired entries
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items() 
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        # If we still need to evict, use LRU policy
        if len(self.cache) >= self.max_size:
            # Find least recently used keys
            lru_keys = sorted(
                self.access_times.keys(),
                key=lambda k: self.access_times.get(k, 0)
            )
            
            # Remove 10% of the cache or at least one entry
            remove_count = max(1, self.max_size // 10)
            for key in lru_keys[:remove_count]:
                if key in self.cache:
                    del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
            
            logger.info(f"Evicted {min(remove_count, len(lru_keys))} entries from embedding cache")
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            # Create a copy to avoid serialization issues during concurrent access
            cache_copy = {k: v for k, v in self.cache.items()}
            
            with open(self.cache_path, 'wb') as f:
                pickle.dump((cache_copy, self.access_times), f)
                
            logger.info(f"Saved {len(cache_copy)} embeddings to disk cache")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {str(e)}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            if not os.path.exists(self.cache_path):
                logger.info("No disk cache found, starting with empty cache")
                return
                
            with open(self.cache_path, 'rb') as f:
                loaded_cache, loaded_access_times = pickle.load(f)
            
            # Filter out expired entries
            current_time = time.time()
            valid_entries = {
                key: (embedding, timestamp)
                for key, (embedding, timestamp) in loaded_cache.items()
                if current_time - timestamp <= self.ttl
            }
            
            valid_access_times = {
                key: access_time
                for key, access_time in loaded_access_times.items()
                if key in valid_entries
            }
            
            self.cache = valid_entries
            self.access_times = valid_access_times
            
            logger.info(f"Loaded {len(self.cache)} embeddings from disk cache")
            
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {str(e)}")
            self.cache = {}
            self.access_times = {}
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_times.clear()
        logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "expired_count": sum(1 for _, timestamp in self.cache.values() if current_time - timestamp > self.ttl),
            "avg_age": sum(current_time - timestamp for _, timestamp in self.cache.values()) / max(1, len(self.cache)),
        }


# Global instances
_model: Optional[SentenceTransformer] = None
_model_name: Optional[str] = None
_embedding_cache = EmbeddingCache()


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
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            # Fallback to default model if specified model fails
            if model_name != DEFAULT_MODEL:
                logger.info(f"Falling back to default model: {DEFAULT_MODEL}")
                return load_model(DEFAULT_MODEL)
            else:
                raise
    
    return _model


def get_embedding(text: str, normalize: bool = True, model_name: str = None) -> np.ndarray:
    """
    Get embedding vector for text with caching.
    
    Args:
        text: Text to embed
        normalize: Whether to normalize the embedding vector
        model_name: Optional model name to use
        
    Returns:
        Embedding vector as numpy array
    """
    if not text:
        logger.warning("Empty text provided for embedding")
        # Return zero vector of appropriate dimension
        model = load_model(model_name or DEFAULT_MODEL)
        dim = model.get_sentence_embedding_dimension()
        return np.zeros(dim)
    
    model_name = model_name or DEFAULT_MODEL
    
    # Check cache first
    cached_embedding = _embedding_cache.get(text, model_name)
    if cached_embedding is not None:
        return cached_embedding
    
    try:
        # Not in cache, compute embedding
        model = load_model(model_name)
        embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=normalize)
        
        # Add to cache
        _embedding_cache.put(text, model_name, embedding)
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embedding for text: {str(e)}")
        # Return zero vector as fallback
        model = load_model(model_name)
        dim = model.get_sentence_embedding_dimension()
        return np.zeros(dim)


def batch_embed(texts: List[str], normalize: bool = True, model_name: str = None) -> Dict[str, np.ndarray]:
    """
    Create embeddings for multiple texts efficiently with caching.
    
    Args:
        texts: List of texts to embed
        normalize: Whether to normalize the embeddings
        model_name: Optional model name to use
        
    Returns:
        Dictionary mapping texts to embedding vectors
    """
    if not texts:
        return {}
    
    model_name = model_name or DEFAULT_MODEL
    
    try:
        # Filter out empty texts and duplicates
        unique_texts = list(set(filter(None, texts)))
        
        # Check which texts are already cached
        cached_embeddings = {}
        texts_to_embed = []
        
        for text in unique_texts:
            cached_embedding = _embedding_cache.get(text, model_name)
            if cached_embedding is not None:
                cached_embeddings[text] = cached_embedding
            else:
                texts_to_embed.append(text)
        
        # If we have any uncached texts, compute their embeddings
        if texts_to_embed:
            model = load_model(model_name)
            
            # Process in batches to avoid memory issues with large datasets
            all_new_embeddings = {}
            
            for i in range(0, len(texts_to_embed), MAX_BATCH_SIZE):
                batch = texts_to_embed[i:i+MAX_BATCH_SIZE]
                batch_embeddings = model.encode(batch, convert_to_numpy=True, normalize_embeddings=normalize)
                
                # Map embeddings back to texts
                for j, text in enumerate(batch):
                    all_new_embeddings[text] = batch_embeddings[j]
                    _embedding_cache.put(text, model_name, batch_embeddings[j])
            
            # Combine with cached embeddings
            cached_embeddings.update(all_new_embeddings)
        
        # Create the final result dictionary, maintaining original order
        result = {}
        for text in texts:
            if text in cached_embeddings:
                result[text] = cached_embeddings[text]
        
        return result
        
    except Exception as e:
        logger.error(f"Error in batch_embed: {str(e)}")
        return {}


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


def semantic_search(query: str, candidates: List[str], threshold: float = 0.6, model_name: str = None) -> List[Tuple[int, str, float]]:
    """
    Find semantically similar items from a list of candidates.
    
    Args:
        query: The search query
        candidates: List of text items to search through
        threshold: Minimum similarity score to include in results
        model_name: Optional model name to use
        
    Returns:
        List of tuples containing (index, text, score) sorted by similarity score
    """
    if not query or not candidates:
        return []
        
    try:
        # Get query embedding
        query_embedding = get_embedding(query, model_name=model_name)
        
        # Get embeddings for all candidates
        candidate_embeddings = batch_embed(candidates, model_name=model_name)
        
        results = []
        
        for i, text in enumerate(candidates):
            # Skip empty candidates or those that failed to embed
            if not text or text not in candidate_embeddings:
                continue
                
            # Calculate similarity
            similarity = cosine_similarity(query_embedding, candidate_embeddings[text])
            
            # Add to results if above threshold
            if similarity >= threshold:
                results.append((i, text, similarity))
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []


@register_plugin(
    plugin_type=PluginType.VECTOR_ALIGNMENT,
    name="sentence_transformer_aligner",
    version="1.0.0",
    description="Vector alignment component using SentenceTransformer models",
    interfaces=["VectorAlignmentInterface"],
    is_default=True
)
class SentenceTransformerAligner(VectorAlignmentInterface):
    """
    Vector alignment component using SentenceTransformer models.
    
    This component provides embedding generation, similarity calculation,
    and semantic search functionality using the SentenceTransformer library.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL, normalize: bool = True):
        """
        Initialize the vector alignment component.
        
        Args:
            model_name: Name of the embedding model to use
            normalize: Whether to normalize embeddings by default
        """
        self.model_name = model_name
        self.normalize = normalize
        
        # Load model on initialization to avoid lazy loading delays
        load_model(model_name)
        
        logger.info(f"Initialized SentenceTransformerAligner with model {model_name}")
    
    def get_embedding(self, text: str, normalize: bool = None) -> np.ndarray:
        """
        Generate embedding vector for input text.
        
        Args:
            text: The text to create an embedding for
            normalize: Whether to normalize the embedding vector
            
        Returns:
            Embedding vector as numpy array
        """
        norm = self.normalize if normalize is None else normalize
        return get_embedding(text, normalize=norm, model_name=self.model_name)
    
    def batch_embed(self, texts: List[str], normalize: bool = None) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize the embedding vectors
            
        Returns:
            Dictionary mapping each text to its embedding
        """
        norm = self.normalize if normalize is None else normalize
        return batch_embed(texts, normalize=norm, model_name=self.model_name)
    
    def calculate_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        Calculate similarity score between two vectors.
        
        Args:
            vector_a: First embedding vector
            vector_b: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        return cosine_similarity(vector_a, vector_b)
    
    def find_similar(
        self, 
        query: Union[str, np.ndarray],
        candidates: Union[List[str], Dict[str, Any], Dict[str, np.ndarray]],
        threshold: float = 0.6
    ) -> List[Tuple[Union[int, str], Any, float]]:
        """
        Find semantically similar items from a collection of candidates.
        
        Args:
            query: Query text or embedding vector
            candidates: Collection of items to search through (texts, objects, or pre-computed embeddings)
            threshold: Minimum similarity score to include in results
            
        Returns:
            List of tuples containing (id, item, score) sorted by similarity score
        """
        try:
            # Convert query to embedding if it's a string
            query_embedding = query if isinstance(query, np.ndarray) else self.get_embedding(query)
            
            results = []
            
            # Handle different candidate formats
            if isinstance(candidates, list):
                # List of strings
                if all(isinstance(c, str) for c in candidates):
                    # Use batch_embed for efficiency
                    embeddings = self.batch_embed(candidates)
                    
                    for i, text in enumerate(candidates):
                        if text in embeddings:
                            similarity = self.calculate_similarity(query_embedding, embeddings[text])
                            if similarity >= threshold:
                                results.append((i, text, similarity))
                
                # List of objects
                else:
                    for i, item in enumerate(candidates):
                        # For objects, try to extract text for embedding
                        if hasattr(item, '__str__'):
                            text = str(item)
                            embedding = self.get_embedding(text)
                            similarity = self.calculate_similarity(query_embedding, embedding)
                            if similarity >= threshold:
                                results.append((i, item, similarity))
            
            # Dictionary of items
            elif isinstance(candidates, dict):
                # Dictionary of embeddings
                if all(isinstance(v, np.ndarray) for v in candidates.values()):
                    for key, embedding in candidates.items():
                        similarity = self.calculate_similarity(query_embedding, embedding)
                        if similarity >= threshold:
                            results.append((key, embedding, similarity))
                
                # Dictionary of objects
                else:
                    for key, item in candidates.items():
                        # For objects, try to extract text for embedding
                        if hasattr(item, '__str__'):
                            text = str(item)
                            embedding = self.get_embedding(text)
                            similarity = self.calculate_similarity(query_embedding, embedding)
                            if similarity >= threshold:
                                results.append((key, item, similarity))
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x[2], reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Error in find_similar: {str(e)}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the underlying embedding model.
        
        Returns:
            Dictionary containing model metadata
        """
        model = load_model(self.model_name)
        
        return {
            "model_name": self.model_name,
            "dimension": model.get_sentence_embedding_dimension(),
            "cache_stats": _embedding_cache.get_stats(),
        }
