import logging
import functools
import os
import time
import pickle
import hashlib
import json
import numpy as np
import traceback
from enum import Enum
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta
import platform
import sys
import threading
import re

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
FALLBACK_MODELS = ["paraphrase-MiniLM-L3-v2", "distiluse-base-multilingual-cased-v1"]  # Fallback models
VERSION = "1.3.0"  # Current version of the embedding utilities

# Define cache signature hash methods
class SignatureMethod(str, Enum):
    """Methods for computing cache signature hashes."""
    SIMPLE_HASH = "simple_hash"  # Simple MD5 hash
    CONTENT_LENGTH = "content_length"  # Hash that includes content length
    SEMANTIC_FINGERPRINT = "semantic_fingerprint"  # Hash that includes semantic key terms
    CONTENT_FEATURES = "content_features"  # Smart hash using statistical features of content
    ADAPTIVE = "adaptive"  # Automatically selects best signature method based on content

# Define embedding recovery modes
class RecoveryMode(str, Enum):
    """Recovery strategies for embedding failures."""
    FALLBACK_MODELS = "fallback_models"  # Try alternative models
    ZERO_VECTOR = "zero_vector"  # Return zero vector of appropriate dimension
    WARNING_ONLY = "warning_only"  # Log warning and return None
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Retry with exponential backoff
    CACHED_SIMILAR = "cached_similar"  # Use most similar cached embedding
    FULL_RECOVERY = "full_recovery"  # Try all recovery methods in sequence

# Ensure directories
def ensure_directory(path):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)

# Default cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
ensure_directory(CACHE_DIR)

# Set up version tracking directory
VERSION_DIR = os.path.join(CACHE_DIR, 'versions')
ensure_directory(VERSION_DIR)


class EmbeddingCache:
    """
    Enhanced caching system for embedding vectors with advanced signature hashing,
    version tracking, comprehensive persistence, and intelligent eviction strategies.
    
    This cache implements multiple signature hashing methods, automatic method selection,
    fine-grained version tracking, and graceful fallback mechanisms.
    """
    
    def __init__(
        self, 
        cache_dir: str = CACHE_DIR, 
        max_size: int = MAX_CACHE_SIZE, 
        ttl: int = CACHE_TTL,
        signature_method: SignatureMethod = SignatureMethod.ADAPTIVE,
        version_tracking: bool = True,
        thread_safe: bool = True,
        auto_save_interval: int = 100,
        cache_segments: int = 4  # Divide cache into segments for better performance
    ):
        """
        Initialize the enhanced embedding cache.
        
        Args:
            cache_dir: Directory for disk persistence
            max_size: Maximum number of items in memory cache
            ttl: Time-to-live for cache entries in seconds
            signature_method: Method to use for computing cache keys
            version_tracking: Whether to enable version tracking
            thread_safe: Whether to make the cache thread-safe
            auto_save_interval: Number of writes before auto-saving
            cache_segments: Number of segments to divide the cache into
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl = ttl
        self.signature_method = signature_method
        self.version_tracking = version_tracking
        self.thread_safe = thread_safe
        self.auto_save_interval = auto_save_interval
        self.cache_segments = cache_segments
        
        # In-memory cache: key -> (embedding, timestamp, metadata)
        self.cache: Dict[str, Tuple[np.ndarray, float, Dict[str, Any]]] = {}
        
        # Access tracking for LRU eviction
        self.access_times: Dict[str, float] = {}
        
        # Cache hit/miss tracking
        self.hits = 0
        self.misses = 0
        self.version = VERSION
        
        # Method usage tracking
        self.method_performance: Dict[SignatureMethod, Dict[str, float]] = {
            method: {"hits": 0, "misses": 0, "latency": 0, "calls": 0} 
            for method in SignatureMethod
        }
        
        # Disk cache paths
        self.cache_path = os.path.join(cache_dir, f"embedding_cache_{self.version}.pkl")
        self.version_log_path = os.path.join(VERSION_DIR, "embedding_version_log.json")
        
        # Failed embedding tracking (text -> model -> {count, last_failure, errors})
        self.failure_tracking: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Similar text groups for potential fallbacks (hash bucket -> list of texts)
        self.similar_texts: Dict[str, List[str]] = {}
        
        # Operations counter for auto-save
        self.operations_count = 0
        
        # Thread lock for thread safety
        self.lock = threading.RLock() if thread_safe else None
        
        # Environment info for version tracking
        self.environment_info = self._get_environment_info()
        
        # Load cache from disk if available
        self._load_cache()
        
        # Record version info if tracking is enabled
        if version_tracking:
            self._record_version_info()
        
        logger.info(f"Initialized embedding cache v{VERSION} (max_size={max_size}, ttl={ttl}s, method={signature_method})")
    
    def _get_environment_info(self) -> Dict[str, str]:
        """Get information about the runtime environment."""
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "processor": platform.processor(),
            "sentence_transformers_version": SentenceTransformer.__version__ if hasattr(SentenceTransformer, "__version__") else "unknown"
        }
    
    def _compute_content_features(self, text: str) -> str:
        """
        Compute statistical features of text for robust signature.
        
        Args:
            text: Text to analyze
            
        Returns:
            Feature signature string
        """
        # Skip processing for very short texts
        if len(text) < 10:
            return f"short_{len(text)}_{hashlib.md5(text.encode('utf-8')).hexdigest()[:16]}"
        
        # Extract features
        char_count = len(text)
        word_count = len(text.split())
        avg_word_len = char_count / max(1, word_count)
        
        # Character class distribution
        letter_count = sum(c.isalpha() for c in text)
        digit_count = sum(c.isdigit() for c in text)
        space_count = sum(c.isspace() for c in text)
        punct_count = char_count - letter_count - digit_count - space_count
        
        # Line statistics
        lines = text.split('\n')
        line_count = len(lines)
        avg_line_len = char_count / max(1, line_count)
        
        # Vocabulary richness (approximate)
        unique_words = len(set(w.lower() for w in re.findall(r'\b\w+\b', text)))
        vocab_richness = unique_words / max(1, word_count)
        
        # Structural elements
        has_html = '<' in text and '>' in text and ('</' in text or '/>' in text)
        has_json = ('{' in text and '}' in text) or ('[' in text and ']' in text)
        has_code = ('def ' in text or 'function(' in text or 'class ' in text or 
                   ';' in text or '{' in text or 'import ' in text)
        
        # Create feature vector
        feature_vector = [
            char_count, 
            word_count, 
            round(avg_word_len, 2),
            round(letter_count / max(1, char_count), 2),
            round(digit_count / max(1, char_count), 2),
            round(punct_count / max(1, char_count), 2),
            line_count, 
            round(avg_line_len, 2),
            round(vocab_richness, 2),
            1 if has_html else 0,
            1 if has_json else 0,
            1 if has_code else 0
        ]
        
        # Add first few and last few characters as anchors
        prefix = text[:20].replace('\n', ' ')
        suffix = text[-20:].replace('\n', ' ') if len(text) > 20 else ""
        
        # Create hash from features
        feature_str = f"{feature_vector}|{prefix}|{suffix}"
        feature_hash = hashlib.sha256(feature_str.encode('utf-8')).hexdigest()[:32]
        
        return feature_hash
    
    def _select_adaptive_method(self, text: str) -> SignatureMethod:
        """
        Adaptively select the best signature method based on content characteristics.
        
        Args:
            text: The text to analyze
            
        Returns:
            The selected signature method
        """
        # For very short texts, simple hash is sufficient
        if len(text) < 20:
            return SignatureMethod.SIMPLE_HASH
        
        # For very long texts, use content features
        if len(text) > 5000:
            return SignatureMethod.CONTENT_FEATURES
        
        # For structured content (likely code, JSON, etc.)
        if ('def ' in text or 'function(' in text or 'class ' in text or 
            '{' in text or '}' in text or '[' in text or ']' in text or
            '<' in text and '>' in text):
            return SignatureMethod.CONTENT_FEATURES
        
        # For natural language text
        words = text.split()
        if len(words) > 10 and max(len(w) for w in words) < 20:
            # Semantic fingerprint works better for natural language
            return SignatureMethod.SEMANTIC_FINGERPRINT
        
        # Default to content length for other cases
        return SignatureMethod.CONTENT_LENGTH
    
    def _compute_signature(self, text: str, model_name: str) -> str:
        """
        Compute a smart cache signature based on the selected method.
        
        Args:
            text: Text to compute signature for
            model_name: Model name to include in signature
            
        Returns:
            Cache signature string
        """
        start_time = time.time()
        
        # Adaptive method selection
        effective_method = self.signature_method
        if effective_method == SignatureMethod.ADAPTIVE:
            effective_method = self._select_adaptive_method(text)
        
        # Basic components included in all signatures
        model_hash = hashlib.md5(model_name.encode('utf-8')).hexdigest()[:8]
        
        if effective_method == SignatureMethod.SIMPLE_HASH:
            # Simple MD5 hash
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            
        elif effective_method == SignatureMethod.CONTENT_LENGTH:
            # Include content length in the hash calculation
            content_info = f"{len(text)}_{text[:50]}_{text[-50:] if len(text) > 50 else ''}"
            text_hash = hashlib.md5(content_info.encode('utf-8')).hexdigest()
            
        elif effective_method == SignatureMethod.SEMANTIC_FINGERPRINT:
            # Create a semantic fingerprint based on key terms
            # This is a simplified approximation - could be enhanced with NLP
            words = [w.lower() for w in text.split() if len(w) > 3]
            if len(words) > 10:
                # Use important words (beginning, middle, end)
                sample_words = words[:3] + words[len(words)//2-1:len(words)//2+2] + words[-3:]
                fingerprint = "_".join(sorted(sample_words))
            else:
                fingerprint = "_".join(sorted(words))
            
            # Combine with length for better uniqueness
            content_info = f"{len(text)}_{fingerprint}"
            text_hash = hashlib.md5(content_info.encode('utf-8')).hexdigest()
            
        elif effective_method == SignatureMethod.CONTENT_FEATURES:
            # Use statistical content features for robust signature
            text_hash = self._compute_content_features(text)
            
        else:
            # Fallback to simple hash
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Track method performance
        latency = time.time() - start_time
        if self.thread_safe:
            with self.lock:
                self.method_performance[effective_method]["calls"] += 1
                self.method_performance[effective_method]["latency"] += latency
        else:
            self.method_performance[effective_method]["calls"] += 1
            self.method_performance[effective_method]["latency"] += latency
        
        # Store hash bucket for similar text tracking
        short_hash = text_hash[:8]
        if self.thread_safe:
            with self.lock:
                if short_hash not in self.similar_texts:
                    self.similar_texts[short_hash] = []
                if text not in self.similar_texts[short_hash]:
                    self.similar_texts[short_hash].append(text)
        else:
            if short_hash not in self.similar_texts:
                self.similar_texts[short_hash] = []
            if text not in self.similar_texts[short_hash]:
                self.similar_texts[short_hash].append(text)
        
        # Combine with version for versioned caching
        return f"{text_hash}_{model_hash}_{self.version.split('.')[0]}"
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache if available and not expired.
        
        Args:
            text: Text to get embedding for
            model_name: Name of embedding model
            
        Returns:
            Embedding vector if in cache and valid, None otherwise
        """
        if not text:
            return None
            
        key = self._compute_signature(text, model_name)
        
        # Check if in memory cache
        if key in self.cache:
            embedding, timestamp, metadata = self.cache[key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl:
                # Expired, remove from cache
                logger.debug(f"Cache entry expired: {key}")
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                self.misses += 1
                return None
            
            # Update access time for LRU tracking
            self.access_times[key] = time.time()
            self.hits += 1
            
            return embedding
        
        self.misses += 1
        return None
    
    def put(
        self, 
        text: str, 
        model_name: str, 
        embedding: np.ndarray, 
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Store embedding in cache with optional metadata.
        
        Args:
            text: Text the embedding is for
            model_name: Name of embedding model
            embedding: Embedding vector to store
            metadata: Optional metadata to store with the embedding
        """
        if not text or embedding is None:
            return
            
        key = self._compute_signature(text, model_name)
        
        # Check cache size and evict if needed
        if len(self.cache) >= self.max_size:
            self._evict_entries()
        
        # Prepare metadata
        meta = metadata or {}
        meta.update({
            "model": model_name,
            "timestamp": time.time(),
            "version": self.version,
            "dimension": embedding.shape[0],
            "norm": float(np.linalg.norm(embedding)),
        })
        
        # Store in cache with current timestamp and metadata
        current_time = time.time()
        self.cache[key] = (embedding, current_time, meta)
        self.access_times[key] = current_time
        
        # Clear failure tracking for this text if it exists
        if text in self.failure_tracking and model_name in self.failure_tracking[text]:
            del self.failure_tracking[text][model_name]
            if not self.failure_tracking[text]:
                del self.failure_tracking[text]
        
        # Schedule periodic saving
        if len(self.cache) % 100 == 0:
            self._save_cache()
    
    def _evict_entries(self) -> None:
        """
        Evict cache entries using a smart strategy:
        1. First, remove expired entries
        2. Then, remove outlier entries (abnormally large vectors)
        3. Finally, use LRU policy for remaining evictions
        """
        # First try to evict expired entries
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp, _) in self.cache.items() 
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        # If we still need to evict, try to identify outlier entries
        if len(self.cache) >= self.max_size:
            # Look for vectors with unusually large norms (potential outliers)
            vector_norms = {
                key: float(np.linalg.norm(embedding))
                for key, (embedding, _, _) in self.cache.items()
            }
            
            # Simple outlier detection - vectors with norms > 2x median
            if vector_norms:
                median_norm = sorted(vector_norms.values())[len(vector_norms) // 2]
                outlier_keys = [
                    key for key, norm in vector_norms.items()
                    if norm > 2.0 * median_norm and norm > 1.5  # Must be significantly larger
                ]
                
                # Remove up to 5% of cache as outliers
                max_outliers = max(1, self.max_size // 20)
                for key in outlier_keys[:max_outliers]:
                    if key in self.cache:
                        del self.cache[key]
                    if key in self.access_times:
                        del self.access_times[key]
                
                if outlier_keys[:max_outliers]:
                    logger.debug(f"Evicted {len(outlier_keys[:max_outliers])} outlier entries")
        
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
        """Save cache to disk with versioning."""
        try:
            # Create a copy to avoid serialization issues during concurrent access
            cache_copy = {k: v for k, v in self.cache.items()}
            
            # Create a version-specific cache file
            with open(self.cache_path, 'wb') as f:
                pickle.dump({
                    "cache": cache_copy,
                    "access_times": self.access_times,
                    "hits": self.hits,
                    "misses": self.misses,
                    "version": self.version,
                    "timestamp": time.time(),
                    "signature_method": self.signature_method
                }, f)
                
            logger.info(f"Saved {len(cache_copy)} embeddings to disk cache (version {self.version})")
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {str(e)}")
            logger.debug(traceback.format_exc())
    
    def _load_cache(self) -> None:
        """Load cache from disk with version compatibility check."""
        try:
            if not os.path.exists(self.cache_path):
                logger.info(f"No disk cache found for version {self.version}, starting with empty cache")
                return
                
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Handle both old and new format
            if isinstance(cache_data, tuple):
                # Old format
                loaded_cache, loaded_access_times = cache_data
                loaded_hits = 0
                loaded_misses = 0
                loaded_version = "1.0.0"
                
                # Convert to new format
                converted_cache = {
                    key: (embedding, timestamp, {"version": loaded_version})
                    for key, (embedding, timestamp) in loaded_cache.items()
                }
                loaded_cache = converted_cache
            else:
                # New format
                loaded_cache = cache_data.get("cache", {})
                loaded_access_times = cache_data.get("access_times", {})
                loaded_hits = cache_data.get("hits", 0)
                loaded_misses = cache_data.get("misses", 0)
                loaded_version = cache_data.get("version", "1.0.0")
            
            # Only use cache if compatible with current version
            current_major = self.version.split('.')[0]
            loaded_major = loaded_version.split('.')[0]
            
            if current_major != loaded_major:
                logger.warning(f"Cache version mismatch: {loaded_version} vs {self.version}. Starting with empty cache.")
                return
            
            # Filter out expired entries
            current_time = time.time()
            valid_entries = {
                key: entry
                for key, entry in loaded_cache.items()
                if current_time - entry[1] <= self.ttl
            }
            
            valid_access_times = {
                key: access_time
                for key, access_time in loaded_access_times.items()
                if key in valid_entries
            }
            
            self.cache = valid_entries
            self.access_times = valid_access_times
            self.hits = loaded_hits
            self.misses = loaded_misses
            
            logger.info(f"Loaded {len(self.cache)} embeddings from disk cache (version {loaded_version})")
            
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {str(e)}")
            logger.debug(traceback.format_exc())
            self.cache = {}
            self.access_times = {}
    
    def _record_version_info(self) -> None:
        """Record version information for tracking and reproducibility."""
        if not self.version_tracking:
            return
            
        try:
            # Load existing version log if available
            version_log = {}
            if os.path.exists(self.version_log_path):
                with open(self.version_log_path, 'r') as f:
                    version_log = json.load(f)
            
            # Add current version info
            timestamp = datetime.now().isoformat()
            version_log[self.version] = {
                "timestamp": timestamp,
                "signature_method": self.signature_method,
                "cache_size": len(self.cache),
                "stats": self.get_stats()
            }
            
            # Save updated version log
            with open(self.version_log_path, 'w') as f:
                json.dump(version_log, f, indent=2)
                
            logger.debug(f"Recorded version info for embedding cache v{self.version}")
            
        except Exception as e:
            logger.warning(f"Failed to record version info: {str(e)}")
    
    def record_failure(self, text: str, model_name: str) -> int:
        """
        Record a failure to generate embedding for tracking purposes.
        
        Args:
            text: Text that failed to embed
            model_name: Model that failed
            
        Returns:
            Number of failures for this text/model
        """
        if text not in self.failure_tracking:
            self.failure_tracking[text] = {}
            
        if model_name not in self.failure_tracking[text]:
            self.failure_tracking[text][model_name] = 0
            
        self.failure_tracking[text][model_name] += 1
        return self.failure_tracking[text][model_name]
    
    def get_failure_count(self, text: str, model_name: str) -> int:
        """Get the number of times this text/model has failed to embed."""
        if text not in self.failure_tracking:
            return 0
        return self.failure_tracking[text].get(model_name, 0)
    
    def clear(self, version_info: bool = False) -> None:
        """
        Clear the cache.
        
        Args:
            version_info: Whether to also clear version information
        """
        self.cache.clear()
        self.access_times.clear()
        self.failure_tracking.clear()
        self.hits = 0
        self.misses = 0
        
        if version_info and os.path.exists(self.version_log_path):
            os.remove(self.version_log_path)
            
        logger.info(f"Embedding cache cleared (version_info={version_info})")
    
    def get_version_history(self) -> Dict[str, Any]:
        """Get version history information from the log."""
        if not os.path.exists(self.version_log_path):
            return {}
            
        try:
            with open(self.version_log_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load version history: {str(e)}")
            return {}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        current_time = time.time()
        
        # Calculate statistics on vector norms
        norms = [float(np.linalg.norm(embedding)) for embedding, _, _ in self.cache.values()]
        avg_norm = sum(norms) / max(1, len(norms)) if norms else 0
        
        # Calculate model distribution
        model_counts = {}
        for _, _, metadata in self.cache.values():
            model = metadata.get("model", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "hits": self.hits,
            "misses": self.misses,
            "hit_ratio": self.hits / max(1, self.hits + self.misses),
            "expired_count": sum(1 for _, timestamp, _ in self.cache.values() if current_time - timestamp > self.ttl),
            "avg_age": sum(current_time - timestamp for _, timestamp, _ in self.cache.values()) / max(1, len(self.cache)),
            "avg_vector_norm": avg_norm,
            "failure_count": sum(sum(counts.values()) for counts in self.failure_tracking.values()),
            "version": self.version,
            "signature_method": self.signature_method,
            "model_distribution": model_counts,
        }


# Global instances
_models: Dict[str, SentenceTransformer] = {}
_embedding_cache = EmbeddingCache()
_model_locks: Dict[str, bool] = {}  # Simple locking mechanism

# Type for model version information
class ModelVersionInfo(dict):
    """Structure for tracking model version information."""
    pass

# Global registry of model version info
_model_versions: Dict[str, ModelVersionInfo] = {}


def load_model(
    model_name: str = DEFAULT_MODEL, 
    fallback: bool = True,
    record_version: bool = True
) -> SentenceTransformer:
    """
    Load the sentence transformer model with caching and versioning.
    
    Args:
        model_name: Name of the model to load
        fallback: Whether to try fallback models if loading fails
        record_version: Whether to record version information
        
    Returns:
        Loaded SentenceTransformer model
    """
    global _models, _model_versions, _model_locks
    
    # Check if model is already loaded
    if model_name in _models:
        return _models[model_name]
    
    # Check if model is currently being loaded by another thread
    if model_name in _model_locks and _model_locks[model_name]:
        # Try a few times with small delays
        retries = 3
        while retries > 0 and _model_locks.get(model_name, False):
            time.sleep(0.5)
            retries -= 1
            if model_name in _models:
                return _models[model_name]
        
        # If still locked, use fallback
        if fallback and model_name != DEFAULT_MODEL:
            logger.warning(f"Model {model_name} is locked for loading, using default model")
            return load_model(DEFAULT_MODEL, fallback=False)
    
    # Set lock
    _model_locks[model_name] = True
    
    try:
        logger.info(f"Loading embedding model: {model_name}")
        start_time = time.time()
        
        # Load the model
        model = SentenceTransformer(model_name)
        load_time = time.time() - start_time
        logger.info(f"Model {model_name} loaded in {load_time:.2f} seconds")
        
        # Store model in cache
        _models[model_name] = model
        
        # Record version information
        if record_version:
            version_info = ModelVersionInfo({
                "load_time": datetime.now().isoformat(),
                "model_name": model_name,
                "dimension": model.get_sentence_embedding_dimension(),
                "version": VERSION,
                "load_duration_seconds": load_time
            })
            _model_versions[model_name] = version_info
            
            # Save to disk for tracking
            try:
                version_path = os.path.join(VERSION_DIR, f"model_{model_name.replace('/', '_')}.json")
                with open(version_path, 'w') as f:
                    json.dump(version_info, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to save model version info: {str(e)}")
        
        return model
            
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Try fallback models
        if fallback:
            if model_name != DEFAULT_MODEL:
                logger.info(f"Falling back to default model: {DEFAULT_MODEL}")
                _model_locks[model_name] = False
                return load_model(DEFAULT_MODEL, fallback=True)
            else:
                # Try other fallback models
                for fallback_model in FALLBACK_MODELS:
                    try:
                        logger.info(f"Trying fallback model: {fallback_model}")
                        _model_locks[model_name] = False
                        return load_model(fallback_model, fallback=False)
                    except Exception:
                        continue
                
                # If all else fails, raise the original error
                raise
        else:
            raise
    finally:
        # Release lock
        _model_locks[model_name] = False


def find_similar_text_embedding(
    text: str,
    model_name: str,
    threshold: float = 0.8
) -> Optional[Tuple[str, np.ndarray]]:
    """
    Find a similar text that has been successfully embedded previously.
    
    Args:
        text: Text to find similar texts for
        model_name: Model name to use for lookup
        threshold: Similarity threshold for considering texts similar
        
    Returns:
        Tuple of (similar_text, embedding) if found, None otherwise
    """
    # Use the cache's similar_texts feature to find potential candidates
    text_hash = _embedding_cache._compute_signature(text, model_name)
    short_hash = text_hash[:8]
    
    # Look for exact signature match first (would be unusual but possible)
    exact_match = _embedding_cache.get(text, model_name)
    if exact_match is not None:
        return (text, exact_match)
    
    # Check for similar texts in the same hash bucket
    if short_hash in _embedding_cache.similar_texts:
        candidates = _embedding_cache.similar_texts[short_hash]
        
        # Simple similarity check based on length and character overlap
        def text_similarity(a: str, b: str) -> float:
            # Length similarity
            len_sim = min(len(a), len(b)) / max(len(a), len(b))
            
            # Character overlap (basic n-gram approach)
            a_set = set(a[i:i+3] for i in range(len(a)-2))
            b_set = set(b[i:i+3] for i in range(len(b)-2))
            
            if not a_set or not b_set:
                return len_sim
            
            overlap = len(a_set.intersection(b_set)) / max(1, min(len(a_set), len(b_set)))
            return (len_sim * 0.3) + (overlap * 0.7)
        
        # Find the most similar text
        best_match = None
        best_sim = 0
        
        for candidate in candidates:
            if candidate == text:
                continue
                
            sim = text_similarity(text, candidate)
            if sim > best_sim:
                best_sim = sim
                best_match = candidate
        
        # If we found a good match, return its embedding
        if best_match and best_sim >= threshold:
            embedding = _embedding_cache.get(best_match, model_name)
            if embedding is not None:
                return (best_match, embedding)
    
    return None

def get_embedding(
    text: str, 
    normalize: bool = True, 
    model_name: str = None,
    fallback_models: List[str] = None,
    retry_count: int = 2,
    record_failures: bool = True,
    metadata: Dict[str, Any] = None,
    recovery_mode: RecoveryMode = RecoveryMode.FULL_RECOVERY,
    similarity_threshold: float = 0.8,
    force_recompute: bool = False,
    retries_with_backoff: int = 3,
    backoff_factor: float = 2.0
) -> np.ndarray:
    """
    Get embedding vector for text with comprehensive caching and recovery mechanisms.
    
    This enhanced version supports multiple recovery strategies, auto-adaptive
    retries, similar text fallbacks, and detailed version tracking.
    
    Args:
        text: Text to embed
        normalize: Whether to normalize the embedding vector
        model_name: Optional model name to use
        fallback_models: List of models to try if primary model fails
        retry_count: Number of times to retry immediately before trying recovery
        record_failures: Whether to record embedding failures
        metadata: Optional metadata to store with the embedding
        recovery_mode: Strategy to use for recovering from failures
        similarity_threshold: Threshold for considering texts similar for fallback
        force_recompute: Whether to bypass cache and force recomputation
        retries_with_backoff: Number of retries with exponential backoff
        backoff_factor: Factor for exponential backoff (wait time multiplier)
        
    Returns:
        Embedding vector as numpy array
    """
    # Handle empty text
    if not text:
        logger.warning("Empty text provided for embedding")
        # Return zero vector of appropriate dimension
        model = load_model(model_name or DEFAULT_MODEL)
        dim = model.get_sentence_embedding_dimension()
        return np.zeros(dim)
    
    # Use provided model name or default
    selected_model = model_name or DEFAULT_MODEL
    
    # Check cache first (unless forced to recompute)
    if not force_recompute:
        cached_embedding = _embedding_cache.get(text, selected_model)
        if cached_embedding is not None:
            return cached_embedding
    
    # Process fallback models list
    if fallback_models is None:
        # Use default fallbacks if none provided
        fallback_models = [DEFAULT_MODEL] + FALLBACK_MODELS if selected_model != DEFAULT_MODEL else FALLBACK_MODELS
    elif selected_model in fallback_models:
        # Remove the selected model from fallbacks if it's included
        fallback_models = [m for m in fallback_models if m != selected_model]
    
    # Setup metadata
    meta = metadata or {}
    meta.update({
        "generation_time": datetime.now().isoformat(),
        "version": VERSION,
        "recovery_mode": recovery_mode,
        "environment": _embedding_cache.environment_info
    })
    
    # Track failures for the primary model
    failure_count = _embedding_cache.get_failure_count(text, selected_model)
    failure_threshold = 3  # Threshold for trying alternative approaches
    
    # If this text has consistently failed with this model, try a different approach
    if failure_count >= failure_threshold and not force_recompute:
        logger.warning(f"Text has failed {failure_count} times with model {selected_model}")
        
        # Try a different recovery strategy based on the recovery mode
        if recovery_mode in (RecoveryMode.FALLBACK_MODELS, RecoveryMode.FULL_RECOVERY) and fallback_models:
            logger.info(f"Using fallback model: {fallback_models[0]}")
            meta["used_fallback"] = True
            meta["fallback_reason"] = f"previous_failures_{failure_count}"
            meta["original_model"] = selected_model
            
            return get_embedding(
                text, normalize, fallback_models[0], 
                fallback_models[1:], retry_count, record_failures, meta,
                recovery_mode, similarity_threshold, force_recompute
            )
        
        # Try using a similar text that succeeded previously
        if recovery_mode in (RecoveryMode.CACHED_SIMILAR, RecoveryMode.FULL_RECOVERY):
            similar = find_similar_text_embedding(text, selected_model, similarity_threshold)
            if similar:
                similar_text, similar_embedding = similar
                logger.info(f"Using embedding from similar text (similarity >= {similarity_threshold})")
                
                # Clone the embedding to avoid accidental modification
                embedding_copy = similar_embedding.copy()
                
                # Add metadata about this recovery
                sim_meta = meta.copy()
                sim_meta.update({
                    "recovered_from_similar": True,
                    "similar_text_hash": hashlib.md5(similar_text.encode('utf-8')).hexdigest()[:16],
                    "similarity_threshold": similarity_threshold,
                    "recovery_timestamp": time.time()
                })
                
                # Store in cache with recovery metadata
                _embedding_cache.put(text, selected_model, embedding_copy, sim_meta)
                
                return embedding_copy
    
    # Try to get the embedding with multiple layers of recovery
    try:
        # Load the selected model
        model = load_model(selected_model)
        
        # Try multiple times with immediate retries
        for attempt in range(retry_count):
            try:
                embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=normalize)
                
                # Add to cache with metadata
                meta["attempts"] = attempt + 1
                meta["embedding_source"] = "direct"
                _embedding_cache.put(text, selected_model, embedding, meta)
                
                return embedding
            except Exception as retry_err:
                if attempt < retry_count - 1:
                    logger.warning(f"Embedding attempt {attempt+1}/{retry_count} failed: {str(retry_err)}. Retrying...")
                    time.sleep(0.5)  # Small delay before retry
                else:
                    # Last regular attempt failed, try recovery
                    raise retry_err
        
    except Exception as first_error:
        error_msg = f"Error generating embedding for text with model {selected_model}: {str(first_error)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        
        # Record failure if tracking is enabled
        if record_failures:
            _embedding_cache.record_failure(text, selected_model, str(first_error))
        
        # Apply recovery strategies based on the selected mode
        recovery_attempted = False
        
        # 1. Retry with exponential backoff
        if recovery_mode in (RecoveryMode.RETRY_WITH_BACKOFF, RecoveryMode.FULL_RECOVERY):
            recovery_attempted = True
            
            for backoff_attempt in range(retries_with_backoff):
                # Calculate wait time with exponential backoff
                wait_time = (backoff_factor ** backoff_attempt) * 0.5
                logger.info(f"Retrying with backoff: waiting {wait_time:.2f}s before attempt {backoff_attempt+1}/{retries_with_backoff}")
                
                try:
                    time.sleep(wait_time)
                    model = load_model(selected_model, fallback=False)  # Try to reload model
                    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=normalize)
                    
                    # Success after backoff
                    meta["recovery_method"] = "backoff_retry"
                    meta["backoff_attempts"] = backoff_attempt + 1
                    meta["embedding_source"] = "backoff_recovery"
                    _embedding_cache.put(text, selected_model, embedding, meta)
                    
                    return embedding
                except Exception as backoff_err:
                    logger.warning(f"Backoff retry {backoff_attempt+1} failed: {str(backoff_err)}")
                    # Continue to next attempt or strategy
        
        # 2. Try fallback models
        if recovery_mode in (RecoveryMode.FALLBACK_MODELS, RecoveryMode.FULL_RECOVERY) and fallback_models:
            recovery_attempted = True
            fallback_model = fallback_models[0]
            
            logger.info(f"Trying fallback model: {fallback_model}")
            meta["used_fallback"] = True
            meta["fallback_reason"] = "primary_model_failure"
            meta["original_model"] = selected_model
            meta["fallback_tier"] = 1
            
            try:
                return get_embedding(
                    text, normalize, fallback_model, 
                    fallback_models[1:], retry_count, record_failures, meta,
                    recovery_mode, similarity_threshold, False
                )
            except Exception as fallback_err:
                logger.warning(f"Fallback model {fallback_model} also failed: {str(fallback_err)}")
                # Fall through to next strategy
        
        # 3. Try finding a similar text in cache
        if recovery_mode in (RecoveryMode.CACHED_SIMILAR, RecoveryMode.FULL_RECOVERY):
            recovery_attempted = True
            
            similar = find_similar_text_embedding(text, selected_model, similarity_threshold)
            if similar:
                similar_text, similar_embedding = similar
                logger.info(f"Using embedding from similar text after failures")
                
                # Clone the embedding to avoid accidental modification
                embedding_copy = similar_embedding.copy()
                
                # Add metadata about this recovery
                sim_meta = meta.copy()
                sim_meta.update({
                    "recovered_from_similar": True,
                    "recovery_method": "similar_text",
                    "similar_text_hash": hashlib.md5(similar_text.encode('utf-8')).hexdigest()[:16],
                    "similarity_threshold": similarity_threshold,
                    "recovery_timestamp": time.time(),
                    "embedding_source": "similarity_recovery"
                })
                
                # Store in cache with recovery metadata
                _embedding_cache.put(text, selected_model, embedding_copy, sim_meta)
                
                return embedding_copy
        
        # 4. Warning-only mode just logs the error and returns None
        if recovery_mode == RecoveryMode.WARNING_ONLY:
            logger.warning(f"Embedding generation failed and WARNING_ONLY recovery mode is set. Returning None.")
            return None
        
        # 5. If all recovery methods failed or none attempted, return zero vector
        if not recovery_attempted or recovery_mode in (RecoveryMode.ZERO_VECTOR, RecoveryMode.FULL_RECOVERY):
            # Try to get the correct dimension
            try:
                model = load_model(selected_model)
                dim = model.get_sentence_embedding_dimension()
            except Exception:
                # If we can't load the model, try a fallback model for dimension
                try:
                    if fallback_models:
                        model = load_model(fallback_models[0])
                        dim = model.get_sentence_embedding_dimension()
                    else:
                        # Use a standard dimension as last resort
                        dim = 384  # Common dimension for many models
                except Exception:
                    dim = 384
            
            # Create a zero vector
            zero_vector = np.zeros(dim)
            
            # Add metadata about this recovery
            zero_meta = meta.copy()
            zero_meta.update({
                "recovery_method": "zero_vector",
                "embedding_source": "recovery_fallback",
                "vector_dimension": dim,
                "recovery_timestamp": time.time(),
                "is_fallback": True
            })
            
            # Store in cache with fallback metadata
            _embedding_cache.put(text, selected_model, zero_vector, zero_meta)
            
            logger.warning(f"All recovery methods failed, returning zero vector with dimension {dim}")
            return zero_vector
        
        # This should not happen, but just in case
        logger.error("Unexpected flow in embedding recovery - returning None")
        return None


def batch_embed(
    texts: List[str], 
    normalize: bool = True, 
    model_name: str = None,
    fallback_models: List[str] = None,
    batch_size: int = MAX_BATCH_SIZE,
    retry_count: int = 1,
    record_failures: bool = True,
    metadata: Dict[str, Any] = None
) -> Dict[str, np.ndarray]:
    """
    Create embeddings for multiple texts efficiently with enhanced caching and failure recovery.
    
    Args:
        texts: List of texts to embed
        normalize: Whether to normalize the embeddings
        model_name: Optional model name to use
        fallback_models: List of models to try if primary model fails
        batch_size: Size of batches for processing
        retry_count: Number of times to retry failed embeddings
        record_failures: Whether to record embedding failures
        metadata: Optional metadata to store with embeddings
        
    Returns:
        Dictionary mapping texts to embedding vectors
    """
    if not texts:
        return {}
    
    # Use provided model name or default
    selected_model = model_name or DEFAULT_MODEL
    
    # Setup metadata
    meta = metadata or {}
    meta.update({
        "generation_time": datetime.now().isoformat(),
        "version": VERSION,
        "batch_size": min(batch_size, MAX_BATCH_SIZE),
    })
    
    try:
        # Filter out empty texts and duplicates while preserving order
        unique_texts = []
        seen = set()
        for text in texts:
            if text and text not in seen:
                unique_texts.append(text)
                seen.add(text)
        
        # Check which texts are already cached
        cached_embeddings = {}
        texts_to_embed = []
        
        for text in unique_texts:
            cached_embedding = _embedding_cache.get(text, selected_model)
            if cached_embedding is not None:
                cached_embeddings[text] = cached_embedding
            else:
                # Check if this text has repeatedly failed with this model
                failure_count = _embedding_cache.get_failure_count(text, selected_model)
                if failure_count >= 3 and fallback_models:
                    # If this text has failed multiple times, try with first fallback model
                    fallback_embed = get_embedding(
                        text, normalize, fallback_models[0], 
                        fallback_models[1:], retry_count, record_failures, meta
                    )
                    cached_embeddings[text] = fallback_embed
                else:
                    texts_to_embed.append(text)
        
        # If we have any uncached texts, compute their embeddings
        if texts_to_embed:
            try:
                model = load_model(selected_model)
                
                # Process in batches to avoid memory issues with large datasets
                all_new_embeddings = {}
                failed_texts = []
                
                for i in range(0, len(texts_to_embed), min(batch_size, MAX_BATCH_SIZE)):
                    batch = texts_to_embed[i:i+min(batch_size, MAX_BATCH_SIZE)]
                    
                    # Try multiple times if needed
                    for attempt in range(retry_count):
                        try:
                            batch_embeddings = model.encode(
                                batch, 
                                convert_to_numpy=True, 
                                normalize_embeddings=normalize
                            )
                            
                            # Map embeddings back to texts
                            for j, text in enumerate(batch):
                                all_new_embeddings[text] = batch_embeddings[j]
                                batch_meta = meta.copy()
                                batch_meta["batch_index"] = i // batch_size
                                batch_meta["attempts"] = attempt + 1
                                _embedding_cache.put(text, selected_model, batch_embeddings[j], batch_meta)
                                
                            break  # Success, exit retry loop
                            
                        except Exception as batch_err:
                            if attempt < retry_count - 1:
                                logger.warning(f"Batch embedding attempt {attempt+1}/{retry_count} "
                                              f"failed for batch {i//batch_size}: {str(batch_err)}. Retrying...")
                                time.sleep(1)  # Delay before retry
                            else:
                                # Last attempt failed, record failures and try individually
                                logger.error(f"Batch embedding failed after {retry_count} attempts: {str(batch_err)}")
                                failed_texts.extend(batch)
                
                # Try failed texts individually
                if failed_texts:
                    logger.info(f"Trying to embed {len(failed_texts)} failed texts individually")
                    for text in failed_texts:
                        try:
                            # Try with individual embedding function that has its own fallback logic
                            embedding = get_embedding(
                                text, normalize, selected_model, 
                                fallback_models, retry_count, record_failures, meta
                            )
                            all_new_embeddings[text] = embedding
                        except Exception as individual_err:
                            logger.error(f"Individual embedding also failed for text: {str(individual_err)}")
                            if record_failures:
                                _embedding_cache.record_failure(text, selected_model)
                
                # Combine with cached embeddings
                cached_embeddings.update(all_new_embeddings)
                
            except Exception as model_err:
                logger.error(f"Error with model {selected_model}: {str(model_err)}")
                logger.debug(traceback.format_exc())
                
                # Try fallback model for all remaining texts
                if fallback_models:
                    fallback_model = fallback_models[0]
                    logger.info(f"Trying fallback model {fallback_model} for {len(texts_to_embed)} texts")
                    fallback_results = batch_embed(
                        texts_to_embed, normalize, fallback_model,
                        fallback_models[1:], batch_size, retry_count, record_failures, meta
                    )
                    cached_embeddings.update(fallback_results)
        
        # Create the final result dictionary, maintaining original order
        result = {}
        for text in texts:
            if text in cached_embeddings:
                result[text] = cached_embeddings[text]
        
        return result
        
    except Exception as e:
        logger.error(f"Critical error in batch_embed: {str(e)}")
        logger.debug(traceback.format_exc())
        
        # Try fallback model for everything
        if fallback_models:
            fallback_model = fallback_models[0]
            logger.info(f"Attempting complete fallback to model {fallback_model}")
            return batch_embed(
                texts, normalize, fallback_model,
                fallback_models[1:], batch_size, retry_count, record_failures, meta
            )
        
        return {}


def cosine_similarity(
    a: np.ndarray, 
    b: np.ndarray, 
    fallback_value: float = 0.0
) -> float:
    """
    Calculate cosine similarity between two vectors with enhanced error handling.
    
    Args:
        a: First vector
        b: Second vector
        fallback_value: Value to return in case of error
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    try:
        # Check if inputs are valid numpy arrays
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            logger.warning(f"Invalid input types for cosine similarity: {type(a)} and {type(b)}")
            return fallback_value
            
        # Check for dimension mismatch
        if a.shape != b.shape:
            logger.warning(f"Dimension mismatch in cosine similarity: {a.shape} vs {b.shape}")
            # If shapes are different but sizes match, try to reshape
            if a.size == b.size:
                a = a.reshape(-1)
                b = b.reshape(-1)
            else:
                return fallback_value
        
        # Handle zero vectors
        if np.all(a == 0) or np.all(b == 0):
            return fallback_value
            
        # Check for NaN values
        if np.isnan(a).any() or np.isnan(b).any():
            logger.warning("NaN values detected in vectors, using fallback value")
            return fallback_value
            
        # Calculate cosine similarity with numerical stability
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        
        if a_norm < 1e-10 or b_norm < 1e-10:
            return fallback_value
            
        similarity = float(np.dot(a, b) / (a_norm * b_norm + 1e-8))
        
        # Ensure result is in valid range
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        logger.debug(traceback.format_exc())
        return fallback_value


def semantic_search(
    query: str, 
    candidates: List[str], 
    threshold: float = 0.6, 
    model_name: str = None,
    fallback_models: List[str] = None,
    retry_count: int = 1,
    max_results: int = 100,
    include_metadata: bool = False
) -> List[Tuple[int, str, float, Optional[Dict[str, Any]]]]:
    """
    Find semantically similar items from a list of candidates with enhanced reliability.
    
    Args:
        query: The search query
        candidates: List of text items to search through
        threshold: Minimum similarity score to include in results
        model_name: Optional model name to use
        fallback_models: Models to try if primary model fails
        retry_count: Number of times to retry embedding
        max_results: Maximum number of results to return
        include_metadata: Whether to include embedding metadata in results
        
    Returns:
        List of tuples containing (index, text, score, metadata) sorted by similarity score
    """
    if not query or not candidates:
        return []
        
    try:
        # Get query embedding with fallback options
        query_embedding = get_embedding(
            query, 
            model_name=model_name,
            fallback_models=fallback_models,
            retry_count=retry_count,
            metadata={"query": True, "search_timestamp": time.time()}
        )
        
        # Get embeddings for all candidates
        candidate_embeddings = batch_embed(
            candidates, 
            model_name=model_name,
            fallback_models=fallback_models,
            retry_count=retry_count,
            batch_size=min(MAX_BATCH_SIZE, 32),  # Smaller batch size for search
            metadata={"is_candidate": True, "search_timestamp": time.time()}
        )
        
        results = []
        
        for i, text in enumerate(candidates):
            # Skip empty candidates or those that failed to embed
            if not text or text not in candidate_embeddings:
                continue
                
            # Calculate similarity
            similarity = cosine_similarity(query_embedding, candidate_embeddings[text])
            
            # Add to results if above threshold
            if similarity >= threshold:
                # Get embedding metadata if available and requested
                metadata = None
                if include_metadata:
                    # Try to get metadata - not critical if this fails
                    try:
                        # We'd need to extend the cache to support this, just a placeholder
                        key = _embedding_cache._compute_signature(text, model_name or DEFAULT_MODEL)
                        if key in _embedding_cache.cache:
                            _, _, metadata = _embedding_cache.cache[key]
                    except Exception:
                        pass
                        
                # Add result with optional metadata
                if include_metadata:
                    results.append((i, text, similarity, metadata))
                else:
                    results.append((i, text, similarity, None))
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Limit to max_results
        return results[:max_results]
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        logger.debug(traceback.format_exc())
        return []


# Additional utility functions

def get_model_dimensions() -> Dict[str, int]:
    """
    Get dimensions for all loaded models.
    
    Returns:
        Dictionary mapping model names to their embedding dimensions
    """
    dimensions = {}
    for model_name, model in _models.items():
        try:
            dimensions[model_name] = model.get_sentence_embedding_dimension()
        except Exception:
            dimensions[model_name] = -1  # Error indicator
    return dimensions


def get_version_info() -> Dict[str, Any]:
    """
    Get version information for the embedding utilities.
    
    Returns:
        Dictionary containing version metadata
    """
    return {
        "version": VERSION,
        "models_loaded": list(_models.keys()),
        "cache_stats": _embedding_cache.get_stats(),
        "default_model": DEFAULT_MODEL,
        "fallback_models": FALLBACK_MODELS,
        "model_versions": _model_versions,
        "version_history": _embedding_cache.get_version_history()
    }


def clear_cache(clear_version_info: bool = False) -> None:
    """
    Clear the embedding cache.
    
    Args:
        clear_version_info: Whether to also clear version tracking information
    """
    _embedding_cache.clear(version_info=clear_version_info)
    logger.info(f"Cleared embedding cache (version_info={clear_version_info})")


@register_plugin(
    plugin_type=PluginType.VECTOR_ALIGNMENT,
    name="sentence_transformer_aligner",
    version=VERSION,  # Use the global version
    description="Advanced vector alignment component with multi-strategy recovery and version tracking",
    interfaces=["VectorAlignmentInterface"],
    is_default=True
)
class SentenceTransformerAligner(VectorAlignmentInterface):
    """
    Advanced vector alignment component using SentenceTransformer models.
    
    This component provides embedding generation, similarity calculation,
    and semantic search functionality with comprehensive caching, multi-strategy
    recovery, version tracking, and configurable performance options.
    """
    
    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL, 
        normalize: bool = True,
        fallback_models: List[str] = None,
        signature_method: SignatureMethod = SignatureMethod.ADAPTIVE,
        recovery_mode: RecoveryMode = RecoveryMode.FULL_RECOVERY,
        retry_count: int = 2,
        thread_safe: bool = True,
        cache_ttl: int = CACHE_TTL,
        similarity_threshold: float = 0.8,
        retries_with_backoff: int = 3,
        version_tracking: bool = True
    ):
        """
        Initialize the advanced vector alignment component.
        
        Args:
            model_name: Name of the embedding model to use
            normalize: Whether to normalize embeddings by default
            fallback_models: List of models to try if primary model fails
            signature_method: Method for computing cache signatures
            recovery_mode: Strategy to use for recovering from failures
            retry_count: Number of times to retry failed operations
            thread_safe: Whether to use thread-safe operations
            cache_ttl: Time-to-live for cache entries in seconds
            similarity_threshold: Threshold for similar text recovery
            retries_with_backoff: Number of retries with exponential backoff
            version_tracking: Whether to track version information
        """
        self.model_name = model_name
        self.normalize = normalize
        self.fallback_models = fallback_models or FALLBACK_MODELS
        self.signature_method = signature_method
        self.recovery_mode = recovery_mode
        self.retry_count = retry_count
        self.thread_safe = thread_safe
        self.cache_ttl = cache_ttl
        self.similarity_threshold = similarity_threshold
        self.retries_with_backoff = retries_with_backoff
        self.version_tracking = version_tracking
        
        # Configuration for the component
        self.config = {
            "model_name": model_name,
            "normalize": normalize,
            "fallback_models": self.fallback_models,
            "version": VERSION,
            "signature_method": signature_method,
            "recovery_mode": recovery_mode,
            "retry_count": retry_count,
            "thread_safe": thread_safe,
            "cache_ttl": cache_ttl,
            "similarity_threshold": similarity_threshold,
            "retries_with_backoff": retries_with_backoff,
            "version_tracking": version_tracking,
            "initialization_time": datetime.now().isoformat(),
            "environment": _embedding_cache.environment_info
        }
        
        # Performance statistics
        self.stats = {
            "embeddings_generated": 0,
            "total_latency": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "recovery_counts": {r.value: 0 for r in RecoveryMode},
            "errors": 0
        }
        
        # Load model on initialization to avoid lazy loading delays
        try:
            load_model(model_name)
        except Exception as e:
            logger.warning(f"Failed to preload model {model_name}: {str(e)}")
            # We'll try fallbacks later when needed
        
        logger.info(f"Initialized SentenceTransformerAligner v{VERSION} with model {model_name} and {recovery_mode} recovery")
    
    def get_embedding(
        self, 
        text: str, 
        normalize: bool = None,
        retry_count: int = None,
        recovery_mode: RecoveryMode = None,
        metadata: Dict[str, Any] = None,
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        Generate embedding vector for input text with comprehensive recovery strategies.
        
        Args:
            text: The text to create an embedding for
            normalize: Whether to normalize the embedding vector
            retry_count: Number of times to retry if embedding fails
            recovery_mode: Strategy to use for recovering from failures
            metadata: Additional metadata to store with the embedding
            force_recompute: Whether to bypass cache and force recomputation
            
        Returns:
            Embedding vector as numpy array
        """
        start_time = time.time()
        
        # Use instance defaults if not specified
        norm = self.normalize if normalize is None else normalize
        tries = retry_count if retry_count is not None else self.retry_count
        mode = recovery_mode if recovery_mode is not None else self.recovery_mode
        
        # Add component identifier to metadata
        meta = metadata or {}
        meta.update({
            "component": "SentenceTransformerAligner",
            "config": {k: v for k, v in self.config.items() if k != "environment"},
            "component_version": VERSION,
            "request_timestamp": datetime.now().isoformat()
        })
        
        try:
            # Get embedding with comprehensive recovery strategies
            embedding = get_embedding(
                text, 
                normalize=norm, 
                model_name=self.model_name,
                fallback_models=self.fallback_models,
                retry_count=tries,
                metadata=meta,
                recovery_mode=mode,
                similarity_threshold=self.similarity_threshold,
                force_recompute=force_recompute,
                retries_with_backoff=self.retries_with_backoff
            )
            
            # Update stats
            if self.thread_safe:
                with threading.RLock():
                    self.stats["embeddings_generated"] += 1
                    self.stats["total_latency"] += time.time() - start_time
                    
                    # Check if this was a recovery
                    if "recovery_method" in meta:
                        self.stats["recovery_counts"][mode.value] += 1
            else:
                self.stats["embeddings_generated"] += 1
                self.stats["total_latency"] += time.time() - start_time
                
                # Check if this was a recovery
                if "recovery_method" in meta:
                    self.stats["recovery_counts"][mode.value] += 1
            
            return embedding
            
        except Exception as e:
            # Update error stats
            if self.thread_safe:
                with threading.RLock():
                    self.stats["errors"] += 1
            else:
                self.stats["errors"] += 1
                
            logger.error(f"Error in get_embedding: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # For WARNING_ONLY mode, just return None
            if mode == RecoveryMode.WARNING_ONLY:
                return None
                
            # For other modes, return zero vector as last resort
            dim = 384  # Default dimension as fallback
            try:
                model = load_model(self.model_name)
                dim = model.get_sentence_embedding_dimension()
            except Exception:
                pass
                
            return np.zeros(dim)
    
    def batch_embed(
        self, 
        texts: List[str], 
        normalize: bool = None,
        batch_size: int = MAX_BATCH_SIZE,
        retry_count: int = None,
        recovery_mode: RecoveryMode = None,
        metadata: Dict[str, Any] = None,
        force_recompute: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently with comprehensive recovery.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize the embedding vectors
            batch_size: Size of batches for processing
            retry_count: Number of times to retry if embedding fails
            recovery_mode: Strategy to use for recovering from failures
            metadata: Additional metadata to store with the embeddings
            force_recompute: Whether to bypass cache and force recomputation
            
        Returns:
            Dictionary mapping each text to its embedding
        """
        start_time = time.time()
        
        # Use instance defaults if not specified
        norm = self.normalize if normalize is None else normalize
        tries = retry_count if retry_count is not None else self.retry_count
        mode = recovery_mode if recovery_mode is not None else self.recovery_mode
        
        # Add component identifier to metadata
        meta = metadata or {}
        meta.update({
            "component": "SentenceTransformerAligner",
            "batch_operation": True,
            "batch_size": batch_size,
            "text_count": len(texts),
            "config": {k: v for k, v in self.config.items() if k != "environment"},
            "component_version": VERSION,
            "request_timestamp": datetime.now().isoformat()
        })
        
        try:
            # Get embeddings with comprehensive recovery strategies
            embeddings = batch_embed(
                texts, 
                normalize=norm, 
                model_name=self.model_name,
                fallback_models=self.fallback_models,
                batch_size=batch_size,
                retry_count=tries,
                metadata=meta,
                record_failures=True
            )
            
            # Update stats
            if self.thread_safe:
                with threading.RLock():
                    self.stats["embeddings_generated"] += len(embeddings)
                    self.stats["total_latency"] += time.time() - start_time
                    
                    # Update cache hit/miss counts
                    self.stats["cache_hits"] += _embedding_cache.hits
                    self.stats["cache_misses"] += _embedding_cache.misses
            else:
                self.stats["embeddings_generated"] += len(embeddings)
                self.stats["total_latency"] += time.time() - start_time
                
                # Update cache hit/miss counts
                self.stats["cache_hits"] += _embedding_cache.hits
                self.stats["cache_misses"] += _embedding_cache.misses
            
            return embeddings
            
        except Exception as e:
            # Update error stats
            if self.thread_safe:
                with threading.RLock():
                    self.stats["errors"] += 1
            else:
                self.stats["errors"] += 1
                
            logger.error(f"Error in batch_embed: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Return whatever we could successfully embed
            return {}
    
    def calculate_similarity(
        self, 
        vector_a: np.ndarray, 
        vector_b: np.ndarray,
        fallback_value: float = 0.0
    ) -> float:
        """
        Calculate similarity score between two vectors with error handling.
        
        Args:
            vector_a: First embedding vector
            vector_b: Second embedding vector
            fallback_value: Value to return in case of error
            
        Returns:
            Similarity score between 0 and 1
        """
        return cosine_similarity(vector_a, vector_b, fallback_value)
    
    def find_similar(
        self, 
        query: Union[str, np.ndarray],
        candidates: Union[List[str], Dict[str, Any], Dict[str, np.ndarray]],
        threshold: float = 0.6,
        max_results: int = 100,
        include_metadata: bool = False,
        retry_count: int = None
    ) -> List[Tuple[Union[int, str], Any, float, Optional[Dict[str, Any]]]]:
        """
        Find semantically similar items from a collection of candidates.
        
        Args:
            query: Query text or embedding vector
            candidates: Collection of items to search through (texts, objects, or pre-computed embeddings)
            threshold: Minimum similarity score to include in results
            max_results: Maximum number of results to return
            include_metadata: Whether to include metadata in results
            retry_count: Number of times to retry if embedding fails
            
        Returns:
            List of tuples containing (id, item, score, metadata) sorted by similarity score
        """
        tries = retry_count if retry_count is not None else self.retry_count
        
        try:
            # Convert query to embedding if it's a string
            query_embedding = query if isinstance(query, np.ndarray) else self.get_embedding(
                query, retry_count=tries, metadata={"query": True}
            )
            
            results = []
            
            # Handle different candidate formats
            if isinstance(candidates, list):
                # List of strings
                if all(isinstance(c, str) for c in candidates):
                    # Use batch_embed for efficiency
                    embeddings = self.batch_embed(
                        candidates, 
                        retry_count=tries,
                        metadata={"candidates": True, "count": len(candidates)}
                    )
                    
                    for i, text in enumerate(candidates):
                        if text in embeddings:
                            similarity = self.calculate_similarity(query_embedding, embeddings[text])
                            if similarity >= threshold:
                                # Get metadata if requested
                                metadata = None
                                if include_metadata:
                                    try:
                                        key = _embedding_cache._compute_signature(text, self.model_name)
                                        if key in _embedding_cache.cache:
                                            _, _, metadata = _embedding_cache.cache[key]
                                    except Exception:
                                        pass
                                
                                if include_metadata:
                                    results.append((i, text, similarity, metadata))
                                else:
                                    results.append((i, text, similarity, None))
                
                # List of objects
                else:
                    for i, item in enumerate(candidates):
                        # For objects, try to extract text for embedding
                        if hasattr(item, '__str__'):
                            text = str(item)
                            embedding = self.get_embedding(
                                text, 
                                retry_count=tries,
                                metadata={"object_candidate": True, "index": i}
                            )
                            similarity = self.calculate_similarity(query_embedding, embedding)
                            if similarity >= threshold:
                                if include_metadata:
                                    results.append((i, item, similarity, {"object_type": type(item).__name__}))
                                else:
                                    results.append((i, item, similarity, None))
            
            # Dictionary of items
            elif isinstance(candidates, dict):
                # Dictionary of embeddings
                if all(isinstance(v, np.ndarray) for v in candidates.values()):
                    for key, embedding in candidates.items():
                        similarity = self.calculate_similarity(query_embedding, embedding)
                        if similarity >= threshold:
                            if include_metadata:
                                results.append((key, embedding, similarity, {"pre_computed": True}))
                            else:
                                results.append((key, embedding, similarity, None))
                
                # Dictionary of objects
                else:
                    for key, item in candidates.items():
                        # For objects, try to extract text for embedding
                        if hasattr(item, '__str__'):
                            text = str(item)
                            embedding = self.get_embedding(
                                text, 
                                retry_count=tries,
                                metadata={"dict_candidate": True, "key": key}
                            )
                            similarity = self.calculate_similarity(query_embedding, embedding)
                            if similarity >= threshold:
                                if include_metadata:
                                    results.append(
                                        (key, item, similarity, {"object_type": type(item).__name__})
                                    )
                                else:
                                    results.append((key, item, similarity, None))
            
            # Sort by similarity score (descending)
            results.sort(key=lambda x: x[2], reverse=True)
            
            # Limit to max_results
            limited_results = results[:max_results]
            
            # If metadata was not requested, remove it from the results
            if not include_metadata:
                limited_results = [(id_, item, score, None) for id_, item, score, _ in limited_results]
            
            return limited_results
            
        except Exception as e:
            logger.error(f"Error in find_similar: {str(e)}")
            logger.debug(traceback.format_exc())
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the underlying embedding model.
        
        Returns:
            Dictionary containing model metadata and statistics
        """
        try:
            model = load_model(self.model_name)
            
            # Basic info
            info = {
                "model_name": self.model_name,
                "dimension": model.get_sentence_embedding_dimension(),
                "version": VERSION,
                "normalize": self.normalize,
                "fallback_models": self.fallback_models,
                "signature_method": self.signature_method,
                "retry_count": self.retry_count,
            }
            
            # Add cache stats
            info["cache_stats"] = _embedding_cache.get_stats()
            
            # Add version info if available
            if self.model_name in _model_versions:
                info["version_info"] = _model_versions[self.model_name]
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {
                "model_name": self.model_name,
                "error": str(e),
                "version": VERSION,
                "cache_stats": _embedding_cache.get_stats(),
            }
