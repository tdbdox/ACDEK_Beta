import logging
import numpy as np
import os
import pickle
import time
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set
from collections import defaultdict

from registry.registry_types import PluginType
from registry.plugin_registry import register_plugin
from interfaces.vector_alignment_interface import VectorAlignmentInterface
from storage.vector_store.vector_store_interface import VectorStoreInterface

logger = logging.getLogger(__name__)


@register_plugin(
    plugin_type=PluginType.STORAGE,
    name="in_memory_vector_store",
    version="1.0.0",
    description="In-memory vector store with persistence",
    interfaces=["VectorStoreInterface"],
    is_default=True
)
class InMemoryVectorStore(VectorStoreInterface):
    """
    Simple in-memory vector store with optional disk persistence.
    
    This component stores vectors and metadata in memory with optional
    persistence to disk, providing efficient similarity search and retrieval.
    """
    
    def __init__(
        self,
        persistence_path: Optional[str] = None,
        auto_save: bool = True,
        auto_save_threshold: int = 100
    ):
        """
        Initialize the vector store.
        
        Args:
            persistence_path: Optional path for disk persistence
            auto_save: Whether to automatically save on updates
            auto_save_threshold: Frequency of auto-saves (number of changes)
        """
        # Storage for vectors and metadata
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Track modifications
        self.modified: Set[str] = set()
        self.update_count: int = 0
        
        # Persistence settings
        self.persistence_path = persistence_path
        self.auto_save = auto_save
        self.auto_save_threshold = auto_save_threshold
        
        # Statistics
        self.stats = {
            "total_vectors": 0,
            "total_searches": 0,
            "created_at": time.time(),
            "last_modified": None
        }
        
        # Load from disk if persistence path provided
        if persistence_path:
            self._load()
        
        logger.info(f"Initialized InMemoryVectorStore with {len(self.vectors)} vectors")
    
    def store(
        self,
        vectors: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Store vectors with optional metadata.
        
        Args:
            vectors: Dictionary mapping IDs to vector embeddings
            metadata: Optional dictionary mapping IDs to metadata dictionaries
            
        Returns:
            List of stored vector IDs
        """
        stored_ids = []
        metadata = metadata or {}
        
        for id, vector in vectors.items():
            # Store vector
            self.vectors[id] = vector
            
            # Store metadata
            if id in metadata:
                self.metadata[id] = metadata[id]
            elif id not in self.metadata:
                self.metadata[id] = {}
            
            # Mark as modified
            self.modified.add(id)
            stored_ids.append(id)
        
        # Update stats
        self.stats["total_vectors"] = len(self.vectors)
        self.stats["last_modified"] = time.time()
        self.update_count += len(vectors)
        
        # Auto-save if enabled
        if self.auto_save and self.persistence_path and self.update_count >= self.auto_save_threshold:
            self._save()
            self.update_count = 0
        
        return stored_ids
    
    def retrieve(self, ids: List[str]) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve vectors and metadata by IDs.
        
        Args:
            ids: List of vector IDs to retrieve
            
        Returns:
            Dictionary mapping IDs to tuples of (vector, metadata)
        """
        result = {}
        
        for id in ids:
            if id in self.vectors:
                vector = self.vectors[id]
                meta = self.metadata.get(id, {})
                result[id] = (vector, meta)
        
        return result
    
    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        threshold: float = 0.0,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Vector to search for
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            filter_fn: Optional function to filter results by metadata
            
        Returns:
            List of tuples containing (id, similarity, metadata)
        """
        # Track search count
        self.stats["total_searches"] += 1
        
        # Filter IDs by metadata if filter function provided
        ids_to_search = list(self.vectors.keys())
        if filter_fn:
            ids_to_search = [
                id for id in ids_to_search
                if id in self.metadata and filter_fn(self.metadata[id])
            ]
        
        # Calculate similarities for all vectors
        similarities = []
        
        for id in ids_to_search:
            vector = self.vectors[id]
            
            # Calculate cosine similarity
            similarity = self._calculate_similarity(query_vector, vector)
            
            # Apply threshold filter
            if similarity >= threshold:
                similarities.append((id, similarity, self.metadata.get(id, {})))
        
        # Sort by similarity (descending) and limit results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:limit]
    
    def delete(self, ids: List[str]) -> int:
        """
        Delete vectors by IDs.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            Number of vectors deleted
        """
        deleted_count = 0
        
        for id in ids:
            if id in self.vectors:
                del self.vectors[id]
                if id in self.metadata:
                    del self.metadata[id]
                if id in self.modified:
                    self.modified.remove(id)
                deleted_count += 1
        
        if deleted_count > 0:
            # Update stats
            self.stats["total_vectors"] = len(self.vectors)
            self.stats["last_modified"] = time.time()
            self.update_count += deleted_count
            
            # Auto-save if enabled
            if self.auto_save and self.persistence_path and self.update_count >= self.auto_save_threshold:
                self._save()
                self.update_count = 0
        
        return deleted_count
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a vector.
        
        Args:
            id: Vector ID to update
            metadata: New metadata to apply (merged with existing)
            
        Returns:
            True if successful, False otherwise
        """
        if id not in self.vectors:
            return False
        
        # Initialize metadata if not exists
        if id not in self.metadata:
            self.metadata[id] = {}
        
        # Update metadata (merge)
        self.metadata[id].update(metadata)
        
        # Mark as modified
        self.modified.add(id)
        self.update_count += 1
        self.stats["last_modified"] = time.time()
        
        # Auto-save if enabled
        if self.auto_save and self.persistence_path and self.update_count >= self.auto_save_threshold:
            self._save()
            self.update_count = 0
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics (count, size, etc.)
        """
        # Calculate memory usage (approximate)
        vector_mem = sum(v.nbytes for v in self.vectors.values())
        
        return {
            **self.stats,
            "vector_count": len(self.vectors),
            "metadata_count": len(self.metadata),
            "modified_count": len(self.modified),
            "memory_usage_bytes": vector_mem,
            "persistence_enabled": bool(self.persistence_path),
            "auto_save_enabled": self.auto_save
        }
    
    def clear(self) -> bool:
        """
        Clear all vectors from the store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.vectors.clear()
            self.metadata.clear()
            self.modified.clear()
            
            # Update stats
            self.stats["total_vectors"] = 0
            self.stats["last_modified"] = time.time()
            self.update_count += 1
            
            # Auto-save if enabled
            if self.auto_save and self.persistence_path:
                self._save()
                self.update_count = 0
            
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def save(self) -> bool:
        """
        Save the vector store to disk.
        
        Returns:
            True if successful, False otherwise
        """
        return self._save()
    
    def _save(self) -> bool:
        """Internal method to save to disk."""
        if not self.persistence_path:
            logger.warning("No persistence path specified, cannot save")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            
            # Prepare data to save
            data = {
                "vectors": self.vectors,
                "metadata": self.metadata,
                "stats": self.stats
            }
            
            # Save to disk
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Saved {len(self.vectors)} vectors to {self.persistence_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def _load(self) -> bool:
        """Internal method to load from disk."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return False
        
        try:
            with open(self.persistence_path, 'rb') as f:
                data = pickle.load(f)
            
            self.vectors = data.get("vectors", {})
            self.metadata = data.get("metadata", {})
            
            # Keep only stats that make sense to load
            loaded_stats = data.get("stats", {})
            self.stats["total_vectors"] = loaded_stats.get("total_vectors", len(self.vectors))
            self.stats["total_searches"] = loaded_stats.get("total_searches", 0)
            
            logger.info(f"Loaded {len(self.vectors)} vectors from {self.persistence_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def _calculate_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        # Handle zero vectors
        if np.all(vector_a == 0) or np.all(vector_b == 0):
            return 0.0
            
        # Calculate cosine similarity with numerical stability
        similarity = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b) + 1e-8)
        
        # Ensure result is in valid range
        return max(0.0, min(1.0, float(similarity)))