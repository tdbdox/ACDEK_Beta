from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Generator
import numpy as np


class VectorStoreInterface(ABC):
    """
    Interface for vector storage and retrieval components.
    
    Components implementing this interface are responsible for:
    1. Storing vector embeddings with metadata
    2. Retrieving vectors by ID or similarity search
    3. Managing vector indexes for efficient retrieval
    4. Handling persistence and caching of embeddings
    """
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def retrieve(self, ids: List[str]) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve vectors and metadata by IDs.
        
        Args:
            ids: List of vector IDs to retrieve
            
        Returns:
            Dictionary mapping IDs to tuples of (vector, metadata)
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """
        Delete vectors by IDs.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            Number of vectors deleted
        """
        pass
    
    @abstractmethod
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a vector.
        
        Args:
            id: Vector ID to update
            metadata: New metadata to apply (merged with existing)
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics (count, size, etc.)
        """
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """
        Clear all vectors from the store.
        
        Returns:
            True if successful, False otherwise
        """
        pass