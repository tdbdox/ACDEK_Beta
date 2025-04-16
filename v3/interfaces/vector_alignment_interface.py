from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np


class VectorAlignmentInterface(ABC):
    """
    Interface for vector alignment and semantic similarity components.
    
    Components implementing this interface are responsible for:
    1. Creating and managing embeddings
    2. Finding semantically similar content
    3. Calculating similarity scores
    4. Managing embedding caches
    """
    
    @abstractmethod
    def get_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding vector for input text.
        
        Args:
            text: The text to create an embedding for
            normalize: Whether to normalize the embedding vector
            
        Returns:
            Embedding vector as numpy array
        """
        pass
    
    @abstractmethod
    def batch_embed(self, texts: List[str], normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize the embedding vectors
            
        Returns:
            Dictionary mapping each text to its embedding
        """
        pass
    
    @abstractmethod
    def calculate_similarity(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        Calculate similarity score between two vectors.
        
        Args:
            vector_a: First embedding vector
            vector_b: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the underlying embedding model.
        
        Returns:
            Dictionary containing model metadata
        """
        pass

