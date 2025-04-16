from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, TypeVar, Generic

# Generic type variables for input and output types
T_Input = TypeVar('T_Input')
T_Output = TypeVar('T_Output')


class EnrichmentInterface(Generic[T_Input, T_Output], ABC):
    """
    Interface for data enrichment components.
    
    Components implementing this interface are responsible for:
    1. Processing and enriching input data
    2. Adding metadata, confidence scores, and other contextual information
    3. Preparing data for downstream decision components
    """
    
    @abstractmethod
    def enrich(self, input_data: T_Input, context: Optional[Dict[str, Any]] = None) -> T_Output:
        """
        Enrich input data with additional metadata, scores, and context.
        
        Args:
            input_data: The raw input data to enrich
            context: Optional contextual information to use for enrichment
            
        Returns:
            Enriched data with additional metadata and scores
        """
        pass
    
    @abstractmethod
    def get_confidence(self, enriched_data: T_Output) -> float:
        """
        Extract the overall confidence score from enriched data.
        
        Args:
            enriched_data: The enriched data to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        pass
    
    @abstractmethod
    def apply_constraints(
        self, 
        enriched_data: T_Output, 
        constraints: Dict[str, Any]
    ) -> T_Output:
        """
        Apply additional constraints to already enriched data.
        
        Args:
            enriched_data: The previously enriched data
            constraints: Dictionary of constraints to apply
            
        Returns:
            Enriched data with constraints applied
        """
        pass
    
    @abstractmethod
    def supports_input_type(self, input_type: str) -> bool:
        """
        Check if this enrichment component supports a given input type.
        
        Args:
            input_type: String identifier of input type
            
        Returns:
            Boolean indicating support
        """
        pass

