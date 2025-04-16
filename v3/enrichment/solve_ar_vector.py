import logging
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from collections import defaultdict
from dataclasses import dataclass, field

from storage.vector_store.embed_utils import get_embedding, batch_embed, cosine_similarity

# Registry imports
from registry.registry_types import PluginType
from registry.plugin_registry import register_plugin
from interfaces.vector_alignment_interface import VectorAlignmentInterface
from interfaces.enrichment_interface import EnrichmentInterface

logger = logging.getLogger(__name__)


@dataclass
class ArItem:
    """Data slot with value and metadata."""
    key: str
    value: Union[str, int, float, Dict[str, Any]]
    confidence: float
    age: float = 0
    source: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "age": self.age,
            "source": self.source
        }


@dataclass
class ArEnriched:
    """Collection of enriched data slots."""
    slots: Dict[str, ArItem] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert to dictionary representation."""
        return {key: slot.to_dict() for key, slot in self.slots.items()}
    
    def get_highest_confidence_slots(self, limit: int = 5, threshold: float = 0.0) -> Dict[str, ArItem]:
        """Get the highest confidence slots above threshold."""
        items = [(key, slot) for key, slot in self.slots.items() if slot.confidence >= threshold]
        items.sort(key=lambda x: x[1].confidence, reverse=True)
        return {key: slot for key, slot in items[:limit]}


@dataclass
class QEnriched:
    """Enriched query with intent and context."""
    intent: str
    dependencies: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    ambiguity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "intent": self.intent,
            "dependencies": self.dependencies,
            "constraints": self.constraints,
            "ambiguity_score": self.ambiguity_score
        }


@register_plugin(
    plugin_type=PluginType.ENRICHMENT,
    name="ar_vector_enricher",
    version="1.0.0",
    description="Enriches data sources by selecting relevant data slots based on query intent",
    interfaces=["EnrichmentInterface"],
    is_default=True
)
class ArVectorEnricher(EnrichmentInterface[Dict[str, Any], ArEnriched]):
    """
    Vector-based data enrichment component that selects and ranks data slots.
    
    This component enriches raw data by:
    1. Embedding the query intent and dependencies
    2. Embedding the data slot values
    3. Computing similarity scores between query and data
    4. Applying confidence adjustments based on age and constraints
    5. Selecting the most relevant data slots
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.2,
        max_slots: int = 5,
        decay_rate: float = 0.01,
        constraint_boost: float = 1.2,
        intent_weight: float = 0.7,
        vector_provider: Optional[VectorAlignmentInterface] = None
    ):
        """
        Initialize the enricher with configuration.
        
        Args:
            confidence_threshold: Minimum confidence score for slot selection
            max_slots: Maximum number of slots to return
            decay_rate: Time decay rate for confidence (lambda parameter)
            constraint_boost: Boost factor for slots matching constraints
            intent_weight: Weight of intent match vs. dependency matches
            vector_provider: Optional vector alignment component to use
        """
        self.confidence_threshold = confidence_threshold
        self.max_slots = max_slots
        self.decay_rate = decay_rate
        self.constraint_boost = constraint_boost
        self.intent_weight = intent_weight
        
        # If not provided, use direct functions
        self.vector_provider = vector_provider
    
    def enrich(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ArEnriched:
        """
        Enrich data sources by selecting relevant data slots based on query intent.
        
        Args:
            input_data: Dictionary containing:
                - ar_raw: Raw data sources
                - q_enriched: Enriched query with intent and dependencies
            context: Optional additional context
            
        Returns:
            Enriched data with confidence scoring
        """
        # Extract required input
        ar_raw = input_data.get("ar_raw", {})
        q_enriched = input_data.get("q_enriched")
        
        if not isinstance(q_enriched, QEnriched):
            # Try to convert from dict
            if isinstance(q_enriched, dict):
                q_enriched = QEnriched(
                    intent=q_enriched.get("intent", ""),
                    dependencies=q_enriched.get("dependencies", []),
                    constraints=q_enriched.get("constraints", {}),
                    ambiguity_score=q_enriched.get("ambiguity_score", 0.0)
                )
            else:
                logger.error("Missing or invalid q_enriched in input data")
                return ArEnriched(slots={})
        
        try:
            # Process query embeddings
            query_texts = [q_enriched.intent] + q_enriched.dependencies
            query_embeddings = self._get_embeddings(query_texts)
            
            if not query_embeddings:
                logger.warning("Failed to generate query embeddings")
                return ArEnriched(slots={})
            
            # Primary embedding is the intent embedding
            query_embedding = query_embeddings.get(q_enriched.intent)
            if query_embedding is None:
                logger.warning("Missing intent embedding, using first available")
                query_embedding = next(iter(query_embeddings.values()))
            
            # Extract slots from ar_raw based on its structure
            slots = self._extract_slots(ar_raw)
            
            if not slots:
                logger.warning("No data slots found in ar_raw")
                return ArEnriched(slots={})
            
            logger.info(f"Processing {len(slots)} data slots")
            
            # Process each slot and compute confidence scores
            candidates = []
            for key, data in slots.items():
                try:
                    processed_item = self._process_slot(
                        key,
                        data,
                        query_embedding,
                        query_embeddings,
                        q_enriched
                    )
                    if processed_item:
                        candidates.append((processed_item.confidence, processed_item))
                except Exception as e:
                    logger.error(f"Error processing slot '{key}': {str(e)}")
            
            # Sort candidates by confidence score (descending)
            candidates.sort(reverse=True, key=lambda x: x[0])
            
            # Determine confidence threshold based on query ambiguity
            adjusted_threshold = self._get_adjusted_threshold(q_enriched.ambiguity_score)
            
            # Select top slots based on confidence and threshold
            top_slots = {}
            count = 0
            for conf, item in candidates:
                if conf >= adjusted_threshold:
                    top_slots[item.key] = item
                    count += 1
                
                # Adaptive limit - take more slots if confidence is high
                adaptive_max = self.max_slots * 2 if conf > 0.6 else self.max_slots
                if count >= adaptive_max:
                    break
            
            logger.info(f"Selected {len(top_slots)} slots with confidence threshold {adjusted_threshold:.2f}")
            return ArEnriched(slots=top_slots)
            
        except Exception as e:
            logger.exception(f"Error in enrich: {str(e)}")
            # Return empty result on error
            return ArEnriched(slots={})
    
    def get_confidence(self, enriched_data: ArEnriched) -> float:
        """
        Extract the overall confidence score from enriched data.
        
        Args:
            enriched_data: The enriched data to analyze
            
        Returns:
            Confidence score between 0 and 1
        """
        if not enriched_data.slots:
            return 0.0
        
        # Average of top 3 confidence scores
        confidences = [slot.confidence for slot in enriched_data.slots.values()]
        confidences.sort(reverse=True)
        top_n = min(3, len(confidences))
        
        if top_n == 0:
            return 0.0
            
        return sum(confidences[:top_n]) / top_n
    
    def apply_constraints(
        self,
        enriched_data: ArEnriched,
        constraints: Dict[str, Any]
    ) -> ArEnriched:
        """
        Apply additional constraints to already enriched data.
        
        Args:
            enriched_data: The previously enriched data
            constraints: Dictionary of constraints to apply
            
        Returns:
            Enriched data with constraints applied
        """
        if not constraints or not enriched_data.slots:
            return enriched_data
        
        # Create a new set of slots with adjusted confidence
        updated_slots = {}
        
        for key, slot in enriched_data.slots.items():
            # Start with original confidence
            confidence = slot.confidence
            value_str = str(slot.value).lower()
            
            # Check if slot matches any constraint
            for constraint_key, constraint_value in constraints.items():
                constraint_value_str = str(constraint_value).lower()
                
                # If the slot key or value matches a constraint, boost confidence
                if (constraint_key.lower() in key.lower() or
                    constraint_value_str in value_str):
                    confidence = min(1.0, confidence * self.constraint_boost)
                    break
            
            # Create updated slot with new confidence
            updated_slots[key] = ArItem(
                key=slot.key,
                value=slot.value,
                confidence=confidence,
                age=slot.age,
                source=slot.source,
                embedding=slot.embedding
            )
        
        return ArEnriched(slots=updated_slots)
    
    def supports_input_type(self, input_type: str) -> bool:
        """
        Check if this enrichment component supports a given input type.
        
        Args:
            input_type: String identifier of input type
            
        Returns:
            Boolean indicating support
        """
        return input_type in ["ar_raw", "ar_data", "data_sources"]
    
    def _get_embeddings(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for a list of texts using the vector provider."""
        if self.vector_provider:
            # Use registered vector provider
            embeddings = {}
            for text in texts:
                embeddings[text] = self.vector_provider.get_embedding(text)
            return embeddings
        else:
            # Use direct embedding function
            return batch_embed(texts)
    
    def _extract_slots(self, ar_raw: Any) -> Dict[str, Any]:
        """Extract slots from ar_raw based on its structure."""
        slots = {}
        
        # Handle slots directly if present
        if hasattr(ar_raw, 'slots') and ar_raw.slots:
            slots = ar_raw.slots
        # Handle items if present
        elif hasattr(ar_raw, 'items') and ar_raw.items:
            # Convert items format to slots format
            for key, item_data in ar_raw.items.items():
                if isinstance(item_data, dict):
                    slots[key] = item_data
        # Handle dictionary format
        elif isinstance(ar_raw, dict):
            # Try slots at top level
            if 'slots' in ar_raw and isinstance(ar_raw['slots'], dict):
                slots = ar_raw['slots']
            # Try items at top level
            elif 'items' in ar_raw and isinstance(ar_raw['items'], dict):
                for key, item_data in ar_raw['items'].items():
                    if isinstance(item_data, dict):
                        slots[key] = item_data
            # Assume the dict itself contains the data
            else:
                slots = ar_raw
        
        return slots
    
    def _process_slot(
        self,
        key: str,
        data: Any,
        query_embedding: np.ndarray,
        query_embeddings: Dict[str, np.ndarray],
        q_enriched: QEnriched
    ) -> Optional[ArItem]:
        """Process a single slot and compute its confidence score."""
        # Extract value and age
        value = None
        age = 0
        
        if isinstance(data, dict):
            value = data.get("value")
            age = data.get("age", 0)
        else:
            value = data
        
        if value is None:
            return None
            
        # Convert value to string for embedding
        value_text = str(value)
        
        # Skip empty values
        if not value_text.strip():
            return None
        
        # Get embedding for the value
        if self.vector_provider:
            value_embedding = self.vector_provider.get_embedding(value_text)
        else:
            value_embedding = get_embedding(value_text)
        
        # Calculate similarity with query intent
        if self.vector_provider:
            intent_sim_score = self.vector_provider.calculate_similarity(query_embedding, value_embedding)
        else:
            intent_sim_score = cosine_similarity(query_embedding, value_embedding)
        
        # Calculate similarities with dependencies if present
        dep_sim_scores = []
        for dep in q_enriched.dependencies:
            if dep in query_embeddings:
                dep_embedding = query_embeddings[dep]
                if self.vector_provider:
                    dep_score = self.vector_provider.calculate_similarity(dep_embedding, value_embedding)
                else:
                    dep_score = cosine_similarity(dep_embedding, value_embedding)
                dep_sim_scores.append(dep_score)
        
        # Combine intent and dependency scores
        # Higher weight for intent match, but boost if matches dependencies
        base_confidence = intent_sim_score
        if dep_sim_scores:
            max_dep_score = max(dep_sim_scores)
            # Weighted average: intent_weight% intent, (1-intent_weight)% dependencies
            base_confidence = (self.intent_weight * intent_sim_score) + ((1 - self.intent_weight) * max_dep_score)
        
        # Apply time decay to confidence
        decayed_confidence = base_confidence * math.exp(-self.decay_rate * age)
        
        # Apply constraints boost if present
        constraint_boost = 1.0
        for constraint_key, constraint_value in q_enriched.constraints.items():
            # If the data key or value matches a constraint, boost confidence
            if (constraint_key.lower() in key.lower() or 
                str(constraint_value).lower() in value_text.lower()):
                constraint_boost = self.constraint_boost
                break
        
        final_confidence = min(1.0, decayed_confidence * constraint_boost)
        
        logger.info(f"Slot '{key}': intent_sim={intent_sim_score:.4f}, age={age}, final_conf={final_confidence:.4f}")
        
        # Create ArItem
        return ArItem(
            key=key,
            value=value,
            confidence=final_confidence,
            age=age,
            embedding=value_embedding
        )
    
    def _get_adjusted_threshold(self, ambiguity_score: float) -> float:
        """Get confidence threshold adjusted for query ambiguity."""
        # If query is ambiguous, lower the threshold to include more data
        if ambiguity_score > 0:
            return max(0.05, self.confidence_threshold - (ambiguity_score * 0.3))
        return self.confidence_threshold


def solve_Ar(ar_raw: Any, q_enriched: QEnriched) -> ArEnriched:
    """
    Legacy compatibility function for v2 code.
    
    Args:
        ar_raw: Raw data sources
        q_enriched: Enriched query with intent and dependencies
        
    Returns:
        Enriched data with confidence scoring
    """
    enricher = ArVectorEnricher()
    return enricher.enrich({"ar_raw": ar_raw, "q_enriched": q_enriched})