from acdek.types import ArItem, ArEnriched, QEnriched
from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as np
import logging
from collections import defaultdict
from embeddings.embed_utils import get_embedding, cosine_similarity, batch_embed

# Configure logging
logger = logging.getLogger(__name__)

def solve_Ar(Ar_raw, Q_enriched: QEnriched) -> ArEnriched:
    """
    Enrich data sources by selecting relevant data slots based on query intent.
    
    Args:
        Ar_raw: Raw data sources
        Q_enriched: Enriched query with intent and dependencies
        
    Returns:
        Enriched data with confidence scoring
    """
    candidates = []
    
    try:
        # Get query embedding based on intent and dependencies
        query_texts = [Q_enriched.intent]
        query_texts.extend(Q_enriched.dependencies)
        
        # Batch embed all query texts
        query_embeddings = batch_embed(query_texts)
        
        if not query_embeddings:
            logger.warning("Failed to generate query embeddings")
            return ArEnriched(slots={})
            
        # Primary embedding is the intent embedding
        query_embedding = query_embeddings.get(Q_enriched.intent)
        if query_embedding is None:
            logger.warning("Missing intent embedding, using first available")
            query_embedding = next(iter(query_embeddings.values()))
        
        # Extract slots from Ar_raw based on its structure
        slots = {}
        
        # Handle slots directly if present
        if hasattr(Ar_raw, 'slots') and Ar_raw.slots:
            slots = Ar_raw.slots
        # Handle items if present
        elif hasattr(Ar_raw, 'items') and Ar_raw.items:
            # Convert items format to slots format
            for key, item_data in Ar_raw.items.items():
                if isinstance(item_data, dict):
                    slots[key] = item_data
        # Handle dictionary format
        elif isinstance(Ar_raw, dict):
            # Try slots at top level
            if 'slots' in Ar_raw and isinstance(Ar_raw['slots'], dict):
                slots = Ar_raw['slots']
            # Try items at top level
            elif 'items' in Ar_raw and isinstance(Ar_raw['items'], dict):
                for key, item_data in Ar_raw['items'].items():
                    if isinstance(item_data, dict):
                        slots[key] = item_data
            # Assume the dict itself contains the data
            else:
                slots = Ar_raw
        
        if not slots:
            logger.warning("No data slots found in Ar_raw")
            return ArEnriched(slots={})
        
        logger.info(f"Processing {len(slots)} data slots")
        
        # For each data slot, calculate its similarity to the query
        for key, data in slots.items():
            try:
                # Extract value
                if isinstance(data, dict):
                    value = data.get("value", "")
                    age = data.get("age", 0)
                else:
                    value = data
                    age = 0
                
                # Convert value to string for embedding
                value_text = str(value)
                
                # Skip empty values
                if not value_text.strip():
                    continue
                
                # Get embedding for the value
                value_embedding = get_embedding(value_text)
                
                # Calculate similarity with query intent
                intent_sim_score = cosine_similarity(query_embedding, value_embedding)
                
                # Calculate similarities with dependencies if present
                dep_sim_scores = []
                for dep in Q_enriched.dependencies:
                    if dep in query_embeddings:
                        dep_embedding = query_embeddings[dep]
                        dep_score = cosine_similarity(dep_embedding, value_embedding)
                        dep_sim_scores.append(dep_score)
                
                # Combine intent and dependency scores
                # Higher weight for intent match, but boost if matches dependencies
                base_confidence = intent_sim_score
                if dep_sim_scores:
                    max_dep_score = max(dep_sim_scores)
                    # Weighted average: 70% intent, 30% dependencies
                    base_confidence = (0.7 * intent_sim_score) + (0.3 * max_dep_score)
                
                # Apply time decay to confidence
                decay_lambda = 0.01  # Decay rate parameter
                decayed_confidence = base_confidence * math.exp(-decay_lambda * age)
                
                # Apply constraints boost if present
                constraint_boost = 1.0
                for constraint_key, constraint_value in Q_enriched.constraints.items():
                    # If the data key or value matches a constraint, boost confidence
                    if (constraint_key.lower() in key.lower() or 
                        str(constraint_value).lower() in value_text.lower()):
                        constraint_boost = 1.2
                        break
                
                final_confidence = min(1.0, decayed_confidence * constraint_boost)
                
                logger.info(f"Slot '{key}': intent_sim={intent_sim_score:.4f}, age={age}, final_conf={final_confidence:.4f}")
                
                # Create ArItem and add to candidates
                candidates.append((final_confidence, ArItem(
                    key=key,
                    value=value,
                    confidence=final_confidence,
                    age=age
                )))
                
            except Exception as e:
                logger.error(f"Error processing slot '{key}': {str(e)}")
        
        # Sort candidates by confidence score (descending)
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        # Return top slots with confidence above threshold
        # More adaptive thresholding based on query ambiguity
        if hasattr(Q_enriched, 'ambiguity_score'):
            # If query is ambiguous, lower the threshold to include more data
            confidence_threshold = max(0.05, 0.3 - (Q_enriched.ambiguity_score * 0.3))
        else:
            confidence_threshold = 0.2
            
        # Take top 5 slots but require minimum confidence
        top_slots = {}
        count = 0
        for conf, item in candidates:
            if conf >= confidence_threshold:
                top_slots[item.key] = item
                count += 1
            # Adaptive limit - take more slots if confidence is high
            max_slots = 10 if conf > 0.2 else 5
            if count >= max_slots:
                break
        
        logger.info(f"Selected {len(top_slots)} slots with confidence threshold {confidence_threshold:.2f}")
        return ArEnriched(slots=top_slots)
        
    except Exception as e:
        logger.exception(f"Error in solve_Ar: {str(e)}")
        # Return empty result on error
        return ArEnriched(slots={})
