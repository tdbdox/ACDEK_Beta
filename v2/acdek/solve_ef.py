from acdek.types import QEnriched, EfRaw
from typing import Dict, Any, List
import logging

# Configure logging
logger = logging.getLogger(__name__)

def solve_ef(ef_raw: EfRaw, Q_enriched: QEnriched, Cu_enriched: dict) -> Dict[str, Any]:
    """
    Process environment factors and filter/score them based on query and context relevance.
    
    Args:
        ef_raw: Raw environment factors
        Q_enriched: Enriched query information
        Cu_enriched: Enriched user context
        
    Returns:
        Dictionary of environment factors with relevance scores
    """
    if ef_raw is None:
        logger.warning("No environment factors provided")
        return {}
    
    try:
        # Step 1: Extract relevant keywords from query and context
        keywords = []
        
        # Add keywords from intent
        if hasattr(Q_enriched, 'intent') and Q_enriched.intent:
            keywords.extend(Q_enriched.intent.lower().split())
        
        # Add keywords from dependencies
        if hasattr(Q_enriched, 'dependencies') and Q_enriched.dependencies:
            keywords.extend([str(dep).lower() for dep in Q_enriched.dependencies])
        
        # Add keywords from constraints
        if hasattr(Q_enriched, 'constraints') and Q_enriched.constraints:
            for k, v in Q_enriched.constraints.items():
                keywords.append(str(k).lower())
                keywords.append(str(v).lower())
        
        # Add keywords from user context
        if Cu_enriched and isinstance(Cu_enriched, dict):
            # Only include primitive values, not complex objects
            for k, v in Cu_enriched.items():
                if isinstance(v, (str, int, float, bool)):
                    keywords.append(str(k).lower())
                    keywords.append(str(v).lower())
        
        # Remove duplicates and empty strings
        keywords = list(set(filter(None, keywords)))
        
        logger.info(f"Extracted {len(keywords)} keywords for environment matching")
        
        # Step 2: Extract variables from ef_raw
        ef_variables = {}
        
        # Handle the case where ef_raw has 'variables' attribute
        if hasattr(ef_raw, 'variables') and ef_raw.variables:
            ef_variables = ef_raw.variables
        # Handle the case where ef_raw exposes variables directly
        elif hasattr(ef_raw, 'urgency') or hasattr(ef_raw, 'volatility'):
            ef_variables = {
                k: getattr(ef_raw, k)
                for k in dir(ef_raw)
                if not k.startswith('_') and not callable(getattr(ef_raw, k))
            }
        # Handle dictionary input
        elif isinstance(ef_raw, dict):
            ef_variables = ef_raw
            
        if not ef_variables:
            logger.warning("No environment variables found in ef_raw")
            return {}
            
        # Step 3: Relevance filter — keep keys that appear in query or context
        filtered_ef = {}
        
        for k, v in ef_variables.items():
            # Check if any keyword is in the key or value
            if any(word in str(v).lower() or word in k.lower() for word in keywords):
                filtered_ef[k] = v
                
        # Step 4: Merge with query context info for more complete environment picture
        merged = {
            **filtered_ef,
            "intent": Q_enriched.intent if hasattr(Q_enriched, 'intent') else "unknown"
        }
        
        # Add ambiguity if available
        if hasattr(Q_enriched, 'ambiguity_score'):
            merged["ambiguity"] = Q_enriched.ambiguity_score
            
        # Add semantic class if available
        if Cu_enriched and isinstance(Cu_enriched, dict) and Cu_enriched.get("semantic_class"):
            merged["user_semantic_class"] = Cu_enriched["semantic_class"]
            
        # Step 5: Score factors by relevance and importance
        scored = {}
        
        for k, v in merged.items():
            # Calculate keyword match score (how many keywords match)
            keyword_match_count = sum(1 for word in keywords if word in str(k).lower() or word in str(v).lower())
            
            # Importance weights for different types of factors
            importance_weight = 1.0
            if k in ["urgency", "deadline", "priority"]:
                importance_weight = 2.0
            elif k in ["volatility", "risk", "sensitivity"]:
                importance_weight = 1.5
                
            # Calculate final score
            match_score = min(1.0, keyword_match_count / max(1, len(keywords)))
            final_score = match_score * importance_weight
            
            scored[k] = {
                "value": v,
                "score": final_score,
                "keywords_matched": keyword_match_count
            }
            
        logger.info(f"Processed {len(scored)} environment factors with scores")
        return scored
        
    except Exception as e:
        logger.exception(f"Error in solve_ef: {str(e)}")
        # Return minimal valid output
        return {
            "error": str(e),
            "urgency": getattr(ef_raw, "urgency", "medium") if ef_raw else "medium",
            "volatility": getattr(ef_raw, "volatility", "low") if ef_raw else "low"
        }
