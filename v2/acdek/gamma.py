from acdek.types import QEnriched, ArEnriched, ArItem, Theta
from embeddings.embed_utils import get_embedding, cosine_similarity, batch_embed
import logging
from typing import Dict, Tuple, List, Any, Optional, Union
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def verify(Q: QEnriched, Ar: ArEnriched, psi: Union[dict, Theta], gamma_adjusted: float = 0.7) -> Tuple[bool, Dict]:
    """
    Verify that all dependencies and constraints are satisfied in the data slots.
    
    Args:
        Q: Enriched query
        Ar: Enriched data slots
        psi: Modulation vector (theta) as either a dict or Theta object
        gamma_adjusted: Adjusted confidence threshold
        
    Returns:
        Tuple of (verification_passed, diagnostics)
    """
    failed_dependencies = []
    failed_constraints = []
    
    try:
        # Early exit if there are no slots
        if not Ar.slots:
            logger.warning("No data slots available for verification")
            # Use a message without expecting tuple unpacking later
            return False, {"failed_dependencies": [("No data slots available", None)]}
        
        # Early exit if there are no dependencies
        if not Q.dependencies:
            logger.info("No dependencies to verify")
            # Continue to check constraints
        else:
            # Batch embed all dependencies for efficiency
            dependency_embeddings = batch_embed(Q.dependencies)
            
            # Create embeddings for all slot keys and values
            slot_keys = list(Ar.slots.keys())
            slot_values = [str(item.value) for item in Ar.slots.values()]
            
            # Batch embed slot keys and values
            key_embeddings = batch_embed(slot_keys)
            value_embeddings = batch_embed(slot_values)
            
            # Process each dependency
            for dep in Q.dependencies:
                logger.info(f"Verifying dependency: {dep}")
                
                if dep not in dependency_embeddings:
                    logger.warning(f"Missing embedding for dependency: {dep}")
                    failed_dependencies.append((dep, None))
                    continue
                
                dep_embedding = dependency_embeddings[dep]
                best_match = None
                highest_score = 0.0
                
                # Check each slot for a match
                for i, (key, item) in enumerate(Ar.slots.items()):
                    # Get key and value embeddings
                    key_embedding = key_embeddings.get(key)
                    value_embedding = value_embeddings.get(str(item.value))
                    
                    if key_embedding is None or value_embedding is None:
                        logger.warning(f"Missing embedding for slot {key}")
                        continue
                    
                    # Calculate similarity scores
                    key_score = cosine_similarity(dep_embedding, key_embedding)
                    value_score = cosine_similarity(dep_embedding, value_embedding)
                    score = max(key_score, value_score)
                    
                    if score > highest_score:
                        highest_score = score
                        best_match = item
                
                # Evaluate the best match
                if best_match:
                    logger.info(f"Evaluating: confidence={best_match.confidence:.4f}, highest_score={highest_score:.4f}")
                    
                    # More flexible confidence threshold based on similarity score
                    confidence_threshold = gamma_adjusted
                    
                    # If semantic match is very strong, allow slightly lower confidence
                    if highest_score >= 0.8:
                        confidence_threshold = max(0.5, gamma_adjusted - 0.2)
                    # If semantic match is strong, allow slightly lower confidence
                    elif highest_score >= 0.65:
                        confidence_threshold = max(0.6, gamma_adjusted - 0.1)
                    
                    if best_match.confidence >= confidence_threshold:
                        logger.info(f"Matched to slot '{best_match.key}' with score {highest_score:.4f} and confidence {best_match.confidence:.2f}")
                    else:
                        logger.warning(f"Match found but did not meet confidence threshold (required: {confidence_threshold:.2f})")
                        failed_dependencies.append((dep, best_match.confidence))
                else:
                    logger.warning("No suitable slot found")
                    failed_dependencies.append((dep, None))
        
        # Verify constraints
        if Q.constraints:
            for k, v in Q.constraints.items():
                # First check for direct match
                if k in Ar.slots and str(Ar.slots[k].value) == str(v):
                    logger.info(f"Constraint {k}={v} satisfied directly")
                    continue
                
                # If no direct match, check for semantic matches in slot keys or values
                constraint_satisfied = False
                for slot_key, slot_item in Ar.slots.items():
                    # Check if constraint key is in slot key or if this is the specific time_period constraint
                    if k.lower() in slot_key.lower() or (k == "time_period" and slot_key == "time_period"):
                        # For time_period=last_quarter constraint, we should do a more specialized check
                        if k == "time_period" and v == "last_quarter":
                            # If value contains any quarter-related terms, consider it satisfied
                            quarter_terms = ["last quarter", "q1", "q2", "q3", "q4", "quarter"]
                            slot_value_lower = str(slot_item.value).lower()
                            
                            if any(term in slot_value_lower for term in quarter_terms):
                                logger.info(f"Constraint {k}={v} satisfied semantically via {slot_key} with value '{slot_value_lower}'")
                                constraint_satisfied = True
                                break
                        # Otherwise do the normal check if constraint value is in slot value
                        elif str(v).lower() in str(slot_item.value).lower():
                            logger.info(f"Constraint {k}={v} satisfied semantically via {slot_key}")
                            constraint_satisfied = True
                            break
                
                if not constraint_satisfied:
                    logger.warning(f"Constraint {k}={v} not satisfied")
                    failed_constraints.append((k, v))
        
        # Return verification result
        if failed_dependencies or failed_constraints:
            return False, {
                "failed_dependencies": failed_dependencies,
                "failed_constraints": failed_constraints
            }
        
        return True, {}
        
    except Exception as e:
        logger.exception(f"Error in verification: {str(e)}")
        return False, {"error": str(e)}


def gamma(Q: QEnriched, Ar: ArEnriched, psi: Union[dict, Theta]) -> Dict[str, Any]:
    """
    Verify dependencies and constraints and form the payload for execution.
    
    Args:
        Q: Enriched query
        Ar: Enriched data
        psi: Modulation vector (theta) as either a dict or Theta object
        
    Returns:
        Dictionary with payload or fallback information
    """
    try:
        # Get confidence floor from modulation vector
        gamma_adjusted = psi["confidence_floor"] if isinstance(psi, dict) else getattr(psi, "confidence_floor", 0.7)
        logger.info(f"Using confidence floor: {gamma_adjusted}")
        
        # Verify dependencies and constraints
        passed, diagnostics = verify(Q, Ar, psi, gamma_adjusted)
        logger.info(f"Verification passed: {passed}, diagnostics: {diagnostics}")
        
        if passed:
            # Form successful payload
            payload = {
                "intent": Q.intent,
                "slots": {k: v.value for k, v in Ar.slots.items()},
                "constraints": Q.constraints,
                "confidence": {k: v.confidence for k, v in Ar.slots.items()},
                "verification_status": "passed"
            }
            
            # Add metadata about payload
            payload["slot_count"] = len(Ar.slots)
            payload["average_confidence"] = sum(v.confidence for v in Ar.slots.values()) / max(1, len(Ar.slots))
            
            # Calculate a composite confidence score for the entire payload
            if Q.dependencies:
                # Weight by dependency coverage
                dependency_count = len(Q.dependencies)
                missing_dependencies = len([d for d, _ in diagnostics.get("failed_dependencies", [])])
                coverage_ratio = (dependency_count - missing_dependencies) / dependency_count
                payload["dependency_coverage"] = coverage_ratio
            
            logger.info(f"Generated payload with {len(Ar.slots)} slots")
            return {"payload": payload}
        else:
            # Form fallback response with detailed diagnostics
            fallback_reason = "Verification failed"
            
            # Add more specific reason if available
            if "failed_dependencies" in diagnostics and diagnostics["failed_dependencies"]:
                # Check if failed_dependencies is a list of tuples or a simple list
                if isinstance(diagnostics["failed_dependencies"], list) and all(isinstance(x, tuple) for x in diagnostics["failed_dependencies"] if isinstance(x, tuple)):
                    missing_deps = [d for d, _ in diagnostics["failed_dependencies"]]
                    fallback_reason = f"Missing required dependencies: {', '.join(missing_deps)}"
                else:
                    # Handle case where failed_dependencies is a simple list of strings
                    fallback_reason = f"Missing required dependencies: {', '.join(diagnostics['failed_dependencies'])}"
            elif "failed_constraints" in diagnostics and diagnostics["failed_constraints"]:
                missing_constraints = [f"{k}={v}" for k, v in diagnostics["failed_constraints"]]
                fallback_reason = f"Constraints not satisfied: {', '.join(missing_constraints)}"
            elif "error" in diagnostics:
                fallback_reason = f"Verification error: {diagnostics['error']}"
            
            logger.warning(f"Fallback triggered: {fallback_reason}")
            
            return {
                "fallback": True,
                "reason": fallback_reason,
                "details": diagnostics,
                "available_slots": list(Ar.slots.keys()) if Ar.slots else []
            }
    
    except Exception as e:
        logger.exception(f"Error in gamma: {str(e)}")
        return {
            "fallback": True,
            "reason": f"Error in payload formation: {str(e)}",
            "details": {"error": str(e)}
        }
