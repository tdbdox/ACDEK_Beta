"""
Memory-Conditioned Modulation Demo
This simple benchmark demonstrates how memory conditions modulation parameters.
"""

from acdek.types import QRaw, CuRaw, EfRaw, Theta
from mock_db.query_requests import memory_conditioned_query
from acdek.omega import omega, extract_memory_trace, get_theta_values
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_demo():
    """Run a simple demo of memory-conditioned modulation."""
    print("=== MEMORY-CONDITIONED MODULATION DEMO ===")
    
    # Create sample user context with RiskProfile class and memory trace
    Cu_enriched = {
        "semantic_class": "RiskProfile",
        "confidence": 0.92,
        "coverage": 1.0,
        "status": "success",
        "traits": {
            "risk_tolerance": memory_conditioned_query["memory_trace"]["learned_traits"]["risk_tolerance"],
            "analytical_depth": memory_conditioned_query["memory_trace"]["learned_traits"]["analytical_depth"],
            "decision_style": memory_conditioned_query["decision_style"],
            "evidence_threshold": memory_conditioned_query["memory_trace"]["learned_traits"]["evidence_threshold"],
            "assertiveness": memory_conditioned_query["memory_trace"]["learned_traits"]["assertiveness"],
            "role": "risk_analyst",
            "department": memory_conditioned_query["department"],
            "memory_trace": memory_conditioned_query["memory_trace"]
        }
    }
    
    # Simple environment factors 
    ef_enriched = {
        "urgency": "low",
        "volatility": "high"
    }
    
    # Original theta parameters
    original_theta = Theta(
        confidence_floor=0.6,
        tone=0.6,
        assertiveness=0.7,
        fallback_policy="clarify"
    )
    
    print("\n=== Original Modulation Parameters ===")
    print(f"Confidence Floor: {original_theta.confidence_floor:.2f}")
    print(f"Assertiveness: {original_theta.assertiveness:.2f}")
    print(f"Tone: {original_theta.tone:.2f}")
    print(f"Fallback Policy: {original_theta.fallback_policy}")
    
    # Apply memory conditioning
    conditioned_theta = omega(Cu_enriched, ef_enriched, override=original_theta)
    
    if conditioned_theta == original_theta:
        print("\n⚠️ WARNING: Modulation parameters were not changed by memory conditioning!")
    
    print("\n=== Memory-Conditioned Modulation Parameters ===")
    
    # Extract values for display
    if isinstance(conditioned_theta, dict):
        print(f"Confidence Floor: {conditioned_theta.get('confidence_floor', 'N/A')}")
        print(f"Assertiveness: {conditioned_theta.get('assertiveness', 'N/A')}")
        print(f"Tone: {conditioned_theta.get('tone', 'N/A')}")
        print(f"Fallback Policy: {conditioned_theta.get('fallback_policy', 'N/A')}")
    else:
        print(f"Confidence Floor: {conditioned_theta.confidence_floor:.2f}")
        print(f"Assertiveness: {conditioned_theta.assertiveness:.2f}")
        print(f"Tone: {conditioned_theta.tone:.2f}")
        print(f"Fallback Policy: {conditioned_theta.fallback_policy}")
    
    # Check for changes
    original_values = get_theta_values(original_theta)
    conditioned_values = get_theta_values(conditioned_theta)
    
    print("\n=== Analysis of Memory Conditioning Effects ===")
    for param_name in ["confidence_floor", "assertiveness", "tone", "fallback_policy"]:
        original = original_values.get(param_name)
        conditioned = conditioned_values.get(param_name)
        
        if original != conditioned:
            print(f"✓ {param_name.title()}: {original} → {conditioned} (CHANGED)")
        else:
            print(f"✗ {param_name.title()}: {original} → {conditioned} (unchanged)")
    
    # Extract memory trace info
    memory_trace = extract_memory_trace(Cu_enriched)
    learned_traits = memory_trace.get("learned_traits", {})
    
    print("\n=== User Memory Trace (learned traits) ===")
    for trait, value in learned_traits.items():
        print(f"{trait}: {value}")
    
    return conditioned_theta

if __name__ == "__main__":
    run_demo()