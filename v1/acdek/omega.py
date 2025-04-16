from acdek.types import ModulationVector

def omega(Cu_enriched: dict, ef_enriched: dict) -> dict:
    # Derive tone and assertiveness based on role or environmental signal
    tone = 0.5
    if Cu_enriched.get("role") == "executive":
        tone = 0.8
    elif Cu_enriched.get("role") == "intern":
        tone = 0.3

    assertiveness = 0.6
    if ef_enriched.get("urgency") == "high":
        assertiveness = 0.9

    confidence_floor = 0.7
    if ef_enriched.get("volatility") == "high":
        confidence_floor = 0.85

    fallback_policy = "clarify"
    if Cu_enriched.get("risk_tolerance") == "low":
        fallback_policy = "safe"

    return {
        "tone": tone,
        "assertiveness": assertiveness,
        "confidence_floor": confidence_floor,
        "fallback_policy": fallback_policy
    }
