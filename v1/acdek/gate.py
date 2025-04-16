def gate(payload: dict, theta: dict) -> str:
    """
    Returns:
    - "execute" if all criteria are met
    - "clarify" if ambiguity is high and fallback policy is set to clarify
    - "fallback" otherwise
    """
    dependencies = payload.get("dependencies", list(payload.get("confidence", {}).keys()))
    confidence_map = payload.get("confidence", {})
    confidence_values = [confidence_map.get(dep, 0) for dep in dependencies]
    min_confidence = min(confidence_values) if confidence_values else 0
    confidence_floor = theta.get("confidence_floor", 0.7)
    fallback_policy = theta.get("fallback_policy", "clarify")
    ambiguity_score = payload.get("ambiguity_score", 0.0)

    print(f"🧪 gate(): min_confidence={min_confidence}, floor={confidence_floor}")
    print(f"🧪 gate(): ambiguity_score={ambiguity_score}, policy={fallback_policy}")

    if min_confidence >= confidence_floor:
        return "execute"
    elif ambiguity_score > 0.15 and fallback_policy == "clarify":
        return "clarify"
    else:
        return "fallback"
