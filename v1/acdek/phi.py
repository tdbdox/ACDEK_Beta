def phi(gate_decision: str, payload: dict, theta: dict) -> dict:
    if gate_decision == "execute":
        return {
            "status": "executed",
            "intent": payload.get("intent"),
            "output": payload.get("slots"),
            "constraints": payload.get("constraints"),
            "confidence": payload.get("confidence"),
            "modulation": theta
        }
    elif gate_decision == "clarify":
        return {
            "status": "clarify_required",
            "reason": "Ambiguity exceeds threshold",
            "clarify_fields": list(payload.get("slots", {}).keys()),
            "modulation": theta
        }
    else:  # fallback
        return {
            "status": "fallback_triggered",
            "reason": "Confidence or constraints unsatisfied",
            "justification": "Returning safe fallback due to unsatisfied conditions.",
            "modulation": theta
        }
