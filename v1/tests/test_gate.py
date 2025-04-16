from acdek.gate import gate

def test_gate_execute():
    payload = {
        "confidence": {"a": 0.9, "b": 0.85},
        "ambiguity_score": 0.05
    }
    theta = {
        "confidence_floor": 0.7,
        "fallback_policy": "clarify"
    }
    assert gate(payload, theta) == "execute"

def test_gate_clarify_due_to_ambiguity():
    payload = {
        "confidence": {"a": 0.65},
        "ambiguity_score": 0.2
    }
    theta = {
        "confidence_floor": 0.7,
        "fallback_policy": "clarify"
    }
    assert gate(payload, theta) == "clarify"

def test_gate_fallback_due_to_low_confidence_and_no_clarify_policy():
    payload = {
        "confidence": {"a": 0.6},
        "ambiguity_score": 0.1
    }
    theta = {
        "confidence_floor": 0.7,
        "fallback_policy": "safe"
    }
    assert gate(payload, theta) == "fallback"

def test_gate_clarify_due_to_fallback_policy():
    payload = {
        "confidence": {"a": 0.6},
        "ambiguity_score": 0.2  # Raised to trigger clarify condition
    }
    theta = {
        "confidence_floor": 0.7,
        "fallback_policy": "clarify"
    }
    assert gate(payload, theta) == "clarify"
