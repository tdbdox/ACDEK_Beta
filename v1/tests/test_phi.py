from acdek.phi import phi

def test_phi_execute_path():
    gate_decision = "execute"
    payload = {
        "intent": "analyze data",
        "slots": {"report_id": "123"},
        "constraints": {"format": "pdf"},
        "confidence": {"report_id": 0.9}
    }
    theta = {
        "tone": 0.7,
        "assertiveness": 0.8,
        "confidence_floor": 0.75,
        "fallback_policy": "clarify"
    }

    result = phi(gate_decision, payload, theta)
    assert result["status"] == "executed"
    assert result["intent"] == "analyze data"
    assert "modulation" in result

def test_phi_clarify_path():
    gate_decision = "clarify"
    payload = {
        "slots": {"field1": "value1"},
    }
    theta = {
        "tone": 0.5,
        "assertiveness": 0.5,
        "confidence_floor": 0.7,
        "fallback_policy": "clarify"
    }

    result = phi(gate_decision, payload, theta)
    assert result["status"] == "clarify_required"
    assert "clarify_fields" in result
    assert result["clarify_fields"] == ["field1"]

def test_phi_fallback_path():
    gate_decision = "fallback"
    payload = {}
    theta = {
        "tone": 0.4,
        "assertiveness": 0.3,
        "confidence_floor": 0.8,
        "fallback_policy": "safe"
    }

    result = phi(gate_decision, payload, theta)
    assert result["status"] == "fallback_triggered"
    assert "reason" in result
    assert result["modulation"]["fallback_policy"] == "safe"
