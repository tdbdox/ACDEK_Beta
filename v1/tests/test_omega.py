from acdek.omega import omega

def test_omega_generates_correct_modulation_vector():
    Cu_enriched = {
        "role": "executive",
        "risk_tolerance": "low"
    }

    ef_enriched = {
        "urgency": "high",
        "volatility": "high"
    }

    result = omega(Cu_enriched, ef_enriched)

    assert isinstance(result, dict)
    assert result["tone"] == 0.8
    assert result["assertiveness"] == 0.9
    assert result["confidence_floor"] == 0.85
    assert result["fallback_policy"] == "safe"

def test_omega_with_different_conditions():
    Cu_enriched = {
        "role": "manager",
        "risk_tolerance": "high"
    }

    ef_enriched = {
        "urgency": "low",
        "volatility": "medium"
    }

    result = omega(Cu_enriched, ef_enriched)

    assert isinstance(result, dict)
    assert result["tone"] == 0.5
    assert result["assertiveness"] == 0.6
    assert result["confidence_floor"] == 0.7
    assert result["fallback_policy"] == "clarify"
