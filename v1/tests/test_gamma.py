from acdek.gamma import gamma
from acdek.types import QEnriched, ArEnriched, ArItem

def test_gamma_verifies_payload_success():
    Q = QEnriched(
        intent="fetch user report",
        dependencies=["report_id"],
        constraints={"report_id": "123"},
        ambiguity_score=0.1
    )

    Ar = ArEnriched(slots={
        "report_id": ArItem(key="report_id", value="123", confidence=0.95, age=1)
    })

    psi = {"confidence_floor": 0.7}
    result = gamma(Q, Ar, psi)

    assert "fallback" not in result
    assert result["intent"] == "fetch user report"
    assert result["slots"]["report_id"] == "123"
    assert result["confidence"]["report_id"] >= 0.7

def test_gamma_triggers_fallback_on_low_confidence():
    Q = QEnriched(
        intent="fetch user report",
        dependencies=["report_id"],
        constraints={"report_id": "123"},
        ambiguity_score=0.1
    )

    Ar = ArEnriched(slots={
        "report_id": ArItem(key="report_id", value="123", confidence=0.5, age=1)
    })

    psi = {"confidence_floor": 0.7}
    result = gamma(Q, Ar, psi)

    assert result["fallback"] is True
    assert result["reason"] == "Verification failed"
    assert "report_id" in result["missing_or_low_confidence"]
