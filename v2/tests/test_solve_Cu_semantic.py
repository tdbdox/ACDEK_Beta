from acdek.types import CuRaw
from acdek.solve_Cu import solve_Cu
from acdek.registry import register_semantic_class
from acdek.semantic_defs import CognitiveState

register_semantic_class("CognitiveState", CognitiveState)

def test_semantic_class_assignment():
    Cu_raw = CuRaw(traits={
        "focus_level": 0.9,
        "attention_span": 3.2,
        "stress_index": 0.1
    })

    enriched = solve_Cu(Cu_raw, Q_enriched=None, Ar_enriched=None)

    print(f"Matched class: {enriched['semantic_class']}")
    print(f"Confidence: {enriched['confidence']:.4f}")
    print(f"Object: {enriched['object']}")

    assert enriched["semantic_class"] == "CognitiveState"
    assert enriched["confidence"] > 0.75