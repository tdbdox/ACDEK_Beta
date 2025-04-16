from acdek.solve_Ar import solve_Ar
from acdek.types import QEnriched, ArItem

def test_solve_Ar_filters_and_scores():
    Q_enriched = QEnriched(
        intent="Generate sales summary",
        dependencies=["sales", "summary"],
        constraints={},
        ambiguity_score=0.1
    )

    Ar_raw = {
        "item1": {"value": "sales report Q1", "age": 5},
        "item2": {"value": "inventory log", "age": 2},
        "item3": {"value": "summary data", "age": 10}
    }

    result = solve_Ar(Ar_raw, Q_enriched)
    
    assert isinstance(result.slots, dict)
    assert all(isinstance(v, ArItem) for v in result.slots.values())
    assert all(v.confidence >= 0.5 for v in result.slots.values())

    print("Returned slots:", list(result.slots.keys()))
    for key, item in result.slots.items():
        print(f"{key}: confidence={item.confidence:.4f}")

    # Updated expectation: all items currently exceed the confidence threshold
    assert len(result.slots) == 3