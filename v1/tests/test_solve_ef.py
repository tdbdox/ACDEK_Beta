from acdek.solve_ef import solve_ef
from acdek.types import EfRaw, QEnriched

def test_solve_ef_filters_merges_and_scores():
    ef_raw = EfRaw(variables={
        "season": "spring",
        "event": "product launch",
        "location": "Las Vegas",
        "currency": "USD"
    })

    Q_enriched = QEnriched(
        intent="analyze sales trends from spring",
        dependencies=["sales", "trends"],
        constraints={},
        ambiguity_score=0.1
    )

    Cu_enriched = {
        "role": "analyst",
        "focus": "seasonal sales"
    }

    result = solve_ef(ef_raw, Q_enriched, Cu_enriched)

    assert isinstance(result, dict)
    assert "season" in result  # expected to match on "spring"
    assert "role" in result  # from Cu_enriched
    assert "intent" in result  # from Q_enriched
    assert all("score" in v for v in result.values())
