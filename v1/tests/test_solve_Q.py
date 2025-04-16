from acdek.solve_Q import solve_Q
from acdek.types import CuRaw, EfRaw

def test_solve_Q_basics():
    Q = "Generate the financial summary from last quarter"
    Cu = CuRaw(traits={})
    ef = EfRaw(variables={})

    enriched = solve_Q(Q, Cu, ef)
    assert enriched.intent is not None
    assert enriched.ambiguity_score <= 1.0

def test_solve_Q_returns_full_structure():
    Q = "Summarize customer insights from March sales data"
    Cu = CuRaw(traits={"role": "analyst"})
    ef = EfRaw(variables={"season": "spring"})

    enriched = solve_Q(Q, Cu, ef)

    assert isinstance(enriched.intent, str)
    assert isinstance(enriched.dependencies, list)
    assert isinstance(enriched.constraints, dict)
    assert isinstance(enriched.ambiguity_score, float)
    assert 0.0 <= enriched.ambiguity_score <= 1.0
