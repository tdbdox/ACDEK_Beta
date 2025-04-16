from acdek.solve_Cu import solve_Cu
from acdek.types import CuRaw, QEnriched, ArEnriched, ArItem

def test_solve_Cu_with_context():
    Cu_raw = CuRaw(traits={"role": "analyst"})
    Q_enriched = QEnriched(
        intent="generate insights",
        dependencies=["insight"],
        constraints={"format": "pdf"},
        ambiguity_score=0.1
    )
    Ar_enriched = ArEnriched(slots={
        "sales_data": ArItem(key="sales_data", value="Q1 report", confidence=0.9, age=3)
    })

    result = solve_Cu(Cu_raw, Q_enriched, Ar_enriched)

    assert isinstance(result, dict)
    assert "role" in result
    assert "intent" in result
    assert "ar_sales_data" in result
    assert result["ar_sales_data"] == "Q1 report"

def test_solve_Cu_without_context():
    Cu_raw = CuRaw(traits={})
    Q_enriched = QEnriched(
        intent="generate insights",
        dependencies=["insight"],
        constraints={"format": "pdf"},
        ambiguity_score=0.1
    )
    Ar_enriched = ArEnriched(slots={})

    result = solve_Cu(Cu_raw, Q_enriched, Ar_enriched)

    assert isinstance(result, dict)
    assert result == {}

def test_solve_Cu_missing_Cu_raw():
    Q_enriched = QEnriched(
        intent="generate insights",
        dependencies=["insight"],
        constraints={"format": "pdf"},
        ambiguity_score=0.1
    )
    Ar_enriched = ArEnriched(slots={})

    result = solve_Cu(None, Q_enriched, Ar_enriched)

    assert isinstance(result, dict)
    assert result == {}

def test_solve_Cu_empty_Cu_raw():
    Cu_raw = CuRaw(traits={})
    Q_enriched = QEnriched(
        intent="generate insights",
        dependencies=["insight"],
        constraints={"format": "pdf"},
        ambiguity_score=0.1
    )
    Ar_enriched = ArEnriched(slots={})

    result = solve_Cu(Cu_raw, Q_enriched, Ar_enriched)

    assert isinstance(result, dict)
    assert result == {}
