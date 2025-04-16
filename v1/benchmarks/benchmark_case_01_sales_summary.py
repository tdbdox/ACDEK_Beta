from acdek.acdek_kernel import run_acdek_pipeline
from acdek.types import CuRaw, EfRaw

scenarios = [
    {
        "name": "Execute Scenario",
        "Q_raw": "Generate a report using sales",
        "Ar_raw": {
            "report_id": {"value": "Q1 sales report", "age": 1},
            "format": {"value": "pdf", "age": 1}
        },
        "expected": "executed"
    },
    {
        "name": "Fallback Scenario (Low Confidence)",
        "Q_raw": "Generate a report using sales",
        "Ar_raw": {
            "sales": {"value": "fuzzy value", "age": 40}
        },
        "expected": "fallback_triggered"
    },
    {
        "name": "Clarify Scenario (High Ambiguity)",
        "Q_raw": "Handle it",
        "Ar_raw": {
            "task": {"value": "review process", "age": 1}
        },
        "expected": "clarify_required"
    }
]

Cu_raw = CuRaw(traits={"role": "analyst", "risk_tolerance": "medium"})
ef_raw = EfRaw(variables={"urgency": "normal", "volatility": "low"})

print("=== ACDEK BENCHMARK CASE 01 ===")
for case in scenarios:
    print(f"\n--- {case['name']} ---")
    result = run_acdek_pipeline(case["Q_raw"], case["Ar_raw"], Cu_raw, ef_raw)
    status = result.get("status")
    modulation = result.get("modulation", {})

    # Updated fallback-safe access
    payload = result.get("payload", {})
    ambiguity_score = result.get("ambiguity_score", payload.get("ambiguity_score", "N/A"))
    gate_decision = result.get("gate_decision", "N/A")
    reasoning = result.get("reason", "N/A")

    print(f"Expected: {case['expected']}")
    print(f"Status: {status}")
    print(f"Gate Decision: {gate_decision}")
    print(f"Ambiguity Score: {ambiguity_score}")
    print(f"Modulation Vector: {modulation}")
    print(f"Reasoning: {reasoning}")
    print(f"Payload: {payload}")
