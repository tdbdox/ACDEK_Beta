from acdek.acdek_kernel import run_acdek_pipeline
from acdek.types import CuRaw, EfRaw

print("=== ACDEK BENCHMARK CASE 01 — SEMANTIC PIPELINE ===")

# Simple semantic scenario
Q_raw = "Can you create a quick report based on the latest sales?"
Ar_raw = {
    "sales": {"value": "Q1 sales summary report", "age": 1},
    "format": {"value": "pdf", "age": 0}
}
Cu_raw = CuRaw(
    traits={
        "role": "manager",
        "risk_tolerance": "medium"
    }
)
ef_raw = EfRaw(
    variables={
        "urgency": "normal",
        "volatility": "low"
    }
)

theta = {
    "tone": 0.7,
    "assertiveness": 0.75,
    "confidence_floor": 0.6,
    "fallback_policy": "clarify"
}

result = run_acdek_pipeline(Q_raw, Ar_raw, Cu_raw, ef_raw, theta)

print("\n--- Simple Semantic Scenario ---")
print("Expected: executed")
print("Status:", result["status"])
print("Gate Decision:", result.get("gate_decision", "N/A"))
print("Ambiguity Score:", result.get("ambiguity_score", "N/A"))
print("Modulation Vector:", result.get("modulation", {}))
print("Reasoning:", result.get("reason", "N/A"))
print("Payload:", result.get("output", {}))
