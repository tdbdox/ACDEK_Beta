# /benchmarks/benchmark_case_02_ambiguity.py

from acdek.acdek_kernel import run_acdek_pipeline
from acdek.types import CuRaw, EfRaw

print("=== ACDEK BENCHMARK CASE 02 — AMBIGUITY STRESS TEST ===")

# Simulated ambiguous query
Q_raw = "Generate a performance summary for Q1"

# Multiple semantically related slot candidates
Ar_raw = {
    "sales_q1": {"value": "Sales data Q1", "age": 2},
    "revenue_q1": {"value": "Revenue breakdown Q1", "age": 3},
    "expenses_q1": {"value": "Expense report Q1", "age": 1},
    "summary_q1": {"value": "Team summary notes", "age": 1}
}

# Moderate ambiguity user profile
Cu_raw = CuRaw(
    traits={
        "domain": "finance",
        "risk_tolerance": "medium"
    }
)

ef_raw = EfRaw(
    urgency="medium",
    volatility="low"
)

# Optional: Override theta to loosen fallback thresholds
theta = {
    "confidence_floor": 0.6,
    "fallback_policy": "clarify",
    "tone": 0.65,
    "assertiveness": 0.7
}

# Run ACDEK pipeline
result = run_acdek_pipeline(Q_raw, Ar_raw, Cu_raw, ef_raw, theta)

# Display structured output
print("\n--- Ambiguity Stress Scenario ---")
print(f"Expected: clarify_required OR fallback_triggered")
print(f"Status: {result.get('status')}")
print(f"Gate Decision: {result.get('gate_decision')}")
print(f"Ambiguity Score: {result.get('ambiguity_score')}")
print(f"Modulation Vector: {result.get('modulation')}")
print(f"Reasoning: {result.get('reasoning')}")
print(f"Payload: {result.get('payload')}") 