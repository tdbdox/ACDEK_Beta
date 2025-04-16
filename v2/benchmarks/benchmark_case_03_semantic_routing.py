from acdek.acdek_kernel import run_acdek_pipeline
from acdek.types import QRaw, ArRaw, CuRaw, EfRaw, Theta

print("=== ACDEK BENCHMARK CASE 03 — SEMANTIC ROUTING TEST ===")

Q_raw = QRaw(query="Summarize user behavior metrics for report")
Ar_raw = ArRaw(items={
    "activity_summary": {"value": "User active 6.3 hrs/day", "age": 1},
    "focus_score": {"value": "0.84", "age": 1},
    "sleep_quality": {"value": "high", "age": 1},
})

Cu_raw = CuRaw(traits={
    "focus_level": 0.84,
    "attention_span": 3.2,
    "stress_index": 0.2
})

ef_raw = EfRaw(
    urgency="low",
    volatility="medium"
)

theta = Theta(
    confidence_floor=0.6,
    tone=0.5,
    assertiveness=0.6,
    fallback_policy="clarify"
)

result = run_acdek_pipeline(Q_raw, Ar_raw, Cu_raw, ef_raw, theta)

print("\n--- Semantic Class Routing Check ---")
print(f"Semantic Class: {result.get('semantic_class')}")

# Handle the case where semantic_class_confidence might be None
class_confidence = result.get('semantic_class_confidence')
if class_confidence is not None:
    print(f"Class Confidence: {class_confidence:.2f}")
else:
    print("Class Confidence: None")

print(f"Final Status: {result.get('status', 'N/A')}")
print(f"Payload: {result.get('payload', 'N/A')}")
print(f"Reasoning: {result.get('reasoning', 'N/A')}")
print(f"Modulation: {result.get('modulation', 'N/A')}")