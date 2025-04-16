from acdek.solve_Q import solve_Q
from acdek.solve_Ar import solve_Ar
from acdek.solve_Cu import solve_Cu
from acdek.solve_ef import solve_ef
from acdek.gamma import gamma
from acdek.omega import omega
from acdek.gate import gate
from acdek.phi import phi
from acdek.types import CuRaw, EfRaw

def run_acdek_pipeline(Q_raw: str, Ar_raw: dict, Cu_raw: CuRaw, ef_raw: EfRaw) -> dict:
    # Step 1: Enrich Query
    Q_enriched = solve_Q(Q_raw, Cu_raw, ef_raw)
    print("🔍 Q_enriched.intent:", Q_enriched.intent)
    print("🔍 Q_enriched.dependencies:", Q_enriched.dependencies)
    print("🔍 Q_enriched.ambiguity_score:", Q_enriched.ambiguity_score)

    # Step 2: Enrich Data
    Ar_enriched = solve_Ar(Ar_raw, Q_enriched)
    print("🔍 Ar_enriched.slots:", list(Ar_enriched.slots.keys()))
    for k, v in Ar_enriched.slots.items():
        print(f"     ↳ {k} → value: {v.value}, confidence: {v.confidence:.2f}")

    # Step 3: Enrich User Context
    Cu_enriched = solve_Cu(Cu_raw, Q_enriched, Ar_enriched)

    # Step 4: Enrich Environment
    ef_enriched = solve_ef(ef_raw, Q_enriched, Cu_enriched)

    # Step 5: Modulation Vector
    theta = omega(Cu_enriched, ef_enriched)

    # Step 6: Verification / Payload Formation
    payload = gamma(Q_enriched, Ar_enriched, theta)

    # Handle fallback directly if payload indicates failure
    if "fallback" in payload:
        return {
            "status": "fallback_triggered",
            "reason": payload.get("reason", "Unknown failure"),
            "justification": "Returning safe fallback due to unsatisfied conditions.",
            "modulation": theta,
            "gate_decision": "fallback",
            "ambiguity_score": Q_enriched.ambiguity_score
        }

    # Unwrap payload from gamma
    payload = payload["payload"]
    print("📦 Final payload keys before gate:", payload.keys())
    print("📦 Confidence values before gate:", payload.get('confidence', {}))

    payload["ambiguity_score"] = Q_enriched.ambiguity_score
    payload["dependencies"] = Q_enriched.dependencies

    # Step 7: Gating Decision
    decision = gate(payload, theta)

    # Step 8: Execute or Fallback
    return phi(decision, payload, theta)
