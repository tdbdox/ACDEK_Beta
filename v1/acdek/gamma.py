from acdek.types import QEnriched, ArEnriched

def verify(Q: QEnriched, Ar: ArEnriched, psi: dict, gamma_adjusted: float = 0.7) -> tuple[bool, dict]:
    failed_dependencies = []
    failed_constraints = []

    # Check all dependencies exist in Ar and meet confidence threshold
    for dep in Q.dependencies:
        print(f"🧪 Verifying dependency: {dep}")
        ar_item = Ar.slots.get(dep)
        if ar_item:
            print(f"  ↳ Found in Ar.slots with confidence: {ar_item.confidence}")
            if ar_item.confidence < gamma_adjusted:
                failed_dependencies.append((dep, ar_item.confidence))
        else:
            print("  ↳ NOT found in Ar.slots")
            failed_dependencies.append((dep, None))

    # Check constraints (if any) are satisfied
    for k, v in Q.constraints.items():
        if k not in Ar.slots or str(Ar.slots[k].value) != str(v):
            failed_constraints.append((k, v))

    if failed_dependencies or failed_constraints:
        return False, {
            "failed_dependencies": failed_dependencies,
            "failed_constraints": failed_constraints
        }

    return True, {}

def gamma(Q: QEnriched, Ar: ArEnriched, psi: dict) -> dict:
    gamma_adjusted = psi.get("confidence_floor", 0.7)
    passed, diagnostics = verify(Q, Ar, psi, gamma_adjusted)
    print(f"✅ gamma(): verify() returned passed={passed}, diagnostics={diagnostics}")
    if passed:
        print("🎯 gamma payload keys:", {
            "intent": Q.intent,
            "slots": {k: v.value for k, v in Ar.slots.items()},
            "constraints": Q.constraints,
            "confidence": {k: v.confidence for k, v in Ar.slots.items()}
        }.keys())
        return {
            "payload": {
                "intent": Q.intent,
                "slots": {k: v.value for k, v in Ar.slots.items()},
                "constraints": Q.constraints,
                "confidence": {k: v.confidence for k, v in Ar.slots.items()}
            }
        }
    else:
        return {
            "fallback": True,
            "reason": "Verification failed",
            "details": diagnostics
        }
