from acdek.types import CuRaw, QEnriched, ArEnriched

def solve_Cu(Cu_raw: CuRaw, Q_enriched: QEnriched, Ar_enriched: ArEnriched) -> dict:
    if Cu_raw is None or not Cu_raw.traits:
        return {}  # Cu∅

    # Extract content from Q_enriched and Ar_enriched for fusion
    fused_traits = {
        "intent": Q_enriched.intent,
        "dependencies": Q_enriched.dependencies,
        "constraints": Q_enriched.constraints
    }

    for slot_key, ar_item in Ar_enriched.slots.items():
        fused_traits[f"ar_{slot_key}"] = ar_item.value

    # Combine raw user traits with enriched fusion
    return {**Cu_raw.traits, **fused_traits}
