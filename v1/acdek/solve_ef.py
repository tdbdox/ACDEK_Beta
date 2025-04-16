from acdek.types import QEnriched, EfRaw

def solve_ef(ef_raw: EfRaw, Q_enriched: QEnriched, Cu_enriched: dict) -> dict:
    # Step 1: Relevance filter — keep keys that appear in query or context
    keywords = Q_enriched.intent.lower().split()
    keywords += [str(dep).lower() for dep in Q_enriched.dependencies]
    keywords += [str(val).lower() for val in Cu_enriched.values()]

    filtered_ef = {
        k: v for k, v in ef_raw.variables.items()
        if any(word in str(v).lower() or word in k.lower() for word in keywords)
    }

    # Step 2: Merge with query and context
    merged = {**filtered_ef, **Cu_enriched, "intent": Q_enriched.intent}

    # Step 3: Score (very basic relevance proxy)
    scored = {
        k: {"value": v, "score": len(str(v)) / 10.0}  # dummy scoring by length
        for k, v in merged.items()
    }

    return scored
