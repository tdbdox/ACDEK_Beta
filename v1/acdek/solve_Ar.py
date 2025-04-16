from acdek.types import ArItem, ArEnriched, QEnriched
from typing import List, Dict
import math

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(i*j for i, j in zip(a, b))
    norm_a = math.sqrt(sum(i*i for i in a))
    norm_b = math.sqrt(sum(j*j for j in b))
    return dot / (norm_a * norm_b + 1e-8)

def dummy_embed(text: str) -> List[float]:
    return [float(ord(c)) for c in text][:10]  # Simplified embedding

def solve_Ar(Ar_raw: Dict[str, Dict], Q_enriched: QEnriched) -> ArEnriched:
    result = {}
    query_embedding = dummy_embed(Q_enriched.intent)

    for key, data in Ar_raw.items():
        value_text = str(data.get("value", ""))
        value_embedding = dummy_embed(value_text)

        sim_score = cosine_similarity(query_embedding, value_embedding)
        raw_confidence = sim_score  # similarity as proxy for confidence
        decay_lambda = 0.01
        age = data.get("age", 0)
        decayed_confidence = raw_confidence * math.exp(-decay_lambda * age)

        if decayed_confidence >= 0.5:  # threshold
            result[key] = ArItem(
                key=key,
                value=data.get("value"),
                confidence=decayed_confidence,
                age=age
            )

    return ArEnriched(slots=result)

