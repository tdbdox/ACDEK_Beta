# Modulation Profiles — Simulated user personas with different interaction needs

modulation_profiles = {
    "default": {
        "tone": 0.5,
        "assertiveness": 0.5,
        "confidence_floor": 0.7,
        "fallback_policy": "clarify"
    },
    "executive_summary": {
        "tone": 0.6,
        "assertiveness": 0.8,
        "confidence_floor": 0.75,
        "fallback_policy": "streamline"
    },
    "casual_briefing": {
        "tone": 0.4,
        "assertiveness": 0.3,
        "confidence_floor": 0.5,
        "fallback_policy": "clarify"
    },
    "technical_analysis": {
        "tone": 0.7,
        "assertiveness": 0.9,
        "confidence_floor": 0.85,
        "fallback_policy": "expand"
    }
}
