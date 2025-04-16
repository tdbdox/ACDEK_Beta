from typing import List, Dict, Tuple, Optional
from acdek.types import QEnriched, CuRaw, EfRaw, QRaw
from embeddings.embed_utils import get_embedding, cosine_similarity, semantic_search, batch_embed
import logging
import re
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

# Intent definitions with descriptions
INTENTS = {
    "analyze_report": "Generate a detailed analysis of a business report.",
    "summarize_sales": "Provide a summary of recent sales data.",
    "compare_quarters": "Compare performance across different quarters.",
    "track_inventory": "Track the current inventory levels and trends."
}

# Common dependencies that queries might require
DEPENDENCY_CANDIDATES = [
    "report_id", "sales", "summary", "inventory", "comparison", 
    "quarter", "date_range", "product_id", "category", "region"
]

# Common formats or outputs users might request
FORMAT_PATTERNS = {
    "pdf": [r"(?:in|as|to|export to)\s+pdf", r"pdf\s+(?:format|file|document|report)", r"\bpdf\b"],
    "excel": [r"(?:in|as|to|export to)\s+excel", r"excel\s+(?:format|file|sheet|workbook)", r"\bexcel\b", r"\.xlsx?"],
    "csv": [r"(?:in|as|to|export to)\s+csv", r"csv\s+(?:format|file)", r"\bcsv\b", r"\.csv"],
    "json": [r"(?:in|as|to|export to)\s+json", r"json\s+(?:format|file)", r"\bjson\b", r"\.json"],
    "chart": [r"(?:as|in|with)\s+(?:a\s+)?chart", r"(?:visualize|plot|graph)", r"(?:bar|pie|line)\s+(?:chart|graph)"],
    "table": [r"(?:as|in|with)\s+(?:a\s+)?table", r"tabular\s+(?:format|view)"]
}

def detect_format_constraints(query_text: str) -> Dict[str, str]:
    """
    Detect output format constraints from query text.
    
    Args:
        query_text: The raw query text
        
    Returns:
        Dictionary of detected format constraints
    """
    constraints = {}
    query_lower = query_text.lower()
    
    # Check for each format pattern
    for format_name, patterns in FORMAT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                constraints["format"] = format_name
                logger.info(f"Detected format constraint: {format_name}")
                return constraints
    
    return constraints

def detect_time_constraints(query_text: str) -> Dict[str, str]:
    """
    Detect time-related constraints from query text.
    
    Args:
        query_text: The raw query text
        
    Returns:
        Dictionary of detected time constraints
    """
    constraints = {}
    query_lower = query_text.lower()
    
    # Check for specific time periods
    time_periods = {
        "last_week": [r"last\s+week", r"previous\s+week"],
        "last_month": [r"last\s+month", r"previous\s+month"],
        "last_quarter": [r"last\s+quarter", r"previous\s+quarter", r"q[1-4]"],
        "last_year": [r"last\s+year", r"previous\s+year", r"20\d{2}"],
        "ytd": [r"year\s+to\s+date", r"\bytd\b"],
        "mtd": [r"month\s+to\s+date", r"\bmtd\b"]
    }
    
    for period_name, patterns in time_periods.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                constraints["time_period"] = period_name
                logger.info(f"Detected time constraint: {period_name}")
                return constraints
    
    return constraints

def solve_Q(Q: QRaw, Cu: CuRaw, ef: EfRaw) -> QEnriched:
    """
    Process raw query to extract intent, dependencies, and constraints.
    
    Args:
        Q: Raw query input
        Cu: User context
        ef: Environment factors
        
    Returns:
        Enriched query with intent, dependencies and constraints
    """
    logger.info(f"Processing query: {Q.query}")
    
    # Get query embedding
    query_vec = get_embedding(Q.query)
    
    # Intent classification with batch embedding
    intent_candidates = list(INTENTS.values())
    intent_embeddings = batch_embed(intent_candidates)
    
    intent_scores = {}
    for intent_name, intent_desc in INTENTS.items():
        if intent_desc in intent_embeddings:
            score = cosine_similarity(query_vec, intent_embeddings[intent_desc])
            intent_scores[intent_name] = score
    
    # Get the intent with highest score
    if intent_scores:
        intent = max(intent_scores, key=intent_scores.get)
        best_score = intent_scores[intent]
        logger.info(f"Classified intent: {intent} with score {best_score:.4f}")
    else:
        # Fallback intent if something goes wrong
        intent = "analyze_report"
        intent_scores[intent] = 0.5
        logger.warning(f"No intent scores available, using fallback: {intent}")
    
    # Dependency extraction - get all candidates in batch for efficiency
    dependency_embeddings = batch_embed(DEPENDENCY_CANDIDATES)
    dependencies = []
    
    for dep in DEPENDENCY_CANDIDATES:
        if dep in dependency_embeddings:
            score = cosine_similarity(query_vec, dependency_embeddings[dep])
            if score >= 0.45:  # Threshold for semantic match
                dependencies.append(dep)
                logger.info(f"Matched dependency: {dep} with score {score:.4f}")
    
    # Constraint extraction
    constraints = {}
    
    # Check for format constraints
    format_constraints = detect_format_constraints(Q.query)
    if format_constraints:
        constraints.update(format_constraints)
    
    # Check for time constraints
    time_constraints = detect_time_constraints(Q.query)
    if time_constraints:
        constraints.update(time_constraints)
    
    # Calculate ambiguity score
    if intent_scores:
        # Lower score means more confident (less ambiguous)
        max_intent_score = max(intent_scores.values())
        # Average of top scores compared to max score
        top_intents = sorted(intent_scores.values(), reverse=True)[:2]
        if len(top_intents) > 1:
            # If top two scores are close, there's ambiguity
            score_gap = top_intents[0] - top_intents[1]
            ambiguity_score = 1.0 - (max_intent_score * score_gap)
        else:
            ambiguity_score = 1.0 - max_intent_score
    else:
        ambiguity_score = 0.5  # Default moderate ambiguity
    
    # Cap ambiguity between 0 and 1
    ambiguity_score = max(0.0, min(1.0, ambiguity_score))
    
    logger.info(f"Query ambiguity score: {ambiguity_score:.4f}")
    logger.info(f"Extracted constraints: {constraints}")
    
    # Return enriched query
    return QEnriched(
        intent=intent,
        dependencies=dependencies,
        constraints=constraints,
        ambiguity_score=ambiguity_score
    )