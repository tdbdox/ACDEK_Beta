#!/usr/bin/env python3
"""
Enhanced Contradiction Detection Module for ACDEK v3 Grounding Engine

This module provides sophisticated contradiction detection between 
response and reference texts to improve hallucination detection.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from pathlib import Path
import sys
import json

# Add parent directories to path to enable imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK data: {str(e)}. Tokenization may be limited.")

# Domain-specific contradiction patterns
MEDICAL_CONTRADICTION_PATTERNS = [
    {
        "name": "cure_claim",
        "pattern": r"(?i)(cure|cures|cured|curative|healing|heals|treats|reverses|eliminates)",
        "context": r"(?i)(cancer|diabetes|alzheimer|arthritis|dementia|autism|hiv|aids)",
        "match_type": "positive_claim",
        "confidence": 0.85,
        "reason": "Claims about curing serious medical conditions"
    },
    {
        "name": "always_never",
        "pattern": r"(?i)(always|never|all patients|no patients|100\% of cases|in every case)",
        "context": r"(?i)(symptoms|treatment|response|side effects|outcomes|causes)",
        "match_type": "absolute_claim",
        "confidence": 0.80,
        "reason": "Absolute claims about medical treatments or outcomes"
    },
    {
        "name": "opposing_actions",
        "pattern": r"(?i)(increases|decreases|raises|lowers|improves|worsens|strengthens|weakens)",
        "context": r"(?i)(blood pressure|heart rate|cholesterol|immune system|inflammation)",
        "match_type": "opposing_actions",
        "confidence": 0.75,
        "reason": "Contradictory claims about physiological effects"
    },
    {
        "name": "safety_claim",
        "pattern": r"(?i)(safe|unsafe|harmless|dangerous|risk-free|no side effects)",
        "context": r"(?i)(medication|treatment|procedure|vaccine|surgery|therapy)",
        "match_type": "absolute_safety",
        "confidence": 0.85,
        "reason": "Absolute safety claims about medical interventions"
    },
    {
        "name": "causation",
        "pattern": r"(?i)(causes|caused by|leads to|results in|directly affects|prevents)",
        "context": r"(?i)(disease|condition|disorder|illness|syndrome|symptoms)",
        "match_type": "causation",
        "confidence": 0.75,
        "reason": "Strong causation claims in medicine"
    }
]

FINANCIAL_CONTRADICTION_PATTERNS = [
    {
        "name": "guaranteed_returns",
        "pattern": r"(?i)(guaranteed|certain|assured|definite|always|never loses)",
        "context": r"(?i)(returns|profit|gains|investment|market|stocks|bonds)",
        "match_type": "absolute_claim",
        "confidence": 0.90,
        "reason": "Claims about guaranteed financial returns"
    },
    {
        "name": "tax_absolutes",
        "pattern": r"(?i)(tax-free|no taxes|avoid all taxes|completely deductible)",
        "context": r"(?i)(investment|income|gains|withdrawal|distribution|account)",
        "match_type": "tax_claim",
        "confidence": 0.85,
        "reason": "Absolute claims about tax treatments"
    },
    {
        "name": "market_predictions",
        "pattern": r"(?i)(will rise|will fall|is certain to|bound to|definitely|undoubtedly)",
        "context": r"(?i)(market|stock|index|price|rate|economy)",
        "match_type": "prediction",
        "confidence": 0.80,
        "reason": "Certain predictions about market movements"
    },
    {
        "name": "risk_claims",
        "pattern": r"(?i)(risk-free|no risk|zero risk|completely safe|complete protection)",
        "context": r"(?i)(investment|strategy|portfolio|fund|product)",
        "match_type": "risk_claim",
        "confidence": 0.90,
        "reason": "Claims about absence of financial risk"
    },
    {
        "name": "regulation_claims",
        "pattern": r"(?i)(unregulated|regulated|oversight|compliance|reporting|disclosure)",
        "context": r"(?i)(account|investment|bank|broker|advisor|transaction)",
        "match_type": "regulation",
        "confidence": 0.75,
        "reason": "Contradictory claims about financial regulation"
    }
]

LEGAL_CONTRADICTION_PATTERNS = [
    {
        "name": "legal_absolutes",
        "pattern": r"(?i)(always|never|in all cases|in every jurisdiction|universal)",
        "context": r"(?i)(law|legal|court|judge|ruling|precedent|statute|regulation)",
        "match_type": "legal_absolute",
        "confidence": 0.85,
        "reason": "Absolute claims about legal applications"
    },
    {
        "name": "legal_rights",
        "pattern": r"(?i)(right to|entitled to|guaranteed|protected by law|legal obligation)",
        "context": r"(?i)(constitution|amendment|statute|law|regulation|code)",
        "match_type": "rights_claim",
        "confidence": 0.80,
        "reason": "Claims about absolute legal rights"
    },
    {
        "name": "jurisdiction_claims",
        "pattern": r"(?i)(all states|every country|universally|internationally|globally)",
        "context": r"(?i)(law|legal system|courts|enforcement|application|statute)",
        "match_type": "jurisdiction",
        "confidence": 0.85,
        "reason": "Overgeneralized claims about legal jurisdictions"
    },
    {
        "name": "precedent_claims",
        "pattern": r"(?i)(establishes|overturns|reverses|confirms|denies|creates precedent)",
        "context": r"(?i)(case|ruling|decision|court|supreme court|appellate)",
        "match_type": "precedent",
        "confidence": 0.75,
        "reason": "Claims about legal precedent"
    },
    {
        "name": "liability_claims",
        "pattern": r"(?i)(always liable|never liable|complete immunity|absolute protection)",
        "context": r"(?i)(damages|lawsuit|claim|tort|negligence|responsibility)",
        "match_type": "liability",
        "confidence": 0.80,
        "reason": "Absolute claims about legal liability"
    }
]

# Numerical pattern detection
NUMERIC_PATTERN = re.compile(r'(?<!\w)(\d+(?:\.\d+)?)\s*(%|percent|thousand|million|billion|trillion)?')
YEAR_PATTERN = re.compile(r'\b(19\d{2}|20\d{2})\b')  # Years 1900-2099
MONETARY_PATTERN = re.compile(r'(?<!\w)[$€£¥]\s*(\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)')
RATIO_PATTERN = re.compile(r'(?<!\w)(\d+)\s*(?::|to|:to:)\s*(\d+)')

class ConflictDetector:
    """
    Enhanced contradiction detection between response and reference texts.
    """
    
    def __init__(self):
        """
        Initialize the conflict detector with appropriate data structures.
        """
        # Load domain-specific patterns
        self.domain_patterns = {
            "medical": MEDICAL_CONTRADICTION_PATTERNS,
            "legal": LEGAL_CONTRADICTION_PATTERNS,
            "financial": FINANCIAL_CONTRADICTION_PATTERNS,
            # Default to empty list for unknown domains
        }
        
        # Common negation words
        self.negation_words = {
            "not", "no", "never", "none", "nobody", "nothing", "nowhere", 
            "neither", "nor", "without", "absence", "lack", "cannot", "can't",
            "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
            "doesn't", "don't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't",
            "mightn't", "mustn't", "deny", "denies", "denied", "reject", "rejects",
            "rejected", "exclude", "excludes", "excluded", "false", "untrue", "incorrectly"
        }
        
        # Common contradiction pairs
        self.contradiction_pairs = [
            ("increase", "decrease"),
            ("improve", "worsen"),
            ("raise", "lower"),
            ("higher", "lower"),
            ("more", "less"),
            ("larger", "smaller"),
            ("better", "worse"),
            ("positive", "negative"),
            ("present", "absent"),
            ("include", "exclude"),
            ("required", "optional"),
            ("mandatory", "voluntary"),
            ("always", "never"),
            ("all", "none"),
            ("everyone", "no one"),
            ("everywhere", "nowhere"),
            ("everything", "nothing"),
            ("cause", "prevent"),
            ("create", "destroy"),
            ("begin", "end"),
            ("start", "stop"),
            ("open", "close"),
            ("accept", "reject"),
            ("allow", "prohibit"),
            ("permit", "forbid"),
            ("legal", "illegal"),
            ("safe", "dangerous"),
            ("healthy", "unhealthy"),
            ("effective", "ineffective"),
            ("success", "failure"),
            ("true", "false"),
            ("correct", "incorrect"),
            ("right", "wrong"),
            ("advantage", "disadvantage"),
            ("benefit", "harm"),
            ("help", "hurt"),
            ("gain", "loss"),
            ("profit", "deficit"),
            ("increase", "reduction"),
            ("addition", "subtraction"),
            ("expansion", "contraction")
        ]
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text for comparison.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = ''.join(c for c in text if c not in string.punctuation)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_key_claims(self, text: str) -> List[str]:
        """
        Extract key claims from text for contradiction analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of key claim sentences
        """
        # Start with sentence tokenization
        try:
            sentences = sent_tokenize(text)
        except Exception:
            # Fallback to simple splitting
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter for sentences with strong claim language
        claim_indicators = [
            r'\b(?:is|are|was|were|will be|must be|should be|always|never)\b',
            r'\b(?:cause|causes|lead|leads|result|results)\b',
            r'\b(?:increase|decrease|improve|reduce|prevent|treat|cure)\b',
            r'\b(?:show|demonstrate|prove|confirm|verify|establish)\b',
            r'\b(?:require|obligate|mandate|necessitate|demand)\b'
        ]
        
        claim_pattern = '|'.join(claim_indicators)
        
        key_claims = []
        for sentence in sentences:
            if re.search(claim_pattern, sentence, re.IGNORECASE):
                key_claims.append(sentence)
        
        return key_claims
    
    def _extract_negations(self, text: str) -> Set[str]:
        """
        Extract negated terms from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Set of negated terms
        """
        negated_terms = set()
        
        # First tokenize into sentences
        try:
            sentences = sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            
            # Find negation words
            for i, word in enumerate(words):
                if word in self.negation_words:
                    # Look at nearby words (context window of 5)
                    start = max(0, i - 1)
                    end = min(len(words), i + 6)
                    
                    # Extract the negated terms
                    for j in range(start, end):
                        if j != i and words[j] not in self.negation_words and len(words[j]) > 3:
                            negated_terms.add(words[j])
        
        return negated_terms
    
    def _detect_term_contradictions(self, response: str, reference: str) -> Tuple[bool, float, str]:
        """
        Detect contradictions based on term-level analysis.
        
        Args:
            response: Response text
            reference: Reference text
            
        Returns:
            Tuple of (contradiction_found, confidence, reason)
        """
        # Clean texts
        clean_response = self._clean_text(response)
        clean_reference = self._clean_text(reference)
        
        # Extract negated terms
        response_negations = self._extract_negations(response)
        reference_negations = self._extract_negations(reference)
        
        # Check for terms that are negated in one text but affirmed in the other
        response_words = set(clean_response.split())
        reference_words = set(clean_reference.split())
        
        contradiction_terms = []
        
        # Check for terms negated in reference but affirmed in response
        for term in reference_negations:
            if term in response_words and term not in response_negations:
                contradiction_terms.append(term)
        
        # Check for terms negated in response but affirmed in reference
        for term in response_negations:
            if term in reference_words and term not in reference_negations:
                contradiction_terms.append(term)
        
        if contradiction_terms:
            term_list = ", ".join(contradiction_terms[:3])  # List top 3 terms
            return True, 0.7, f"Contradictory negation of terms: {term_list}"
        
        # Check for presence of direct contradictions
        for term1, term2 in self.contradiction_pairs:
            # Check if contradictory terms appear in response and reference
            if (term1 in clean_response and term2 in clean_reference) or \
               (term2 in clean_response and term1 in clean_reference):
                return True, 0.75, f"Contradictory terms: '{term1}' vs '{term2}'"
        
        return False, 0.0, "No term-level contradictions found"
    
    def _extract_numerical_values(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """
        Extract numerical values from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary mapping value types to lists of (context, value) tuples
        """
        results = {
            "numbers": [],
            "years": [],
            "monetary": [],
            "ratios": [],
            "percentages": []
        }
        
        # Extract numbers with context
        for match in NUMERIC_PATTERN.finditer(text):
            value = float(match.group(1))
            unit = match.group(2) if match.group(2) else ""
            
            # Get context (5 words before and after)
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]
            
            if unit and "percent" in unit.lower() or unit == "%":
                results["percentages"].append((context, value))
            else:
                results["numbers"].append((context, value))
        
        # Extract years
        for match in YEAR_PATTERN.finditer(text):
            year = int(match.group(1))
            
            # Get context
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]
            
            results["years"].append((context, year))
        
        # Extract monetary values
        for match in MONETARY_PATTERN.finditer(text):
            value_str = match.group(1).replace(',', '')
            value = float(value_str)
            
            # Get context
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]
            
            results["monetary"].append((context, value))
        
        # Extract ratios
        for match in RATIO_PATTERN.finditer(text):
            num1 = int(match.group(1))
            num2 = int(match.group(2))
            ratio = num1 / num2 if num2 != 0 else float('inf')
            
            # Get context
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]
            
            results["ratios"].append((context, ratio))
        
        return results
    
    def _detect_numeric_contradictions(self, response: str, reference: str) -> Tuple[bool, float, str]:
        """
        Detect contradictions based on numerical values.
        
        Args:
            response: Response text
            reference: Reference text
            
        Returns:
            Tuple of (contradiction_found, confidence, reason)
        """
        # Extract numerical values from both texts
        response_nums = self._extract_numerical_values(response)
        reference_nums = self._extract_numerical_values(reference)
        
        # Check for year contradictions
        for r_context, r_year in response_nums["years"]:
            for ref_context, ref_year in reference_nums["years"]:
                # Only compare if contexts are similar
                if self._context_similarity(r_context, ref_context) > 0.5:
                    # Check for significant year difference
                    if abs(r_year - ref_year) > 1:  # Allow 1 year tolerance
                        return True, 0.9, f"Year contradiction: {r_year} vs {ref_year} in similar contexts"
        
        # Check for percentage contradictions
        for r_context, r_pct in response_nums["percentages"]:
            for ref_context, ref_pct in reference_nums["percentages"]:
                if self._context_similarity(r_context, ref_context) > 0.5:
                    # Check for significant percentage difference
                    if abs(r_pct - ref_pct) > 5:  # 5% tolerance
                        return True, 0.85, f"Percentage contradiction: {r_pct}% vs {ref_pct}% in similar contexts"
        
        # Check for monetary contradictions
        for r_context, r_value in response_nums["monetary"]:
            for ref_context, ref_value in reference_nums["monetary"]:
                if self._context_similarity(r_context, ref_context) > 0.5:
                    # Check for significant monetary difference (>10%)
                    if ref_value > 0 and abs((r_value - ref_value) / ref_value) > 0.1:
                        return True, 0.85, f"Monetary value contradiction in similar contexts"
        
        # Check for general numeric contradictions
        for r_context, r_value in response_nums["numbers"]:
            for ref_context, ref_value in reference_nums["numbers"]:
                if self._context_similarity(r_context, ref_context) > 0.6:  # Higher threshold for general numbers
                    # Check for significant difference (>20%)
                    if ref_value > 0 and abs((r_value - ref_value) / ref_value) > 0.2:
                        return True, 0.7, f"Numeric contradiction in similar contexts"
        
        return False, 0.0, "No numerical contradictions found"
    
    def _context_similarity(self, context1: str, context2: str) -> float:
        """
        Calculate simple context similarity for numeric comparison.
        
        Args:
            context1: First context string
            context2: Second context string
            
        Returns:
            Similarity score (0-1)
        """
        # Simple word overlap similarity
        words1 = set(self._clean_text(context1).split())
        words2 = set(self._clean_text(context2).split())
        
        # Remove very common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "of", "is", "are"}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def detect_domain_contradictions(self, response: str, reference: str, domain: str) -> Tuple[bool, float, str]:
        """
        Detect domain-specific contradictions.
        
        Args:
            response: Response text
            reference: Reference text
            domain: Domain for selecting appropriate patterns
            
        Returns:
            Tuple of (contradiction_found, confidence, reason)
        """
        # Get domain-specific patterns
        patterns = self.domain_patterns.get(domain.lower(), [])
        
        if not patterns:
            return False, 0.0, f"No patterns defined for domain '{domain}'"
        
        # Check each pattern
        for pattern_def in patterns:
            pattern = pattern_def["pattern"]
            context_pattern = pattern_def["context"]
            match_type = pattern_def["match_type"]
            confidence = pattern_def["confidence"]
            reason = pattern_def["reason"]
            
            # Check if pattern appears in response
            response_matches = re.findall(pattern, response)
            if not response_matches:
                continue
                
            # Check if context appears in both texts
            response_context = re.findall(context_pattern, response)
            reference_context = re.findall(context_pattern, reference)
            
            if not response_context or not reference_context:
                continue
            
            # Special handling based on match type
            if match_type == "absolute_claim":
                # Check if reference contains qualifying language that contradicts absolute claim
                qualifiers = r"(?i)(usually|often|sometimes|may|might|can|could|in some cases|rarely|not always)"
                if re.search(qualifiers, reference) and response_matches:
                    return True, confidence, reason
                    
            elif match_type == "opposing_actions":
                # Check for specific opposing verbs
                for oppose_pair in self.contradiction_pairs:
                    term1, term2 = oppose_pair
                    if (term1 in response.lower() and term2 in reference.lower()) or \
                       (term2 in response.lower() and term1 in reference.lower()):
                        return True, confidence, reason
            
            elif match_type == "causation":
                # Check for causal language in response but disclaimers in reference
                causal_terms = r"(?i)(causes|caused by|leads to|results in)"
                disclaimer_terms = r"(?i)(associated with|correlation|not proven|may be linked|hypothesis)"
                
                if re.search(causal_terms, response) and re.search(disclaimer_terms, reference):
                    return True, confidence, reason
            
            else:
                # Generic check: pattern in response contradicts something in reference
                for resp_match in response_matches:
                    resp_context = resp_match.lower()
                    
                    # Check if negated in reference
                    for neg in self.negation_words:
                        if re.search(r'\b' + re.escape(neg) + r'\b.*?\b' + re.escape(resp_context) + r'\b', 
                                    reference.lower()) or \
                           re.search(r'\b' + re.escape(resp_context) + r'\b.*?\b' + re.escape(neg) + r'\b', 
                                    reference.lower()):
                            return True, confidence, reason
        
        return False, 0.0, "No domain-specific contradictions found"
    
    def detect_contextual_contradictions(self, response: str, reference: str, domain: str) -> Tuple[bool, float, str]:
        """
        Main method to detect contextual contradictions between response and reference.
        
        Args:
            response: Response text
            reference: Reference text
            domain: Domain for selecting appropriate patterns
            
        Returns:
            Tuple of (contradiction_found, confidence, reason)
        """
        if not response or not reference:
            return False, 0.0, "Empty input"
        
        # Check domain-specific contradictions first
        domain_result = self.detect_domain_contradictions(response, reference, domain)
        if domain_result[0]:
            return domain_result
        
        # Check for term-based contradictions
        term_result = self._detect_term_contradictions(response, reference)
        if term_result[0]:
            return term_result
        
        # Check for numerical contradictions
        num_result = self._detect_numeric_contradictions(response, reference)
        if num_result[0]:
            return num_result
        
        # If no contradictions found
        return False, 0.0, "No contradictions detected"

def detect_contradictions(response: str, reference: str, domain: str) -> Dict[str, Any]:
    """
    Helper function to detect contradictions and return structured result.
    
    Args:
        response: Response text
        reference: Reference text
        domain: Domain for contradiction patterns
        
    Returns:
        Dictionary with contradiction detection results
    """
    detector = ConflictDetector()
    
    has_contradiction, confidence, reason = detector.detect_contextual_contradictions(
        response, reference, domain
    )
    
    result = {
        "has_contradiction": has_contradiction,
        "contradiction_confidence": confidence,
        "contradiction_reason": reason,
        "domain": domain
    }
    
    return result

if __name__ == "__main__":
    # Example usage when run directly
    detector = ConflictDetector()
    
    test_pairs = [
        {
            "domain": "medical",
            "reference": "Type 2 diabetes is a chronic condition that affects the way your body metabolizes sugar. While there are treatments available to manage the condition, there is currently no known cure for type 2 diabetes.",
            "response": "Type 2 diabetes can be completely cured with proper diet and exercise in most cases."
        },
        {
            "domain": "financial", 
            "reference": "The Federal Deposit Insurance Corporation (FDIC) insures deposits in member banks up to $250,000 per depositor, per insured bank, for each account ownership category.",
            "response": "The FDIC provides insurance coverage of up to $500,000 for all deposits in any bank."
        },
        {
            "domain": "legal",
            "reference": "The legality of recording conversations varies by jurisdiction. In the United States, some states require consent from all parties while others require consent from only one party.",
            "response": "Recording a conversation is legal in all 50 states as long as one party consents to the recording."
        }
    ]
    
    for test in test_pairs:
        result = detect_contradictions(test["response"], test["reference"], test["domain"])
        
        print(f"\nDomain: {test['domain']}")
        print(f"Reference: {test['reference'][:100]}...")
        print(f"Response: {test['response']}")
        print(f"Has contradiction: {result['has_contradiction']}")
        print(f"Confidence: {result['contradiction_confidence']}")
        print(f"Reason: {result['contradiction_reason']}")