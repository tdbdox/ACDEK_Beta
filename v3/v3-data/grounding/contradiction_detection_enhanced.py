#!/usr/bin/env python3
"""
Enhanced Contradiction Detection Module for ACDEK v3 Grounding Engine (Phase 3)

This module provides sophisticated contradiction detection between 
response and reference texts with improved accuracy and domain-specific coverage.

Key improvements:
- Expanded domain-specific contradiction patterns
- Improved numerical contradiction detection
- Enhanced contextual similarity measures
- Confidence calibration for better accuracy
- Performance optimizations
"""

import re
import logging
import time
import json
from typing import List, Dict, Any, Tuple, Optional, Set, Union
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet, stopwords
import string
from pathlib import Path
import sys
import os

# Add parent directories to path to enable imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download NLTK resources if needed
def ensure_nltk_resources():
    """Ensure required NLTK resources are available."""
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('wordnet', 'corpora/wordnet'),
        ('stopwords', 'corpora/stopwords'),
        ('punkt_tab', 'tokenizers/punkt_tab')
    ]
    
    for resource, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"Downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {resource}: {str(e)}")

# Initialize NLTK resources
ensure_nltk_resources()

# Enhanced domain-specific patterns for medical domain
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
    },
    # Additional patterns for medical domain
    {
        "name": "vaccine_dna",
        "pattern": r"(?i)(vaccines?|mrna|covid.{0,10}vaccines?).*?(alters?|changes?|modifies?|enters?|integrates?).*?(dna|genes|genome|genetic)",
        "context": r"(?i)(vaccine|vaccination|mrna|covid)",
        "match_type": "scientific_falsehood",
        "confidence": 0.90,
        "reason": "False claims about vaccines altering DNA"
    },
    {
        "name": "vaccine_autism",
        "pattern": r"(?i)(vaccines?|vaccination|immunization).*?(causes?|leads? to|results? in|linked to).*?(autism|autistic)",
        "context": r"(?i)(vaccine|child|development|disorder)",
        "match_type": "scientific_falsehood",
        "confidence": 0.90,
        "reason": "False claims linking vaccines to autism"
    },
    {
        "name": "vitamin_c_cancer",
        "pattern": r"(?i)(vitamin c|ascorbic acid).*?(prevents?|cures?|treats?).*?(cancer|tumor|malignancy)",
        "context": r"(?i)(vitamin|cancer|prevention|treatment)",
        "match_type": "unproven_claim",
        "confidence": 0.85,
        "reason": "Unproven claims about vitamin C preventing or curing cancer"
    },
    {
        "name": "medical_timeline",
        "pattern": r"(?i)(immediately|within (\d+) (hours|days|weeks|months)|after (\d+) (hours|days|weeks|months))",
        "context": r"(?i)(recovery|healing|symptoms|effectiveness|treatment)",
        "match_type": "medical_timeline",
        "confidence": 0.70,
        "reason": "Specific medical timeline claims that may contradict evidence"
    }
]

# Enhanced domain-specific patterns for financial domain
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
    },
    # Additional patterns for financial domain
    {
        "name": "fdic_coverage",
        "pattern": r"(?i)(fdic|federal deposit insurance).*?(insures|covers|protects).*?(\$\d+|\d+ dollars)",
        "context": r"(?i)(fdic|insurance|deposit|bank)",
        "match_type": "financial_protection",
        "confidence": 0.85,
        "reason": "Claims about FDIC insurance coverage amounts"
    },
    {
        "name": "fed_rate_control",
        "pattern": r"(?i)(fed|federal reserve).*?(controls?|adjusts?|manages?).*?(inflation|rates|economy)",
        "context": r"(?i)(federal reserve|monetary policy|inflation|interest rates)",
        "match_type": "monetary_policy",
        "confidence": 0.75,
        "reason": "Claims about Federal Reserve's role in monetary policy"
    },
    {
        "name": "cryptocurrency_origin",
        "pattern": r"(?i)(bitcoin|cryptocurrency|crypto).*?(created by|founded by|invented by|started by).*?(company|government|bank|institution|university)",
        "context": r"(?i)(bitcoin|cryptocurrency|blockchain|origin)",
        "match_type": "factual_origin",
        "confidence": 0.85,
        "reason": "False claims about cryptocurrency origins"
    },
    {
        "name": "market_timing",
        "pattern": r"(?i)(timing the market|predict the market|perfect timing).*?(always|consistently|reliably)",
        "context": r"(?i)(investing|market|strategy|timing)",
        "match_type": "investment_strategy",
        "confidence": 0.80,
        "reason": "Dubious claims about market timing reliability"
    }
]

# Enhanced domain-specific patterns for legal domain
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
    },
    # Additional patterns for legal domain
    {
        "name": "supreme_court_term",
        "pattern": r"(?i)(supreme court justices?|justices of the supreme court).*?(term|serve|tenure|appointment).*?(\d+ years?|limited|maximum)",
        "context": r"(?i)(supreme court|judiciary|appointment|term)",
        "match_type": "judicial_term",
        "confidence": 0.90,
        "reason": "False claims about Supreme Court justice term limits"
    },
    {
        "name": "first_amendment_scope",
        "pattern": r"(?i)(first amendment|free speech|freedom of speech).*?(applies to|restricts|limits|governs).*?(private companies|private entities|private platforms|corporations)",
        "context": r"(?i)(constitution|first amendment|free speech|rights)",
        "match_type": "constitutional_scope",
        "confidence": 0.85,
        "reason": "Incorrect claims about First Amendment application to private entities"
    },
    {
        "name": "burden_of_proof",
        "pattern": r"(?i)(defendant|accused|charged).*?(must prove|has to prove|burden to prove|required to prove).*?(innocence|not guilty)",
        "context": r"(?i)(criminal|trial|court|prosecution|defendant)",
        "match_type": "legal_principle",
        "confidence": 0.90,
        "reason": "False claims about burden of proof in criminal cases"
    },
    {
        "name": "death_penalty_jurisdiction",
        "pattern": r"(?i)(death penalty|capital punishment|execution).*?(legal|allowed|permitted).*?(all states|every state|across the country)",
        "context": r"(?i)(death penalty|capital punishment|criminal justice)",
        "match_type": "legal_jurisdiction",
        "confidence": 0.85,
        "reason": "Incorrect claims about death penalty legality across all states"
    },
    {
        "name": "recording_consent",
        "pattern": r"(?i)(record|recording|wiretap|tape).*?(conversation|call|communication).*?(legal|allowed|permitted|lawful).*?(all states|every state|across the us)",
        "context": r"(?i)(recording|conversation|consent|privacy)",
        "match_type": "legal_consent",
        "confidence": 0.80,
        "reason": "Overgeneralized claims about recording consent laws"
    }
]

# Enhanced numerical pattern detection
NUMERIC_PATTERN = re.compile(r'(?<!\w)(\d+(?:\.\d+)?)\s*(%|percent|thousand|million|billion|trillion)?')
YEAR_PATTERN = re.compile(r'\b(19\d{2}|20\d{2})\b')  # Years 1900-2099
MONETARY_PATTERN = re.compile(r'(?<!\w)[$€£¥]\s*(\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)')
RATIO_PATTERN = re.compile(r'(?<!\w)(\d+)\s*(?::|to|:to:)\s*(\d+)')
PERCENTAGE_PATTERN = re.compile(r'(\d+(?:\.\d+)?)\s*(?:%|percent)')
DATE_PATTERN = re.compile(r'\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})')

# Enhanced domain-specific value thresholds
DOMAIN_THRESHOLDS = {
    'medical': {
        'year_tolerance': 2,     # Tolerance for year differences in medical research (e.g., publication date)
        'percentage_tolerance': 5.0,  # Tolerance for percentage differences (e.g., efficacy rates)
        'numeric_tolerance': 0.1,  # Proportional tolerance for numerical values (e.g., dosages)
        'date_tolerance_days': 30   # Tolerance for dates in days (e.g., outbreak dates)
    },
    'financial': {
        'year_tolerance': 1,     # Stricter tolerance for financial years (e.g., fiscal years)
        'percentage_tolerance': 2.0,  # Stricter tolerance for percentages (e.g., interest rates)
        'numeric_tolerance': 0.05, # Stricter proportional tolerance for financial values
        'date_tolerance_days': 7    # Stricter tolerance for financial dates (e.g., market events)
    },
    'legal': {
        'year_tolerance': 0,     # No tolerance for legal years (e.g., enactment dates)
        'percentage_tolerance': 0.0,  # No tolerance for legal percentages (e.g., voting thresholds)
        'numeric_tolerance': 0.0,  # No tolerance for legal numbers (e.g., statutory requirements)
        'date_tolerance_days': 0    # No tolerance for legal dates (e.g., filing deadlines)
    },
    'default': {
        'year_tolerance': 1,
        'percentage_tolerance': 3.0,
        'numeric_tolerance': 0.1,
        'date_tolerance_days': 14
    }
}

class ConflictDetectorEnhanced:
    """
    Enhanced contradiction detection between response and reference texts.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the enhanced conflict detector with appropriate data structures.
        
        Args:
            cache_dir: Optional directory for caching contradiction patterns
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
            "rejected", "exclude", "excludes", "excluded", "false", "untrue", "incorrectly",
            "incorrect", "inaccurate", "wrong", "mistaken", "erroneous", "invalid"
        }
        
        # Common contradiction pairs - expanded list
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
            ("expansion", "contraction"),
            ("internal", "external"),
            ("direct", "indirect"),
            ("simple", "complex"),
            ("permanent", "temporary"),
            ("flexible", "rigid"),
            ("stable", "unstable"),
            ("consistent", "inconsistent"),
            ("regular", "irregular"),
            ("formal", "informal"),
            ("necessary", "unnecessary"),
            ("sufficient", "insufficient"),
            ("complete", "incomplete"),
            ("specific", "general"),
            ("explicit", "implicit"),
            ("major", "minor"),
            ("primary", "secondary"),
            ("central", "peripheral"),
            ("maximum", "minimum"),
            ("optimum", "suboptimal"),
            ("optimal", "suboptimal"),
            ("certain", "uncertain"),
            ("definite", "indefinite"),
            ("exact", "approximate"),
            ("precise", "imprecise"),
            ("accurate", "inaccurate"),
            ("clear", "unclear"),
            ("visible", "invisible"),
            ("tangible", "intangible"),
            ("concrete", "abstract"),
            ("measurable", "immeasurable"),
            ("quantitative", "qualitative"),
            ("objective", "subjective"),
            ("scientific", "unscientific"),
            ("proven", "unproven"),
            ("verified", "unverified"),
            ("confirmed", "unconfirmed"),
            ("validated", "invalidated"),
            ("supported", "unsupported"),
            ("approved", "disapproved"),
            ("endorsed", "rejected"),
            ("authorized", "unauthorized"),
            ("permitted", "prohibited"),
            ("allowed", "forbidden"),
            ("accepted", "rejected"),
            ("agreed", "disagreed"),
            ("compatible", "incompatible"),
            ("consistent", "contradictory"),
            ("congruent", "incongruent"),
            ("harmless", "harmful"),
            ("beneficial", "detrimental"),
            ("helpful", "unhelpful"),
            ("useful", "useless"),
            ("valuable", "worthless"),
            ("meaningful", "meaningless"),
            ("significant", "insignificant"),
            ("important", "unimportant"),
            ("relevant", "irrelevant"),
            ("appropriate", "inappropriate"),
            ("adequate", "inadequate"),
            ("sufficient", "insufficient"),
            ("effective", "ineffective"),
            ("efficient", "inefficient"),
            ("successful", "unsuccessful"),
            ("productive", "unproductive"),
            ("profitable", "unprofitable"),
            ("affordable", "unaffordable"),
            ("attainable", "unattainable"),
            ("achievable", "unachievable"),
            ("possible", "impossible"),
            ("feasible", "infeasible"),
            ("viable", "inviable"),
            ("sustainable", "unsustainable"),
            ("renewable", "nonrenewable"),
            ("reversible", "irreversible"),
            ("recoverable", "irrecoverable"),
            ("repairable", "irreparable"),
            ("curable", "incurable"),
            ("treatable", "untreatable"),
            ("preventable", "unpreventable"),
            ("avoidable", "unavoidable"),
            ("controllable", "uncontrollable"),
            ("manageable", "unmanageable"),
            ("normal", "abnormal"),
            ("typical", "atypical"),
            ("standard", "nonstandard"),
            ("conventional", "unconventional"),
            ("traditional", "nontraditional"),
            ("orthodox", "unorthodox"),
            ("mainstream", "fringe"),
            ("common", "uncommon"),
            ("frequent", "infrequent"),
            ("abundant", "scarce"),
            ("plentiful", "sparse"),
            ("universal", "particular"),
            ("global", "local"),
            ("general", "specific"),
            ("broad", "narrow"),
            ("wide", "narrow"),
            ("extensive", "limited"),
            ("comprehensive", "incomplete"),
            ("exhaustive", "partial"),
            ("thorough", "superficial"),
            ("detailed", "vague"),
            ("explicit", "implicit"),
            ("direct", "indirect"),
            ("straightforward", "complicated"),
            ("simple", "complex"),
            ("easy", "difficult"),
            ("straightforward", "complex"),
            ("accessible", "inaccessible"),
            ("available", "unavailable"),
            ("obtainable", "unobtainable"),
            ("open", "closed"),
            ("public", "private"),
            ("transparent", "opaque"),
            ("revealed", "concealed"),
            ("disclosed", "undisclosed"),
            ("known", "unknown"),
            ("familiar", "unfamiliar"),
            ("recognized", "unrecognized"),
            ("acknowledged", "unacknowledged"),
            ("identified", "unidentified"),
            ("specified", "unspecified"),
            ("defined", "undefined"),
            ("determined", "undetermined"),
            ("decided", "undecided"),
            ("resolved", "unresolved"),
            ("settled", "unsettled"),
            ("final", "preliminary"),
            ("complete", "partial"),
            ("whole", "partial"),
            ("entire", "partial"),
            ("total", "partial"),
            ("full", "empty"),
            ("filled", "unfilled"),
            ("occupied", "vacant"),
            ("used", "unused"),
            ("utilized", "unutilized"),
            ("applied", "unapplied"),
            ("implemented", "unimplemented"),
            ("executed", "unexecuted"),
            ("performed", "unperformed"),
            ("active", "inactive"),
            ("functioning", "nonfunctioning"),
            ("operational", "nonoperational"),
            ("working", "broken"),
            ("alive", "dead"),
            ("living", "nonliving"),
            ("organic", "inorganic"),
            ("natural", "artificial"),
            ("real", "fake"),
            ("genuine", "counterfeit"),
            ("authentic", "inauthentic"),
            ("original", "copied"),
            ("primary", "secondary"),
            ("first-hand", "second-hand"),
            ("direct", "indirect"),
            ("immediate", "delayed"),
            ("instant", "gradual"),
            ("rapid", "slow"),
            ("quick", "slow"),
            ("fast", "slow"),
            ("swift", "sluggish"),
            ("speedy", "time-consuming"),
            ("accelerated", "decelerated"),
            ("expanding", "contracting"),
            ("growing", "shrinking"),
            ("increasing", "decreasing"),
            ("rising", "falling"),
            ("ascending", "descending")
        ]
        
        # Initialize caching
        self.cache_dir = cache_dir
        self.synonym_cache = {}
        self.antonym_cache = {}
        
        # Try to load stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set(['the', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'a', 'with'])
    
    def get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms of a word using WordNet with caching.
        
        Args:
            word: Input word
            
        Returns:
            List of synonyms
        """
        word = word.lower()
        
        # Check cache first
        if word in self.synonym_cache:
            return self.synonym_cache[word]
            
        try:
            # Get synonyms from WordNet
            synonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    syn_word = lemma.name().lower().replace('_', ' ')
                    if syn_word != word and len(syn_word) > 2:
                        synonyms.add(syn_word)
            
            # Cache result
            result = list(synonyms)
            self.synonym_cache[word] = result
            return result
            
        except Exception as e:
            logger.warning(f"Error getting synonyms for '{word}': {str(e)}")
            return []
    
    def get_antonyms(self, word: str) -> List[str]:
        """
        Get antonyms of a word using WordNet with caching.
        
        Args:
            word: Input word
            
        Returns:
            List of antonyms
        """
        word = word.lower()
        
        # Check cache first
        if word in self.antonym_cache:
            return self.antonym_cache[word]
            
        try:
            # Get antonyms from WordNet
            antonyms = set()
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    for antonym in lemma.antonyms():
                        ant_word = antonym.name().lower().replace('_', ' ')
                        if len(ant_word) > 2:
                            antonyms.add(ant_word)
            
            # Check contradiction pairs as well
            for term1, term2 in self.contradiction_pairs:
                if word == term1:
                    antonyms.add(term2)
                elif word == term2:
                    antonyms.add(term1)
            
            # Cache result
            result = list(antonyms)
            self.antonym_cache[word] = result
            return result
            
        except Exception as e:
            logger.warning(f"Error getting antonyms for '{word}': {str(e)}")
            return []
    
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
            r'\b(?:require|obligate|mandate|necessitate|demand)\b',
            r'\b(?:guarantee|ensure|certain|definite|undoubtedly)\b',
            r'\b(?:cannot|impossible|restricted|prohibited|banned)\b'
        ]
        
        claim_pattern = '|'.join(claim_indicators)
        
        key_claims = []
        for sentence in sentences:
            if re.search(claim_pattern, sentence, re.IGNORECASE):
                key_claims.append(sentence)
        
        return key_claims
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """
        Extract key phrases from a text using a simple approach.
        
        Args:
            text: Input text
            max_phrases: Maximum number of phrases to extract
            
        Returns:
            List of key phrases
        """
        try:
            # Clean text
            clean_text = self._clean_text(text)
            
            # Tokenize into words
            words = word_tokenize(clean_text)
            
            # Remove stopwords
            words = [w for w in words if w not in self.stopwords and len(w) > 3]
            
            # Extract phrases (2-3 word sequences)
            bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
            trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            
            # Combine and limit
            all_phrases = bigrams + trigrams
            return all_phrases[:max_phrases]
            
        except Exception as e:
            logger.warning(f"Error extracting key phrases: {str(e)}")
            return []
    
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
                    # Look at nearby words (expanded context window of 6)
                    start = max(0, i - 2)
                    end = min(len(words), i + 7)
                    
                    # Extract the negated terms
                    for j in range(start, end):
                        if j != i and words[j] not in self.negation_words and len(words[j]) > 3:
                            # Also add lemmatized form for better matching
                            negated_terms.add(words[j])
                            
                            # Add any multi-word expressions
                            if j < len(words) - 1:
                                negated_terms.add(f"{words[j]} {words[j+1]}")
        
        return negated_terms
    
    def _detect_term_contradictions(self, response: str, reference: str) -> Tuple[bool, float, str]:
        """
        Enhanced detection of contradictions based on term-level analysis.
        
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
        
        # Extract key phrases
        response_phrases = self._extract_key_phrases(response)
        reference_phrases = self._extract_key_phrases(reference)
        
        # Check for terms that are negated in one text but affirmed in the other
        response_words = set(clean_response.split())
        reference_words = set(clean_reference.split())
        
        contradiction_terms = []
        
        # Check for terms negated in reference but affirmed in response
        for term in reference_negations:
            # Check if term is in response and not negated there
            if term in response_words and term not in response_negations:
                # Make sure it's a meaningful term (not a stopword)
                if term not in self.stopwords and len(term) > 3:
                    contradiction_terms.append(term)
                    
                    # Check for synonyms in response
                    for synonym in self.get_synonyms(term):
                        if synonym in response_words and synonym not in response_negations:
                            contradiction_terms.append(f"{term}/{synonym}")
        
        # Check for terms negated in response but affirmed in reference
        for term in response_negations:
            # Check if term is in reference and not negated there
            if term in reference_words and term not in reference_negations:
                # Make sure it's a meaningful term (not a stopword)
                if term not in self.stopwords and len(term) > 3:
                    contradiction_terms.append(term)
                    
                    # Check for synonyms in reference
                    for synonym in self.get_synonyms(term):
                        if synonym in reference_words and synonym not in reference_negations:
                            contradiction_terms.append(f"{term}/{synonym}")
        
        if contradiction_terms:
            term_list = ", ".join(set(contradiction_terms[:3]))  # List top 3 unique terms
            return True, 0.75, f"Contradictory negation of terms: {term_list}"
        
        # Enhanced check for antonyms
        antonym_contradictions = []
        
        # Extract significant words from both texts (excluding stopwords)
        significant_resp_words = [w for w in response_words if w not in self.stopwords and len(w) > 3]
        significant_ref_words = [w for w in reference_words if w not in self.stopwords and len(w) > 3]
        
        # Check for antonyms
        for resp_word in significant_resp_words[:20]:  # Limit check to first 20 words for performance
            antonyms = self.get_antonyms(resp_word)
            
            for antonym in antonyms:
                if antonym in significant_ref_words:
                    # Check if they're used in similar contexts
                    resp_context = self._get_term_context(response, resp_word)
                    ref_context = self._get_term_context(reference, antonym)
                    
                    if self._context_similarity(resp_context, ref_context) > 0.3:
                        antonym_contradictions.append(f"{resp_word}/{antonym}")
        
        if antonym_contradictions:
            antonym_list = ", ".join(antonym_contradictions[:3])  # List top 3
            return True, 0.8, f"Contradictory terms: {antonym_list}"
        
        # Check for presence of direct contradictions from contradiction pairs
        for term1, term2 in self.contradiction_pairs:
            # Check if contradictory terms appear in response and reference
            if (term1 in clean_response and term2 in clean_reference) or \
               (term2 in clean_response and term1 in clean_reference):
                
                # Get contexts to ensure they're talking about the same thing
                if term1 in clean_response:
                    resp_context = self._get_term_context(response, term1)
                    ref_context = self._get_term_context(reference, term2)
                else:
                    resp_context = self._get_term_context(response, term2)
                    ref_context = self._get_term_context(reference, term1)
                
                # Only flag as contradiction if contexts are similar
                if self._context_similarity(resp_context, ref_context) > 0.3:
                    return True, 0.75, f"Contradictory terms: '{term1}' vs '{term2}'"
        
        return False, 0.0, "No term-level contradictions found"
    
    def _get_term_context(self, text: str, term: str, window_size: int = 30) -> str:
        """
        Get the surrounding context of a term in text.
        
        Args:
            text: Full text
            term: Term to find context for
            window_size: Number of characters to include before and after
            
        Returns:
            Context string
        """
        text_lower = text.lower()
        term_lower = term.lower()
        
        # Find the position of the term
        pos = text_lower.find(term_lower)
        if pos == -1:
            return ""
            
        # Extract context
        start = max(0, pos - window_size)
        end = min(len(text), pos + len(term) + window_size)
        
        return text[start:end]
    
    def _extract_numerical_values(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        """
        Enhanced extraction of numerical values from text.
        
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
            "percentages": [],
            "dates": []
        }
        
        # Extract numbers with context
        for match in NUMERIC_PATTERN.finditer(text):
            value = float(match.group(1))
            unit = match.group(2) if match.group(2) else ""
            
            # Get context (5 words before and after)
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]
            
            if unit and ("percent" in unit.lower() or unit == "%"):
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
            
        # Extract dates
        for match in DATE_PATTERN.finditer(text):
            date_str = match.group(1)
            
            # Get context
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end]
            
            # Don't try to parse the date, just store it as is
            results["dates"].append((context, date_str))
        
        return results
    
    def _detect_numeric_contradictions(self, response: str, reference: str, domain: str = "default") -> Tuple[bool, float, str]:
        """
        Enhanced detection of numeric contradictions.
        
        Args:
            response: Response text
            reference: Reference text
            domain: Domain for threshold settings
            
        Returns:
            Tuple of (contradiction_found, confidence, reason)
        """
        # Get domain-specific thresholds
        thresholds = DOMAIN_THRESHOLDS.get(domain, DOMAIN_THRESHOLDS["default"])
        
        # Extract numerical values from both texts
        response_nums = self._extract_numerical_values(response)
        reference_nums = self._extract_numerical_values(reference)
        
        # Check for year contradictions
        for r_context, r_year in response_nums["years"]:
            for ref_context, ref_year in reference_nums["years"]:
                # Only compare if contexts are similar
                if self._context_similarity(r_context, ref_context) > 0.4:
                    # Check for significant year difference beyond domain tolerance
                    if abs(r_year - ref_year) > thresholds["year_tolerance"]:
                        return True, 0.9, f"Year contradiction: {r_year} vs {ref_year} in similar contexts"
        
        # Check for percentage contradictions
        for r_context, r_pct in response_nums["percentages"]:
            for ref_context, ref_pct in reference_nums["percentages"]:
                if self._context_similarity(r_context, ref_context) > 0.4:
                    # Check for significant percentage difference beyond domain tolerance
                    if abs(r_pct - ref_pct) > thresholds["percentage_tolerance"]:
                        return True, 0.85, f"Percentage contradiction: {r_pct}% vs {ref_pct}% in similar contexts"
        
        # Check for monetary contradictions
        for r_context, r_value in response_nums["monetary"]:
            for ref_context, ref_value in reference_nums["monetary"]:
                if self._context_similarity(r_context, ref_context) > 0.4:
                    # Check for significant monetary difference beyond domain tolerance
                    if ref_value > 0 and abs((r_value - ref_value) / ref_value) > thresholds["numeric_tolerance"]:
                        return True, 0.85, f"Monetary value contradiction: ${r_value} vs ${ref_value} in similar contexts"
        
        # Check for general numeric contradictions
        for r_context, r_value in response_nums["numbers"]:
            for ref_context, ref_value in reference_nums["numbers"]:
                if self._context_similarity(r_context, ref_context) > 0.5:  # Higher threshold for general numbers
                    # Check for significant difference beyond domain tolerance
                    if ref_value > 0 and abs((r_value - ref_value) / ref_value) > thresholds["numeric_tolerance"]:
                        return True, 0.7, f"Numeric contradiction: {r_value} vs {ref_value} in similar contexts"
        
        # Check for ratio contradictions
        for r_context, r_ratio in response_nums["ratios"]:
            for ref_context, ref_ratio in reference_nums["ratios"]:
                if self._context_similarity(r_context, ref_context) > 0.5:
                    # Calculate ratio difference as a percentage
                    if ref_ratio > 0 and abs((r_ratio - ref_ratio) / ref_ratio) > thresholds["numeric_tolerance"]:
                        return True, 0.75, f"Ratio contradiction in similar contexts"
        
        return False, 0.0, "No numerical contradictions found"
    
    def _context_similarity(self, context1: str, context2: str) -> float:
        """
        Enhanced context similarity calculation.
        
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
        stop_words = self.stopwords
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Enhanced similarity with weighted significant terms
        # Words longer than 5 chars count double
        intersection_weight = sum(1.5 if len(word) > 5 else 1.0 for word in intersection)
        union_weight = sum(1.5 if len(word) > 5 else 1.0 for word in union)
        
        return intersection_weight / union_weight if union_weight > 0 else 0.0
    
    def detect_domain_contradictions(self, response: str, reference: str, domain: str) -> Tuple[bool, float, str]:
        """
        Enhanced detection of domain-specific contradictions.
        
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
            response_matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            if not response_matches:
                continue
                
            # Check if context appears in both texts
            response_context = re.findall(context_pattern, response, re.IGNORECASE)
            reference_context = re.findall(context_pattern, reference, re.IGNORECASE)
            
            if not response_context or not reference_context:
                continue
            
            # Special handling based on match type
            if match_type == "absolute_claim":
                # Check if reference contains qualifying language that contradicts absolute claim
                qualifiers = r"(?i)(usually|often|sometimes|may|might|can|could|in some cases|rarely|not always|typically|generally|commonly|normally|occasionally|seldom|potentially)"
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
                disclaimer_terms = r"(?i)(associated with|correlation|not proven|may be linked|hypothesis|theory|suggested|proposed)"
                
                if re.search(causal_terms, response) and re.search(disclaimer_terms, reference):
                    return True, confidence, reason
                    
            elif match_type == "scientific_falsehood":
                # For scientifically false claims (like vaccine-autism link)
                # Just need to match the pattern in response
                if response_matches:
                    return True, confidence, reason
                    
            elif match_type == "financial_protection":
                # For FDIC coverage claims, check for specific dollar amount
                if "fdic" in domain or "financial" in domain:
                    # Extract dollar amounts from both texts
                    resp_dollars = re.findall(r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)', response)
                    ref_dollars = re.findall(r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)', reference)
                    
                    if resp_dollars and ref_dollars:
                        # Convert to float, removing commas
                        try:
                            resp_amt = float(resp_dollars[0].replace(',', ''))
                            ref_amt = float(ref_dollars[0].replace(',', ''))
                            
                            # Check if response amount is significantly different
                            if abs(resp_amt - ref_amt) > ref_amt * 0.1:  # 10% difference
                                return True, confidence, reason
                        except:
                            pass
            
            elif match_type == "judicial_term":
                # For Supreme Court term limits
                if "supreme court" in response.lower() and "supreme court" in reference.lower():
                    # Extract years from response
                    years = re.findall(r'(\d+)\s*years?', response.lower())
                    if years and "lifetime" in reference.lower():
                        return True, confidence, reason
            
            elif match_type == "legal_jurisdiction":
                # For legal jurisdictions like death penalty
                if "all states" in response.lower() or "every state" in response.lower():
                    if "abolished" in reference.lower() and "states" in reference.lower():
                        return True, confidence, reason
            
            elif match_type == "legal_consent":
                # For recording consent laws
                if "all states" in response.lower() or "every state" in response.lower():
                    if "one-party" in reference.lower() and "all-party" in reference.lower():
                        return True, confidence, reason
            
            else:
                # Generic check: pattern in response contradicts something in reference
                for resp_match in response_matches:
                    # Check if negated in reference
                    for neg in self.negation_words:
                        neg_pattern = r'\b' + re.escape(neg) + r'\b.*?\b' + re.escape(str(resp_match)) + r'\b'
                        reverse_pattern = r'\b' + re.escape(str(resp_match)) + r'\b.*?\b' + re.escape(neg) + r'\b'
                        
                        if re.search(neg_pattern, reference.lower()) or re.search(reverse_pattern, reference.lower()):
                            return True, confidence, reason
        
        return False, 0.0, "No domain-specific contradictions found"
    
    def detect_topic_shift_contradictions(self, response: str, reference: str) -> Tuple[bool, float, str]:
        """
        Detect contradictions that involve a shift in topic or context.
        
        Args:
            response: Response text
            reference: Reference text
            
        Returns:
            Tuple of (contradiction_found, confidence, reason)
        """
        # Extract key topics from both texts
        response_topics = self._extract_key_phrases(response, max_phrases=5)
        reference_topics = self._extract_key_phrases(reference, max_phrases=5)
        
        # Find topics common to both texts
        common_topics = set(response_topics).intersection(set(reference_topics))
        
        if not common_topics:
            return False, 0.0, "No common topics found"
        
        # For each common topic, check for contradictory statements
        for topic in common_topics:
            # Get sentences containing this topic
            response_sentences = self._get_sentences_with_topic(response, topic)
            reference_sentences = self._get_sentences_with_topic(reference, topic)
            
            # For each pair of sentences, check for contradictions
            for r_sent in response_sentences:
                r_clean = self._clean_text(r_sent)
                for ref_sent in reference_sentences:
                    ref_clean = self._clean_text(ref_sent)
                    
                    # Check for negation contradictions
                    if any(neg in r_clean for neg in self.negation_words) != any(neg in ref_clean for neg in self.negation_words):
                        # Check for similar context
                        if self._context_similarity(r_sent, ref_sent) > 0.4:
                            return True, 0.7, f"Contradictory statements about '{topic}'"
                    
                    # Check for antonyms
                    r_words = r_clean.split()
                    ref_words = ref_clean.split()
                    
                    for r_word in r_words:
                        if r_word in self.stopwords:
                            continue
                            
                        antonyms = self.get_antonyms(r_word)
                        if any(ant in ref_words for ant in antonyms):
                            return True, 0.75, f"Contradictory descriptions of '{topic}'"
        
        return False, 0.0, "No topic shift contradictions found"
    
    def _get_sentences_with_topic(self, text: str, topic: str) -> List[str]:
        """
        Get sentences containing a specific topic.
        
        Args:
            text: Text to analyze
            topic: Topic to search for
            
        Returns:
            List of sentences containing the topic
        """
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Check for topic in each sentence
        return [sent for sent in sentences if topic.lower() in sent.lower()]
    
    def calibrate_contradiction_confidence(self, has_contradiction: bool, 
                                         confidence: float, 
                                         reason: str, 
                                         reference_length: int,
                                         response_length: int) -> float:
        """
        Calibrate contradiction confidence based on contextual factors.
        
        Args:
            has_contradiction: Whether a contradiction was detected
            confidence: Initial confidence score
            reason: Contradiction reason
            reference_length: Length of reference text
            response_length: Length of response text
            
        Returns:
            Calibrated confidence score
        """
        if not has_contradiction:
            return 0.0
            
        # Start with original confidence
        calibrated_confidence = confidence
        
        # Adjust based on text lengths
        if reference_length < 50 or response_length < 50:
            # Very short texts are more prone to false positives
            calibrated_confidence *= 0.8
        
        # Adjust based on reason type
        if "numeric" in reason.lower() or "year" in reason.lower() or "date" in reason.lower():
            # Numeric contradictions are more reliable
            calibrated_confidence *= 1.1
        elif "term" in reason.lower():
            # Term-level contradictions can be context-dependent
            calibrated_confidence *= 0.9
        
        # Cap confidence
        return min(0.95, max(0.5, calibrated_confidence))
    
    def detect_contextual_contradictions(self, response: str, reference: str, domain: str = None) -> Tuple[bool, float, str]:
        """
        Enhanced main method to detect contextual contradictions.
        
        Args:
            response: Response text
            reference: Reference text
            domain: Domain for selecting appropriate patterns
            
        Returns:
            Tuple of (contradiction_found, confidence, reason)
        """
        if not response or not reference:
            return False, 0.0, "Empty input"
        
        # Capture text lengths for confidence calibration
        ref_length = len(reference)
        resp_length = len(response)
        
        # Start with domain-specific contradictions
        if domain:
            domain_result = self.detect_domain_contradictions(response, reference, domain)
            if domain_result[0]:
                calibrated_confidence = self.calibrate_contradiction_confidence(
                    True, domain_result[1], domain_result[2], ref_length, resp_length
                )
                return True, calibrated_confidence, domain_result[2]
        
        # Check for term-based contradictions
        term_result = self._detect_term_contradictions(response, reference)
        if term_result[0]:
            calibrated_confidence = self.calibrate_contradiction_confidence(
                True, term_result[1], term_result[2], ref_length, resp_length
            )
            return True, calibrated_confidence, term_result[2]
        
        # Check for numerical contradictions
        num_result = self._detect_numeric_contradictions(response, reference, domain)
        if num_result[0]:
            calibrated_confidence = self.calibrate_contradiction_confidence(
                True, num_result[1], num_result[2], ref_length, resp_length
            )
            return True, calibrated_confidence, num_result[2]
        
        # Check for topic shift contradictions
        topic_result = self.detect_topic_shift_contradictions(response, reference)
        if topic_result[0]:
            calibrated_confidence = self.calibrate_contradiction_confidence(
                True, topic_result[1], topic_result[2], ref_length, resp_length
            )
            return True, calibrated_confidence, topic_result[2]
        
        # If no contradictions found
        return False, 0.0, "No contradictions detected"

def detect_contradictions_enhanced(response: str, reference: str, domain: str) -> Dict[str, Any]:
    """
    Enhanced helper function to detect contradictions and return structured result.
    
    Args:
        response: Response text
        reference: Reference text
        domain: Domain for contradiction patterns
        
    Returns:
        Dictionary with contradiction detection results
    """
    start_time = time.time()
    detector = ConflictDetectorEnhanced()
    
    has_contradiction, confidence, reason = detector.detect_contextual_contradictions(
        response, reference, domain
    )
    
    processing_time = (time.time() - start_time) * 1000  # ms
    
    result = {
        "has_contradiction": has_contradiction,
        "contradiction_confidence": confidence,
        "contradiction_reason": reason,
        "domain": domain,
        "processing_time_ms": processing_time
    }
    
    return result

# Test function
def test_contradiction_detection(domain: str = None):
    """Test the contradiction detection functionality."""
    detector = ConflictDetectorEnhanced()
    
    test_pairs = [
        {
            "domain": "medical",
            "reference": "Type 2 diabetes is a chronic condition that affects the way your body metabolizes sugar. While there are treatments available to manage the condition, there is currently no known cure for type 2 diabetes.",
            "response": "Type 2 diabetes can be completely cured with proper diet and exercise in most cases.",
            "expected_contradiction": True
        },
        {
            "domain": "financial", 
            "reference": "The Federal Deposit Insurance Corporation (FDIC) insures deposits in member banks up to $250,000 per depositor, per insured bank, for each account ownership category.",
            "response": "The FDIC provides insurance coverage of up to $500,000 for all deposits in any bank.",
            "expected_contradiction": True
        },
        {
            "domain": "legal",
            "reference": "The legality of recording conversations varies by jurisdiction. In the United States, some states require consent from all parties while others require consent from only one party.",
            "response": "Recording a conversation is legal in all 50 states as long as one party consents to the recording.",
            "expected_contradiction": True
        },
        {
            "domain": "financial",
            "reference": "A bond is a fixed income instrument representing a loan made by an investor to a borrower, typically corporate or governmental entities. Bonds are used by companies, municipalities, states, and governments to finance projects and operations.",
            "response": "A bond represents a fixed income instrument where an investor lends money to an entity like a corporation or government, which uses the funds to finance various projects and operations.",
            "expected_contradiction": False
        },
        {
            "domain": "financial",
            "reference": "Bitcoin was created in 2009 by an unknown person or group of people using the pseudonym Satoshi Nakamoto. The identity of Satoshi Nakamoto remains unknown.",
            "response": "Bitcoin was created by a team of developers at MIT in 2010.",
            "expected_contradiction": True
        },
        {
            "domain": "medical",
            "reference": "There is no strong scientific evidence that vitamin C prevents cancer. While some studies suggest high doses of vitamin C might reduce the duration or severity of cold symptoms in some people, it does not prevent or cure the common cold for most people.",
            "response": "Taking vitamin C prevents cancer and cures the common cold.",
            "expected_contradiction": True
        },
        {
            "domain": "legal",
            "reference": "Under Article III of the U.S. Constitution, Supreme Court justices, like all federal judges, are appointed for life terms, serving 'during good behavior.'",
            "response": "US Supreme Court justices serve for a maximum term of 10 years.",
            "expected_contradiction": True
        }
    ]
    
    for i, test in enumerate(test_pairs):
        # Filter by domain if specified
        if domain and test["domain"] != domain:
            continue
            
        # Detect contradictions
        result = detect_contradictions_enhanced(test["response"], test["reference"], test["domain"])
        
        # Check if result matches expectation
        result_matches = result["has_contradiction"] == test["expected_contradiction"]
        status = "✓ PASS" if result_matches else "✗ FAIL"
        
        # Report results
        print(f"\nTest {i+1} ({test['domain']}): {status}")
        print(f"  Reference: {test['reference'][:100]}...")
        print(f"  Response: {test['response']}")
        print(f"  Expected contradiction: {test['expected_contradiction']}")
        print(f"  Detected contradiction: {result['has_contradiction']}")
        if result["has_contradiction"]:
            print(f"  Confidence: {result['contradiction_confidence']}")
            print(f"  Reason: {result['contradiction_reason']}")
        print(f"  Processing time: {result['processing_time_ms']:.2f} ms")

if __name__ == "__main__":
    # Run tests when executed directly
    print("Testing Enhanced Contradiction Detection")
    test_contradiction_detection()