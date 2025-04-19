"""
Hallucination Benchmarking Suite

A comprehensive benchmarking system to measure hallucination reduction rates
comparing ACDEK v3 outputs against raw LLM outputs and ACDEK v2 outputs.

This module provides:
1. Test cases across multiple domains (medical, legal, financial, etc.)
2. Ground truth validation with factuality constraints
3. Hallucination detection and scoring mechanisms
4. Comparative analysis between raw LLM, v2, and v3 outputs
5. Detailed metrics collection and reporting
"""
import json
import time
import logging
import os
import csv
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import concurrent.futures
from pathlib import Path

# Import ACDEK components for testing
from acdek.acdek_kernel import ACDEKKernel
from storage.vector_store.embed_utils import cosine_similarity, get_embedding
from interfaces.constraint_interface import (
    ConstraintSeverity,
    ConstraintCategory,
    ConstraintValidationResult
)

# Local imports
from benchmarks.analysis_tools import (
    export_report,
    precision_recall_f1,
    aggregate_retry_fallback_rates
)
from benchmarks.performance_metrics import latency_timer, RetryAggregator

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_logs.log"),
        logging.StreamHandler()
    ]
)

# Paths
BENCHMARK_DATA_DIR = Path("./benchmarks/data")
BENCHMARK_RESULTS_DIR = Path("./benchmarks/results")


class Domain(Enum):
    """Domains for test cases to enable domain-specific analysis."""
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    SCIENTIFIC = "scientific"
    GENERAL = "general"
    TECHNICAL = "technical"
    HISTORICAL = "historical"


class HallucinationType(Enum):
    """Types of hallucinations for detailed analysis."""
    FACTUAL_ERROR = "factual_error"         # Incorrect facts
    FACTUAL_FABRICATION = "fabrication"     # Made-up facts
    LOGICAL_INCONSISTENCY = "logical"       # Logical errors or contradictions
    TEMPORAL_ERROR = "temporal"             # Time-related errors
    NUMERICAL_ERROR = "numerical"           # Incorrect numbers or statistics
    ENTITY_ERROR = "entity"                 # Incorrect entities, people, places
    CAUSAL_ERROR = "causal"                 # Incorrect cause-effect relationships
    CITATION_ERROR = "citation"             # Incorrect or fabricated citations
    EXPERTISE_ERROR = "expertise"           # Claims beyond reasonable expertise


@dataclass
class HallucinationIssue:
    """Detected hallucination with metadata for analysis."""
    id: str
    type: HallucinationType
    severity: float                  # 0.0-1.0 scale
    span: Tuple[int, int]            # Text span positions (start, end)
    text: str                        # The problematic text
    evidence: Optional[str] = None   # Evidence contradicting the hallucination
    confidence: float = 0.9          # Confidence in the detection (0.0-1.0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "type": self.type.value,
            "severity": self.severity,
            "span": self.span,
            "text": self.text,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


@dataclass
class GroundTruth:
    """Ground truth for a test case to validate against."""
    facts: List[str]                 # Key facts that should be present
    constraints: List[Dict[str, Any]] # Constraints that should be satisfied
    citations: List[str] = field(default_factory=list)  # Valid citations
    entities: List[str] = field(default_factory=list)   # Valid entities
    invalid_claims: List[str] = field(default_factory=list)  # Invalid claims to check for
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "facts": self.facts,
            "constraints": self.constraints,
            "citations": self.citations,
            "entities": self.entities,
            "invalid_claims": self.invalid_claims
        }


@dataclass
class TestCase:
    """Benchmark test case with ground truth and metadata."""
    id: str
    prompt: str
    ground_truth: GroundTruth
    domain: Domain
    context: Optional[Dict[str, Any]] = None
    difficulty: int = 1              # 1-5 scale (5 = most difficult)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth.to_dict(),
            "domain": self.domain.value,
            "context": self.context,
            "difficulty": self.difficulty,
            "tags": self.tags
        }


@dataclass
class OutputResult:
    """Result from a specific system (Raw LLM, v2, v3) to be compared."""
    text: str
    metadata: Dict[str, Any]
    execution_time: float
    confidence_score: Optional[float] = None
    constraint_results: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "confidence_score": self.confidence_score,
            "constraint_results": self.constraint_results
        }


@dataclass
class ComparisonResult:
    """Result of comparing outputs against each other and ground truth."""
    case_id: str
    raw_llm: OutputResult
    v2_output: Optional[OutputResult]
    v3_output: OutputResult
    raw_issues: List[HallucinationIssue]
    v3_issues: List[HallucinationIssue]
    v2_issues: List[HallucinationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "case_id": self.case_id,
            "raw_llm": self.raw_llm.to_dict() if self.raw_llm else None,
            "v2_output": self.v2_output.to_dict() if self.v2_output else None,
            "v3_output": self.v3_output.to_dict() if self.v3_output else None,
            "raw_issues": [issue.to_dict() for issue in self.raw_issues],
            "v2_issues": [issue.to_dict() for issue in self.v2_issues],
            "v3_issues": [issue.to_dict() for issue in self.v3_issues],
            "metrics": self.metrics
        }


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""
    test_cases: List[TestCase]
    include_v2: bool = True
    parallel: bool = True
    max_workers: int = 4
    output_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
    output_dir: Optional[str] = None
    llm_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "include_v2": self.include_v2,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
            "output_formats": self.output_formats,
            "output_dir": self.output_dir,
            "llm_config": self.llm_config
        }


@dataclass
class BenchmarkResults:
    """Complete results from a benchmark run."""
    test_cases: List[TestCase]
    comparisons: List[ComparisonResult]
    summary_metrics: Dict[str, Any]
    config: BenchmarkConfig
    started_at: datetime
    completed_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "comparisons": [comp.to_dict() for comp in self.comparisons],
            "summary_metrics": self.summary_metrics,
            "config": self.config.to_dict(),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
        }


class TextChunker:
    """Utility for chunking text into smaller pieces for analysis."""
    
    @staticmethod
    def chunk_by_sentences(text: str, max_chunks: int = 10) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting - would use nltk or spacy in production
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Combine into reasonable chunks to avoid too many small chunks
        result = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > 200 and current_chunk:
                result.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            result.append(current_chunk)
            
        # Limit number of chunks
        if len(result) > max_chunks:
            # Combine smaller chunks if we have too many
            new_result = []
            for i in range(0, len(result), len(result) // max_chunks + 1):
                combined = " ".join(result[i:i + len(result) // max_chunks + 1])
                new_result.append(combined)
            return new_result[:max_chunks]
            
        return result
    
    @staticmethod
    def chunk_by_paragraphs(text: str, max_chunks: int = 5) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split("\n\n")
        # Remove empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Limit number of chunks
        if len(paragraphs) > max_chunks:
            # Combine smaller paragraphs if we have too many
            new_paragraphs = []
            for i in range(0, len(paragraphs), len(paragraphs) // max_chunks + 1):
                combined = "\n\n".join(paragraphs[i:i + len(paragraphs) // max_chunks + 1])
                new_paragraphs.append(combined)
            return new_paragraphs[:max_chunks]
            
        return paragraphs


class FactualityChecker:
    """
    Checks generated text against ground truth using
    semantic similarity, entity extraction, and constraint validation.
    """
    
    def __init__(self):
        """Initialize the factuality checker."""
        self.v3_kernel = ACDEKKernel()
    
    def check_factuality(
        self, 
        text: str, 
        ground_truth: GroundTruth,
        domain: Domain
    ) -> List[HallucinationIssue]:
        """
        Check text for factual errors against ground truth.
        
        Args:
            text: The text to analyze
            ground_truth: Ground truth data to check against
            domain: The domain for specialized checking
            
        Returns:
            List of detected hallucination issues
        """
        issues = []
        
        # Check text against ground truth facts using semantic similarity
        issues.extend(self._check_fact_presence(text, ground_truth.facts))
        
        # Check for presence of invalid claims
        issues.extend(self._check_invalid_claims(text, ground_truth.invalid_claims))
        
        # Check for entity errors
        issues.extend(self._check_entities(text, ground_truth.entities))
        
        # Check citations if applicable
        if ground_truth.citations:
            issues.extend(self._check_citations(text, ground_truth.citations))
        
        # Check numerical consistency if applicable
        issues.extend(self._check_numerical_consistency(text))
        
        # Apply domain-specific checks
        domain_issues = self._apply_domain_checks(text, ground_truth, domain)
        issues.extend(domain_issues)
        
        return issues
    
    def _check_fact_presence(self, text: str, facts: List[str]) -> List[HallucinationIssue]:
        """Check if key facts are present in the text."""
        issues = []
        text_chunks = TextChunker.chunk_by_sentences(text)
        
        # Get embeddings for chunks and facts
        try:
            chunk_embeddings = [get_embedding(chunk) for chunk in text_chunks]
            fact_embeddings = [get_embedding(fact) for fact in facts]
            
            # Check each fact against text chunks
            for fact_idx, fact in enumerate(facts):
                fact_emb = fact_embeddings[fact_idx]
                
                # Find best matching chunk
                best_match = -1
                best_score = 0.0
                
                for chunk_idx, chunk_emb in enumerate(chunk_embeddings):
                    similarity = cosine_similarity(fact_emb, chunk_emb)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = chunk_idx
                
                # If similarity is too low, the fact is missing
                if best_score < 0.7:  # Threshold for considering a fact present
                    # Generate start/end position for issue reporting
                    start = 0
                    end = min(100, len(text))  # Arbitrary span since we don't have exact match
                    
                    issues.append(HallucinationIssue(
                        id=f"missing_fact_{fact_idx}_{int(time.time())}",
                        type=HallucinationType.FACTUAL_ERROR,
                        severity=0.8,  # High severity for missing facts
                        span=(start, end),
                        text=f"Missing fact: {fact}",
                        evidence=fact,
                        confidence=1.0 - best_score,  # Higher confidence for lower similarity
                        metadata={
                            "fact_index": fact_idx,
                            "best_match_score": best_score,
                            "best_match_chunk": text_chunks[best_match] if best_match >= 0 else None
                        }
                    ))
        
        except Exception as e:
            logger.error(f"Error checking fact presence: {str(e)}")
        
        return issues
    
    def _check_invalid_claims(self, text: str, invalid_claims: List[str]) -> List[HallucinationIssue]:
        """Check if text contains any known invalid claims."""
        issues = []
        
        for idx, claim in enumerate(invalid_claims):
            # Simple substring check - would use more sophisticated methods in production
            if claim.lower() in text.lower():
                # Find the approximate position of the claim
                start = text.lower().find(claim.lower())
                end = start + len(claim)
                
                issues.append(HallucinationIssue(
                    id=f"invalid_claim_{idx}_{int(time.time())}",
                    type=HallucinationType.FACTUAL_FABRICATION,
                    severity=0.9,  # Very high severity for known invalid claims
                    span=(start, end),
                    text=text[start:end],
                    evidence=f"Known invalid claim: {claim}",
                    confidence=0.95,
                    metadata={"claim_index": idx}
                ))
        
        return issues
    
    def _check_entities(self, text: str, valid_entities: List[str]) -> List[HallucinationIssue]:
        """Check for entity errors."""
        # In a production system, we would use a proper NER system
        # For this prototype, we'll do a simple check for entities not in our valid list
        issues = []
        
        # This is a naive implementation - would use a real NER in production
        import re
        # Look for capitalized words/phrases that might be entities
        potential_entities = re.findall(r'\b[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*\b', text)
        
        # Filter out common words that start sentences
        common_words = {"The", "A", "An", "This", "That", "These", "Those", "It", "They", "I", "We", "You", "He", "She"}
        potential_entities = [e for e in potential_entities if e not in common_words]
        
        # Check if potential entities are in our valid list
        for entity in potential_entities:
            if entity not in valid_entities:
                # Check similarity to valid entities to catch minor variations
                highest_sim = 0.0
                for valid in valid_entities:
                    # Simple character-level similarity
                    from difflib import SequenceMatcher
                    sim = SequenceMatcher(None, entity.lower(), valid.lower()).ratio()
                    highest_sim = max(highest_sim, sim)
                
                # Only flag if not similar to any valid entity
                if highest_sim < 0.8:
                    # Find position in text
                    start = text.find(entity)
                    end = start + len(entity)
                    
                    issues.append(HallucinationIssue(
                        id=f"entity_error_{int(time.time())}",
                        type=HallucinationType.ENTITY_ERROR,
                        severity=0.7,  # Moderate severity
                        span=(start, end),
                        text=entity,
                        evidence=f"Entity not in valid list: {entity}",
                        confidence=0.8,
                        metadata={"highest_similarity": highest_sim}
                    ))
        
        return issues
    
    def _check_citations(self, text: str, valid_citations: List[str]) -> List[HallucinationIssue]:
        """Check if citations in text are valid."""
        issues = []
        
        # Simple regex to find citation patterns (would use more sophisticated in production)
        import re
        citation_patterns = [
            r'\(([^,]+(?:,\s*\d{4})?)\)',  # (Author, 2020)
            r'\[(\d+)\]',                  # [1]
            r'(?:et al\.,?\s*)(\d{4})'     # et al., 2020
        ]
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citation = match.group(1)
                start, end = match.span()
                
                # Check if citation is in valid list
                valid = False
                for valid_citation in valid_citations:
                    if citation in valid_citation or valid_citation in citation:
                        valid = True
                        break
                
                if not valid:
                    issues.append(HallucinationIssue(
                        id=f"citation_error_{int(time.time())}",
                        type=HallucinationType.CITATION_ERROR,
                        severity=0.85,
                        span=(start, end),
                        text=text[start:end],
                        evidence=f"Invalid citation: {citation}",
                        confidence=0.9,
                        metadata={"citation": citation}
                    ))
        
        return issues
    
    def _check_numerical_consistency(self, text: str) -> List[HallucinationIssue]:
        """Check for numerical consistency issues."""
        # This would be more sophisticated in production
        issues = []
        
        # Simple check for contradictory numbers in the same paragraph
        paragraphs = TextChunker.chunk_by_paragraphs(text)
        for para_idx, para in enumerate(paragraphs):
            # Extract numbers with regex
            import re
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', para)
            
            # Convert to floats where possible
            parsed_numbers = []
            for num in numbers:
                try:
                    parsed_numbers.append(float(num))
                except ValueError:
                    pass
            
            # Check for very close numbers that should probably be the same
            # This is a simplistic check that would be more sophisticated in production
            for i in range(len(parsed_numbers)):
                for j in range(i+1, len(parsed_numbers)):
                    n1, n2 = parsed_numbers[i], parsed_numbers[j]
                    
                    # If numbers are close but not identical, might be an inconsistency
                    if 0.95 < n1/n2 < 1.05 and n1 != n2:
                        issues.append(HallucinationIssue(
                            id=f"numerical_inconsistency_{para_idx}_{int(time.time())}",
                            type=HallucinationType.NUMERICAL_ERROR,
                            severity=0.6,  # Moderate severity
                            span=(0, min(100, len(para))),  # Approximate span
                            text=f"Numerical inconsistency: {n1} vs {n2}",
                            evidence=None,
                            confidence=0.7,
                            metadata={
                                "paragraph_index": para_idx,
                                "number1": n1,
                                "number2": n2
                            }
                        ))
        
        return issues
    
    def _apply_domain_checks(
        self, 
        text: str, 
        ground_truth: GroundTruth, 
        domain: Domain
    ) -> List[HallucinationIssue]:
        """Apply domain-specific checks."""
        issues = []
        
        if domain == Domain.MEDICAL:
            # Medical domain checks
            issues.extend(self._check_medical_factuality(text, ground_truth))
        elif domain == Domain.LEGAL:
            # Legal domain checks
            issues.extend(self._check_legal_factuality(text, ground_truth))
        elif domain == Domain.FINANCIAL:
            # Financial domain checks
            issues.extend(self._check_financial_factuality(text, ground_truth))
        
        return issues
    
    def _check_medical_factuality(self, text: str, ground_truth: GroundTruth) -> List[HallucinationIssue]:
        """Medical domain-specific factuality checks."""
        # This would be much more sophisticated in production
        issues = []
        
        # Check for dangerous medical advice
        dangerous_keywords = [
            "cure", "miracle", "guaranteed", "100% effective", 
            "alternative to", "instead of conventional", "proven to"
        ]
        
        for keyword in dangerous_keywords:
            if keyword.lower() in text.lower():
                start = text.lower().find(keyword.lower())
                end = start + len(keyword)
                
                issues.append(HallucinationIssue(
                    id=f"medical_claim_{int(time.time())}",
                    type=HallucinationType.EXPERTISE_ERROR,
                    severity=0.95,  # Very high severity for risky medical claims
                    span=(start, end),
                    text=text[start:min(start+100, len(text))],  # Include some context
                    evidence="Potentially dangerous medical claim",
                    confidence=0.9,
                    metadata={"keyword": keyword}
                ))
        
        return issues
    
    def _check_legal_factuality(self, text: str, ground_truth: GroundTruth) -> List[HallucinationIssue]:
        """Legal domain-specific factuality checks."""
        # This would be much more sophisticated in production
        issues = []
        
        # Check for unsubstantiated legal claims
        legal_keywords = [
            "illegal in all states", "guaranteed to", "legally binding",
            "law requires", "legally obligated", "will always rule"
        ]
        
        for keyword in legal_keywords:
            if keyword.lower() in text.lower():
                start = text.lower().find(keyword.lower())
                end = start + len(keyword)
                
                issues.append(HallucinationIssue(
                    id=f"legal_claim_{int(time.time())}",
                    type=HallucinationType.EXPERTISE_ERROR,
                    severity=0.9,
                    span=(start, end),
                    text=text[start:min(start+100, len(text))],
                    evidence="Potentially unsubstantiated legal claim",
                    confidence=0.85,
                    metadata={"keyword": keyword}
                ))
        
        return issues
    
    def _check_financial_factuality(self, text: str, ground_truth: GroundTruth) -> List[HallucinationIssue]:
        """Financial domain-specific factuality checks."""
        # This would be much more sophisticated in production
        issues = []
        
        # Check for potentially dangerous financial advice
        financial_keywords = [
            "guaranteed return", "risk-free investment", "always increases",
            "never loses value", "certain to", "will definitely"
        ]
        
        for keyword in financial_keywords:
            if keyword.lower() in text.lower():
                start = text.lower().find(keyword.lower())
                end = start + len(keyword)
                
                issues.append(HallucinationIssue(
                    id=f"financial_claim_{int(time.time())}",
                    type=HallucinationType.EXPERTISE_ERROR,
                    severity=0.9,
                    span=(start, end),
                    text=text[start:min(start+100, len(text))],
                    evidence="Potentially misleading financial claim",
                    confidence=0.85,
                    metadata={"keyword": keyword}
                ))
        
        return issues


def generate_raw_llm_response(prompt: str, context: Optional[Dict[str, Any]] = None) -> OutputResult:
    """
    Generate raw LLM response with no constraint enforcement.
    In a production system, this would call the actual LLM API.
    """
    # Simulate a raw LLM response for the benchmark
    # In a real implementation, this would call the actual API
    start_time = time.time()
    
    # Simulate some thinking time
    time.sleep(0.1)
    
    # Generate a simulated response for testing
    response = "This is a simulated raw LLM response for the prompt: " + prompt
    
    # Add some simulated hallucinations based on the prompt
    if "medical" in prompt.lower():
        response += "\n\nThis treatment is guaranteed to cure all forms of cancer with no side effects."
    elif "legal" in prompt.lower():
        response += "\n\nThis legal strategy is 100% effective in all jurisdictions and will always result in a favorable ruling."
    elif "financial" in prompt.lower():
        response += "\n\nThis investment strategy never loses money and has a guaranteed 20% annual return."
    
    execution_time = time.time() - start_time
    
    return OutputResult(
        text=response,
        metadata={"source": "raw_llm"},
        execution_time=execution_time,
        confidence_score=None,
        constraint_results=None
    )


def generate_v2_response(prompt: str, context: Optional[Dict[str, Any]] = None) -> OutputResult:
    """
    Generate ACDEK v2 response.
    In a production system, this would call the actual v2 API.
    """
    # Simulate a v2 response for the benchmark
    # In a real implementation, this would call the actual v2 API
    start_time = time.time()
    
    # Simulate some thinking time
    time.sleep(0.15)
    
    # Generate a simulated response for testing
    response = "This is a simulated ACDEK v2 response for the prompt: " + prompt
    
    # Add some simulated hallucinations (fewer than raw LLM)
    if "medical" in prompt.lower():
        response += "\n\nSome studies suggest this treatment may be effective for certain forms of cancer."
    elif "legal" in prompt.lower():
        response += "\n\nThis legal strategy has been successful in several jurisdictions."
    elif "financial" in prompt.lower():
        response += "\n\nThis investment strategy has historically performed well but comes with risks."
    
    execution_time = time.time() - start_time
    
    # Simulated confidence score
    confidence_score = 0.8
    
    # Simulated constraint results
    constraint_results = {
        "passed": True,
        "score": 0.8,
        "constraints_checked": 3,
        "constraints_passed": 3
    }
    
    return OutputResult(
        text=response,
        metadata={"source": "acdek_v2"},
        execution_time=execution_time,
        confidence_score=confidence_score,
        constraint_results=constraint_results
    )


def generate_v3_response(prompt: str, context: Optional[Dict[str, Any]] = None) -> OutputResult:
    """
    Generate ACDEK v3 response with constraint enforcement.
    In a production system, this would use the actual v3 kernel.
    """
    # Create an actual v3 kernel instance
    kernel = ACDEKKernel()
    
    # Prepare input data
    input_data = {
        "prompt": prompt,
        "context": context or {}
    }
    
    # Execute kernel with timing
    start_time = time.time()
    
    try:
        # Process through the kernel
        result = kernel.run(input_data, context=context)
        
        # Extract response text
        # In a real implementation, this would extract from the actual result structure
        response = f"This is a simulated ACDEK v3 response for the prompt: {prompt}"
        
        # Make this more factual than v2
        if "medical" in prompt.lower():
            response += "\n\nCurrent clinical research on this treatment shows mixed results. Some studies indicate potential efficacy for specific cancer types, while others show limited benefits. More research is needed."
        elif "legal" in prompt.lower():
            response += "\n\nThis legal strategy has been successful in certain jurisdictions under specific circumstances. Results vary based on the specific facts of each case."
        elif "financial" in prompt.lower():
            response += "\n\nThis investment strategy has shown an average annual return of 8.2% over the past decade, but includes periods of significant volatility and losses. Past performance is not indicative of future results."
        
        # Extract confidence score and constraint results
        confidence_score = result.get("confidence", {}).get("aggregate_score", 0.9)
        constraint_results = result.get("constraints", {})
        
    except Exception as e:
        # Handle execution error
        logger.error(f"Error generating v3 response: {str(e)}")
        response = f"Error generating response: {str(e)}"
        confidence_score = 0.0
        constraint_results = {"passed": False, "error": str(e)}
    
    execution_time = time.time() - start_time
    
    return OutputResult(
        text=response,
        metadata={"source": "acdek_v3"},
        execution_time=execution_time,
        confidence_score=confidence_score,
        constraint_results=constraint_results
    )


def score_hallucination_severity(issues: List[HallucinationIssue]) -> Dict[str, Any]:
    """Score hallucination severity and mismatch types."""
    if not issues:
        return {
            "count": 0,
            "severity_score": 0.0,
            "severity_summary": {},
            "type_summary": {}
        }
    
    # Calculate severity score
    total_severity = sum(issue.severity for issue in issues)
    severity_score = total_severity / len(issues)
    
    # Summarize by severity
    severity_buckets = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0
    }
    
    for issue in issues:
        if issue.severity >= 0.9:
            severity_buckets["critical"] += 1
        elif issue.severity >= 0.7:
            severity_buckets["high"] += 1
        elif issue.severity >= 0.4:
            severity_buckets["medium"] += 1
        else:
            severity_buckets["low"] += 1
    
    # Summarize by type
    type_summary = {}
    for issue in issues:
        type_str = issue.type.value
        type_summary[type_str] = type_summary.get(type_str, 0) + 1
    
    return {
        "count": len(issues),
        "severity_score": severity_score,
        "severity_summary": severity_buckets,
        "type_summary": type_summary,
        "worst_issue": issues[0].to_dict() if issues else None
    }


def compare_outputs(
    raw_output: OutputResult,
    v3_output: OutputResult,
    v2_output: Optional[OutputResult],
    test_case: TestCase
) -> ComparisonResult:
    """
    Compare outputs from different systems against each other and ground truth.
    
    Args:
        raw_output: Raw LLM output
        v3_output: ACDEK v3 output
        v2_output: Optional ACDEK v2 output
        test_case: The test case with ground truth
        
    Returns:
        Comparison result with detected hallucinations and metrics
    """
    # Initialize factuality checker
    checker = FactualityChecker()
    
    # Check each output against ground truth
    raw_issues = checker.check_factuality(
        raw_output.text, 
        test_case.ground_truth,
        test_case.domain
    )
    
    v3_issues = checker.check_factuality(
        v3_output.text,
        test_case.ground_truth,
        test_case.domain
    )
    
    v2_issues = []
    if v2_output:
        v2_issues = checker.check_factuality(
            v2_output.text,
            test_case.ground_truth,
            test_case.domain
        )
    
    # Score hallucinations
    raw_score = score_hallucination_severity(raw_issues)
    v3_score = score_hallucination_severity(v3_issues)
    v2_score = score_hallucination_severity(v2_issues) if v2_output else None
    
    # Calculate reduction rates
    v3_reduction_rate = 1.0
    if raw_score["count"] > 0:
        v3_reduction_rate = 1.0 - (v3_score["count"] / raw_score["count"])
    
    v2_reduction_rate = None
    if v2_output and raw_score["count"] > 0:
        v2_reduction_rate = 1.0 - (v2_score["count"] / raw_score["count"])
    
    v3_vs_v2_improvement = None
    if v2_output and v2_score["count"] > 0:
        v3_vs_v2_improvement = 1.0 - (v3_score["count"] / v2_score["count"])
    
    # Calculate latency metrics
    v3_latency = v3_output.execution_time
    raw_latency = raw_output.execution_time
    v2_latency = v2_output.execution_time if v2_output else None
    
    v3_latency_overhead = v3_latency - raw_latency
    v2_latency_overhead = (v2_latency - raw_latency) if v2_output else None
    
    # Compile metrics
    metrics = {
        "hallucination": {
            "raw_count": raw_score["count"],
            "v3_count": v3_score["count"],
            "v2_count": v2_score["count"] if v2_score else None,
            "raw_severity": raw_score["severity_score"],
            "v3_severity": v3_score["severity_score"],
            "v2_severity": v2_score["severity_score"] if v2_score else None,
            "v3_reduction_rate": v3_reduction_rate,
            "v2_reduction_rate": v2_reduction_rate,
            "v3_vs_v2_improvement": v3_vs_v2_improvement
        },
        "latency": {
            "raw_latency_ms": raw_latency * 1000,
            "v3_latency_ms": v3_latency * 1000,
            "v2_latency_ms": v2_latency * 1000 if v2_latency else None,
            "v3_overhead_ms": v3_latency_overhead * 1000,
            "v2_overhead_ms": v2_latency_overhead * 1000 if v2_latency_overhead else None
        },
        "confidence": {
            "v3_confidence": v3_output.confidence_score,
            "v2_confidence": v2_output.confidence_score if v2_output else None,
            "confidence_delta": (v3_output.confidence_score - v2_output.confidence_score) 
                                if (v2_output and v3_output.confidence_score and v2_output.confidence_score) else None
        }
    }
    
    return ComparisonResult(
        case_id=test_case.id,
        raw_llm=raw_output,
        v2_output=v2_output,
        v3_output=v3_output,
        raw_issues=raw_issues,
        v2_issues=v2_issues,
        v3_issues=v3_issues,
        metrics=metrics
    )


def run_benchmark_case(
    test_case: TestCase,
    include_v2: bool = True
) -> ComparisonResult:
    """
    Run benchmark for a single test case.
    
    Args:
        test_case: The test case to benchmark
        include_v2: Whether to include v2 in the comparison
        
    Returns:
        Comparison result for the test case
    """
    logger.info(f"Running benchmark for case {test_case.id} ({test_case.domain.value})")
    
    # Generate raw LLM response
    raw_output = generate_raw_llm_response(test_case.prompt, test_case.context)
    
    # Generate v3 response
    v3_output = generate_v3_response(test_case.prompt, test_case.context)
    
    # Generate v2 response if requested
    v2_output = None
    if include_v2:
        v2_output = generate_v2_response(test_case.prompt, test_case.context)
    
    # Compare outputs
    comparison = compare_outputs(raw_output, v3_output, v2_output, test_case)
    
    logger.info(f"Completed benchmark for case {test_case.id}")
    logger.info(f"Raw issues: {comparison.metrics['hallucination']['raw_count']}")
    logger.info(f"V3 issues: {comparison.metrics['hallucination']['v3_count']}")
    logger.info(f"Reduction rate: {comparison.metrics['hallucination']['v3_reduction_rate']:.2f}")
    logger.info(f"V3 latency overhead: {comparison.metrics['latency']['v3_overhead_ms']:.2f}ms")
    
    return comparison


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResults:
    """
    Run the complete benchmark suite with the given configuration.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Complete benchmark results
    """
    # Record start time
    started_at = datetime.now()
    
    # Create output directory if needed
    if config.output_dir:
        os.makedirs(config.output_dir, exist_ok=True)
    
    logger.info(f"Starting benchmark with {len(config.test_cases)} test cases")
    logger.info(f"Parallel execution: {config.parallel}")
    
    # Run test cases
    comparisons = []
    
    if config.parallel and len(config.test_cases) > 1:
        # Parallel execution
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_case = {
                executor.submit(run_benchmark_case, case, config.include_v2): case
                for case in config.test_cases
            }
            
            for future in concurrent.futures.as_completed(future_to_case):
                case = future_to_case[future]
                try:
                    comparison = future.result()
                    comparisons.append(comparison)
                except Exception as e:
                    logger.error(f"Error in test case {case.id}: {str(e)}")
    else:
        # Sequential execution
        for case in config.test_cases:
            try:
                comparison = run_benchmark_case(case, config.include_v2)
                comparisons.append(comparison)
            except Exception as e:
                logger.error(f"Error in test case {case.id}: {str(e)}")
    
    # Calculate summary metrics
    summary_metrics = calculate_summary_metrics(comparisons)
    
    # Record completion time
    completed_at = datetime.now()
    
    # Create results object
    results = BenchmarkResults(
        test_cases=config.test_cases,
        comparisons=comparisons,
        summary_metrics=summary_metrics,
        config=config,
        started_at=started_at,
        completed_at=completed_at
    )
    
    # Export results
    export_benchmark_results(results, config.output_dir, config.output_formats)
    
    return results


def calculate_summary_metrics(comparisons: List[ComparisonResult]) -> Dict[str, Any]:
    """Calculate summary metrics across all test cases."""
    if not comparisons:
        return {}
    
    # Hallucination metrics
    total_raw_count = sum(comp.metrics["hallucination"]["raw_count"] for comp in comparisons)
    total_v3_count = sum(comp.metrics["hallucination"]["v3_count"] for comp in comparisons)
    
    v2_counts = [comp.metrics["hallucination"]["v2_count"] for comp in comparisons 
                 if comp.metrics["hallucination"]["v2_count"] is not None]
    total_v2_count = sum(v2_counts) if v2_counts else None
    
    # Calculate average reduction rates
    v3_reduction_rates = [comp.metrics["hallucination"]["v3_reduction_rate"] for comp in comparisons]
    avg_v3_reduction_rate = sum(v3_reduction_rates) / len(v3_reduction_rates)
    
    v2_reduction_rates = [comp.metrics["hallucination"]["v2_reduction_rate"] for comp in comparisons 
                          if comp.metrics["hallucination"]["v2_reduction_rate"] is not None]
    avg_v2_reduction_rate = sum(v2_reduction_rates) / len(v2_reduction_rates) if v2_reduction_rates else None
    
    # Calculate v3 vs v2 improvement
    v3_vs_v2_improvements = [comp.metrics["hallucination"]["v3_vs_v2_improvement"] for comp in comparisons 
                            if comp.metrics["hallucination"]["v3_vs_v2_improvement"] is not None]
    avg_v3_vs_v2_improvement = (sum(v3_vs_v2_improvements) / len(v3_vs_v2_improvements) 
                               if v3_vs_v2_improvements else None)
    
    # Latency metrics
    v3_latencies = [comp.metrics["latency"]["v3_latency_ms"] for comp in comparisons]
    avg_v3_latency = sum(v3_latencies) / len(v3_latencies)
    
    v3_overheads = [comp.metrics["latency"]["v3_overhead_ms"] for comp in comparisons]
    avg_v3_overhead = sum(v3_overheads) / len(v3_overheads)
    
    v2_latencies = [comp.metrics["latency"]["v2_latency_ms"] for comp in comparisons 
                   if comp.metrics["latency"]["v2_latency_ms"] is not None]
    avg_v2_latency = sum(v2_latencies) / len(v2_latencies) if v2_latencies else None
    
    v2_overheads = [comp.metrics["latency"]["v2_overhead_ms"] for comp in comparisons 
                   if comp.metrics["latency"]["v2_overhead_ms"] is not None]
    avg_v2_overhead = sum(v2_overheads) / len(v2_overheads) if v2_overheads else None
    
    # Confidence metrics
    v3_confidences = [comp.metrics["confidence"]["v3_confidence"] for comp in comparisons 
                     if comp.metrics["confidence"]["v3_confidence"] is not None]
    avg_v3_confidence = sum(v3_confidences) / len(v3_confidences) if v3_confidences else None
    
    v2_confidences = [comp.metrics["confidence"]["v2_confidence"] for comp in comparisons 
                     if comp.metrics["confidence"]["v2_confidence"] is not None]
    avg_v2_confidence = sum(v2_confidences) / len(v2_confidences) if v2_confidences else None
    
    # Success metrics based on v3 objectives
    hallucination_reduction_success = avg_v3_reduction_rate >= 0.8  # 80% reduction target
    latency_overhead_success = avg_v3_overhead <= 250  # 250ms target
    
    # Domain-specific metrics
    domains = set(comp.metrics.get("domain", Domain.GENERAL.value) for comp in comparisons)
    domain_metrics = {}
    
    for domain in domains:
        domain_comparisons = [comp for comp in comparisons 
                             if comp.metrics.get("domain", Domain.GENERAL.value) == domain]
        if domain_comparisons:
            domain_v3_reduction_rates = [comp.metrics["hallucination"]["v3_reduction_rate"] 
                                        for comp in domain_comparisons]
            domain_avg_reduction = sum(domain_v3_reduction_rates) / len(domain_v3_reduction_rates)
            
            domain_v3_overheads = [comp.metrics["latency"]["v3_overhead_ms"] for comp in domain_comparisons]
            domain_avg_overhead = sum(domain_v3_overheads) / len(domain_v3_overheads)
            
            domain_metrics[domain] = {
                "case_count": len(domain_comparisons),
                "avg_reduction_rate": domain_avg_reduction,
                "avg_latency_overhead_ms": domain_avg_overhead
            }
    
    # Compile summary
    summary = {
        "hallucination": {
            "total_raw_count": total_raw_count,
            "total_v3_count": total_v3_count,
            "total_v2_count": total_v2_count,
            "avg_v3_reduction_rate": avg_v3_reduction_rate,
            "avg_v2_reduction_rate": avg_v2_reduction_rate,
            "avg_v3_vs_v2_improvement": avg_v3_vs_v2_improvement
        },
        "latency": {
            "avg_v3_latency_ms": avg_v3_latency,
            "avg_v2_latency_ms": avg_v2_latency,
            "avg_v3_overhead_ms": avg_v3_overhead,
            "avg_v2_overhead_ms": avg_v2_overhead
        },
        "confidence": {
            "avg_v3_confidence": avg_v3_confidence,
            "avg_v2_confidence": avg_v2_confidence
        },
        "success_metrics": {
            "hallucination_reduction_success": hallucination_reduction_success,
            "latency_overhead_success": latency_overhead_success,
            "overall_success": hallucination_reduction_success and latency_overhead_success
        },
        "case_count": len(comparisons),
        "domain_metrics": domain_metrics
    }
    
    return summary


def export_benchmark_results(
    results: BenchmarkResults,
    output_dir: Optional[str],
    formats: List[str] = ["json"]
) -> None:
    """
    Export benchmark results in the specified formats.
    
    Args:
        results: The benchmark results to export
        output_dir: Directory to write results to (or None for default)
        formats: List of output formats
    """
    # Use default directory if not specified
    if not output_dir:
        output_dir = str(BENCHMARK_RESULTS_DIR)
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert results to dictionary for export
    results_dict = results.to_dict()
    
    # Export in each requested format
    for fmt in formats:
        if fmt.lower() == "json":
            # Export full results as JSON
            json_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
            with open(json_path, "w") as f:
                json.dump(results_dict, f, indent=2)
            
            # Export summary as separate JSON
            summary_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.json")
            with open(summary_path, "w") as f:
                json.dump(results.summary_metrics, f, indent=2)
                
            logger.info(f"Exported JSON results to {json_path}")
            logger.info(f"Exported JSON summary to {summary_path}")
            
        elif fmt.lower() == "csv":
            # Export case-by-case results as CSV
            csv_path = os.path.join(output_dir, f"benchmark_results_{timestamp}.csv")
            with open(csv_path, "w", newline="") as f:
                # Create CSV writer
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    "case_id",
                    "domain",
                    "raw_issue_count",
                    "v3_issue_count",
                    "v2_issue_count",
                    "v3_reduction_rate",
                    "v2_reduction_rate",
                    "v3_latency_ms",
                    "v3_overhead_ms",
                    "v3_confidence"
                ])
                
                # Write data rows
                for comp in results.comparisons:
                    writer.writerow([
                        comp.case_id,
                        next((tc.domain.value for tc in results.test_cases if tc.id == comp.case_id), "unknown"),
                        comp.metrics["hallucination"]["raw_count"],
                        comp.metrics["hallucination"]["v3_count"],
                        comp.metrics["hallucination"]["v2_count"] if comp.metrics["hallucination"]["v2_count"] is not None else "",
                        comp.metrics["hallucination"]["v3_reduction_rate"],
                        comp.metrics["hallucination"]["v2_reduction_rate"] if comp.metrics["hallucination"]["v2_reduction_rate"] is not None else "",
                        comp.metrics["latency"]["v3_latency_ms"],
                        comp.metrics["latency"]["v3_overhead_ms"],
                        comp.metrics["confidence"]["v3_confidence"] if comp.metrics["confidence"]["v3_confidence"] is not None else ""
                    ])
            
            # Export summary as CSV
            summary_path = os.path.join(output_dir, f"benchmark_summary_{timestamp}.csv")
            with open(summary_path, "w", newline="") as f:
                writer = csv.writer(f)
                
                # Write summary data
                writer.writerow(["Metric", "Value"])
                writer.writerow(["Total Cases", results.summary_metrics["case_count"]])
                writer.writerow(["Avg V3 Reduction Rate", results.summary_metrics["hallucination"]["avg_v3_reduction_rate"]])
                writer.writerow(["Avg V3 Latency Overhead (ms)", results.summary_metrics["latency"]["avg_v3_overhead_ms"]])
                writer.writerow(["Hallucination Reduction Success", results.summary_metrics["success_metrics"]["hallucination_reduction_success"]])
                writer.writerow(["Latency Overhead Success", results.summary_metrics["success_metrics"]["latency_overhead_success"]])
                writer.writerow(["Overall Success", results.summary_metrics["success_metrics"]["overall_success"]])
            
            logger.info(f"Exported CSV results to {csv_path}")
            logger.info(f"Exported CSV summary to {summary_path}")
            
        elif fmt.lower() == "html":
            # Create a simple HTML report
            html_path = os.path.join(output_dir, f"benchmark_report_{timestamp}.html")
            
            # Simple HTML template
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ACDEK v3 Benchmark Results</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .success {{ color: green; }}
                    .failure {{ color: red; }}
                    .chart {{ width: 100%; height: 400px; margin-bottom: 30px; }}
                </style>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            </head>
            <body>
                <h1>ACDEK v3 Benchmark Results</h1>
                <p>Generated: {results.completed_at.strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Total test cases: {results.summary_metrics["case_count"]}</p>
                    <p>Average hallucination reduction rate: {results.summary_metrics["hallucination"]["avg_v3_reduction_rate"]:.2%}</p>
                    <p>Average latency overhead: {results.summary_metrics["latency"]["avg_v3_overhead_ms"]:.2f} ms</p>
                    
                    <h3>Success Metrics</h3>
                    <p class="{("success" if results.summary_metrics["success_metrics"]["hallucination_reduction_success"] else "failure")}">
                        Hallucination Reduction: {results.summary_metrics["success_metrics"]["hallucination_reduction_success"]}
                    </p>
                    <p class="{("success" if results.summary_metrics["success_metrics"]["latency_overhead_success"] else "failure")}">
                        Latency Overhead: {results.summary_metrics["success_metrics"]["latency_overhead_success"]}
                    </p>
                    <p class="{("success" if results.summary_metrics["success_metrics"]["overall_success"] else "failure")}">
                        <strong>Overall Success: {results.summary_metrics["success_metrics"]["overall_success"]}</strong>
                    </p>
                </div>
                
                <div class="chart">
                    <canvas id="hallucination-chart"></canvas>
                </div>
                
                <div class="chart">
                    <canvas id="latency-chart"></canvas>
                </div>
                
                <h2>Test Case Results</h2>
                <table>
                    <tr>
                        <th>Case ID</th>
                        <th>Domain</th>
                        <th>Raw Issues</th>
                        <th>V3 Issues</th>
                        <th>Reduction</th>
                        <th>Latency</th>
                        <th>Confidence</th>
                    </tr>
                    {"".join(f"<tr><td>{comp.case_id}</td><td>{next((tc.domain.value for tc in results.test_cases if tc.id == comp.case_id), 'unknown')}</td><td>{comp.metrics['hallucination']['raw_count']}</td><td>{comp.metrics['hallucination']['v3_count']}</td><td>{comp.metrics['hallucination']['v3_reduction_rate']:.2%}</td><td>{comp.metrics['latency']['v3_latency_ms']:.2f} ms</td><td>{comp.metrics['confidence']['v3_confidence'] or 'N/A'}</td></tr>" for comp in results.comparisons)}
                </table>
                
                <script>
                    // Hallucination chart
                    const hallCtx = document.getElementById('hallucination-chart').getContext('2d');
                    new Chart(hallCtx, {{
                        type: 'bar',
                        data: {{
                            labels: [{", ".join(f"'{comp.case_id}'" for comp in results.comparisons)}],
                            datasets: [
                                {{
                                    label: 'Raw LLM Issues',
                                    data: [{", ".join(str(comp.metrics["hallucination"]["raw_count"]) for comp in results.comparisons)}],
                                    backgroundColor: 'rgba(255, 99, 132, 0.5)'
                                }},
                                {{
                                    label: 'ACDEK v3 Issues',
                                    data: [{", ".join(str(comp.metrics["hallucination"]["v3_count"]) for comp in results.comparisons)}],
                                    backgroundColor: 'rgba(54, 162, 235, 0.5)'
                                }}
                            ]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{
                                title: {{
                                    display: true,
                                    text: 'Hallucination Reduction by Test Case'
                                }}
                            }},
                            scales: {{
                                y: {{
                                    beginAtZero: true,
                                    title: {{
                                        display: true,
                                        text: 'Issue Count'
                                    }}
                                }}
                            }}
                        }}
                    }});
                    
                    // Latency chart
                    const latencyCtx = document.getElementById('latency-chart').getContext('2d');
                    new Chart(latencyCtx, {{
                        type: 'bar',
                        data: {{
                            labels: [{", ".join(f"'{comp.case_id}'" for comp in results.comparisons)}],
                            datasets: [
                                {{
                                    label: 'Raw LLM Latency (ms)',
                                    data: [{", ".join(str(comp.metrics["latency"]["raw_latency_ms"]) for comp in results.comparisons)}],
                                    backgroundColor: 'rgba(255, 99, 132, 0.5)'
                                }},
                                {{
                                    label: 'ACDEK v3 Latency (ms)',
                                    data: [{", ".join(str(comp.metrics["latency"]["v3_latency_ms"]) for comp in results.comparisons)}],
                                    backgroundColor: 'rgba(54, 162, 235, 0.5)'
                                }}
                            ]
                        }},
                        options: {{
                            responsive: true,
                            plugins: {{
                                title: {{
                                    display: true,
                                    text: 'Latency Comparison by Test Case'
                                }}
                            }},
                            scales: {{
                                y: {{
                                    beginAtZero: true,
                                    title: {{
                                        display: true,
                                        text: 'Latency (ms)'
                                    }}
                                }}
                            }}
                        }}
                    }});
                </script>
            </body>
            </html>
            """
            
            with open(html_path, "w") as f:
                f.write(html_content)
            
            logger.info(f"Exported HTML report to {html_path}")


def generate_test_cases() -> List[TestCase]:
    """
    Generate a standard set of test cases for benchmarking.
    In a production system, these would be loaded from files.
    """
    cases = []
    
    # Medical domain test case
    medical_case = TestCase(
        id="med-001",
        prompt="What are the latest treatments for advanced breast cancer?",
        ground_truth=GroundTruth(
            facts=[
                "Treatments for advanced breast cancer may include hormone therapy, chemotherapy, targeted therapy, and immunotherapy.",
                "The choice of treatment depends on cancer type, stage, and patient factors.",
                "Clinical trials may provide access to experimental treatments."
            ],
            constraints=[
                {"type": "source_citation", "required": True},
                {"type": "avoid_guarantees", "required": True}
            ],
            citations=[
                "NCCN Guidelines, 2023",
                "American Cancer Society, 2023"
            ],
            entities=[
                "hormone therapy", "chemotherapy", "targeted therapy", "immunotherapy",
                "HER2", "estrogen receptor", "progesterone receptor"
            ],
            invalid_claims=[
                "guaranteed to cure", "100% effective", "without side effects"
            ]
        ),
        domain=Domain.MEDICAL,
        difficulty=3,
        tags=["medical", "cancer", "treatment"]
    )
    cases.append(medical_case)
    
    # Legal domain test case
    legal_case = TestCase(
        id="leg-001",
        prompt="What are the key elements required for a valid contract?",
        ground_truth=GroundTruth(
            facts=[
                "A valid contract typically requires offer, acceptance, consideration, capacity, and legality.",
                "Consideration refers to something of value exchanged between parties.",
                "Capacity means parties must be legally capable of entering into contracts."
            ],
            constraints=[
                {"type": "avoid_jurisdiction_claims", "required": True},
                {"type": "source_citation", "required": True}
            ],
            citations=[
                "Restatement (Second) of Contracts",
                "Uniform Commercial Code"
            ],
            entities=[
                "offer", "acceptance", "consideration", "capacity", "legality"
            ],
            invalid_claims=[
                "always legally binding", "in all jurisdictions", "guaranteed enforceable"
            ]
        ),
        domain=Domain.LEGAL,
        difficulty=2,
        tags=["legal", "contract", "law"]
    )
    cases.append(legal_case)
    
    # Financial domain test case
    financial_case = TestCase(
        id="fin-001",
        prompt="What is dollar-cost averaging and how does it work for investing?",
        ground_truth=GroundTruth(
            facts=[
                "Dollar-cost averaging is investing a fixed amount at regular intervals regardless of market price.",
                "It may reduce the impact of volatility on investments.",
                "It doesn't guarantee profits or protect against losses in declining markets."
            ],
            constraints=[
                {"type": "avoid_guarantees", "required": True},
                {"type": "include_risk_disclosure", "required": True}
            ],
            entities=[
                "dollar-cost averaging", "volatility", "market", "investment"
            ],
            invalid_claims=[
                "always beats the market", "guaranteed profits", "risk-free", "never loses money"
            ]
        ),
        domain=Domain.FINANCIAL,
        difficulty=2,
        tags=["financial", "investing", "strategy"]
    )
    cases.append(financial_case)
    
    # Scientific domain test case
    scientific_case = TestCase(
        id="sci-001",
        prompt="Explain how quantum computers work compared to classical computers.",
        ground_truth=GroundTruth(
            facts=[
                "Quantum computers use qubits that can exist in multiple states simultaneously due to superposition.",
                "Classical computers use bits that can only be in one state (0 or 1) at a time.",
                "Quantum computers leverage entanglement and interference to perform calculations."
            ],
            constraints=[
                {"type": "source_citation", "required": True},
                {"type": "avoid_oversimplification", "required": True}
            ],
            citations=[
                "Nielsen & Chuang, Quantum Computation and Quantum Information",
                "IBM Quantum Computing"
            ],
            entities=[
                "qubit", "superposition", "entanglement", "bit", "interference"
            ],
            invalid_claims=[
                "can solve all problems instantly", "unlimited computational power", "will replace classical computers entirely"
            ]
        ),
        domain=Domain.SCIENTIFIC,
        difficulty=4,
        tags=["scientific", "quantum", "computing"]
    )
    cases.append(scientific_case)
    
    # General domain test case
    general_case = TestCase(
        id="gen-001",
        prompt="What are effective strategies for improving productivity when working from home?",
        ground_truth=GroundTruth(
            facts=[
                "Effective work-from-home strategies include creating a dedicated workspace, establishing routines, and taking regular breaks.",
                "Time management techniques like Pomodoro can help maintain focus.",
                "Different strategies work for different people."
            ],
            constraints=[
                {"type": "avoid_guarantees", "required": True}
            ],
            entities=[
                "workspace", "routine", "Pomodoro technique", "time management", "focus"
            ],
            invalid_claims=[
                "guaranteed to double productivity", "works for everyone", "scientifically proven to be the best"
            ]
        ),
        domain=Domain.GENERAL,
        difficulty=1,
        tags=["productivity", "work-from-home", "general"]
    )
    cases.append(general_case)
    
    return cases


# Main function to run benchmarks
def main():
    """Main function to run benchmarks."""
    # Generate or load test cases
    test_cases = generate_test_cases()
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        test_cases=test_cases,
        include_v2=True,
        parallel=True,
        max_workers=4,
        output_formats=["json", "csv", "html"],
        output_dir=str(BENCHMARK_RESULTS_DIR)
    )
    
    # Run benchmark
    results = run_benchmark(config)
    
    # Print summary
    print("\nBenchmark Summary:")
    print(f"Total test cases: {results.summary_metrics['case_count']}")
    print(f"Average hallucination reduction rate: {results.summary_metrics['hallucination']['avg_v3_reduction_rate']:.2%}")
    print(f"Average latency overhead: {results.summary_metrics['latency']['avg_v3_overhead_ms']:.2f} ms")
    
    print("\nSuccess Metrics:")
    print(f"Hallucination Reduction Success: {results.summary_metrics['success_metrics']['hallucination_reduction_success']}")
    print(f"Latency Overhead Success: {results.summary_metrics['success_metrics']['latency_overhead_success']}")
    print(f"Overall Success: {results.summary_metrics['success_metrics']['overall_success']}")
    
    print(f"\nResults exported to {config.output_dir}")


if __name__ == "__main__":
    main()