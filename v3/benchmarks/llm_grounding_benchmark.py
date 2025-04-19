#!/usr/bin/env python3
"""
LLM Grounding Benchmark Suite

A specialized benchmark for evaluating the effectiveness of ACDEK's grounding engine
in mitigating hallucinations across different LLM responses.

This benchmark focuses specifically on:
1. Factuality verification with reference data
2. Hallucination detection and classification
3. Confidence correlation with factual accuracy
4. Cross-domain robustness
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import concurrent.futures

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import ACDEK components
from acdek.acdek_kernel import ACDEKKernel
from v3.v3.grounding.grounding_engine import (
    grounded_validation_pipeline,
    score_factuality,
    retrieve_reference,
    DOMAIN_THRESHOLDS,
    FACTUALITY_THRESHOLD
)
from llm_adapters.openai_adapter import OpenAIAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grounding_benchmark.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class GroundingTestCase:
    """Test case for evaluating grounding effectiveness."""
    id: str
    prompt: str
    domain: str
    expected_hallucinations: bool = False  # Whether we expect hallucinations in raw LLM response
    difficulty: int = 1  # 1-5 scale
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "domain": self.domain,
            "expected_hallucinations": self.expected_hallucinations,
            "difficulty": self.difficulty,
            "tags": self.tags
        }


@dataclass
class LLMResponse:
    """LLM response with metadata."""
    text: str
    prompt: str
    model: str
    accuracy_focused: bool
    latency: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "prompt": self.prompt,
            "model": self.model,
            "accuracy_focused": self.accuracy_focused,
            "latency": self.latency,
            "timestamp": self.timestamp
        }


@dataclass
class GroundingResult:
    """Result of grounding evaluation."""
    factuality_score: float
    hallucination_flag: bool
    reference_text: str
    debug_metadata: Dict[str, Any]
    domain: str
    threshold_used: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "factuality_score": self.factuality_score,
            "hallucination_flag": self.hallucination_flag,
            "reference_text": self.reference_text,
            "debug_metadata": self.debug_metadata,
            "domain": self.domain,
            "threshold_used": self.threshold_used
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a test case."""
    test_case: GroundingTestCase
    standard_response: LLMResponse
    accuracy_response: LLMResponse
    standard_grounding: GroundingResult
    accuracy_grounding: GroundingResult
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_case": self.test_case.to_dict(),
            "standard_response": self.standard_response.to_dict(),
            "accuracy_response": self.accuracy_response.to_dict(),
            "standard_grounding": self.standard_grounding.to_dict(),
            "accuracy_grounding": self.accuracy_grounding.to_dict(),
            "metrics": self.metrics
        }


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark."""
    model: str = "gpt-4"
    domains: List[str] = field(default_factory=lambda: ["financial", "medical", "legal"])
    parallel: bool = True
    max_workers: int = 4
    output_dir: str = "./benchmarks/results"
    output_file: str = "grounding_benchmark_results.json"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "domains": self.domains,
            "parallel": self.parallel,
            "max_workers": self.max_workers,
            "output_dir": self.output_dir,
            "output_file": self.output_file
        }


class GroundingBenchmark:
    """
    Benchmark for evaluating ACDEK's grounding engine.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize the benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.llm_adapter = OpenAIAdapter(model=config.model)
        self.test_cases = self._load_test_cases()
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _load_test_cases(self) -> List[GroundingTestCase]:
        """
        Load or generate test cases.
        
        Returns:
            List of test cases
        """
        test_cases = []
        
        # Medical domain test cases
        if "medical" in self.config.domains:
            test_cases.extend([
                GroundingTestCase(
                    id="med-001",
                    prompt="What are the primary symptoms of diabetes?",
                    domain="medical",
                    expected_hallucinations=False,
                    difficulty=1,
                    tags=["diabetes", "symptoms", "common"]
                ),
                GroundingTestCase(
                    id="med-002",
                    prompt="Can diabetes be completely cured?",
                    domain="medical",
                    expected_hallucinations=True,
                    difficulty=2,
                    tags=["diabetes", "cure", "misleading"]
                ),
                GroundingTestCase(
                    id="med-003",
                    prompt="What is a rare genetic disorder that affects fewer than 1000 people worldwide?",
                    domain="medical",
                    expected_hallucinations=True,
                    difficulty=4,
                    tags=["rare", "genetic", "specific"]
                )
            ])
        
        # Financial domain test cases
        if "financial" in self.config.domains:
            test_cases.extend([
                GroundingTestCase(
                    id="fin-001",
                    prompt="What is the difference between an asset and a liability?",
                    domain="financial",
                    expected_hallucinations=False,
                    difficulty=1,
                    tags=["asset", "liability", "basic"]
                ),
                GroundingTestCase(
                    id="fin-002",
                    prompt="Is an asset in financial terms a type of liability?",
                    domain="financial",
                    expected_hallucinations=True,
                    difficulty=2,
                    tags=["asset", "liability", "misleading"]
                ),
                GroundingTestCase(
                    id="fin-003",
                    prompt="What investment strategy guarantees a 15% annual return with no risk?",
                    domain="financial",
                    expected_hallucinations=True,
                    difficulty=3,
                    tags=["investment", "guarantee", "misleading"]
                )
            ])
            
        # Legal domain test cases
        if "legal" in self.config.domains:
            test_cases.extend([
                GroundingTestCase(
                    id="leg-001",
                    prompt="What are the essential elements of a valid contract?",
                    domain="legal",
                    expected_hallucinations=False,
                    difficulty=1,
                    tags=["contract", "elements", "basic"]
                ),
                GroundingTestCase(
                    id="leg-002",
                    prompt="Is there a legal strategy that guarantees success in all court cases?",
                    domain="legal",
                    expected_hallucinations=True,
                    difficulty=3,
                    tags=["guarantee", "court", "misleading"]
                )
            ])
        
        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the benchmark.
        
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Starting grounding benchmark with {len(self.test_cases)} test cases")
        start_time = time.time()
        
        # Run test cases
        results = []
        
        if self.config.parallel and len(self.test_cases) > 1:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_case = {
                    executor.submit(self._run_test_case, case): case
                    for case in self.test_cases
                }
                
                for future in concurrent.futures.as_completed(future_to_case):
                    case = future_to_case[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed test case {case.id}")
                    except Exception as e:
                        logger.error(f"Error in test case {case.id}: {str(e)}")
        else:
            # Sequential execution
            for case in self.test_cases:
                try:
                    result = self._run_test_case(case)
                    results.append(result)
                    logger.info(f"Completed test case {case.id}")
                except Exception as e:
                    logger.error(f"Error in test case {case.id}: {str(e)}")
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(results)
        
        # Compile benchmark results
        benchmark_results = {
            "config": self.config.to_dict(),
            "results": [r.to_dict() for r in results],
            "summary_metrics": summary_metrics,
            "timestamp": datetime.now().isoformat(),
            "execution_time_seconds": time.time() - start_time
        }
        
        # Export results
        self._export_results(benchmark_results)
        
        return benchmark_results
    
    def _run_test_case(self, test_case: GroundingTestCase) -> BenchmarkResult:
        """
        Run a single test case.
        
        Args:
            test_case: The test case to run
            
        Returns:
            Benchmark result
        """
        logger.info(f"Running test case {test_case.id} ({test_case.domain})")
        
        # Get standard response (higher temperature)
        standard_start_time = time.time()
        standard_response_text = self.llm_adapter.call_llm(test_case.prompt)
        standard_latency = time.time() - standard_start_time
        
        standard_response = LLMResponse(
            text=standard_response_text,
            prompt=test_case.prompt,
            model=self.config.model,
            accuracy_focused=False,
            latency=standard_latency
        )
        
        # Get accuracy-focused response (lower temperature)
        accuracy_start_time = time.time()
        accuracy_response_text = self.llm_adapter.call_with_accuracy_focus(test_case.prompt)
        accuracy_latency = time.time() - accuracy_start_time
        
        accuracy_response = LLMResponse(
            text=accuracy_response_text,
            prompt=test_case.prompt,
            model=self.config.model,
            accuracy_focused=True,
            latency=accuracy_latency
        )
        
        # Apply grounding to standard response
        standard_grounding_result = grounded_validation_pipeline(
            test_case.prompt,
            standard_response_text,
            test_case.domain,
            include_debug=True
        )
        
        standard_reference = retrieve_reference(test_case.prompt, test_case.domain)
        standard_reference_text = standard_reference["text"]
        
        standard_grounding = GroundingResult(
            factuality_score=standard_grounding_result.get("factuality_score", 0.0),
            hallucination_flag=standard_grounding_result.get("hallucination_flag", False),
            reference_text=standard_reference_text,
            debug_metadata=standard_grounding_result.get("debug_metadata", {}),
            domain=test_case.domain,
            threshold_used=DOMAIN_THRESHOLDS.get(test_case.domain, FACTUALITY_THRESHOLD)
        )
        
        # Apply grounding to accuracy-focused response
        accuracy_grounding_result = grounded_validation_pipeline(
            test_case.prompt,
            accuracy_response_text,
            test_case.domain,
            include_debug=True
        )
        
        accuracy_reference = retrieve_reference(test_case.prompt, test_case.domain)
        accuracy_reference_text = accuracy_reference["text"]
        
        accuracy_grounding = GroundingResult(
            factuality_score=accuracy_grounding_result.get("factuality_score", 0.0),
            hallucination_flag=accuracy_grounding_result.get("hallucination_flag", False),
            reference_text=accuracy_reference_text,
            debug_metadata=accuracy_grounding_result.get("debug_metadata", {}),
            domain=test_case.domain,
            threshold_used=DOMAIN_THRESHOLDS.get(test_case.domain, FACTUALITY_THRESHOLD)
        )
        
        # Calculate metrics
        metrics = self._calculate_test_case_metrics(
            test_case,
            standard_grounding,
            accuracy_grounding
        )
        
        return BenchmarkResult(
            test_case=test_case,
            standard_response=standard_response,
            accuracy_response=accuracy_response,
            standard_grounding=standard_grounding,
            accuracy_grounding=accuracy_grounding,
            metrics=metrics
        )
    
    def _calculate_test_case_metrics(
        self,
        test_case: GroundingTestCase,
        standard_grounding: GroundingResult,
        accuracy_grounding: GroundingResult
    ) -> Dict[str, Any]:
        """
        Calculate metrics for a single test case.
        
        Args:
            test_case: The test case
            standard_grounding: Grounding result for standard response
            accuracy_grounding: Grounding result for accuracy-focused response
            
        Returns:
            Dictionary with metrics
        """
        # Calculate factuality improvement
        factuality_improvement = accuracy_grounding.factuality_score - standard_grounding.factuality_score
        
        # Calculate hallucination detection accuracy
        expected_hallucination = test_case.expected_hallucinations
        standard_hallucination_accuracy = (expected_hallucination == standard_grounding.hallucination_flag)
        accuracy_hallucination_accuracy = (expected_hallucination == accuracy_grounding.hallucination_flag)
        
        # Check for contradictions
        standard_contradiction = standard_grounding.debug_metadata.get("factuality_details", {}).get("has_contradiction", False)
        accuracy_contradiction = accuracy_grounding.debug_metadata.get("factuality_details", {}).get("has_contradiction", False)
        
        # Calculate vector similarity stats
        standard_similarity = standard_grounding.debug_metadata.get("factuality_details", {}).get("vector_similarity", 0)
        accuracy_similarity = accuracy_grounding.debug_metadata.get("factuality_details", {}).get("vector_similarity", 0)
        similarity_improvement = accuracy_similarity - standard_similarity
        
        return {
            "factuality_improvement": factuality_improvement,
            "standard_hallucination_accuracy": standard_hallucination_accuracy,
            "accuracy_hallucination_accuracy": accuracy_hallucination_accuracy,
            "standard_contradiction_detected": standard_contradiction,
            "accuracy_contradiction_detected": accuracy_contradiction,
            "standard_vector_similarity": standard_similarity,
            "accuracy_vector_similarity": accuracy_similarity,
            "similarity_improvement": similarity_improvement,
            "expected_hallucinations": expected_hallucination,
            "standard_hallucination_detected": standard_grounding.hallucination_flag,
            "accuracy_hallucination_detected": accuracy_grounding.hallucination_flag,
            "standard_factuality_score": standard_grounding.factuality_score,
            "accuracy_factuality_score": accuracy_grounding.factuality_score
        }
    
    def _calculate_summary_metrics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """
        Calculate summary metrics across all results.
        
        Args:
            results: List of benchmark results
            
        Returns:
            Dictionary with summary metrics
        """
        if not results:
            return {}
        
        # Calculate average factuality scores
        avg_standard_factuality = sum(r.metrics["standard_factuality_score"] for r in results) / len(results)
        avg_accuracy_factuality = sum(r.metrics["accuracy_factuality_score"] for r in results) / len(results)
        avg_factuality_improvement = sum(r.metrics["factuality_improvement"] for r in results) / len(results)
        
        # Calculate hallucination detection metrics
        total_expected_hallucinations = sum(1 for r in results if r.metrics["expected_hallucinations"])
        
        standard_correct_detections = sum(1 for r in results 
                                         if r.metrics["expected_hallucinations"] == r.metrics["standard_hallucination_detected"])
        accuracy_correct_detections = sum(1 for r in results 
                                         if r.metrics["expected_hallucinations"] == r.metrics["accuracy_hallucination_detected"])
        
        standard_detection_accuracy = standard_correct_detections / len(results)
        accuracy_detection_accuracy = accuracy_correct_detections / len(results)
        
        # Calculate domain-specific metrics
        domain_metrics = {}
        for domain in self.config.domains:
            domain_results = [r for r in results if r.test_case.domain == domain]
            if domain_results:
                domain_avg_standard_factuality = sum(r.metrics["standard_factuality_score"] for r in domain_results) / len(domain_results)
                domain_avg_accuracy_factuality = sum(r.metrics["accuracy_factuality_score"] for r in domain_results) / len(domain_results)
                
                domain_metrics[domain] = {
                    "case_count": len(domain_results),
                    "avg_standard_factuality": domain_avg_standard_factuality,
                    "avg_accuracy_factuality": domain_avg_accuracy_factuality,
                    "avg_improvement": domain_avg_accuracy_factuality - domain_avg_standard_factuality
                }
        
        # Calculate difficulty-based metrics
        difficulty_metrics = {}
        for level in range(1, 6):  # 1-5 difficulty levels
            level_results = [r for r in results if r.test_case.difficulty == level]
            if level_results:
                level_avg_standard_factuality = sum(r.metrics["standard_factuality_score"] for r in level_results) / len(level_results)
                level_avg_accuracy_factuality = sum(r.metrics["accuracy_factuality_score"] for r in level_results) / len(level_results)
                
                difficulty_metrics[str(level)] = {
                    "case_count": len(level_results),
                    "avg_standard_factuality": level_avg_standard_factuality,
                    "avg_accuracy_factuality": level_avg_accuracy_factuality,
                    "avg_improvement": level_avg_accuracy_factuality - level_avg_standard_factuality
                }
        
        return {
            "overall": {
                "case_count": len(results),
                "avg_standard_factuality": avg_standard_factuality,
                "avg_accuracy_factuality": avg_accuracy_factuality,
                "avg_factuality_improvement": avg_factuality_improvement,
                "standard_detection_accuracy": standard_detection_accuracy,
                "accuracy_detection_accuracy": accuracy_detection_accuracy,
                "expected_hallucination_count": total_expected_hallucinations
            },
            "by_domain": domain_metrics,
            "by_difficulty": difficulty_metrics
        }
    
    def _export_results(self, results: Dict[str, Any]) -> None:
        """
        Export benchmark results to file.
        
        Args:
            results: Benchmark results
        """
        output_path = os.path.join(self.config.output_dir, self.config.output_file)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Exported results to {output_path}")


def main():
    """Run benchmark with default settings."""
    config = BenchmarkConfig(
        model="gpt-4",
        domains=["financial", "medical", "legal"],
        parallel=True,
        max_workers=4,
        output_dir="./benchmarks/results",
        output_file=f"grounding_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    benchmark = GroundingBenchmark(config)
    results = benchmark.run_benchmark()
    
    # Print summary
    summary = results["summary_metrics"]["overall"]
    print("\nGrounding Benchmark Summary:")
    print(f"Total test cases: {summary['case_count']}")
    print(f"Standard response factuality: {summary['avg_standard_factuality']:.4f}")
    print(f"Accuracy-focused factuality: {summary['avg_accuracy_factuality']:.4f}")
    print(f"Average improvement: {summary['avg_factuality_improvement']:.4f}")
    print(f"Standard hallucination detection accuracy: {summary['standard_detection_accuracy']:.2%}")
    print(f"Accuracy-focused hallucination detection accuracy: {summary['accuracy_detection_accuracy']:.2%}")
    
    print("\nDomain-Specific Results:")
    for domain, metrics in results["summary_metrics"]["by_domain"].items():
        print(f"- {domain.capitalize()}: {metrics['avg_improvement']:.4f} improvement")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())