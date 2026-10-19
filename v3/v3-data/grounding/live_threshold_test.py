#!/usr/bin/env python3
"""
Live LLM Testing for Grounding Engine Thresholds

This script provides a comprehensive testing framework for evaluating
the adjusted factuality thresholds against real LLM responses, using
actual domain-specific reference data sets to ensure proper evaluation.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grounding_live_threshold_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to enable imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import dependencies
from grounding_engine import (
    grounded_validation_pipeline,
    retrieve_reference,
    score_factuality,
    build_grounding_context,
    get_available_domains,
    clear_reference_cache,
    FACTUALITY_THRESHOLD,
    DOMAIN_THRESHOLDS
)

# Import OpenAI adapter
from llm_adapters.openai_adapter import OpenAIAdapter

# Test cases across domains with real-world datasets
REAL_WORLD_TEST_CASES = [
    # Financial domain test cases
    {
        "id": "fin-asset-liabilities-1",
        "domain": "financial",
        "prompt": "What is the difference between assets and liabilities?",
        "dataset": "finance.json"
    },
    {
        "id": "fin-balance-sheet-1",
        "domain": "financial",
        "prompt": "Explain what a balance sheet shows.",
        "dataset": "finance.json"
    },
    {
        "id": "fin-income-statement-1",
        "domain": "financial",
        "prompt": "What is an income statement and what does it show?",
        "dataset": "finance.json"
    },
    {
        "id": "fin-equity-1",
        "domain": "financial",
        "prompt": "What is shareholders' equity and how is it calculated?",
        "dataset": "finance.json"
    },
    
    # Legal domain test cases
    {
        "id": "leg-contract-1",
        "domain": "legal",
        "prompt": "What are the essential elements of a valid contract?",
        "dataset": "legal.json"
    },
    {
        "id": "leg-tort-1",
        "domain": "legal",
        "prompt": "What is tort law and what are some common types of torts?",
        "dataset": "legal.json"
    },
    {
        "id": "leg-copyright-1",
        "domain": "legal",
        "prompt": "What is copyright protection and what does it cover?",
        "dataset": "legal.json"
    },
    {
        "id": "leg-evidence-1",
        "domain": "legal",
        "prompt": "What are the rules of evidence in court proceedings?",
        "dataset": "legal.json"
    },
    
    # Medical domain test cases
    {
        "id": "med-diabetes-1",
        "domain": "medical",
        "prompt": "What is diabetes mellitus, its types, and how is it diagnosed?",
        "dataset": "medical.json"
    },
    {
        "id": "med-hypertension-1",
        "domain": "medical",
        "prompt": "What is hypertension and how is it treated?",
        "dataset": "medical.json"
    },
    {
        "id": "med-vaccine-1",
        "domain": "medical",
        "prompt": "How do vaccines work to protect against diseases?",
        "dataset": "medical.json"
    },
    {
        "id": "med-cancer-1",
        "domain": "medical",
        "prompt": "What is cancer and what are the main treatment approaches?",
        "dataset": "medical.json"
    }
]

@dataclass
class TestResult:
    """Container for test results."""
    test_id: str
    domain: str
    prompt: str
    response_type: str  # accurate, creative, hallucination_prone
    response_text: str
    factuality_score: float
    hallucination_flag: bool
    threshold_used: float
    reference_text: str
    processing_time_ms: float
    additional_metadata: Dict[str, Any] = field(default_factory=dict)
    expert_factual: Optional[bool] = None  # Expert judgment of factuality
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_id": self.test_id,
            "domain": self.domain,
            "prompt": self.prompt,
            "response_type": self.response_type,
            "response_text": self.response_text,
            "factuality_score": self.factuality_score,
            "hallucination_flag": self.hallucination_flag,
            "threshold_used": self.threshold_used,
            "reference_text": self.reference_text,
            "processing_time_ms": self.processing_time_ms,
            "additional_metadata": self.additional_metadata,
            "expert_factual": self.expert_factual
        }

def run_live_test_case(
    adapter: OpenAIAdapter,
    test_case: Dict[str, Any],
    include_debug: bool = True
) -> List[TestResult]:
    """
    Run a single test case with live LLM calls.
    
    Args:
        adapter: OpenAI adapter instance
        test_case: Test case configuration
        include_debug: Whether to include debug metadata
        
    Returns:
        List of test results for different response types
    """
    domain = test_case["domain"]
    prompt = test_case["prompt"]
    test_id = test_case["id"]
    results = []
    
    logger.info(f"Running live test for '{test_id}' in domain '{domain}'")
    
    try:
        # Get reference details
        reference = retrieve_reference(prompt, domain)
        reference_text = reference["text"]
        
        # Test responses with different modes
        response_modes = [
            ("accuracy_focused", adapter.call_with_accuracy_focus),
            ("creativity_focused", adapter.call_with_creativity_focus),
            ("hallucination_prone", adapter.call_with_hallucination_risk)
        ]
        
        for mode_name, mode_func in response_modes:
            # Generate response
            logger.info(f"Generating {mode_name} response for test '{test_id}'")
            response = mode_func(prompt)
            
            # Validate response
            start_time = time.time()
            validation_result = grounded_validation_pipeline(
                prompt, response, domain, include_debug=include_debug
            )
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Get the threshold used
            threshold_used = DOMAIN_THRESHOLDS.get(domain, FACTUALITY_THRESHOLD)
            
            # Create test result
            result = TestResult(
                test_id=test_id,
                domain=domain,
                prompt=prompt,
                response_type=mode_name,
                response_text=response,
                factuality_score=validation_result.get("factuality_score", 0),
                hallucination_flag=validation_result.get("hallucination_flag", True),
                threshold_used=threshold_used,
                reference_text=reference_text,
                processing_time_ms=processing_time,
                additional_metadata={
                    "constraint_pass": validation_result.get("constraint_pass", False),
                    "debug": validation_result.get("debug_metadata", {}) if include_debug else {}
                }
            )
            
            results.append(result)
            
            # Log result
            logger.info(f"Result for {mode_name}: Score={result.factuality_score:.4f}, "
                       f"Hallucination={result.hallucination_flag}, "
                       f"Threshold={threshold_used}")
            
        # Clear cache between test cases
        clear_reference_cache()
        
        return results
        
    except Exception as e:
        logger.error(f"Error in test case {test_id}: {str(e)}")
        
        # Create error result
        error_result = TestResult(
            test_id=test_id,
            domain=domain,
            prompt=prompt,
            response_type="error",
            response_text=f"Error: {str(e)}",
            factuality_score=0.0,
            hallucination_flag=True,
            threshold_used=DOMAIN_THRESHOLDS.get(domain, FACTUALITY_THRESHOLD),
            reference_text="Error retrieving reference",
            processing_time_ms=0.0,
            additional_metadata={"error": str(e)}
        )
        
        return [error_result]

def calculate_metrics(results: List[TestResult]) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics from test results.
    
    Args:
        results: List of test results
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        "global": {
            "total_tests": len(results),
            "error_count": len([r for r in results if r.response_type == "error"]),
            "avg_factuality_score": None,
            "hallucination_rate": None,
            "avg_processing_time_ms": None
        },
        "by_domain": {},
        "by_response_type": {},
        "thresholds": {
            "global": FACTUALITY_THRESHOLD,
            "domains": {k: v for k, v in DOMAIN_THRESHOLDS.items()}
        }
    }
    
    # Skip if no results
    if not results:
        return metrics
    
    # Calculate global metrics
    valid_results = [r for r in results if r.response_type != "error"]
    if valid_results:
        metrics["global"]["avg_factuality_score"] = sum(r.factuality_score for r in valid_results) / len(valid_results)
        metrics["global"]["hallucination_rate"] = sum(1 for r in valid_results if r.hallucination_flag) / len(valid_results)
        metrics["global"]["avg_processing_time_ms"] = sum(r.processing_time_ms for r in valid_results) / len(valid_results)
    
    # Calculate domain-specific metrics
    domains = set(r.domain for r in results)
    for domain in domains:
        domain_results = [r for r in results if r.domain == domain and r.response_type != "error"]
        
        if not domain_results:
            continue
            
        metrics["by_domain"][domain] = {
            "count": len(domain_results),
            "avg_factuality_score": sum(r.factuality_score for r in domain_results) / len(domain_results),
            "hallucination_rate": sum(1 for r in domain_results if r.hallucination_flag) / len(domain_results),
            "threshold": DOMAIN_THRESHOLDS.get(domain, FACTUALITY_THRESHOLD),
            "by_response_type": {}
        }
        
        # Domain metrics by response type
        response_types = set(r.response_type for r in domain_results)
        for resp_type in response_types:
            type_results = [r for r in domain_results if r.response_type == resp_type]
            
            if not type_results:
                continue
                
            metrics["by_domain"][domain]["by_response_type"][resp_type] = {
                "count": len(type_results),
                "avg_factuality_score": sum(r.factuality_score for r in type_results) / len(type_results),
                "hallucination_rate": sum(1 for r in type_results if r.hallucination_flag) / len(type_results)
            }
    
    # Calculate response type metrics
    response_types = set(r.response_type for r in results if r.response_type != "error")
    for resp_type in response_types:
        type_results = [r for r in results if r.response_type == resp_type]
        
        if not type_results:
            continue
            
        metrics["by_response_type"][resp_type] = {
            "count": len(type_results),
            "avg_factuality_score": sum(r.factuality_score for r in type_results) / len(type_results),
            "hallucination_rate": sum(1 for r in type_results if r.hallucination_flag) / len(type_results),
            "by_domain": {}
        }
        
        # Response type metrics by domain
        for domain in domains:
            domain_type_results = [r for r in type_results if r.domain == domain]
            
            if not domain_type_results:
                continue
                
            metrics["by_response_type"][resp_type]["by_domain"][domain] = {
                "count": len(domain_type_results),
                "avg_factuality_score": sum(r.factuality_score for r in domain_type_results) / len(domain_type_results),
                "hallucination_rate": sum(1 for r in domain_type_results if r.hallucination_flag) / len(domain_type_results)
            }
    
    return metrics

def save_results(results: List[TestResult], metrics: Dict[str, Any], output_dir: str) -> str:
    """
    Save test results and metrics to files.
    
    Args:
        results: List of test results
        metrics: Metrics dictionary
        output_dir: Output directory
        
    Returns:
        Path to the output directory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamped filenames
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"threshold_test_results_{timestamp}.json")
    metrics_file = os.path.join(output_dir, f"threshold_test_metrics_{timestamp}.json")
    
    # Save results
    with open(results_file, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    
    # Save metrics
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Metrics saved to {metrics_file}")
    
    return output_dir

def run_threshold_evaluation(
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    output_dir: str = "./threshold_test_output",
    test_cases: Optional[List[Dict[str, Any]]] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run comprehensive threshold evaluation with live LLM calls.
    
    Args:
        api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
        model: Model to use (defaults to gpt-4)
        output_dir: Output directory
        test_cases: Test cases to run (defaults to REAL_WORLD_TEST_CASES)
        verbose: Whether to print detailed results
        
    Returns:
        Metrics dictionary
    """
    print("\n=======================================")
    print("ACDEK v3 Grounding Engine - Live Threshold Evaluation")
    print("=======================================\n")
    
    # Use default test cases if none provided
    if test_cases is None:
        test_cases = REAL_WORLD_TEST_CASES
    
    # Initialize OpenAI adapter
    adapter = OpenAIAdapter(api_key=api_key, model=model)
    
    # Check if API key is available
    if not adapter.api_key:
        logger.error("No OpenAI API key available. Set OPENAI_API_KEY environment variable or pass --api-key.")
        return {"error": "No OpenAI API key available"}
    
    print(f"Testing with OpenAI model: {model}")
    print(f"Number of test cases: {len(test_cases)}")
    print(f"Domains: {', '.join(set(case['domain'] for case in test_cases))}")
    print("\nCurrent thresholds:")
    print(f"  Global: {FACTUALITY_THRESHOLD}")
    for domain, threshold in DOMAIN_THRESHOLDS.items():
        print(f"  {domain}: {threshold}")
    
    print("\nRunning tests...")
    
    all_results = []
    
    # Run tests for each case
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}/{len(test_cases)}: {test_case['id']} - {test_case['domain']}")
        
        if verbose:
            print(f"Prompt: {test_case['prompt']}")
        
        # Run the test
        case_results = run_live_test_case(adapter, test_case)
        all_results.extend(case_results)
        
        if verbose:
            for result in case_results:
                print(f"\n  {result.response_type.capitalize()} Response:")
                print(f"  Factuality Score: {result.factuality_score:.4f}")
                print(f"  Hallucination Flag: {result.hallucination_flag}")
                print(f"  Threshold Used: {result.threshold_used}")
                if result.hallucination_flag:
                    print(f"  Detected as hallucination (score < threshold)")
                else:
                    print(f"  Detected as factual (score >= threshold)")
    
    # Calculate metrics
    metrics = calculate_metrics(all_results)
    
    # Print summary
    print("\n=======================================")
    print("Threshold Evaluation Results")
    print("=======================================")
    print(f"Total tests: {metrics['global']['total_tests']}")
    print(f"Errors: {metrics['global']['error_count']}")
    
    if metrics['global']['avg_factuality_score'] is not None:
        print(f"Global avg factuality score: {metrics['global']['avg_factuality_score']:.4f}")
        print(f"Global hallucination rate: {metrics['global']['hallucination_rate']:.2%}")
    
    print("\nResults by Domain:")
    for domain, stats in metrics.get("by_domain", {}).items():
        print(f"  {domain.capitalize()}:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg factuality score: {stats['avg_factuality_score']:.4f}")
        print(f"    Hallucination rate: {stats['hallucination_rate']:.2%}")
        print(f"    Threshold: {stats['threshold']}")
    
    print("\nResults by Response Type:")
    for resp_type, stats in metrics.get("by_response_type", {}).items():
        print(f"  {resp_type.replace('_', ' ').capitalize()}:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg factuality score: {stats['avg_factuality_score']:.4f}")
        print(f"    Hallucination rate: {stats['hallucination_rate']:.2%}")
    
    # Save results
    save_results(all_results, metrics, output_dir)
    
    # Print adapter metrics
    adapter_metrics = adapter.get_metrics()
    print("\nOpenAI API Usage:")
    print(f"  Total Requests: {adapter_metrics.get('request_count', 0)}")
    print(f"  Total Tokens: {adapter_metrics.get('total_tokens', 0)}")
    print(f"  Avg Tokens Per Request: {adapter_metrics.get('avg_tokens_per_request', 0):.1f}")
    print(f"  Avg Latency: {adapter_metrics.get('avg_latency_seconds', 0):.2f}s")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate grounding engine thresholds with live LLM tests")
    parser.add_argument("--api-key", help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use (default: gpt-4)")
    parser.add_argument("--output-dir", default="./threshold_test_output", help="Output directory for results")
    parser.add_argument("--domain", help="Only test a specific domain (financial, legal, or medical)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Filter test cases by domain if specified
    if args.domain:
        test_cases = [case for case in REAL_WORLD_TEST_CASES if case["domain"] == args.domain]
        if not test_cases:
            print(f"No test cases found for domain: {args.domain}")
            print(f"Available domains: {', '.join(set(case['domain'] for case in REAL_WORLD_TEST_CASES))}")
            sys.exit(1)
    else:
        test_cases = REAL_WORLD_TEST_CASES
    
    run_threshold_evaluation(
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        test_cases=test_cases,
        verbose=args.verbose
    )