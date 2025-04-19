#!/usr/bin/env python3
"""
LLM Integration Test for Grounding Engine

This script tests the grounding engine against real responses from
OpenAI models, to evaluate its effectiveness at detecting hallucinations
and ensuring factuality in LLM-generated content.
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grounding_llm_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to enable imports
parent_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(parent_dir))

# Import dependencies
try:
    # Import grounding engine functions
    from v3.v3.grounding import (
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
    from v3.llm_adapters.openai_adapter import OpenAIAdapter
    
    # Test cases across the three domains (financial, legal, medical)
    TEST_CASES = [
        # Financial domain test cases
        {
            "id": "fin-asset-1",
            "domain": "financial",
            "prompt": "What is an asset in financial terms?",
            "reference_id": "fin1"  # Reference to the corresponding entry in finance.json
        },
        {
            "id": "fin-equity-1",
            "domain": "financial",
            "prompt": "How is equity calculated in a company?",
            "reference_id": "fin3"
        },
        {
            "id": "fin-income-1",
            "domain": "financial",
            "prompt": "What does an income statement show?",
            "reference_id": "fin12"
        },
        
        # Legal domain test cases
        {
            "id": "leg-contract-1",
            "domain": "legal",
            "prompt": "What are the requirements for a valid contract?",
            "reference_id": "leg1"
        },
        {
            "id": "leg-tort-1",
            "domain": "legal",
            "prompt": "What is tort law?",
            "reference_id": "leg3"
        },
        {
            "id": "leg-ip-1",
            "domain": "legal",
            "prompt": "What is intellectual property?",
            "reference_id": "leg6"
        },
        
        # Medical domain test cases
        {
            "id": "med-hypertension-1",
            "domain": "medical",
            "prompt": "What is hypertension and how is it diagnosed?",
            "reference_id": "med2"
        },
        {
            "id": "med-diabetes-1",
            "domain": "medical",
            "prompt": "What is diabetes mellitus?",
            "reference_id": "med3"
        },
        {
            "id": "med-cancer-1",
            "domain": "medical",
            "prompt": "What is cancer and how is it treated?",
            "reference_id": "med9"
        }
    ]
    
    @dataclass
    class LLMTestResult:
        """Container for LLM test results."""
        case_id: str
        domain: str
        prompt: str
        response_type: str  # accurate, hallucination_prone, etc.
        response_text: str
        factuality_score: float
        hallucination_flag: bool
        threshold_used: float
        reference_text: str
        processing_time_ms: float
        additional_metadata: Dict[str, Any] = field(default_factory=dict)
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary for serialization."""
            return {
                "case_id": self.case_id,
                "domain": self.domain,
                "prompt": self.prompt,
                "response_type": self.response_type,
                "response_text": self.response_text,
                "factuality_score": self.factuality_score,
                "hallucination_flag": self.hallucination_flag,
                "threshold_used": self.threshold_used,
                "reference_text": self.reference_text,
                "processing_time_ms": self.processing_time_ms,
                "additional_metadata": self.additional_metadata
            }
    
    def run_llm_test(
        adapter: OpenAIAdapter,
        test_case: Dict[str, Any],
        include_debug: bool = True
    ) -> List[LLMTestResult]:
        """
        Run comprehensive test with OpenAI adapter for a single test case.
        
        Args:
            adapter: Initialized OpenAI adapter
            test_case: Test case definition
            include_debug: Whether to include debug metadata
            
        Returns:
            List of test results for different response types
        """
        domain = test_case["domain"]
        prompt = test_case["prompt"]
        case_id = test_case["id"]
        results = []
        
        # Get reference details
        reference = retrieve_reference(prompt, domain)
        reference_text = reference["text"]
        
        # Clear cache between tests
        clear_reference_cache()
        
        try:
            # 1. Test with accuracy-focused response
            logger.info(f"Testing {case_id} with accuracy focus")
            
            # Generate response
            accurate_response = adapter.call_with_accuracy_focus(prompt)
            
            # Validate response
            start_time = time.time()
            accurate_result = grounded_validation_pipeline(
                prompt, accurate_response, domain, include_debug=include_debug
            )
            accurate_time = (time.time() - start_time) * 1000  # ms
            
            # Create test result
            results.append(LLMTestResult(
                case_id=case_id,
                domain=domain,
                prompt=prompt,
                response_type="accuracy_focused",
                response_text=accurate_response,
                factuality_score=accurate_result.get("factuality_score", 0),
                hallucination_flag=accurate_result.get("hallucination_flag", True),
                threshold_used=DOMAIN_THRESHOLDS.get(domain, FACTUALITY_THRESHOLD),
                reference_text=reference_text,
                processing_time_ms=accurate_time,
                additional_metadata={
                    "constraint_pass": accurate_result.get("constraint_pass", False),
                    "debug": accurate_result.get("debug_metadata", {}) if include_debug else {}
                }
            ))
            
            # 2. Test with creativity-focused response
            logger.info(f"Testing {case_id} with creativity focus")
            
            # Generate response
            creative_response = adapter.call_with_creativity_focus(prompt)
            
            # Validate response
            start_time = time.time()
            creative_result = grounded_validation_pipeline(
                prompt, creative_response, domain, include_debug=include_debug
            )
            creative_time = (time.time() - start_time) * 1000  # ms
            
            # Create test result
            results.append(LLMTestResult(
                case_id=case_id,
                domain=domain,
                prompt=prompt,
                response_type="creativity_focused",
                response_text=creative_response,
                factuality_score=creative_result.get("factuality_score", 0),
                hallucination_flag=creative_result.get("hallucination_flag", True),
                threshold_used=DOMAIN_THRESHOLDS.get(domain, FACTUALITY_THRESHOLD),
                reference_text=reference_text,
                processing_time_ms=creative_time,
                additional_metadata={
                    "constraint_pass": creative_result.get("constraint_pass", False),
                    "debug": creative_result.get("debug_metadata", {}) if include_debug else {}
                }
            ))
            
            # 3. Test with hallucination-prone response
            logger.info(f"Testing {case_id} with hallucination risk")
            
            # Generate response
            hallucination_response = adapter.call_with_hallucination_risk(prompt)
            
            # Validate response
            start_time = time.time()
            hallucination_result = grounded_validation_pipeline(
                prompt, hallucination_response, domain, include_debug=include_debug
            )
            hallucination_time = (time.time() - start_time) * 1000  # ms
            
            # Create test result
            results.append(LLMTestResult(
                case_id=case_id,
                domain=domain,
                prompt=prompt,
                response_type="hallucination_prone",
                response_text=hallucination_response,
                factuality_score=hallucination_result.get("factuality_score", 0),
                hallucination_flag=hallucination_result.get("hallucination_flag", True),
                threshold_used=DOMAIN_THRESHOLDS.get(domain, FACTUALITY_THRESHOLD),
                reference_text=reference_text,
                processing_time_ms=hallucination_time,
                additional_metadata={
                    "constraint_pass": hallucination_result.get("constraint_pass", False),
                    "debug": hallucination_result.get("debug_metadata", {}) if include_debug else {}
                }
            ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in test case {case_id}: {str(e)}")
            
            # Create error result
            results.append(LLMTestResult(
                case_id=case_id,
                domain=domain,
                prompt=prompt,
                response_type="error",
                response_text=f"Error: {str(e)}",
                factuality_score=0.0,
                hallucination_flag=True,
                threshold_used=DOMAIN_THRESHOLDS.get(domain, FACTUALITY_THRESHOLD),
                reference_text=reference_text,
                processing_time_ms=0.0,
                additional_metadata={"error": str(e)}
            ))
            
            return results
    
    def calculate_metrics(results: List[LLMTestResult]) -> Dict[str, Any]:
        """
        Calculate summary metrics across all test cases.
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary with metrics
        """
        if not results:
            return {}
        
        # Group by response type
        by_type = {
            "accuracy_focused": [],
            "creativity_focused": [],
            "hallucination_prone": []
        }
        
        for result in results:
            if result.response_type in by_type:
                by_type[result.response_type].append(result)
        
        # Overall metrics
        metrics = {
            "total_tests": len(results),
            "errors": len([r for r in results if r.response_type == "error"]),
            "response_types": {},
            "domains": {}
        }
        
        # Process by response type
        for resp_type, resp_results in by_type.items():
            if not resp_results:
                continue
                
            hallucination_rate = sum(1 for r in resp_results if r.hallucination_flag) / len(resp_results)
            avg_factuality = sum(r.factuality_score for r in resp_results) / len(resp_results)
            avg_processing_time = sum(r.processing_time_ms for r in resp_results) / len(resp_results)
            
            metrics["response_types"][resp_type] = {
                "count": len(resp_results),
                "hallucination_rate": hallucination_rate,
                "avg_factuality_score": avg_factuality,
                "avg_processing_time_ms": avg_processing_time
            }
        
        # Process by domain
        domains = set(r.domain for r in results)
        for domain in domains:
            domain_results = [r for r in results if r.domain == domain]
            
            if not domain_results:
                continue
                
            hallucination_rate = sum(1 for r in domain_results if r.hallucination_flag) / len(domain_results)
            avg_factuality = sum(r.factuality_score for r in domain_results) / len(domain_results)
            avg_processing_time = sum(r.processing_time_ms for r in domain_results) / len(domain_results)
            
            # Domain metrics by response type
            domain_by_type = {}
            for resp_type in by_type.keys():
                type_results = [r for r in domain_results if r.response_type == resp_type]
                
                if not type_results:
                    continue
                    
                domain_by_type[resp_type] = {
                    "count": len(type_results),
                    "hallucination_rate": sum(1 for r in type_results if r.hallucination_flag) / len(type_results),
                    "avg_factuality_score": sum(r.factuality_score for r in type_results) / len(type_results)
                }
            
            metrics["domains"][domain] = {
                "count": len(domain_results),
                "hallucination_rate": hallucination_rate,
                "avg_factuality_score": avg_factuality,
                "avg_processing_time_ms": avg_processing_time,
                "by_response_type": domain_by_type
            }
        
        return metrics
    
    def save_results(results: List[LLMTestResult], metrics: Dict[str, Any], output_dir: str) -> str:
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
        results_file = os.path.join(output_dir, f"llm_test_results_{timestamp}.json")
        metrics_file = os.path.join(output_dir, f"llm_test_metrics_{timestamp}.json")
        
        # Save results
        with open(results_file, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        # Save metrics
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
        logger.info(f"Metrics saved to {metrics_file}")
        
        return output_dir
    
    def run_comprehensive_test(
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        output_dir: str = "./test_output",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run comprehensive LLM integration tests.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Model to use (defaults to gpt-4)
            output_dir: Output directory
            verbose: Whether to print detailed results
            
        Returns:
            Metrics dictionary
        """
        print("\n=======================================")
        print("ACDEK v3 Grounding Engine - LLM Integration Test")
        print("=======================================\n")
        
        # Initialize OpenAI adapter
        adapter = OpenAIAdapter(api_key=api_key, model=model)
        
        # Check if API key is available
        if not adapter.api_key:
            logger.error("No OpenAI API key available. Set OPENAI_API_KEY environment variable or pass --api-key.")
            return {"error": "No OpenAI API key available"}
        
        print(f"Testing with OpenAI model: {model}")
        print(f"Number of test cases: {len(TEST_CASES)}")
        print(f"Domains: {', '.join(set(case['domain'] for case in TEST_CASES))}")
        print("\nRunning tests...")
        
        all_results = []
        
        # Run tests for each case
        for i, test_case in enumerate(TEST_CASES):
            print(f"\nTest case {i+1}/{len(TEST_CASES)}: {test_case['id']} - {test_case['domain']}")
            
            if verbose:
                print(f"Prompt: {test_case['prompt']}")
            
            # Run the test
            results = run_llm_test(adapter, test_case)
            all_results.extend(results)
            
            if verbose:
                for result in results:
                    print(f"\n  {result.response_type.capitalize()} Response:")
                    print(f"  Factuality Score: {result.factuality_score:.4f}")
                    print(f"  Hallucination Detected: {result.hallucination_flag}")
                    print(f"  Processing Time: {result.processing_time_ms:.2f}ms")
        
        # Calculate metrics
        metrics = calculate_metrics(all_results)
        
        # Print summary
        print("\n=======================================")
        print("Test Results Summary")
        print("=======================================")
        print(f"Total tests: {metrics['total_tests']}")
        
        print("\nBy Response Type:")
        for resp_type, stats in metrics["response_types"].items():
            print(f"  {resp_type.replace('_', ' ').capitalize()}:")
            print(f"    Tests: {stats['count']}")
            print(f"    Hallucination Rate: {stats['hallucination_rate']:.2%}")
            print(f"    Avg Factuality Score: {stats['avg_factuality_score']:.4f}")
        
        print("\nBy Domain:")
        for domain, stats in metrics["domains"].items():
            print(f"  {domain.capitalize()}:")
            print(f"    Tests: {stats['count']}")
            print(f"    Hallucination Rate: {stats['hallucination_rate']:.2%}")
            print(f"    Avg Factuality Score: {stats['avg_factuality_score']:.4f}")
        
        # Save results
        save_results(all_results, metrics, output_dir)
        
        # Print adapter metrics
        adapter_metrics = adapter.get_metrics()
        print("\nOpenAI API Usage:")
        print(f"  Total Requests: {adapter_metrics['request_count']}")
        print(f"  Total Tokens: {adapter_metrics['total_tokens']}")
        print(f"  Avg Tokens Per Request: {adapter_metrics['avg_tokens_per_request']:.1f}")
        print(f"  Avg Latency: {adapter_metrics['avg_latency_seconds']:.2f}s")
        
        return metrics

except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    print(f"Error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test grounding engine with real LLM responses")
    parser.add_argument("--api-key", help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use (default: gpt-4)")
    parser.add_argument("--output-dir", default="./test_output", help="Output directory for results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    run_comprehensive_test(
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        verbose=args.verbose
    )