#!/usr/bin/env python3
"""
Real-World Data Integration Test for Grounding Engine

This script tests the grounding engine against real-world test datasets
to evaluate its effectiveness at detecting hallucinations in different domains.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grounding_real_data_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to enable imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Import dependencies
# Import the grounding engine directly from the current directory
from grounding_engine import (
    score_factuality,
    check_hallucination,
    FACTUALITY_THRESHOLD,
    DOMAIN_THRESHOLDS
)

def load_openai_adapter():
    """
    Import and initialize the OpenAI adapter
    """
    try:
        # Try multiple potential import paths
        try:
            from v3.llm_adapters.openai_adapter import OpenAIAdapter
        except ImportError:
            try:
                # Look for it relative to the current file
                sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
                from llm_adapters.openai_adapter import OpenAIAdapter
            except ImportError:
                # If still not found, create a mock adapter for basic functionality
                logger.warning("OpenAI adapter not found. Creating mock adapter.")
                
                class MockOpenAIAdapter:
                    """Simple mock adapter for basic testing without API key"""
                    def __init__(self, *args, **kwargs):
                        self.api_key = None
                    
                    def call_with_accuracy_focus(self, prompt):
                        return f"Mock response to: {prompt}"
                    
                    def call_with_creativity_focus(self, prompt):
                        return f"Creative mock response to: {prompt}"
                        
                    def call_with_hallucination_risk(self, prompt):
                        return f"Risky mock response to: {prompt}"
                        
                    def get_metrics(self):
                        return {"request_count": 0, "total_tokens": 0}
                
                OpenAIAdapter = MockOpenAIAdapter
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("No OpenAI API key found in environment. Use --api-key argument.")
            if "MockOpenAIAdapter" not in str(OpenAIAdapter):
                return None
        
        return OpenAIAdapter(api_key=api_key)
    except Exception as e:
        logger.error(f"Error initializing OpenAI adapter: {e}")
        return None

def load_test_data(domain: str = None) -> List[Dict[str, Any]]:
    """
    Load test data from JSON files in the real_world_test directory
    
    Args:
        domain: Optional domain to filter test cases
        
    Returns:
        List of test cases
    """
    base_dir = Path(__file__).resolve().parent.parent
    test_data_dir = os.path.join(base_dir, "data", "real_world_test")
    
    all_test_cases = []
    
    if not os.path.exists(test_data_dir):
        logger.warning(f"Test data directory not found: {test_data_dir}")
        return []
    
    for file_name in os.listdir(test_data_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(test_data_dir, file_name)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Skip if domain filter is applied and doesn't match
                if domain and data.get('domain') != domain:
                    continue
                    
                test_cases = data.get('test_cases', [])
                for test_case in test_cases:
                    test_case['domain'] = data.get('domain', 'general')
                
                all_test_cases.extend(test_cases)
                logger.info(f"Loaded {len(test_cases)} test cases from {file_name}")
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from {file_path}")
            except Exception as e:
                logger.error(f"Error loading test data from {file_path}: {e}")
    
    return all_test_cases

def run_real_data_test(
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    output_dir: str = "./test_results",
    domains: Optional[List[str]] = None,
    verbose: bool = False
):
    """
    Run real-world data integration test
    
    Args:
        api_key: OpenAI API key
        model: OpenAI model to use
        output_dir: Directory to save results
        domains: List of domains to test (if None, test all)
        verbose: Whether to print verbose output
    """
    print("\n=======================================")
    print("ACDEK v3 Grounding Engine - Real-World Data Integration Test")
    print("=======================================\n")
    
    # Set API key if provided
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load OpenAI adapter
    openai_adapter = load_openai_adapter()
    if not openai_adapter and not api_key:
        print("Warning: No OpenAI API key available. Using static testing only.")
    
    # Determine available domains from test data
    available_domains = set()
    test_data = load_test_data()
    for test_case in test_data:
        available_domains.add(test_case.get('domain', 'general'))
    
    if not available_domains:
        logger.error("No test data available")
        print("Error: No test data available")
        return
    
    # Filter domains if specified
    if domains:
        test_domains = [d for d in domains if d in available_domains]
        if not test_domains:
            logger.error(f"None of the specified domains {domains} are available")
            print(f"Error: None of the specified domains {domains} are available")
            print(f"Available domains: {available_domains}")
            return
    else:
        test_domains = list(available_domains)
    
    print(f"Available domains: {', '.join(available_domains)}")
    print(f"Testing domains: {', '.join(test_domains)}")
    
    print("\nCurrent thresholds:")
    print(f"  Global: {FACTUALITY_THRESHOLD}")
    for domain, threshold in DOMAIN_THRESHOLDS.items():
        print(f"  {domain}: {threshold}")
    
    # Initialize results
    all_results = []
    
    # Process each domain
    for domain in test_domains:
        print(f"\n--- Testing domain: {domain} ---")
        
        # Filter test cases for this domain
        domain_test_cases = [tc for tc in test_data if tc.get('domain') == domain]
        
        print(f"Found {len(domain_test_cases)} test cases")
        
        domain_results = []
        
        # Process each test case
        for i, test_case in enumerate(domain_test_cases):
            test_id = test_case.get('id', f"unknown_{i}")
            prompt = test_case.get('prompt', '')
            reference_text = test_case.get('reference_text', '')
            expert_judgment = test_case.get('expert_judgment')
            
            print(f"\nTest case {i+1}/{len(domain_test_cases)}: {test_id}")
            if verbose:
                print(f"Prompt: {prompt}")
                print(f"Expert judgment: {expert_judgment}")
            
            # Generate LLM response if adapter is available
            if openai_adapter:
                try:
                    # Generate responses with different risk profiles
                    if expert_judgment == "factual":
                        # Use accuracy focus for cases expected to be factual
                        response = openai_adapter.call_with_accuracy_focus(prompt)
                        response_type = "accuracy_focused"
                    elif expert_judgment == "hallucination":
                        # Use hallucination risk for cases expected to be non-factual
                        response = openai_adapter.call_with_hallucination_risk(prompt)
                        response_type = "hallucination_prone"
                    else:
                        # Use balanced approach if no expert judgment provided
                        response = openai_adapter.call_with_creativity_focus(prompt)
                        response_type = "balanced"
                        
                    if verbose:
                        print(f"Response: {response[:100]}...")
                        
                except Exception as e:
                    logger.error(f"Error generating response for test case {test_id}: {e}")
                    print(f"  Error: {str(e)}")
                    response = prompt  # Fall back to using prompt as response
                    response_type = "fallback"
            else:
                # If no LLM adapter, use prompt as response (basic testing)
                response = prompt
                response_type = "static"
            
            # Evaluate factuality
            try:
                # Time the evaluation
                start_time = time.time()
                
                # Score factuality
                score_result = score_factuality(response, reference_text)
                if isinstance(score_result, dict):
                    factuality_score = score_result.get("score", 0.0)
                    debug_info = score_result.get("debug_info", {})
                else:
                    factuality_score = float(score_result)
                    debug_info = {}
                
                # Check if it's a hallucination
                is_hallucination = check_hallucination(factuality_score, domain)
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Get the threshold used
                threshold_used = DOMAIN_THRESHOLDS.get(domain.lower(), FACTUALITY_THRESHOLD)
                
                # Determine if detection matches expert judgment
                detection_correct = None
                if expert_judgment:
                    if expert_judgment == "factual":
                        detection_correct = not is_hallucination
                    elif expert_judgment == "hallucination":
                        detection_correct = is_hallucination
                
                # Create result record
                result = {
                    "test_id": test_id,
                    "domain": domain,
                    "prompt": prompt,
                    "response_type": response_type,
                    "response_snippet": response[:200] + ("..." if len(response) > 200 else ""),
                    "reference_text": reference_text,
                    "expert_judgment": expert_judgment,
                    "factuality_score": factuality_score,
                    "is_hallucination": is_hallucination,
                    "threshold_used": threshold_used,
                    "detection_correct": detection_correct,
                    "processing_time_ms": processing_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                domain_results.append(result)
                all_results.append(result)
                
                # Print result
                print(f"  Factuality Score: {factuality_score:.4f}")
                print(f"  Hallucination: {is_hallucination}")
                print(f"  Threshold Used: {threshold_used}")
                
                if detection_correct is not None:
                    detection_status = "✓ Correct" if detection_correct else "✗ Incorrect"
                    print(f"  Detection: {detection_status}")
                
            except Exception as e:
                logger.error(f"Error evaluating test case {test_id}: {e}")
                print(f"  Error: {str(e)}")
        
        # Calculate domain metrics
        if domain_results:
            total = len(domain_results)
            hallucination_count = sum(1 for r in domain_results if r["is_hallucination"])
            hallucination_rate = hallucination_count / total if total > 0 else 0
            
            judged_results = [r for r in domain_results if r.get("detection_correct") is not None]
            correct_count = sum(1 for r in judged_results if r.get("detection_correct"))
            accuracy = correct_count / len(judged_results) if judged_results else 0
            
            avg_score = sum(r["factuality_score"] for r in domain_results) / total if total > 0 else 0
            
            print(f"\nDomain {domain} Summary:")
            print(f"  Tests: {total}")
            print(f"  Hallucination Rate: {hallucination_rate:.2%}")
            if judged_results:
                print(f"  Detection Accuracy: {accuracy:.2%} ({correct_count}/{len(judged_results)})")
            print(f"  Average Factuality Score: {avg_score:.4f}")
    
    # Calculate overall metrics
    if all_results:
        total = len(all_results)
        hallucination_count = sum(1 for r in all_results if r["is_hallucination"])
        hallucination_rate = hallucination_count / total if total > 0 else 0
        
        judged_results = [r for r in all_results if r.get("detection_correct") is not None]
        correct_count = sum(1 for r in judged_results if r.get("detection_correct"))
        accuracy = correct_count / len(judged_results) if judged_results else 0
        
        avg_score = sum(r["factuality_score"] for r in all_results) / total if total > 0 else 0
        
        print("\n=======================================")
        print("Overall Results")
        print("=======================================")
        print(f"Total Tests: {total}")
        print(f"Hallucination Rate: {hallucination_rate:.2%}")
        if judged_results:
            print(f"Detection Accuracy: {accuracy:.2%} ({correct_count}/{len(judged_results)})")
        print(f"Average Factuality Score: {avg_score:.4f}")
    
    # Calculate confusion matrix metrics
    true_positives = 0  # Correctly identified hallucinations
    false_positives = 0  # Factual incorrectly flagged as hallucination
    true_negatives = 0  # Correctly identified factual
    false_negatives = 0  # Hallucinations incorrectly identified as factual
    
    for result in all_results:
        if result.get("expert_judgment") == "hallucination" and result["is_hallucination"]:
            true_positives += 1
        elif result.get("expert_judgment") == "factual" and result["is_hallucination"]:
            false_positives += 1
        elif result.get("expert_judgment") == "factual" and not result["is_hallucination"]:
            true_negatives += 1
        elif result.get("expert_judgment") == "hallucination" and not result["is_hallucination"]:
            false_negatives += 1
    
    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nConfusion Matrix Metrics:")
    print(f"  True Positives (correctly identified hallucinations): {true_positives}")
    print(f"  False Positives (factual incorrectly flagged): {false_positives}")
    print(f"  True Negatives (correctly identified factual): {true_negatives}")
    print(f"  False Negatives (missed hallucinations): {false_negatives}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1_score:.4f}")
    
    # Save results
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"real_data_test_results_{timestamp}.json")
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "thresholds": {
                "global": FACTUALITY_THRESHOLD,
                "domains": DOMAIN_THRESHOLDS
            },
            "metrics": {
                "total_tests": total,
                "hallucination_rate": hallucination_rate,
                "accuracy": accuracy if judged_results else None,
                "avg_score": avg_score,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score
            },
            "results": all_results
        }
        
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved results to {results_file}")
        print(f"\nResults saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(description="Test grounding engine with real-world datasets")
    parser.add_argument("--api-key", help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use (default: gpt-4)")
    parser.add_argument("--output-dir", default="./test_results", help="Output directory")
    parser.add_argument("--domains", help="Comma-separated list of domains to test (default: all available)")
    parser.add_argument("--create-examples", action="store_true", help="Create example datasets if none exist")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Parse domains if specified
    domains = args.domains.split(",") if args.domains else None
    
    run_real_data_test(
        api_key=args.api_key,
        model=args.model,
        output_dir=args.output_dir,
        domains=domains,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()