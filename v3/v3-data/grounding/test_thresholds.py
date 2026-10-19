#!/usr/bin/env python3
"""
Threshold Evaluation Script for Grounding Engine.

This script tests the impact of the adjusted factuality thresholds
on hallucination detection rates across different domains.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path to enable imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

try:
    # Import grounding functions directly
    from grounding_engine import (
        grounded_validation_pipeline,
        FACTUALITY_THRESHOLD,
        DOMAIN_THRESHOLDS
    )

    # Test cases with known factuality (ground truth)
    TEST_CASES = [
        # Financial domain test cases
        {
            "domain": "financial",
            "prompt": "What is the difference between assets and liabilities?",
            "response": "Assets are resources owned by a company that have future economic value, while liabilities are obligations that require future payment. Assets include cash, inventory, and property, while liabilities include loans, accounts payable, and bonds.",
            "expected_factual": True,  # This is factually correct
            "description": "Correct financial definition"
        },
        {
            "domain": "financial",
            "prompt": "What is the difference between assets and liabilities?",
            "response": "Assets and liabilities are the same thing in accounting, just viewed from different perspectives. An asset is just a liability seen from the other side of the transaction.",
            "expected_factual": False,  # This is factually incorrect
            "description": "Incorrect financial definition"
        },
        {
            "domain": "financial",
            "prompt": "Explain what a balance sheet shows.",
            "response": "A balance sheet shows a company's assets, liabilities, and equity at a specific point in time. The fundamental accounting equation is Assets = Liabilities + Equity, which must always balance.",
            "expected_factual": True,  # This is factually correct
            "description": "Correct balance sheet explanation"
        },
        {
            "domain": "financial",
            "prompt": "Explain what a balance sheet shows.",
            "response": "A balance sheet is primarily used to show the daily cash flow of a business and its profitability trends over time. It's mainly used for short-term budgeting.",
            "expected_factual": False,  # This is incorrect (confusing balance sheet with income statement)
            "description": "Confusing balance sheet with other statements"
        },
        
        # Legal domain test cases
        {
            "domain": "legal",
            "prompt": "What are the requirements for a valid contract?",
            "response": "Generally, a valid contract requires offer, acceptance, consideration, legal capacity of the parties, and lawful purpose. Consideration refers to something of value exchanged between the parties.",
            "expected_factual": True,  # This is factually correct
            "description": "Correct contract requirements"
        },
        {
            "domain": "legal",
            "prompt": "What are the requirements for a valid contract?",
            "response": "A valid contract only requires two people to verbally agree to the terms. There is never a need for written documentation or any exchange of value to make a contract binding in any jurisdiction.",
            "expected_factual": False,  # This is factually incorrect
            "description": "Incorrect contract requirements"
        },
        {
            "domain": "legal",
            "prompt": "Explain what copyright protection covers.",
            "response": "Copyright protection covers original works of authorship fixed in a tangible medium, such as literary, dramatic, musical, and artistic works. It does not protect ideas, procedures, methods, or concepts.",
            "expected_factual": True,  # This is factually correct
            "description": "Correct copyright explanation"
        },
        {
            "domain": "legal",
            "prompt": "Explain what copyright protection covers.",
            "response": "Copyright automatically protects any idea you have, even if you haven't written it down or recorded it in any way. Simply thinking of a business idea gives you full copyright protection against anyone implementing it.",
            "expected_factual": False,  # This is factually incorrect
            "description": "Incorrect copyright explanation"
        },
        
        # Medical domain test cases
        {
            "domain": "medical",
            "prompt": "What is diabetes?",
            "response": "Diabetes is a chronic health condition where the body either doesn't produce enough insulin (Type 1) or can't effectively use the insulin it produces (Type 2), resulting in elevated blood glucose levels. It requires ongoing management and can lead to various health complications if not properly controlled.",
            "expected_factual": True,  # This is factually correct
            "description": "Correct diabetes explanation"
        },
        {
            "domain": "medical",
            "prompt": "What is diabetes?",
            "response": "Diabetes is simply a temporary condition caused by eating too much sugar that can be completely cured by following a sugar-free diet for a few weeks. It has no long-term health implications.",
            "expected_factual": False,  # This is factually incorrect
            "description": "Incorrect diabetes explanation"
        },
        {
            "domain": "medical",
            "prompt": "How do vaccines work?",
            "response": "Vaccines work by introducing a weakened or inactive form of a pathogen (or parts of it) to trigger an immune response. This teaches the immune system to recognize and fight the actual pathogen if exposed later, without causing the disease itself.",
            "expected_factual": True,  # This is factually correct
            "description": "Correct vaccine explanation"
        },
        {
            "domain": "medical",
            "prompt": "How do vaccines work?",
            "response": "Vaccines work by completely replacing your blood with synthetic chemicals that make you immune to all diseases. They permanently alter your DNA to provide immunity.",
            "expected_factual": False,  # This is factually incorrect
            "description": "Incorrect vaccine explanation"
        }
    ]

    def evaluate_thresholds() -> Dict[str, Any]:
        """
        Evaluate the effectiveness of current threshold settings on test cases.
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Current threshold settings:")
        logger.info(f"  Global threshold: {FACTUALITY_THRESHOLD}")
        for domain, threshold in DOMAIN_THRESHOLDS.items():
            logger.info(f"  {domain}: {threshold}")
            
        # Track metrics
        results = {
            "global": {
                "test_count": 0,
                "true_positives": 0,  # Correctly identified as factual
                "true_negatives": 0,  # Correctly identified as hallucinated
                "false_positives": 0,  # Incorrectly identified as factual
                "false_negatives": 0,  # Incorrectly identified as hallucinated
            }
        }
        
        # Initialize domain-specific tracking
        domain_results = {}
        for test_case in TEST_CASES:
            domain = test_case["domain"]
            if domain not in domain_results:
                domain_results[domain] = {
                    "test_count": 0,
                    "true_positives": 0,
                    "true_negatives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                }
                
        # Process each test case
        test_case_results = []
        for idx, test_case in enumerate(TEST_CASES):
            domain = test_case["domain"]
            prompt = test_case["prompt"]
            response = test_case["response"]
            expected_factual = test_case["expected_factual"]
            description = test_case["description"]
            
            # Run through the grounding pipeline
            logger.info(f"Processing test case {idx+1}/{len(TEST_CASES)}: {description}")
            validation_result = grounded_validation_pipeline(
                prompt=prompt,
                response=response,
                domain=domain,
                include_debug=True
            )
            
            # Extract results
            hallucination_detected = validation_result["hallucination_flag"]
            factuality_score = validation_result["factuality_score"]
            threshold_used = validation_result.get("debug_metadata", {}).get("threshold", FACTUALITY_THRESHOLD)
            
            # Update metrics based on expected factuality vs detection
            results["global"]["test_count"] += 1
            domain_results[domain]["test_count"] += 1
            
            if expected_factual and not hallucination_detected:
                # Correctly identified as factual
                results["global"]["true_positives"] += 1
                domain_results[domain]["true_positives"] += 1
                status = "TRUE POSITIVE"
            elif not expected_factual and hallucination_detected:
                # Correctly identified as hallucinated
                results["global"]["true_negatives"] += 1
                domain_results[domain]["true_negatives"] += 1
                status = "TRUE NEGATIVE"
            elif expected_factual and hallucination_detected:
                # Incorrectly identified as hallucinated
                results["global"]["false_negatives"] += 1
                domain_results[domain]["false_negatives"] += 1
                status = "FALSE NEGATIVE"
            else:  # not expected_factual and not hallucination_detected
                # Incorrectly identified as factual
                results["global"]["false_positives"] += 1
                domain_results[domain]["false_positives"] += 1
                status = "FALSE POSITIVE"
                
            # Store detailed result for this test case
            test_case_result = {
                "domain": domain,
                "description": description,
                "expected_factual": expected_factual, 
                "hallucination_detected": hallucination_detected,
                "factuality_score": factuality_score,
                "threshold_used": threshold_used,
                "status": status
            }
            test_case_results.append(test_case_result)
            
            # Log result
            factual_status = "factual" if expected_factual else "hallucinated"
            detected_status = "hallucinated" if hallucination_detected else "factual"
            logger.info(f"  Result: Expected {factual_status}, detected as {detected_status}")
            logger.info(f"  Score: {factuality_score:.4f}, Threshold: {threshold_used:.4f}")
            logger.info(f"  Status: {status}")
            
        # Calculate metrics
        results["domains"] = domain_results
        results["test_case_results"] = test_case_results
        
        # Calculate global metrics
        global_metrics = results["global"]
        total = global_metrics["test_count"]
        true_positives = global_metrics["true_positives"]
        true_negatives = global_metrics["true_negatives"]
        false_positives = global_metrics["false_positives"]
        false_negatives = global_metrics["false_negatives"]
        
        accuracy = (true_positives + true_negatives) / total if total > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        
        results["global"]["accuracy"] = accuracy
        results["global"]["precision"] = precision
        results["global"]["recall"] = recall
        results["global"]["f1_score"] = f1
        results["global"]["specificity"] = specificity
        
        # Calculate domain-specific metrics
        for domain, metrics in domain_results.items():
            domain_total = metrics["test_count"]
            domain_true_positives = metrics["true_positives"]
            domain_true_negatives = metrics["true_negatives"]
            domain_false_positives = metrics["false_positives"]
            domain_false_negatives = metrics["false_negatives"]
            
            domain_accuracy = (domain_true_positives + domain_true_negatives) / domain_total if domain_total > 0 else 0
            domain_precision = domain_true_positives / (domain_true_positives + domain_false_positives) if (domain_true_positives + domain_false_positives) > 0 else 0
            domain_recall = domain_true_positives / (domain_true_positives + domain_false_negatives) if (domain_true_positives + domain_false_negatives) > 0 else 0
            domain_f1 = 2 * (domain_precision * domain_recall) / (domain_precision + domain_recall) if (domain_precision + domain_recall) > 0 else 0
            domain_specificity = domain_true_negatives / (domain_true_negatives + domain_false_positives) if (domain_true_negatives + domain_false_positives) > 0 else 0
            
            metrics["accuracy"] = domain_accuracy
            metrics["precision"] = domain_precision
            metrics["recall"] = domain_recall
            metrics["f1_score"] = domain_f1
            metrics["specificity"] = domain_specificity
            
        return results

    def print_evaluation_summary(results: Dict[str, Any]) -> None:
        """
        Print a summary of the evaluation results.
        
        Args:
            results: Evaluation results dictionary
        """
        global_metrics = results["global"]
        
        print("\n=== Threshold Evaluation Summary ===\n")
        print("Global Metrics:")
        print(f"  Accuracy: {global_metrics['accuracy']:.4f}")
        print(f"  Precision: {global_metrics['precision']:.4f}")
        print(f"  Recall: {global_metrics['recall']:.4f}")
        print(f"  F1 Score: {global_metrics['f1_score']:.4f}")
        print(f"  Specificity (TNR): {global_metrics['specificity']:.4f}")
        print(f"  Tests: {global_metrics['test_count']}")
        print(f"  True Positives: {global_metrics['true_positives']}")
        print(f"  True Negatives: {global_metrics['true_negatives']}")
        print(f"  False Positives: {global_metrics['false_positives']}")
        print(f"  False Negatives: {global_metrics['false_negatives']}")
        
        print("\nDomain-Specific Metrics:")
        for domain, metrics in results["domains"].items():
            print(f"\n  {domain}:")
            print(f"    Accuracy: {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1 Score: {metrics['f1_score']:.4f}")
            print(f"    Specificity (TNR): {metrics['specificity']:.4f}")
            
        print("\nCurrent Thresholds:")
        print(f"  Global: {FACTUALITY_THRESHOLD}")
        for domain, threshold in DOMAIN_THRESHOLDS.items():
            print(f"  {domain}: {threshold}")
            
        print("\nDetailed Results by Test Case:")
        for idx, result in enumerate(results["test_case_results"]):
            print(f"\n  Test {idx+1}: {result['description']} ({result['domain']})")
            print(f"    Expected: {'Factual' if result['expected_factual'] else 'Hallucinated'}")
            print(f"    Result: {'Hallucinated' if result['hallucination_detected'] else 'Factual'}")
            print(f"    Status: {result['status']}")
            print(f"    Score: {result['factuality_score']:.4f}")
            print(f"    Threshold: {result['threshold_used']:.4f}")
            
    def run_evaluation():
        """Run the threshold evaluation."""
        logger.info("=== Starting Threshold Evaluation ===")
        
        # Evaluate current threshold settings
        results = evaluate_thresholds()
        
        # Print results
        print_evaluation_summary(results)
        
        # Save results to file
        output_file = Path(__file__).parent / "threshold_evaluation_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Evaluation results saved to {output_file}")
        logger.info("=== Threshold Evaluation Complete ===")
        
        return results
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

if __name__ == "__main__":
    run_evaluation()