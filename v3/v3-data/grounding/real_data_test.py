#!/usr/bin/env python3
"""
Comprehensive test for the grounding engine with real datasets.

This test validates the grounding engine's performance with real-world 
reference data across multiple domains, ensuring it meets v3 objectives:
1. Accurate reference retrieval
2. Precise factuality scoring
3. Reliable hallucination detection
4. System integration readiness
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

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

try:
    # Import grounding functions
    from v3.grounding import (
        grounded_validation_pipeline,
        retrieve_reference,
        score_factuality,
        build_grounding_context,
        get_available_domains,
        clear_reference_cache,
        FACTUALITY_THRESHOLD
    )

    # Define comprehensive test cases across all domains
    # Each domain has clear correct, incorrect, and edge case examples
    TEST_CASES = [
        # Financial domain test cases
        {
            "id": "fin-asset-1",
            "domain": "financial",
            "prompt": "What is an asset in financial terms?",
            "correct_response": "An asset is a resource with economic value that an individual, corporation, or country owns or controls with the expectation it will provide future benefit.",
            "incorrect_response": "An asset is a type of liability that companies use to secure loans. Assets are always listed on the liability side of the balance sheet.",
            "edge_response": "An asset is something that has value and can potentially provide future benefit to its owner, though some assets may depreciate over time."
        },
        {
            "id": "fin-equity-1",
            "domain": "financial",
            "prompt": "How is equity calculated in a company?",
            "correct_response": "Equity represents ownership in a company and is calculated as the difference between total assets and total liabilities.",
            "incorrect_response": "Equity is calculated by adding up all the cash a company has in its accounts. It has nothing to do with assets or liabilities.",
            "edge_response": "Equity shows ownership interest and is generally determined by subtracting what a company owes from what it owns, though the specific calculation can vary in complex situations."
        },
        {
            "id": "fin-income-1",
            "domain": "financial",
            "prompt": "What does an income statement show?",
            "correct_response": "An income statement is a financial document that shows a company's revenues and expenses over a specific period, indicating how the revenues are transformed into net income or net profit.",
            "incorrect_response": "An income statement only tracks cash flow and has nothing to do with revenues or expenses. It's primarily used for tax planning.",
            "edge_response": "Income statements summarize how much money a company made and spent, though they don't provide a complete picture of a company's financial health without balance sheets and cash flow statements."
        },

        # Legal domain test cases
        {
            "id": "leg-contract-1",
            "domain": "legal",
            "prompt": "What are the requirements for a valid contract?",
            "correct_response": "A contract is a legally binding agreement between two or more parties that creates obligations enforceable by law. Valid contracts require offer, acceptance, consideration, legal capacity, and lawful purpose.",
            "incorrect_response": "A valid contract only requires a verbal agreement between friends. No documentation or consideration is needed, and contracts automatically expire after one year.",
            "edge_response": "Contracts typically need parties to agree to something, exchange something of value, and have legal elements in place, though specific requirements can vary by jurisdiction."
        },
        {
            "id": "leg-tort-1",
            "domain": "legal",
            "prompt": "What is tort law?",
            "correct_response": "Tort law is a body of civil law that addresses wrongful acts resulting in injury or harm to another person, property, reputation, or other legally protected interests, providing remedies through civil litigation.",
            "incorrect_response": "Tort law deals exclusively with car accidents and requires payment to the government rather than affected parties. It's primarily a criminal law matter.",
            "edge_response": "Tort law handles situations where someone causes harm that isn't criminal but still deserves compensation, though many tort cases settle before trial."
        },
        {
            "id": "leg-ip-1",
            "domain": "legal",
            "prompt": "What is intellectual property?",
            "correct_response": "Intellectual property refers to creations of the mind, such as inventions, literary and artistic works, designs, symbols, names, and images used in commerce. It is protected by patents, copyrights, trademarks, and trade secrets.",
            "incorrect_response": "Intellectual property means owning actual physical property like land or buildings. It has nothing to do with creative works or inventions.",
            "edge_response": "Intellectual property covers creative and innovative works, though protection strength and duration varies widely depending on the type and country."
        },

        # Medical domain test cases
        {
            "id": "med-hypertension-1",
            "domain": "medical",
            "prompt": "What is hypertension and how is it diagnosed?",
            "correct_response": "Hypertension (high blood pressure) is a chronic condition where blood pressure against artery walls is consistently elevated. It is diagnosed when readings consistently exceed 130/80 mmHg. Untreated hypertension increases risk of heart disease, stroke, and kidney failure.",
            "incorrect_response": "Hypertension is low blood pressure diagnosed through a single blood test. It's easily cured with over-the-counter medication and has no long-term health effects.",
            "edge_response": "High blood pressure occurs when your heart works too hard to pump blood. While specific numbers are used for diagnosis, the threshold for what's considered high blood pressure has changed over the years."
        },
        {
            "id": "med-diabetes-1",
            "domain": "medical",
            "prompt": "What is diabetes mellitus?",
            "correct_response": "Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels (hyperglycemia) over a prolonged period. The two main types are Type 1 (autoimmune condition where the pancreas produces little or no insulin) and Type 2 (where cells become resistant to insulin's effects).",
            "incorrect_response": "Diabetes is a contagious viral infection that affects the liver and can be completely cured with antibiotics. All types of diabetes are caused by eating too much sugar.",
            "edge_response": "Diabetes is when your body has trouble regulating blood sugar levels, though the specific causes and treatments vary depending on which type you have."
        },
        {
            "id": "med-cancer-1",
            "domain": "medical",
            "prompt": "What is cancer and how is it treated?",
            "correct_response": "Cancer is a disease characterized by the uncontrolled growth and spread of abnormal cells. These cells can invade and destroy normal body tissue and spread throughout the body. Treatment options include surgery, radiation, chemotherapy, immunotherapy, and targeted therapy.",
            "incorrect_response": "Cancer is a bacterial infection that's completely curable with a specific herb that doctors don't want you to know about. All cancers are caused by cell phone radiation.",
            "edge_response": "Cancer happens when cells grow out of control. While there are many treatments available, the effectiveness depends on the cancer type, stage, and individual factors."
        }
    ]

    def run_grounding_test(
        case: Dict[str, Any], 
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Run a comprehensive test for a single case with detailed metrics."""
        domain = case["domain"]
        prompt = case["prompt"]
        results = {}
        
        # Clear cache between tests to ensure fresh retrieval
        clear_reference_cache()
        
        # Test reference retrieval
        retrieval_start = time.time()
        reference = retrieve_reference(prompt, domain)
        retrieval_time = time.time() - retrieval_start
        
        # Test correct response
        correct_start = time.time()
        correct_result = grounded_validation_pipeline(
            prompt, 
            case["correct_response"], 
            domain,
            include_debug=True
        )
        correct_time = time.time() - correct_start
        
        # Test incorrect response
        incorrect_start = time.time()
        incorrect_result = grounded_validation_pipeline(
            prompt, 
            case["incorrect_response"], 
            domain,
            include_debug=True
        )
        incorrect_time = time.time() - incorrect_start
        
        # Test edge case response
        edge_start = time.time()
        edge_result = grounded_validation_pipeline(
            prompt, 
            case["edge_response"], 
            domain,
            include_debug=True
        )
        edge_time = time.time() - edge_start
        
        # Compile results
        results = {
            "case_id": case["id"],
            "domain": domain,
            "prompt": prompt,
            "retrieved_reference": {
                "id": reference.get("id", "unknown"),
                "text": reference.get("text", ""),
                "similarity": reference.get("similarity", 0),
                "retrieval_time_ms": retrieval_time * 1000
            },
            "correct_response": {
                "text": case["correct_response"],
                "factuality_score": correct_result.get("factuality_score", 0),
                "hallucination_flag": correct_result.get("hallucination_flag", True),
                "constraint_pass": correct_result.get("constraint_pass", False),
                "processing_time_ms": correct_time * 1000,
            },
            "incorrect_response": {
                "text": case["incorrect_response"],
                "factuality_score": incorrect_result.get("factuality_score", 0),
                "hallucination_flag": incorrect_result.get("hallucination_flag", True),
                "constraint_pass": incorrect_result.get("constraint_pass", False),
                "processing_time_ms": incorrect_time * 1000
            },
            "edge_response": {
                "text": case["edge_response"],
                "factuality_score": edge_result.get("factuality_score", 0),
                "hallucination_flag": edge_result.get("hallucination_flag", True),
                "constraint_pass": edge_result.get("constraint_pass", False),
                "processing_time_ms": edge_time * 1000
            }
        }
        
        # Calculate validation metrics
        correct_is_factual = correct_result.get("factuality_score", 0) >= FACTUALITY_THRESHOLD
        incorrect_is_hallucination = incorrect_result.get("hallucination_flag", True)
        
        # Calculate accuracy metrics (true positive, true negative)
        results["validation_metrics"] = {
            "correctly_identified_factual": correct_is_factual,
            "correctly_identified_hallucination": incorrect_is_hallucination,
            "retrieval_similarity": reference.get("similarity", 0),
            "total_processing_time_ms": (correct_time + incorrect_time + edge_time) * 1000
        }
        
        if verbose:
            print(f"\n--- Test Case: {case['id']} ---")
            print(f"Prompt: {prompt}")
            print(f"Reference: {reference.get('id', 'unknown')}")
            print(f"Reference similarity: {reference.get('similarity', 0):.4f}")
            print(f"Correct response factuality: {correct_result.get('factuality_score', 0):.4f}")
            print(f"Incorrect response factuality: {incorrect_result.get('factuality_score', 0):.4f}")
            print(f"Edge response factuality: {edge_result.get('factuality_score', 0):.4f}")
            print(f"Correct response hallucination: {correct_result.get('hallucination_flag', True)}")
            print(f"Incorrect response hallucination: {incorrect_result.get('hallucination_flag', True)}")
            print(f"Edge response hallucination: {edge_result.get('hallucination_flag', True)}")
        
        return results

    def calculate_aggregated_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregated metrics across all test cases."""
        total_cases = len(results)
        if total_cases == 0:
            return {}
            
        # Aggregate metrics by domain
        domains = set(r["domain"] for r in results)
        domain_metrics = {}
        
        for domain in domains:
            domain_results = [r for r in results if r["domain"] == domain]
            domain_count = len(domain_results)
            
            if domain_count == 0:
                continue
                
            # Calculate domain-specific metrics
            domain_metrics[domain] = {
                "case_count": domain_count,
                "avg_factuality_correct": sum(r["correct_response"]["factuality_score"] for r in domain_results) / domain_count,
                "avg_factuality_incorrect": sum(r["incorrect_response"]["factuality_score"] for r in domain_results) / domain_count,
                "avg_factuality_edge": sum(r["edge_response"]["factuality_score"] for r in domain_results) / domain_count,
                "true_positive_rate": sum(1 for r in domain_results if r["validation_metrics"]["correctly_identified_factual"]) / domain_count,
                "true_negative_rate": sum(1 for r in domain_results if r["validation_metrics"]["correctly_identified_hallucination"]) / domain_count,
                "avg_retrieval_similarity": sum(r["validation_metrics"]["retrieval_similarity"] for r in domain_results) / domain_count,
                "avg_processing_time_ms": sum(r["validation_metrics"]["total_processing_time_ms"] for r in domain_results) / domain_count
            }
        
        # Overall metrics
        return {
            "total_cases": total_cases,
            "domains": list(domains),
            "true_positive_rate": sum(1 for r in results if r["validation_metrics"]["correctly_identified_factual"]) / total_cases,
            "true_negative_rate": sum(1 for r in results if r["validation_metrics"]["correctly_identified_hallucination"]) / total_cases,
            "avg_retrieval_similarity": sum(r["validation_metrics"]["retrieval_similarity"] for r in results) / total_cases,
            "avg_processing_time_ms": sum(r["validation_metrics"]["total_processing_time_ms"] for r in results) / total_cases,
            "avg_factuality_correct": sum(r["correct_response"]["factuality_score"] for r in results) / total_cases,
            "avg_factuality_incorrect": sum(r["incorrect_response"]["factuality_score"] for r in results) / total_cases,
            "avg_factuality_edge": sum(r["edge_response"]["factuality_score"] for r in results) / total_cases,
            "domain_metrics": domain_metrics
        }

    def evaluate_objectives(metrics: Dict[str, Any]) -> Dict[str, Tuple[bool, str]]:
        """Evaluate if the system meets the v3 objectives."""
        objectives = {}
        
        # Objective 1: Accurate reference retrieval
        # Target: Average similarity score > 0.7
        retrieval_success = metrics.get("avg_retrieval_similarity", 0) > 0.7
        objectives["accurate_retrieval"] = (
            retrieval_success,
            f"Avg similarity: {metrics.get('avg_retrieval_similarity', 0):.4f}"
        )
        
        # Objective 2: Precise factuality scoring
        # Target: True positive rate > 0.9, True negative rate > 0.9
        factuality_success = (
            metrics.get("true_positive_rate", 0) > 0.9 and 
            metrics.get("true_negative_rate", 0) > 0.9
        )
        objectives["factuality_scoring"] = (
            factuality_success,
            f"TPR: {metrics.get('true_positive_rate', 0):.4f}, TNR: {metrics.get('true_negative_rate', 0):.4f}"
        )
        
        # Objective 3: Performance
        # Target: Average processing time < 200ms
        performance_success = metrics.get("avg_processing_time_ms", 1000) < 200
        objectives["performance"] = (
            performance_success,
            f"Avg processing time: {metrics.get('avg_processing_time_ms', 0):.2f}ms"
        )
        
        # Objective 4: Edge case handling
        # We want edge cases to have scores in a middle range, not extremely high or low
        edge_score = metrics.get("avg_factuality_edge", 0)
        edge_success = 0.6 <= edge_score <= 0.9
        objectives["edge_case_handling"] = (
            edge_success,
            f"Avg edge factuality: {edge_score:.4f}"
        )
        
        return objectives

    def run_comprehensive_test(verbose: bool = False) -> None:
        """Run comprehensive test suite and report results."""
        print("\n=======================================")
        print("ACDEK v3 Grounding Engine Real Data Test")
        print("=======================================\n")
        
        results = []
        for case in TEST_CASES:
            try:
                case_result = run_grounding_test(case, verbose)
                results.append(case_result)
            except Exception as e:
                logger.error(f"Error testing case {case['id']}: {str(e)}")
                print(f"Error in case {case['id']}: {str(e)}")
        
        # Calculate aggregated metrics
        metrics = calculate_aggregated_metrics(results)
        
        # Evaluate objectives
        objectives = evaluate_objectives(metrics)
        
        # Print comprehensive results
        print("\n=======================================")
        print("Test Results Summary")
        print("=======================================")
        print(f"Total test cases: {metrics.get('total_cases', 0)}")
        print(f"Domains tested: {', '.join(metrics.get('domains', []))}")
        print("\nPerformance Metrics:")
        print(f"- Average retrieval similarity: {metrics.get('avg_retrieval_similarity', 0):.4f}")
        print(f"- Average processing time: {metrics.get('avg_processing_time_ms', 0):.2f}ms")
        print(f"- True positive rate: {metrics.get('true_positive_rate', 0):.4f}")
        print(f"- True negative rate: {metrics.get('true_negative_rate', 0):.4f}")
        print("\nFactuality Scoring:")
        print(f"- Avg score for correct responses: {metrics.get('avg_factuality_correct', 0):.4f}")
        print(f"- Avg score for incorrect responses: {metrics.get('avg_factuality_incorrect', 0):.4f}")
        print(f"- Avg score for edge case responses: {metrics.get('avg_factuality_edge', 0):.4f}")
        
        # Domain-specific results
        print("\nDomain-Specific Results:")
        for domain, domain_metrics in metrics.get("domain_metrics", {}).items():
            print(f"\n{domain.capitalize()} Domain:")
            print(f"- Test cases: {domain_metrics.get('case_count', 0)}")
            print(f"- True positive rate: {domain_metrics.get('true_positive_rate', 0):.4f}")
            print(f"- True negative rate: {domain_metrics.get('true_negative_rate', 0):.4f}")
            print(f"- Avg retrieval similarity: {domain_metrics.get('avg_retrieval_similarity', 0):.4f}")
            print(f"- Avg processing time: {domain_metrics.get('avg_processing_time_ms', 0):.2f}ms")
        
        # Objectives evaluation
        print("\nV3 Objectives Evaluation:")
        for objective, (met, details) in objectives.items():
            status = "✅ PASS" if met else "❌ FAIL"
            print(f"- {objective}: {status} ({details})")
        
        # Overall assessment
        all_objectives_met = all(met for met, _ in objectives.values())
        print(f"\nOverall Result: {'✅ PASS' if all_objectives_met else '❌ FAIL'}")
        
        # Save results to file
        output_file = "grounding_test_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "metrics": metrics,
                "objectives": {k: v[0] for k, v in objectives.items()},
                "results": results
            }, f, indent=2)
        
        print(f"\nDetailed results saved to {output_file}")

except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test grounding engine with real data")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    run_comprehensive_test(verbose=args.verbose)