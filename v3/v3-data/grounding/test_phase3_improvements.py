#!/usr/bin/env python3
"""
Comprehensive Test Script for Phase 3 Improvements to Grounding Engine

This script evaluates the Phase 3 improvements to the grounding engine,
providing detailed comparisons with previous versions and extensive metrics.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import argparse

# Add parent directory to path to enable imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phase3_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the original and enhanced grounding engines
try:
    # Original engine
    from grounding_engine import (
        score_factuality,
        grounded_validation_pipeline,
        FACTUALITY_THRESHOLD as P1_THRESHOLD,
        DOMAIN_THRESHOLDS as P1_DOMAIN_THRESHOLDS,
        SEMANTIC_CHUNKING_AVAILABLE,
        CONTRADICTION_DETECTION_AVAILABLE
    )
    ORIGINAL_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing original grounding engine: {e}")
    print(f"Error: Failed to import original grounding engine: {e}")
    ORIGINAL_ENGINE_AVAILABLE = False

try:
    # Enhanced engine (Phase 3)
    from grounding_engine_enhanced import (
        score_factuality_enhanced,
        grounded_validation_pipeline_enhanced,
        adaptive_threshold,
        FACTUALITY_THRESHOLD as P3_THRESHOLD,
        DOMAIN_THRESHOLDS as P3_DOMAIN_THRESHOLDS,
        ENHANCED_CHUNKING_AVAILABLE,
        ENHANCED_CONTRADICTION_AVAILABLE
    )
    ENHANCED_ENGINE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing enhanced grounding engine: {e}")
    print(f"Error: Failed to import enhanced grounding engine: {e}")
    ENHANCED_ENGINE_AVAILABLE = False

def load_test_data() -> List[Dict[str, Any]]:
    """
    Load comprehensive test data from domain test files.
    
    Returns:
        List of test cases
    """
    test_cases = []
    
    # Read from domain-specific test files
    domain_files = [
        ("medical", "medical_test.json"),
        ("financial", "finance_test.json"),
        ("legal", "legal_test.json")
    ]
    
    for domain, filename in domain_files:
        file_path = Path(__file__).resolve().parent.parent / 'data' / 'real_world_test' / filename
        
        if not file_path.exists():
            logger.warning(f"Test file {file_path} not found")
            continue
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            if not isinstance(data, dict) or "test_cases" not in data:
                logger.warning(f"Invalid test file format: {file_path}")
                continue
                
            for case in data["test_cases"]:
                # Create standardized test case format
                test_case = {
                    "id": case.get("id", f"test-{len(test_cases)+1}"),
                    "prompt": case.get("prompt", ""),
                    "response": case.get("reference_text", ""),  # Using reference as "good" response
                    "hallucination": False,  # Reference text is never a hallucination
                    "domain": domain
                }
                test_cases.append(test_case)
                
                # If there's expert judgment of "hallucination", create a test case with that
                if case.get("expert_judgment") == "hallucination":
                    hallucination_case = {
                        "id": f"{case.get('id', f'test-{len(test_cases)+1}')}-hallucination",
                        "prompt": case.get("prompt", ""),
                        "response": case.get("response", ""),
                        "hallucination": True,
                        "domain": domain
                    }
                    test_cases.append(hallucination_case)
                
        except Exception as e:
            logger.error(f"Error loading test data from {file_path}: {e}")
    
    # Add additional test cases for specific Phase 3 improvements
    phase3_test_cases = [
        {
            "id": "p3-med-001",
            "prompt": "Can type 2 diabetes be cured?",
            "response": "Yes, type 2 diabetes can be completely cured through diet and exercise alone in most cases, without medication.",
            "hallucination": True,
            "domain": "medical"
        },
        {
            "id": "p3-med-002",
            "prompt": "Do vaccines alter DNA?",
            "response": "Vaccines like the COVID-19 mRNA vaccines enter the nucleus of cells and modify your DNA, permanently altering your genetic code.",
            "hallucination": True,
            "domain": "medical"
        },
        {
            "id": "p3-fin-001",
            "prompt": "What is the FDIC insurance limit?",
            "response": "The FDIC insures all types of investments up to $1 million per account.",
            "hallucination": True,
            "domain": "financial"
        },
        {
            "id": "p3-fin-002",
            "prompt": "Who created Bitcoin?",
            "response": "Bitcoin was created by a team of developers at MIT in 2010.",
            "hallucination": True,
            "domain": "financial"
        },
        {
            "id": "p3-leg-001",
            "prompt": "How long do Supreme Court justices serve?",
            "response": "US Supreme Court justices serve for a maximum term of 10 years.",
            "hallucination": True,
            "domain": "legal"
        },
        {
            "id": "p3-leg-002",
            "prompt": "Is recording a conversation legal in the US?",
            "response": "Recording a conversation is legal in all 50 states as long as one party consents to the recording.",
            "hallucination": True,
            "domain": "legal"
        }
    ]
    
    # Add Phase 3 specific test cases
    test_cases.extend(phase3_test_cases)
    
    logger.info(f"Loaded {len(test_cases)} test cases")
    return test_cases

def check_module_availability() -> Dict[str, bool]:
    """
    Check and report on module availability.
    
    Returns:
        Dictionary with module availability flags
    """
    availability = {
        "original_engine": ORIGINAL_ENGINE_AVAILABLE,
        "enhanced_engine": ENHANCED_ENGINE_AVAILABLE,
        "semantic_chunking": SEMANTIC_CHUNKING_AVAILABLE if "SEMANTIC_CHUNKING_AVAILABLE" in globals() else False,
        "contradiction_detection": CONTRADICTION_DETECTION_AVAILABLE if "CONTRADICTION_DETECTION_AVAILABLE" in globals() else False,
        "enhanced_chunking": ENHANCED_CHUNKING_AVAILABLE if "ENHANCED_CHUNKING_AVAILABLE" in globals() else False,
        "enhanced_contradiction": ENHANCED_CONTRADICTION_AVAILABLE if "ENHANCED_CONTRADICTION_AVAILABLE" in globals() else False
    }
    
    print("\n--- Module Availability ---")
    print(f"Original Engine: {'✓ Available' if availability['original_engine'] else '✗ Not available'}")
    print(f"Enhanced Engine: {'✓ Available' if availability['enhanced_engine'] else '✗ Not available'}")
    print(f"  - Semantic Chunking: {'✓ Available' if availability['semantic_chunking'] else '✗ Not available'}")
    print(f"  - Contradiction Detection: {'✓ Available' if availability['contradiction_detection'] else '✗ Not available'}")
    print(f"  - Enhanced Chunking: {'✓ Available' if availability['enhanced_chunking'] else '✗ Not available'}")
    print(f"  - Enhanced Contradiction: {'✓ Available' if availability['enhanced_contradiction'] else '✗ Not available'}")
    
    return availability

def run_original_pipeline(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the original (Phase 1/2) grounding pipeline on a test case.
    
    Args:
        test_case: Test case data
        
    Returns:
        Dictionary with validation results
    """
    if not ORIGINAL_ENGINE_AVAILABLE:
        return {"error": "Original engine not available"}
    
    try:
        # Run the pipeline with the enhanced settings
        result = grounded_validation_pipeline(
            prompt=test_case["prompt"],
            response=test_case["response"],
            domain=test_case["domain"],
            include_debug=True
        )
        
        # Add metadata
        result["test_id"] = test_case["id"]
        result["domain"] = test_case["domain"]
        result["expected_hallucination"] = test_case["hallucination"]
        result["correct_detection"] = result["hallucination_flag"] == test_case["hallucination"]
        
        return result
    except Exception as e:
        logger.error(f"Error running original pipeline on test {test_case['id']}: {e}")
        return {
            "error": str(e),
            "test_id": test_case["id"],
            "domain": test_case["domain"],
            "expected_hallucination": test_case["hallucination"],
            "correct_detection": False
        }

def run_enhanced_pipeline(test_case: Dict[str, Any], use_adaptive: bool = True) -> Dict[str, Any]:
    """
    Run the enhanced (Phase 3) grounding pipeline on a test case.
    
    Args:
        test_case: Test case data
        use_adaptive: Whether to use adaptive thresholds
        
    Returns:
        Dictionary with validation results
    """
    if not ENHANCED_ENGINE_AVAILABLE:
        return {"error": "Enhanced engine not available"}
    
    try:
        # Run the pipeline with the enhanced settings
        result = grounded_validation_pipeline_enhanced(
            prompt=test_case["prompt"],
            response=test_case["response"],
            domain=test_case["domain"],
            include_debug=True,
            use_enhanced=True,
            use_adaptive_threshold=use_adaptive
        )
        
        # Add metadata
        result["test_id"] = test_case["id"]
        result["domain"] = test_case["domain"]
        result["expected_hallucination"] = test_case["hallucination"]
        result["correct_detection"] = result["hallucination_flag"] == test_case["hallucination"]
        result["adaptive_threshold"] = use_adaptive
        
        return result
    except Exception as e:
        logger.error(f"Error running enhanced pipeline on test {test_case['id']}: {e}")
        return {
            "error": str(e),
            "test_id": test_case["id"],
            "domain": test_case["domain"],
            "expected_hallucination": test_case["hallucination"],
            "correct_detection": False,
            "adaptive_threshold": use_adaptive
        }

def compare_pipelines(test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run and compare all pipeline configurations on test cases.
    
    Args:
        test_cases: List of test cases
        
    Returns:
        Dictionary with comprehensive results and metrics
    """
    # Define configurations to test
    configurations = []
    
    if ORIGINAL_ENGINE_AVAILABLE:
        configurations.append({
            "name": "Phase 1/2 (Baseline)",
            "function": run_original_pipeline,
            "args": {}
        })
    
    if ENHANCED_ENGINE_AVAILABLE:
        configurations.append({
            "name": "Phase 3 (Static)",
            "function": run_enhanced_pipeline,
            "args": {"use_adaptive": False}
        })
        configurations.append({
            "name": "Phase 3 (Adaptive)",
            "function": run_enhanced_pipeline,
            "args": {"use_adaptive": True}
        })
    
    # Run all configurations on all test cases
    all_results = {config["name"]: [] for config in configurations}
    
    print("\n--- Running Tests ---")
    for i, test_case in enumerate(test_cases):
        print(f"Processing test case {i+1}/{len(test_cases)}: {test_case['id']}", end="\r")
        
        for config in configurations:
            result = config["function"](test_case, **config["args"])
            all_results[config["name"]].append(result)
    
    print("\nTests completed                     ")
    
    # Calculate metrics
    metrics = {}
    
    for config_name, results in all_results.items():
        # Overall metrics
        total = len(results)
        correct = sum(1 for r in results if r.get("correct_detection", False))
        accuracy = correct / total if total > 0 else 0
        
        # Handle divided by domain
        domain_metrics = {}
        for domain in ["medical", "financial", "legal"]:
            domain_results = [r for r in results if r.get("domain") == domain]
            domain_total = len(domain_results)
            domain_correct = sum(1 for r in domain_results if r.get("correct_detection", False))
            domain_accuracy = domain_correct / domain_total if domain_total > 0 else 0
            
            domain_metrics[domain] = {
                "total": domain_total,
                "correct": domain_correct,
                "accuracy": domain_accuracy
            }
        
        # True/false positive/negative metrics
        tp = sum(1 for r in results 
                if r.get("expected_hallucination", False) and r.get("hallucination_flag", False))
        fp = sum(1 for r in results 
                if not r.get("expected_hallucination", False) and r.get("hallucination_flag", False))
        tn = sum(1 for r in results 
                if not r.get("expected_hallucination", False) and not r.get("hallucination_flag", False))
        fn = sum(1 for r in results 
                if r.get("expected_hallucination", False) and not r.get("hallucination_flag", False))
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Get average processing time
        processing_times = []
        for r in results:
            if "debug_metadata" in r and "execution_time_ms" in r["debug_metadata"]:
                processing_times.append(r["debug_metadata"]["execution_time_ms"])
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Store all metrics
        metrics[config_name] = {
            "total_tests": total,
            "correct_detections": correct,
            "accuracy": accuracy,
            "domains": domain_metrics,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "avg_processing_time_ms": avg_time
        }
    
    return {
        "configurations": configurations,
        "all_results": all_results,
        "metrics": metrics
    }

def print_comparison_results(comparison: Dict[str, Any]) -> None:
    """
    Print a formatted comparison of pipeline results.
    
    Args:
        comparison: Comparison results from compare_pipelines
    """
    metrics = comparison["metrics"]
    
    print("\n--- Overall Performance Comparison ---")
    print(f"{'Configuration':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Avg Time (ms)':<15}")
    print("-" * 75)
    
    for config_name, stats in metrics.items():
        print(f"{config_name:<20} {stats['accuracy']:<10.2%} {stats['precision']:<10.2%} "
              f"{stats['recall']:<10.2%} {stats['f1_score']:<10.2%} {stats['avg_processing_time_ms']:<15.2f}")
    
    print("\n--- Performance by Domain ---")
    for domain in ["medical", "financial", "legal"]:
        print(f"\n{domain.upper()} Domain:")
        print(f"{'Configuration':<20} {'Accuracy':<10} {'Tests':<10}")
        print("-" * 40)
        
        for config_name, stats in metrics.items():
            domain_stats = stats["domains"][domain]
            print(f"{config_name:<20} {domain_stats['accuracy']:<10.2%} {domain_stats['total']:<10}")
    
    print("\n--- Confusion Matrix Statistics ---")
    print(f"{'Configuration':<20} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8}")
    print("-" * 52)
    
    for config_name, stats in metrics.items():
        print(f"{config_name:<20} {stats['true_positives']:<8} {stats['false_positives']:<8} "
              f"{stats['true_negatives']:<8} {stats['false_negatives']:<8}")

def save_results(comparison: Dict[str, Any], output_dir: str = None) -> str:
    """
    Save test results to JSON and generate visualization.
    
    Args:
        comparison: Comparison results from compare_pipelines
        output_dir: Directory to save results (defaults to current directory)
        
    Returns:
        Path to saved results file
    """
    # Create timestamp for filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Determine output directory
    if not output_dir:
        output_dir = str(Path(__file__).resolve().parent)
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results file path
    results_file = os.path.join(output_dir, f"phase3_test_results_{timestamp}.json")
    
    # Format results for saving
    save_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configurations": [config["name"] for config in comparison["configurations"]],
        "metrics": comparison["metrics"],
        "phase3_thresholds": {
            "global": P3_THRESHOLD if "P3_THRESHOLD" in globals() else None,
            "domains": P3_DOMAIN_THRESHOLDS if "P3_DOMAIN_THRESHOLDS" in globals() else None
        },
        "module_availability": check_module_availability()
    }
    
    # Save results
    try:
        with open(results_file, "w") as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\nTest results saved to {results_file}")
        
        # Generate visualizations if matplotlib is available
        try:
            # Create results visualization
            plot_comparison_results(comparison, output_dir, timestamp)
        except Exception as e:
            logger.warning(f"Could not generate visualization: {e}")
        
        return results_file
        
    except Exception as e:
        logger.error(f"Error saving test results: {e}")
        print(f"Error saving test results: {e}")
        return None

def plot_comparison_results(comparison: Dict[str, Any], output_dir: str, timestamp: str) -> None:
    """
    Generate visualization plots for the test results.
    
    Args:
        comparison: Comparison results from compare_pipelines
        output_dir: Directory to save plots
        timestamp: Timestamp string for filenames
    """
    metrics = comparison["metrics"]
    config_names = list(metrics.keys())
    
    # 1. Overall metrics comparison
    plt.figure(figsize=(12, 6))
    
    # Performance metrics
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(config_names)
    
    for i, config_name in enumerate(config_names):
        values = [metrics[config_name][metric] for metric in metrics_to_plot]
        plt.bar(x + i * width - 0.4 + width/2, values, width, label=config_name)
    
    plt.ylim(0, 1.0)
    plt.xticks(x, metrics_to_plot)
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"phase3_metrics_{timestamp}.png"))
    
    # 2. Domain-specific accuracy
    plt.figure(figsize=(12, 6))
    
    domains = ["medical", "financial", "legal"]
    x = np.arange(len(domains))
    width = 0.8 / len(config_names)
    
    for i, config_name in enumerate(config_names):
        values = [metrics[config_name]["domains"][domain]["accuracy"] for domain in domains]
        plt.bar(x + i * width - 0.4 + width/2, values, width, label=config_name)
    
    plt.ylim(0, 1.0)
    plt.xticks(x, domains)
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Domain')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"phase3_domains_{timestamp}.png"))
    
    # 3. Processing time comparison
    plt.figure(figsize=(10, 5))
    
    times = [metrics[config_name]["avg_processing_time_ms"] for config_name in config_names]
    plt.bar(config_names, times, color='skyblue')
    plt.ylabel('Time (ms)')
    plt.title('Average Processing Time')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"phase3_timing_{timestamp}.png"))
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Run the Phase 3 improvement tests."""
    parser = argparse.ArgumentParser(description='Test Phase 3 improvements to the grounding engine')
    parser.add_argument('--output', '-o', help='Directory to save test results and visualizations')
    parser.add_argument('--quick', '-q', action='store_true', help='Run a smaller subset of tests for quick validation')
    args = parser.parse_args()
    
    print("\n=== ACDEK v3 Grounding Engine - Phase 3 Improvement Tests ===\n")
    
    # Check module availability
    availability = check_module_availability()
    
    if not (availability["original_engine"] or availability["enhanced_engine"]):
        print("Error: Neither original nor enhanced engine is available.")
        sys.exit(1)
    
    # Load test data
    test_cases = load_test_data()
    
    # Optionally use a smaller subset for quick testing
    if args.quick:
        print("Running quick validation with subset of tests")
        # Select a representative sample from each domain
        domains = ["medical", "financial", "legal"]
        selected_cases = []
        
        for domain in domains:
            # Get some hallucination and non-hallucination cases for each domain
            domain_cases = [case for case in test_cases if case["domain"] == domain]
            hallucination_cases = [case for case in domain_cases if case["hallucination"]]
            non_hallucination_cases = [case for case in domain_cases if not case["hallucination"]]
            
            # Select up to 2 of each
            selected_cases.extend(hallucination_cases[:2])
            selected_cases.extend(non_hallucination_cases[:2])
        
        test_cases = selected_cases
        print(f"Selected {len(test_cases)} test cases for quick validation")
    
    # Run comparison of different configurations
    comparison = compare_pipelines(test_cases)
    
    # Print results
    print_comparison_results(comparison)
    
    # Save results
    save_results(comparison, args.output)
    
    print("\n=== Tests Complete ===")

if __name__ == "__main__":
    main()