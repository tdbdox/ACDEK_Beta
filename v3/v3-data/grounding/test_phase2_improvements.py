#!/usr/bin/env python3
"""
Test Script for Phase 2 Improvements to Grounding Engine

This script tests the Phase 2 improvements to the grounding engine,
including semantic chunking and enhanced contradiction detection.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add parent directory to path to enable imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("phase2_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the grounding engine components
try:
    from grounding_engine import (
        score_factuality, 
        grounded_validation_pipeline,
        FACTUALITY_THRESHOLD,
        DOMAIN_THRESHOLDS,
        SEMANTIC_CHUNKING_AVAILABLE,
        CONTRADICTION_DETECTION_AVAILABLE
    )
except ImportError as e:
    logger.error(f"Error importing grounding engine: {e}")
    print(f"Error: Failed to import grounding engine: {e}")
    sys.exit(1)

# Try to import the semantic chunking and contradiction detection modules
try:
    from semantic_chunking import SemanticChunker, chunk_reference_data
    from contradiction_detection import ConflictDetector, detect_contradictions
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import Phase 2 modules directly: {e}")
    MODULES_AVAILABLE = False

def check_module_availability():
    """Check and report on module availability."""
    print("\n--- Module Availability ---")
    print(f"Semantic Chunking: {'✓ Available' if SEMANTIC_CHUNKING_AVAILABLE else '✗ Not available'}")
    print(f"Contradiction Detection: {'✓ Available' if CONTRADICTION_DETECTION_AVAILABLE else '✗ Not available'}")
    print(f"Direct Module Import: {'✓ Successful' if MODULES_AVAILABLE else '✗ Failed'}")
    
    if not SEMANTIC_CHUNKING_AVAILABLE and not CONTRADICTION_DETECTION_AVAILABLE:
        print("\nWARNING: Phase 2 modules are not available. Tests will run with fallback methods.")
        return False
    return True

def test_semantic_chunking():
    """Test the semantic chunking functionality."""
    if not SEMANTIC_CHUNKING_AVAILABLE:
        print("\n--- Semantic Chunking Test: SKIPPED (module not available) ---")
        return
    
    print("\n--- Semantic Chunking Test ---")
    
    # Test text for chunking
    test_text = """
    The Federal Reserve primarily controls inflation through monetary policy tools such as setting the federal funds rate, 
    conducting open market operations, and adjusting reserve requirements for banks. By raising interest rates, the Fed can 
    slow economic growth and reduce inflation by making borrowing more expensive, which reduces spending and investment. 
    Conversely, lowering interest rates can stimulate economic activity when inflation is too low.
    
    When inflation rises above the Fed's target (generally 2%), the FOMC typically increases the federal funds rate. 
    This is the interest rate at which depository institutions lend reserve balances to other depository institutions overnight. 
    This increase cascades through the economy, affecting everything from mortgage rates to credit card interest.
    
    Open market operations involve the buying and selling of government securities to control the money supply. 
    When the Fed sells securities, it reduces the money supply and helps combat inflation. When it buys securities, 
    it increases the money supply, which can combat deflation.
    """
    
    try:
        # Create chunker
        chunker = SemanticChunker()
        
        # Chunk the text
        chunks = chunker.chunk_text(test_text)
        
        # Report results
        print(f"Original text length: {len(test_text)} characters")
        print(f"Generated {len(chunks)} chunks:")
        
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} ({chunk['char_length']} chars, {chunk['word_count']} words):")
            print(f"{chunk['text'][:100]}...")
        
        # Test chunking on a reference item
        print("\nTesting reference data chunking:")
        
        reference_item = {
            "id": "test_ref",
            "text": test_text,
            "source": "Test Source"
        }
        
        chunked_items = chunk_reference_data([reference_item])
        print(f"Generated {len(chunked_items)} reference chunks")
        
        return True
    except Exception as e:
        logger.error(f"Error in semantic chunking test: {e}")
        print(f"Error: {e}")
        return False

def test_contradiction_detection():
    """Test the contradiction detection functionality."""
    if not CONTRADICTION_DETECTION_AVAILABLE:
        print("\n--- Contradiction Detection Test: SKIPPED (module not available) ---")
        return
    
    print("\n--- Contradiction Detection Test ---")
    
    # Test pairs for contradiction detection
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
        }
    ]
    
    try:
        for i, test in enumerate(test_pairs):
            # Detect contradictions
            result = detect_contradictions(test["response"], test["reference"], test["domain"])
            
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
        
        return True
    except Exception as e:
        logger.error(f"Error in contradiction detection test: {e}")
        print(f"Error: {e}")
        return False

def test_integrated_pipeline():
    """Test the integrated grounding pipeline with Phase 2 improvements."""
    print("\n--- Integrated Pipeline Test ---")
    
    # Test cases for the integrated pipeline
    test_cases = [
        {
            "prompt": "What is the role of the Federal Reserve in controlling inflation?",
            "response": "The Federal Reserve controls inflation by adjusting interest rates. When inflation is too high, they raise rates to make borrowing more expensive, which slows economic activity. When inflation is too low, they lower rates to stimulate economic growth.",
            "domain": "financial",
            "expected_hallucination": False
        },
        {
            "prompt": "Can type 2 diabetes be cured?",
            "response": "Yes, type 2 diabetes can be completely cured through diet and exercise alone in most cases, without medication.",
            "domain": "medical",
            "expected_hallucination": True
        },
        {
            "prompt": "What is habeas corpus?",
            "response": "Habeas corpus is a legal procedure that allows individuals to challenge their detention or imprisonment before a court. The term comes from Latin and roughly translates to 'you shall have the body.'",
            "domain": "legal",
            "expected_hallucination": False
        },
        {
            "prompt": "Is recording a conversation legal in the US?",
            "response": "Recording a conversation is legal in all 50 states as long as one party consents to the recording.",
            "domain": "legal",
            "expected_hallucination": True
        }
    ]
    
    # Test configurations to compare
    configurations = [
        {
            "name": "Baseline (Phase 1)",
            "use_chunks": False,
            "use_enhanced_contradiction": False
        },
        {
            "name": "Semantic Chunking Only",
            "use_chunks": True,
            "use_enhanced_contradiction": False
        },
        {
            "name": "Enhanced Contradiction Only",
            "use_chunks": False,
            "use_enhanced_contradiction": True
        },
        {
            "name": "Full Phase 2",
            "use_chunks": True,
            "use_enhanced_contradiction": True
        }
    ]
    
    # Skip configurations that require unavailable modules
    if not SEMANTIC_CHUNKING_AVAILABLE:
        configurations = [c for c in configurations if not c["use_chunks"]]
    if not CONTRADICTION_DETECTION_AVAILABLE:
        configurations = [c for c in configurations if not c["use_enhanced_contradiction"]]
    
    # If no configurations are available, run baseline only
    if not configurations:
        configurations = [{"name": "Baseline (Phase 1)", "use_chunks": False, "use_enhanced_contradiction": False}]
    
    # Run tests for each configuration
    all_results = []
    
    for config in configurations:
        print(f"\n--- Testing Configuration: {config['name']} ---")
        config_results = []
        
        for i, test in enumerate(test_cases):
            # Time the processing
            start_time = time.time()
            
            # Run the pipeline with the current configuration
            result = grounded_validation_pipeline(
                prompt=test["prompt"],
                response=test["response"],
                domain=test["domain"],
                include_debug=True,
                use_chunks=config["use_chunks"],
                use_enhanced_contradiction=config["use_enhanced_contradiction"]
            )
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Check if result matches expectation
            expected_hallucination = test["expected_hallucination"]
            actual_hallucination = result["hallucination_flag"]
            result_matches = expected_hallucination == actual_hallucination
            status = "✓ PASS" if result_matches else "✗ FAIL"
            
            # Report results
            print(f"\nTest {i+1} ({test['domain']}): {status}")
            print(f"  Prompt: {test['prompt']}")
            print(f"  Response: {test['response']}")
            print(f"  Factuality Score: {result['factuality_score']:.4f}")
            print(f"  Hallucination Flag: {result['hallucination_flag']}")
            print(f"  Expected Hallucination: {expected_hallucination}")
            print(f"  Processing Time: {processing_time:.2f} ms")
            
            # Extract debug info
            debug_info = result.get("debug_metadata", {})
            factuality_details = debug_info.get("factuality_details", {})
            
            # Print additional details if available
            if factuality_details.get("has_contradiction"):
                print(f"  Contradiction: {factuality_details.get('contradiction_reason', 'Unknown')}")
            
            if "chunk_scores" in factuality_details:
                print(f"  Chunks Used: {factuality_details.get('chunks_used', 0)}")
            
            # Store result for comparison
            config_results.append({
                "test_id": i,
                "domain": test["domain"],
                "factuality_score": result["factuality_score"],
                "hallucination_flag": result["hallucination_flag"],
                "expected_hallucination": expected_hallucination,
                "correct_detection": result_matches,
                "processing_time_ms": processing_time,
                "config": config["name"]
            })
        
        # Calculate summary metrics for this configuration
        total = len(config_results)
        correct = sum(1 for r in config_results if r["correct_detection"])
        accuracy = correct / total if total > 0 else 0
        avg_processing_time = sum(r["processing_time_ms"] for r in config_results) / total if total > 0 else 0
        
        # Print summary
        print(f"\nConfiguration '{config['name']}' Summary:")
        print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"  Average Processing Time: {avg_processing_time:.2f} ms")
        
        # Store in all results
        all_results.extend(config_results)
    
    # Save results to file
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"phase2_test_results_{timestamp}.json"
        
        # Group by configuration name
        grouped_results = {}
        for result in all_results:
            config_name = result["config"]
            if config_name not in grouped_results:
                grouped_results[config_name] = []
            grouped_results[config_name].append(result)
        
        # Calculate overall metrics for each configuration
        metrics = {}
        for config_name, results in grouped_results.items():
            total = len(results)
            correct = sum(1 for r in results if r["correct_detection"])
            avg_time = sum(r["processing_time_ms"] for r in results) / total if total > 0 else 0
            
            metrics[config_name] = {
                "total_tests": total,
                "correct_detections": correct,
                "accuracy": correct / total if total > 0 else 0,
                "avg_processing_time_ms": avg_time
            }
        
        # Create final output
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "configurations": configurations,
            "results": all_results
        }
        
        # Write to file
        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)
        
        print(f"\nTest results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error saving test results: {e}")
        print(f"Error saving test results: {e}")
    
    return True

def main():
    """Run the Phase 2 improvement tests."""
    print("\n=== ACDEK v3 Grounding Engine - Phase 2 Improvement Tests ===\n")
    
    # Check module availability
    check_module_availability()
    
    # Test semantic chunking
    test_semantic_chunking()
    
    # Test contradiction detection
    test_contradiction_detection()
    
    # Test integrated pipeline
    test_integrated_pipeline()
    
    print("\n=== Tests Complete ===")

if __name__ == "__main__":
    main()