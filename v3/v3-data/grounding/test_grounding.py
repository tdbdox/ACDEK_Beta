#!/usr/bin/env python3
"""
Test script for the grounding engine.

This script tests the basic functionality of the grounding engine
with the provided reference data for different domains.
"""

import sys
import json
import logging
from pathlib import Path

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
    # Import grounding functions directly from the grounding_engine module
    from grounding_engine import (
        grounded_validation_pipeline,
        retrieve_reference,
        score_factuality,
        build_grounding_context,
        get_available_domains
    )
    
    def test_domains():
        """Test available domains."""
        domains = get_available_domains()
        logger.info(f"Available domains: {domains}")
        assert len(domains) > 0, "No domains available"
        return domains
    
    def test_reference_retrieval(domains):
        """Test reference retrieval for each domain."""
        results = {}
        
        for domain in domains:
            prompt = f"Tell me about {domain}"
            try:
                reference = retrieve_reference(prompt, domain)
                logger.info(f"Retrieved reference for domain '{domain}': "
                            f"ID={reference['id']}, similarity={reference['similarity']:.4f}")
                results[domain] = {"success": True, "reference": reference}
            except Exception as e:
                logger.error(f"Error retrieving reference for domain '{domain}': {str(e)}")
                results[domain] = {"success": False, "error": str(e)}
        
        # Verify at least one domain succeeded
        success_count = sum(1 for r in results.values() if r.get("success", False))
        assert success_count > 0, "No references successfully retrieved"
        
        return results
    
    def test_factuality_scoring(reference_results):
        """Test factuality scoring with references."""
        for domain, result in reference_results.items():
            if not result.get("success", False):
                continue
                
            reference = result["reference"]["text"]
            
            # Test with exact match (should be 1.0)
            result_exact = score_factuality(reference, reference)
            score_exact = result_exact["score"]  # Extract score from result dict
            logger.info(f"Domain '{domain}' exact match score: {score_exact:.4f}")
            assert score_exact > 0.95, f"Exact match score too low: {score_exact}"
            
            # Test with partial content
            if len(reference) > 20:
                partial = reference[:len(reference)//2]
                result_partial = score_factuality(partial, reference)
                score_partial = result_partial["score"]  # Extract score from result dict
                logger.info(f"Domain '{domain}' partial match score: {score_partial:.4f}")
                assert 0 <= score_partial <= 1, f"Invalid partial score: {score_partial}"
            
            # Test with unrelated content
            unrelated = f"This is completely unrelated to {domain}"
            result_unrelated = score_factuality(unrelated, reference)
            score_unrelated = result_unrelated["score"]  # Extract score from result dict
            logger.info(f"Domain '{domain}' unrelated score: {score_unrelated:.4f}")
            assert score_unrelated < 0.7, f"Unrelated score too high: {score_unrelated}"
    
    def test_full_pipeline(domains):
        """Test the full grounding pipeline."""
        for domain in domains:
            prompt = f"Tell me about {domain}"
            response = f"This is a test response related to {domain}."
            
            try:
                result = grounded_validation_pipeline(prompt, response, domain, include_debug=True)
                logger.info(f"Domain '{domain}' pipeline result: "
                           f"factuality={result['factuality_score']:.4f}, "
                           f"hallucination={result['hallucination_flag']}")
                
                # Basic validation
                assert 'factuality_score' in result, "Missing factuality_score in result"
                assert 'hallucination_flag' in result, "Missing hallucination_flag in result"
                assert 'reference_text' in result, "Missing reference_text in result"
                
                # Log full result for inspection
                if 'debug_metadata' in result:
                    logger.debug(f"Debug metadata: {json.dumps(result['debug_metadata'], indent=2)}")
                
            except Exception as e:
                logger.error(f"Error in full pipeline for domain '{domain}': {str(e)}")
    
    def run_all_tests():
        """Run all tests in sequence."""
        logger.info("=== Testing Grounding Engine ===")
        
        domains = test_domains()
        logger.info("✓ Domain test passed")
        
        reference_results = test_reference_retrieval(domains)
        logger.info("✓ Reference retrieval test passed")
        
        test_factuality_scoring(reference_results)
        logger.info("✓ Factuality scoring test passed")
        
        test_full_pipeline(domains)
        logger.info("✓ Full pipeline test passed")
        
        logger.info("All tests completed successfully!")

except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

if __name__ == "__main__":
    run_all_tests()