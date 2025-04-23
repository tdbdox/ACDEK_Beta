#!/usr/bin/env python3
"""
Hallucination Edge Case Analysis for Grounding Engine

This script tests the grounding engine against specific edge cases
to analyze its sensitivity to different types of hallucinations:
1. Minor factual errors - small inaccuracies that might be hard to catch
2. Fabrications - completely made-up information
3. Partial truths - mixture of accurate and inaccurate information 
4. Implied falsehoods - statements that imply incorrect information
5. Confident uncertainties - high confidence in uncertain information
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
        logging.FileHandler("grounding_edgecases.log"),
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
    
    # Define specific edge cases to test
    EDGE_CASES = [
        # FINANCIAL DOMAIN EDGE CASES
        # 1. Minor factual error - slightly incorrect definition
        {
            "id": "fin-edge-1",
            "domain": "financial",
            "prompt": "What is an asset in financial terms?",
            "hallucination_type": "minor_factual_error",
            "reference_id": "fin1",
            "crafted_response": "An asset is a resource with financial value that an individual, corporation, or country owns or controls with the expectation it will provide future profit. Assets are always reported on the left side of a balance sheet."
        },
        # 2. Fabrication - completely made up detail
        {
            "id": "fin-edge-2",
            "domain": "financial",
            "prompt": "How is equity calculated in a company?",
            "hallucination_type": "fabrication",
            "reference_id": "fin3",
            "crafted_response": "Equity represents ownership in a company and is calculated using the Hill-Barton formula, which takes the difference between total assets and total liabilities, then applies a valuation multiplier based on industry standards set by the International Equity Board."
        },
        # 3. Partial truth - mix of accurate and inaccurate
        {
            "id": "fin-edge-3",
            "domain": "financial",
            "prompt": "What does an income statement show?",
            "hallucination_type": "partial_truth",
            "reference_id": "fin12",
            "crafted_response": "An income statement is a financial document that shows a company's revenues and expenses over a specific period, indicating how the revenues are transformed into net income or net profit. It also tracks cash flows, asset depreciation schedules, and includes a company's balance sheet accounts at the end."
        },
        
        # LEGAL DOMAIN EDGE CASES
        # 4. Implied falsehood - implies incorrect information
        {
            "id": "leg-edge-1",
            "domain": "legal",
            "prompt": "What are the requirements for a valid contract?",
            "hallucination_type": "implied_falsehood",
            "reference_id": "leg1",
            "crafted_response": "A contract is a legally binding agreement between two or more parties that creates obligations enforceable by law. Valid contracts require offer, acceptance, consideration, legal capacity, and lawful purpose. Unlike many other legal documents, contracts don't typically require witnesses or notarization to be considered valid in most jurisdictions."
        },
        # 5. Confident uncertainty - authoritative but wrong
        {
            "id": "leg-edge-2",
            "domain": "legal",
            "prompt": "What is tort law?",
            "hallucination_type": "confident_uncertainty",
            "reference_id": "leg3",
            "crafted_response": "Tort law is definitively a body of civil law that addresses wrongful acts resulting in injury or harm to another person, property, reputation, or other legally protected interests, providing remedies through civil litigation. In all 50 states, tort law is governed by the Uniform Tort Standards Act which standardizes damage calculations."
        },
        
        # MEDICAL DOMAIN EDGE CASES
        # 6. Subtle numerical error - wrong numbers that seem plausible
        {
            "id": "med-edge-1",
            "domain": "medical",
            "prompt": "What is hypertension and how is it diagnosed?",
            "hallucination_type": "numerical_error",
            "reference_id": "med2",
            "crafted_response": "Hypertension (high blood pressure) is a chronic condition where blood pressure against artery walls is consistently elevated. It is diagnosed when readings consistently exceed 145/95 mmHg. Untreated hypertension increases risk of heart disease, stroke, and kidney failure."
        },
        # 7. Outdated information - was true but no longer is
        {
            "id": "med-edge-2",
            "domain": "medical",
            "prompt": "What is diabetes mellitus?",
            "hallucination_type": "outdated_information",
            "reference_id": "med3",
            "crafted_response": "Diabetes mellitus is a metabolic disorder characterized by high blood sugar levels (hyperglycemia) over a prolonged period. The main types are Type 1 (insulin-dependent) and Type 2 (non-insulin-dependent). According to the latest medical consensus, diabetes is managed primarily through strict dietary control with medication as a secondary intervention."
        }
    ]
    
    @dataclass
    class EdgeCaseResult:
        """Container for edge case test results."""
        case_id: str
        domain: str
        prompt: str
        hallucination_type: str
        crafted_response: str
        factuality_score: float
        hallucination_flag: bool
        threshold_used: float
        reference_text: str
        processing_time_ms: float
        llm_generated_response: Optional[str] = None
        additional_metadata: Dict[str, Any] = field(default_factory=dict)
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary for serialization."""
            result = {
                "case_id": self.case_id,
                "domain": self.domain,
                "prompt": self.prompt,
                "hallucination_type": self.hallucination_type,
                "crafted_response": self.crafted_response,
                "factuality_score": self.factuality_score,
                "hallucination_flag": self.hallucination_flag,
                "threshold_used": self.threshold_used,
                "reference_text": self.reference_text,
                "processing_time_ms": self.processing_time_ms,
                "additional_metadata": self.additional_metadata
            }
            
            if self.llm_generated_response:
                result["llm_generated_response"] = self.llm_generated_response
                
            return result
    
    def test_edge_case(
        case: Dict[str, Any],
        adapter: Optional[OpenAIAdapter] = None,
        include_llm_variant: bool = False,
        include_debug: bool = True
    ) -> EdgeCaseResult:
        """
        Test a specific edge case with the grounding engine.
        
        Args:
            case: Edge case definition
            adapter: Optional OpenAI adapter for LLM variants
            include_llm_variant: Whether to include an LLM-generated variant
            include_debug: Whether to include debug metadata
            
        Returns:
            Edge case test result
        """
        domain = case["domain"]
        prompt = case["prompt"]
        case_id = case["id"]
        crafted_response = case["crafted_response"]
        hallucination_type = case["hallucination_type"]
        
        # Get reference details
        reference = retrieve_reference(prompt, domain)
        reference_text = reference["text"]
        
        # Test the crafted response
        start_time = time.time()
        result = grounded_validation_pipeline(
            prompt, crafted_response, domain, include_debug=include_debug
        )
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Create result
        edge_case_result = EdgeCaseResult(
            case_id=case_id,
            domain=domain,
            prompt=prompt,
            hallucination_type=hallucination_type,
            crafted_response=crafted_response,
            factuality_score=result.get("factuality_score", 0),
            hallucination_flag=result.get("hallucination_flag", True),
            threshold_used=DOMAIN_THRESHOLDS.get(domain, FACTUALITY_THRESHOLD),
            reference_text=reference_text,
            processing_time_ms=processing_time,
            additional_metadata={
                "constraint_pass": result.get("constraint_pass", False),
                "debug": result.get("debug_metadata", {}) if include_debug else {}
            }
        )
        
        # If requested, also test with an LLM-generated version
        if include_llm_variant and adapter and adapter.api_key:
            try:
                # Craft a prompt to generate a similar response with the hallucination type
                generation_prompt = f"""
                Generate a response about the following topic: {prompt}
                
                Your response should include a subtle {hallucination_type.replace('_', ' ')} - 
                meaning it should contain some information that's not entirely accurate.
                
                Keep your response concise (3-5 sentences) and make the inaccuracy subtle but 
                present. Make it sound authoritative and factual.
                """
                
                # Generate response
                llm_response = adapter.call_llm(generation_prompt)
                
                # Test the LLM response
                start_time = time.time()
                llm_result = grounded_validation_pipeline(
                    prompt, llm_response, domain, include_debug=include_debug
                )
                llm_processing_time = (time.time() - start_time) * 1000  # ms
                
                # Add to result
                edge_case_result.llm_generated_response = llm_response
                edge_case_result.additional_metadata["llm_variant"] = {
                    "factuality_score": llm_result.get("factuality_score", 0),
                    "hallucination_flag": llm_result.get("hallucination_flag", True),
                    "processing_time_ms": llm_processing_time,
                    "constraint_pass": llm_result.get("constraint_pass", False),
                    "debug": llm_result.get("debug_metadata", {}) if include_debug else {}
                }
                
            except Exception as e:
                logger.error(f"Error generating LLM variant for {case_id}: {str(e)}")
                edge_case_result.additional_metadata["llm_variant_error"] = str(e)
        
        return edge_case_result
    
    def analyze_edge_cases(results: List[EdgeCaseResult]) -> Dict[str, Any]:
        """
        Analyze edge case results for insights into grounding engine performance.
        
        Args:
            results: List of edge case results
            
        Returns:
            Dictionary with analysis results
        """
        if not results:
            return {}
        
        # Group by hallucination type
        by_type = {}
        for result in results:
            h_type = result.hallucination_type
            if h_type not in by_type:
                by_type[h_type] = []
            by_type[h_type].append(result)
        
        # Group by domain
        by_domain = {}
        for result in results:
            domain = result.domain
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(result)
        
        # Calculate overall detection rate
        total_cases = len(results)
        detected_hallucinations = sum(1 for r in results if r.hallucination_flag)
        overall_detection_rate = detected_hallucinations / total_cases if total_cases > 0 else 0
        
        # Calculate detection rates by type
        detection_by_type = {}
        for h_type, type_results in by_type.items():
            if not type_results:
                continue
                
            type_count = len(type_results)
            type_detected = sum(1 for r in type_results if r.hallucination_flag)
            type_rate = type_detected / type_count if type_count > 0 else 0
            type_scores = [r.factuality_score for r in type_results]
            type_avg_score = sum(type_scores) / len(type_scores) if type_scores else 0
            
            detection_by_type[h_type] = {
                "count": type_count,
                "detected": type_detected,
                "detection_rate": type_rate,
                "avg_factuality_score": type_avg_score
            }
        
        # Calculate detection rates by domain
        detection_by_domain = {}
        for domain, domain_results in by_domain.items():
            if not domain_results:
                continue
                
            domain_count = len(domain_results)
            domain_detected = sum(1 for r in domain_results if r.hallucination_flag)
            domain_rate = domain_detected / domain_count if domain_count > 0 else 0
            domain_scores = [r.factuality_score for r in domain_results]
            domain_avg_score = sum(domain_scores) / len(domain_scores) if domain_scores else 0
            
            # Find hardest and easiest cases for this domain
            domain_results_sorted = sorted(domain_results, key=lambda r: r.factuality_score)
            hardest_case = domain_results_sorted[0] if domain_results_sorted else None
            easiest_case = domain_results_sorted[-1] if domain_results_sorted else None
            
            detection_by_domain[domain] = {
                "count": domain_count,
                "detected": domain_detected,
                "detection_rate": domain_rate,
                "avg_factuality_score": domain_avg_score,
                "hardest_case": hardest_case.case_id if hardest_case else None,
                "hardest_case_type": hardest_case.hallucination_type if hardest_case else None,
                "hardest_case_score": hardest_case.factuality_score if hardest_case else None,
                "easiest_case": easiest_case.case_id if easiest_case else None,
                "easiest_case_type": easiest_case.hallucination_type if easiest_case else None,
                "easiest_case_score": easiest_case.factuality_score if easiest_case else None
            }
        
        # Check for LLM variant results
        llm_variants_count = sum(1 for r in results if "llm_variant" in r.additional_metadata)
        llm_analysis = {}
        
        if llm_variants_count > 0:
            llm_detected = sum(1 for r in results 
                              if "llm_variant" in r.additional_metadata 
                              and r.additional_metadata["llm_variant"].get("hallucination_flag", False))
            llm_detection_rate = llm_detected / llm_variants_count if llm_variants_count > 0 else 0
            
            # Compare detection rates
            crafted_detected = sum(1 for r in results[:llm_variants_count] if r.hallucination_flag)
            crafted_detection_rate = crafted_detected / llm_variants_count if llm_variants_count > 0 else 0
            
            llm_analysis = {
                "count": llm_variants_count,
                "detected": llm_detected,
                "detection_rate": llm_detection_rate,
                "crafted_detection_rate": crafted_detection_rate,
                "detection_rate_difference": llm_detection_rate - crafted_detection_rate
            }
        
        return {
            "total_cases": total_cases,
            "total_detected": detected_hallucinations,
            "overall_detection_rate": overall_detection_rate,
            "by_hallucination_type": detection_by_type,
            "by_domain": detection_by_domain,
            "llm_variant_analysis": llm_analysis if llm_variants_count > 0 else None
        }
    
    def run_edge_case_analysis(
        api_key: Optional[str] = None,
        include_llm_variants: bool = False,
        output_dir: str = "./edge_case_results",
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run analysis on edge cases for hallucination detection.
        
        Args:
            api_key: OpenAI API key for LLM variants
            include_llm_variants: Whether to include LLM-generated variants
            output_dir: Output directory
            verbose: Whether to print detailed results
            
        Returns:
            Analysis results dictionary
        """
        print("\n=======================================")
        print("ACDEK v3 Grounding Engine - Edge Case Analysis")
        print("=======================================\n")
        
        # Initialize OpenAI adapter if needed
        adapter = None
        if include_llm_variants:
            adapter = OpenAIAdapter(api_key=api_key)
            
            if not adapter.api_key:
                logger.warning("No OpenAI API key available. LLM variants will be skipped.")
                include_llm_variants = False
            else:
                print(f"Using OpenAI model for LLM variants: {adapter.model}")
        
        print(f"Number of edge cases: {len(EDGE_CASES)}")
        print(f"Include LLM variants: {include_llm_variants}")
        print("\nRunning edge case analysis...")
        
        results = []
        
        # Test each edge case
        for i, case in enumerate(EDGE_CASES):
            print(f"\nEdge case {i+1}/{len(EDGE_CASES)}: {case['id']} - {case['hallucination_type']}")
            
            if verbose:
                print(f"Prompt: {case['prompt']}")
                print(f"Crafted response: {case['crafted_response']}")
            
            # Run the test
            result = test_edge_case(case, adapter, include_llm_variants)
            results.append(result)
            
            if verbose:
                print(f"Factuality Score: {result.factuality_score:.4f}")
                print(f"Hallucination Detected: {result.hallucination_flag}")
                print(f"Processing Time: {result.processing_time_ms:.2f}ms")
                
                if result.llm_generated_response and verbose:
                    print(f"\nLLM-generated variant:")
                    print(result.llm_generated_response)
                    llm_data = result.additional_metadata.get("llm_variant", {})
                    print(f"LLM Factuality Score: {llm_data.get('factuality_score', 0):.4f}")
                    print(f"LLM Hallucination Detected: {llm_data.get('hallucination_flag', False)}")
        
        # Analyze results
        analysis = analyze_edge_cases(results)
        
        # Print summary
        print("\n=======================================")
        print("Edge Case Analysis Summary")
        print("=======================================")
        print(f"Total edge cases: {analysis['total_cases']}")
        print(f"Overall detection rate: {analysis['overall_detection_rate']:.2%}")
        
        print("\nDetection by Hallucination Type:")
        for h_type, stats in analysis['by_hallucination_type'].items():
            print(f"  {h_type.replace('_', ' ').capitalize()}:")
            print(f"    Count: {stats['count']}")
            print(f"    Detection Rate: {stats['detection_rate']:.2%}")
            print(f"    Avg Factuality Score: {stats['avg_factuality_score']:.4f}")
        
        print("\nDetection by Domain:")
        for domain, stats in analysis['by_domain'].items():
            print(f"  {domain.capitalize()}:")
            print(f"    Count: {stats['count']}")
            print(f"    Detection Rate: {stats['detection_rate']:.2%}")
            print(f"    Avg Factuality Score: {stats['avg_factuality_score']:.4f}")
            print(f"    Hardest Case: {stats['hardest_case']} ({stats['hardest_case_type']}) - {stats['hardest_case_score']:.4f}")
        
        if analysis['llm_variant_analysis']:
            llm_stats = analysis['llm_variant_analysis']
            print("\nLLM Variant Analysis:")
            print(f"  Count: {llm_stats['count']}")
            print(f"  Detection Rate: {llm_stats['detection_rate']:.2%}")
            print(f"  Crafted Detection Rate: {llm_stats['crafted_detection_rate']:.2%}")
            print(f"  Difference: {llm_stats['detection_rate_difference']:.2%}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        results_file = os.path.join(output_dir, f"edge_case_results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
            
        analysis_file = os.path.join(output_dir, f"edge_case_analysis_{timestamp}.json")
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)
            
        print(f"\nResults saved to {results_file}")
        print(f"Analysis saved to {analysis_file}")
        
        return analysis

except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    print(f"Error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze edge cases for hallucination detection")
    parser.add_argument("--api-key", help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)")
    parser.add_argument("--include-llm-variants", action="store_true", help="Include LLM-generated variants")
    parser.add_argument("--output-dir", default="./edge_case_results", help="Output directory for results")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    run_edge_case_analysis(
        api_key=args.api_key,
        include_llm_variants=args.include_llm_variants,
        output_dir=args.output_dir,
        verbose=args.verbose
    )