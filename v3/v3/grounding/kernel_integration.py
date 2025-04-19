#!/usr/bin/env python3
"""
ACDEK Kernel Integration with Grounding Engine

This script demonstrates how to integrate the grounding engine
with the ACDEK kernel for end-to-end factuality verification
in a production-ready pipeline.
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grounding_kernel_integration.log"),
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
    
    # Import ACDEK kernel
    try:
        from v3.acdek.acdek_kernel import ACDEKKernel
    except ImportError:
        from acdek.acdek_kernel import ACDEKKernel
    
    # Import enrichment components
    try:
        from v3.enrichment.constraint_verifier import ConstraintVerifier
        from v3.enrichment.hallucination_connector import HallucinationConnector
    except ImportError:
        try:
            from enrichment.constraint_verifier import ConstraintVerifier
            from enrichment.hallucination_connector import HallucinationConnector
        except ImportError:
            # Create stub classes if imports fail to allow minimal functionality
            class ConstraintVerifier:
                def __init__(self, **kwargs):
                    pass
                def validate(self, data):
                    return {"passed": True}
                    
            class HallucinationConnector:
                def __init__(self):
                    pass
    
    class GroundingEnrichmentPipeline:
        """
        Integration class that connects the grounding engine to the ACDEK kernel
        and provides an enrichment pipeline for factuality verification.
        """
        
        def __init__(
            self,
            llm_adapter: OpenAIAdapter,
            domain_mapping: Optional[Dict[str, str]] = None,
            include_debug: bool = True
        ):
            """
            Initialize the grounding enrichment pipeline.
            
            Args:
                llm_adapter: Initialized OpenAI adapter
                domain_mapping: Optional mapping from topics to domains
                include_debug: Whether to include debug metadata
            """
            self.llm_adapter = llm_adapter
            self.domain_mapping = domain_mapping or {
                "finance": "financial",
                "financial": "financial",
                "accounting": "financial",
                "investment": "financial",
                "legal": "legal",
                "law": "legal",
                "medical": "medical",
                "healthcare": "medical",
                "health": "medical"
            }
            self.include_debug = include_debug
            
            # Detect available domains
            self.available_domains = get_available_domains()
            logger.info(f"Available domains for grounding: {self.available_domains}")
            
            # Initialize kernel components
            self.kernel = ACDEKKernel()
            self.constraint_verifier = ConstraintVerifier()
            
            # Initialize hallucination connector
            self.hallucination_connector = HallucinationConnector()
            
            # Success/error counters
            self.total_requests = 0
            self.successful_requests = 0
            self.total_hallucinations = 0
        
        def detect_domain(self, prompt: str) -> str:
            """
            Detect the appropriate domain for a prompt.
            
            Args:
                prompt: User query
                
            Returns:
                Domain name or "general" if no specific domain detected
            """
            prompt_lower = prompt.lower()
            
            # Check for domain keywords
            for keyword, domain in self.domain_mapping.items():
                if keyword.lower() in prompt_lower:
                    if domain in self.available_domains:
                        return domain
            
            # If no domain detected, try to ask the LLM
            try:
                domain_prompt = f"""
                Determine the domain of the following question. Respond with just one word:
                'financial', 'legal', or 'medical'.
                
                Question: {prompt}
                
                Domain:
                """
                
                llm_domain = self.llm_adapter.call_llm(domain_prompt).strip().lower()
                
                # Check if the LLM's suggestion is valid
                for domain in self.available_domains:
                    if domain in llm_domain or llm_domain in domain:
                        return domain
            except Exception as e:
                logger.warning(f"Error detecting domain with LLM: {str(e)}")
            
            # Default to "financial" for now (adjust as needed)
            if "financial" in self.available_domains:
                return "financial"
            
            # Fall back to first available domain
            if self.available_domains:
                return self.available_domains[0]
            
            return "general"
        
        def process_with_grounding(
            self,
            prompt: str,
            domain: Optional[str] = None,
            accuracy_focus: bool = True
        ) -> Dict[str, Any]:
            """
            Process a prompt through LLM generation and grounding verification.
            
            Args:
                prompt: User query
                domain: Optional domain override
                accuracy_focus: Whether to focus on accuracy in LLM generation
                
            Returns:
                Dictionary with LLM response, grounding results, and metadata
            """
            self.total_requests += 1
            
            # Detect domain if not provided
            if not domain:
                domain = self.detect_domain(prompt)
                logger.info(f"Detected domain: {domain}")
            
            start_time = time.time()
            
            try:
                # 1. Generate LLM response
                if accuracy_focus:
                    response = self.llm_adapter.call_with_accuracy_focus(prompt)
                else:
                    response = self.llm_adapter.call_llm(prompt)
                
                llm_time = time.time() - start_time
                logger.info(f"LLM response generated in {llm_time:.2f}s")
                
                # 2. Apply grounding verification
                grounding_start = time.time()
                grounding_result = grounded_validation_pipeline(
                    prompt, response, domain, include_debug=self.include_debug
                )
                grounding_time = time.time() - grounding_start
                
                # Get reference for context
                reference = retrieve_reference(prompt, domain)
                reference_text = reference["text"]
                
                # 3. Create integrated result
                hallucination_flag = grounding_result.get("hallucination_flag", False)
                if hallucination_flag:
                    self.total_hallucinations += 1
                
                result = {
                    "prompt": prompt,
                    "response": response,
                    "domain": domain,
                    "hallucination_detected": hallucination_flag,
                    "factuality_score": grounding_result.get("factuality_score", 0.0),
                    "reference_text": reference_text,
                    "threshold_used": DOMAIN_THRESHOLDS.get(domain, FACTUALITY_THRESHOLD),
                    "timing": {
                        "llm_generation_seconds": llm_time,
                        "grounding_verification_seconds": grounding_time,
                        "total_seconds": time.time() - start_time
                    }
                }
                
                # Add debug metadata if requested
                if self.include_debug:
                    result["debug_metadata"] = grounding_result.get("debug_metadata", {})
                
                self.successful_requests += 1
                return result
                
            except Exception as e:
                logger.error(f"Error in grounding pipeline: {str(e)}")
                return {
                    "prompt": prompt,
                    "error": str(e),
                    "domain": domain,
                    "timing": {
                        "total_seconds": time.time() - start_time
                    }
                }
        
        def process_with_acdek_kernel(
            self,
            prompt: str,
            domain: Optional[str] = None,
            accuracy_focus: bool = True
        ) -> Dict[str, Any]:
            """
            Process a prompt through the full ACDEK kernel with grounding integration.
            
            Args:
                prompt: User query
                domain: Optional domain override
                accuracy_focus: Whether to focus on accuracy in LLM generation
                
            Returns:
                Dictionary with full ACDEK kernel results
            """
            # Detect domain if not provided
            if not domain:
                domain = self.detect_domain(prompt)
                logger.info(f"Detected domain: {domain}")
            
            try:
                # 1. Generate LLM response
                if accuracy_focus:
                    response = self.llm_adapter.call_with_accuracy_focus(prompt)
                else:
                    response = self.llm_adapter.call_llm(prompt)
                
                # 2. Prepare input for ACDEK kernel
                kernel_input = {
                    "prompt": prompt,
                    "response": response,
                    "domain": domain,
                    "require_grounding": True
                }
                
                # 3. Run kernel pipeline
                kernel_result = self.kernel.run(
                    input_data=kernel_input,
                    context={"domain": domain}
                )
                
                # 4. Extract and return relevant parts of the result
                return kernel_result
                
            except Exception as e:
                logger.error(f"Error in kernel pipeline: {str(e)}")
                return {
                    "prompt": prompt,
                    "error": str(e),
                    "domain": domain
                }
        
        def get_metrics(self) -> Dict[str, Any]:
            """Get usage metrics for the pipeline."""
            hallucination_rate = self.total_hallucinations / max(1, self.successful_requests)
            success_rate = self.successful_requests / max(1, self.total_requests)
            
            return {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "success_rate": success_rate,
                "hallucinations_detected": self.total_hallucinations,
                "hallucination_rate": hallucination_rate,
                "available_domains": self.available_domains,
                "llm_metrics": self.llm_adapter.get_metrics() if hasattr(self.llm_adapter, "get_metrics") else {}
            }
    
    def run_interactive_demo(
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        accuracy_mode: bool = True
    ) -> None:
        """
        Run an interactive demo of the grounding pipeline.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Model to use (defaults to gpt-4)
            accuracy_mode: Whether to focus on accuracy
        """
        print("\n=======================================")
        print("ACDEK v3 Grounding Engine - Interactive Demo")
        print("=======================================\n")
        
        # Initialize OpenAI adapter
        adapter = OpenAIAdapter(api_key=api_key, model=model)
        
        # Check if API key is available
        if not adapter.api_key:
            print("Error: No OpenAI API key available. Set OPENAI_API_KEY environment variable or pass --api-key.")
            return
        
        print(f"Using OpenAI model: {model}")
        print(f"Accuracy mode: {accuracy_mode}")
        
        # Initialize pipeline
        pipeline = GroundingEnrichmentPipeline(
            llm_adapter=adapter,
            include_debug=True
        )
        
        print(f"Available domains: {pipeline.available_domains}")
        print("\nType 'exit' or 'quit' to end the demo")
        print("Type 'metrics' to show pipeline metrics")
        print("Type 'domain:<domain>' to explicitly set the domain (e.g., 'domain:financial')")
        
        # Start interactive loop
        current_domain = None
        
        while True:
            # Get user prompt
            if current_domain:
                print(f"\n[Domain: {current_domain}]")
            prompt = input("\nEnter your prompt: ")
            
            # Check for special commands
            if prompt.lower() in ('exit', 'quit'):
                print("Exiting demo...")
                break
            
            if prompt.lower() == 'metrics':
                metrics = pipeline.get_metrics()
                print("\n--- Pipeline Metrics ---")
                print(f"Total requests: {metrics['total_requests']}")
                print(f"Success rate: {metrics['success_rate']:.2%}")
                print(f"Hallucination rate: {metrics['hallucination_rate']:.2%}")
                
                llm_metrics = metrics.get("llm_metrics", {})
                if llm_metrics:
                    print("\n--- LLM Metrics ---")
                    print(f"Total requests: {llm_metrics.get('request_count', 0)}")
                    print(f"Total tokens: {llm_metrics.get('total_tokens', 0)}")
                    print(f"Avg tokens per request: {llm_metrics.get('avg_tokens_per_request', 0):.1f}")
                    print(f"Avg latency: {llm_metrics.get('avg_latency_seconds', 0):.2f}s")
                
                continue
            
            # Check for domain command
            if prompt.lower().startswith('domain:'):
                new_domain = prompt.split(':', 1)[1].strip().lower()
                if new_domain in pipeline.available_domains:
                    current_domain = new_domain
                    print(f"Domain set to: {current_domain}")
                else:
                    print(f"Invalid domain: {new_domain}")
                    print(f"Available domains: {pipeline.available_domains}")
                continue
            
            # Process the prompt
            print("\nProcessing...")
            result = pipeline.process_with_grounding(
                prompt=prompt,
                domain=current_domain,
                accuracy_focus=accuracy_mode
            )
            
            # Print the result
            if "error" in result:
                print(f"\nError: {result['error']}")
                continue
            
            print("\n--- LLM Response ---")
            print(result["response"])
            
            print("\n--- Grounding Analysis ---")
            print(f"Domain: {result['domain']}")
            print(f"Factuality Score: {result['factuality_score']:.4f}")
            print(f"Threshold: {result['threshold_used']:.4f}")
            print(f"Hallucination Detected: {result['hallucination_detected']}")
            print(f"Total Processing Time: {result['timing']['total_seconds']:.2f}s")
            
            print("\n--- Reference ---")
            print(result["reference_text"])
            
            # Set detected domain as current domain for continuity
            current_domain = result["domain"]
    
    def run_example_queries(
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        output_dir: str = "./integration_results"
    ) -> None:
        """
        Run a set of example queries to demonstrate the pipeline.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Model to use (defaults to gpt-4)
            output_dir: Output directory for results
        """
        print("\n=======================================")
        print("ACDEK v3 Grounding Engine - Example Queries")
        print("=======================================\n")
        
        # Example queries across domains
        EXAMPLE_QUERIES = [
            {"prompt": "What is the difference between assets and liabilities?", "domain": "financial"},
            {"prompt": "How does equity financing work for startups?", "domain": "financial"},
            {"prompt": "What is the difference between a tort and a crime?", "domain": "legal"},
            {"prompt": "What are the requirements for a contract to be legally binding?", "domain": "legal"},
            {"prompt": "What are the symptoms of type 2 diabetes?", "domain": "medical"},
            {"prompt": "How is hypertension diagnosed and treated?", "domain": "medical"}
        ]
        
        # Initialize OpenAI adapter
        adapter = OpenAIAdapter(api_key=api_key, model=model)
        
        # Check if API key is available
        if not adapter.api_key:
            print("Error: No OpenAI API key available. Set OPENAI_API_KEY environment variable or pass --api-key.")
            return
        
        print(f"Using OpenAI model: {model}")
        print(f"Number of example queries: {len(EXAMPLE_QUERIES)}")
        
        # Initialize pipeline
        pipeline = GroundingEnrichmentPipeline(
            llm_adapter=adapter,
            include_debug=True
        )
        
        print(f"Available domains: {pipeline.available_domains}")
        
        # Process each query
        results = []
        print("\nRunning example queries...")
        
        for i, query in enumerate(EXAMPLE_QUERIES):
            print(f"\nQuery {i+1}/{len(EXAMPLE_QUERIES)}: {query['prompt']}")
            
            # Process in both accuracy mode and standard mode
            accurate_result = pipeline.process_with_grounding(
                prompt=query["prompt"],
                domain=query["domain"],
                accuracy_focus=True
            )
            
            standard_result = pipeline.process_with_grounding(
                prompt=query["prompt"],
                domain=query["domain"],
                accuracy_focus=False
            )
            
            # Save both results
            results.append({
                "query": query,
                "accuracy_mode": accurate_result,
                "standard_mode": standard_result
            })
            
            # Print summary
            print(f"Domain: {query['domain']}")
            print(f"Accuracy Mode - Factuality: {accurate_result.get('factuality_score', 0):.4f}, "
                  f"Hallucination: {accurate_result.get('hallucination_detected', True)}")
            print(f"Standard Mode - Factuality: {standard_result.get('factuality_score', 0):.4f}, "
                  f"Hallucination: {standard_result.get('hallucination_detected', True)}")
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(output_dir, f"example_results_{timestamp}.json")
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        
        # Print metrics
        metrics = pipeline.get_metrics()
        print("\n--- Pipeline Metrics ---")
        print(f"Total requests: {metrics['total_requests']}")
        print(f"Success rate: {metrics['success_rate']:.2%}")
        print(f"Hallucination rate: {metrics['hallucination_rate']:.2%}")
        
        llm_metrics = metrics.get("llm_metrics", {})
        if llm_metrics:
            print("\n--- LLM Metrics ---")
            print(f"Total requests: {llm_metrics.get('request_count', 0)}")
            print(f"Total tokens: {llm_metrics.get('total_tokens', 0)}")
            print(f"Avg tokens per request: {llm_metrics.get('avg_tokens_per_request', 0):.1f}")
            print(f"Avg latency: {llm_metrics.get('avg_latency_seconds', 0):.2f}s")

except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    print(f"Error: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACDEK Kernel Integration with Grounding Engine")
    parser.add_argument("--api-key", help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use (default: gpt-4)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--examples", action="store_true", help="Run example queries")
    parser.add_argument("--output-dir", default="./integration_results", help="Output directory for results")
    parser.add_argument("--accuracy-mode", action="store_true", help="Focus on accuracy in LLM generation")
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_demo(
            api_key=args.api_key,
            model=args.model,
            accuracy_mode=args.accuracy_mode
        )
    elif args.examples:
        run_example_queries(
            api_key=args.api_key,
            model=args.model,
            output_dir=args.output_dir
        )
    else:
        print("Please specify --interactive or --examples mode")
        parser.print_help()