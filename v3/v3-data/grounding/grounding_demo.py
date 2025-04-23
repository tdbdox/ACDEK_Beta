#!/usr/bin/env python3
"""
Simplified Grounding Engine Demo

A standalone demo of the grounding engine with LLM integration
that avoids the dependencies on the full ACDEK kernel.
"""

import os
import sys
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
        logging.FileHandler("grounding_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add parent directory to path to enable imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

try:
    # Import OpenAI adapter
    from v3.llm_adapters.openai_adapter import OpenAIAdapter
except ImportError:
    from openai import OpenAI
    
    # Create a simple adapter class if import fails
    class OpenAIAdapter:
        def __init__(self, api_key=None, model="gpt-4", temperature=0.7):
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.model = model
            self.temperature = temperature
            self.client = OpenAI(api_key=self.api_key) if self.api_key else None
            self.request_count = 0
            self.total_tokens = 0
            self.total_latency = 0
            
        def call_llm(self, prompt):
            if not self.api_key or not self.client:
                raise RuntimeError("OpenAI API key not available")
                
            start_time = time.time()
            self.request_count += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1000
            )
            
            latency = time.time() - start_time
            self.total_latency += latency
            self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
            
        def call_with_accuracy_focus(self, prompt):
            if not self.api_key or not self.client:
                raise RuntimeError("OpenAI API key not available")
                
            start_time = time.time()
            self.request_count += 1
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a highly accurate assistant focused on providing factual information. Always prioritize accuracy and indicate when you're uncertain."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1000
            )
            
            latency = time.time() - start_time
            self.total_latency += latency
            self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
            
        def get_metrics(self):
            avg_latency = self.total_latency / max(1, self.request_count)
            avg_tokens = self.total_tokens / max(1, self.request_count)
            
            return {
                "request_count": self.request_count,
                "total_tokens": self.total_tokens,
                "total_latency": self.total_latency,
                "avg_latency_seconds": avg_latency,
                "avg_tokens_per_request": avg_tokens,
                "model": self.model
            }

try:
    # Import grounding engine functions directly
    sys.path.append(str(Path(__file__).resolve().parent))
    from grounding_engine import (
        grounded_validation_pipeline,
        retrieve_reference,
        score_factuality,
        get_available_domains,
        clear_reference_cache,
        FACTUALITY_THRESHOLD,
        DOMAIN_THRESHOLDS
    )
except ImportError as e:
    logger.error(f"Failed to import grounding engine: {e}")
    print(f"Error: {e}")
    print("The grounding engine module is required for this demo.")
    sys.exit(1)

class GroundingDemo:
    """
    Simple demo class for the grounding engine with LLM integration.
    """
    def __init__(
        self,
        llm_adapter: OpenAIAdapter,
        domain_mapping: Optional[Dict[str, str]] = None,
        include_debug: bool = True
    ):
        """
        Initialize the grounding demo.
        
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
        try:
            self.available_domains = get_available_domains()
            logger.info(f"Available domains for grounding: {self.available_domains}")
        except Exception as e:
            logger.error(f"Error getting available domains: {e}")
            self.available_domains = ["financial", "legal", "medical"]
        
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
    
    def process_query(
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
    print("ACDEK v3 Grounding Engine Demo")
    print("=======================================\n")
    
    # Initialize OpenAI adapter
    adapter = OpenAIAdapter(api_key=api_key, model=model)
    
    # Check if API key is available
    if not adapter.api_key:
        print("Error: No OpenAI API key available. Set OPENAI_API_KEY environment variable or pass --api-key.")
        return
    
    print(f"Using OpenAI model: {model}")
    print(f"Accuracy mode: {accuracy_mode}")
    
    # Initialize demo
    demo = GroundingDemo(
        llm_adapter=adapter,
        include_debug=True
    )
    
    print(f"Available domains: {demo.available_domains}")
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
            metrics = demo.get_metrics()
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
            if new_domain in demo.available_domains:
                current_domain = new_domain
                print(f"Domain set to: {current_domain}")
            else:
                print(f"Invalid domain: {new_domain}")
                print(f"Available domains: {demo.available_domains}")
            continue
        
        # Process the prompt
        print("\nProcessing...")
        result = demo.process_query(
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

def run_example_query(
    api_key: Optional[str] = None,
    model: str = "gpt-4",
    accuracy_mode: bool = True,
    query: str = "What is an asset in financial terms?",
    domain: Optional[str] = "financial"
) -> Dict[str, Any]:
    """
    Run a single example query and return the result.
    
    Args:
        api_key: OpenAI API key
        model: Model to use
        accuracy_mode: Whether to focus on accuracy
        query: The query to process
        domain: Optional domain override
        
    Returns:
        Query result
    """
    print(f"\nProcessing query: {query}")
    print(f"Domain: {domain}")
    
    # Initialize OpenAI adapter
    adapter = OpenAIAdapter(api_key=api_key, model=model)
    
    # Check if API key is available
    if not adapter.api_key:
        print("Error: No OpenAI API key available.")
        return {"error": "No OpenAI API key available"}
    
    # Initialize demo
    demo = GroundingDemo(
        llm_adapter=adapter,
        include_debug=True
    )
    
    # Process the query
    result = demo.process_query(
        prompt=query,
        domain=domain,
        accuracy_focus=accuracy_mode
    )
    
    # Print the result
    if "error" in result:
        print(f"\nError: {result['error']}")
        return result
    
    print("\n--- LLM Response ---")
    print(result["response"])
    
    print("\n--- Grounding Analysis ---")
    print(f"Domain: {result['domain']}")
    print(f"Factuality Score: {result['factuality_score']:.4f}")
    print(f"Threshold: {result['threshold_used']:.4f}")
    print(f"Hallucination Detected: {result['hallucination_detected']}")
    print(f"Total Processing Time: {result['timing']['total_seconds']:.2f}s")
    
    # Print debug info if available
    if "debug_metadata" in result and result["debug_metadata"]:
        print("\n--- Debug Info ---")
        debug = result["debug_metadata"]
        if "reference_similarity" in debug:
            print(f"Reference Similarity: {debug['reference_similarity']:.4f}")
            
        # Check for factuality details
        if "factuality_details" in debug:
            facts = debug["factuality_details"]
            print("\nFactuality Details:")
            print(f"  Vector Similarity: {facts.get('vector_similarity', 0):.4f}")
            print(f"  Has Contradiction: {facts.get('has_contradiction', False)}")
            
            if facts.get('has_contradiction', False):
                print(f"  Contradiction Confidence: {facts.get('contradiction_confidence', 0):.4f}")
                print(f"  Contradiction Reason: {facts.get('contradiction_reason', '')}")
                
        if "constraint_results" in debug:
            print(f"\nConstraint Results: {debug['constraint_results']}")
        
        # Let's manually check for contradiction markers
        from sys import path as sys_path
        sys_path.append(str(Path(__file__).resolve().parent))
        
        try:
            # Import the enhanced contains_contradiction function
            from grounding_engine import contains_contradiction
            
            # Check for contradictions with enhanced detection
            text = result["response"]
            reference = result["reference_text"]
            has_contradiction, confidence, reason = contains_contradiction(text, reference)
            
            print(f"\nEnhanced Contradiction Analysis:")
            print(f"  Contains Contradiction: {has_contradiction}")
            if has_contradiction:
                print(f"  Confidence: {confidence:.4f}")
                print(f"  Reason: {reason}")
            
        except ImportError as e:
            print(f"Could not run contradiction analysis: {str(e)}")
    
    print("\n--- Reference ---")
    print(result["reference_text"])
    
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grounding Engine Demo")
    parser.add_argument("--api-key", help="OpenAI API key (defaults to OPENAI_API_KEY environment variable)")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use (default: gpt-4)")
    parser.add_argument("--accuracy-mode", action="store_true", help="Focus on accuracy in LLM generation")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--query", default="What is an asset in financial terms?", help="Query to process in non-interactive mode")
    parser.add_argument("--domain", default="financial", help="Domain for the query")
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_demo(
            api_key=args.api_key,
            model=args.model,
            accuracy_mode=args.accuracy_mode
        )
    else:
        run_example_query(
            api_key=args.api_key,
            model=args.model,
            accuracy_mode=args.accuracy_mode,
            query=args.query,
            domain=args.domain
        )