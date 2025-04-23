#!/usr/bin/env python3
"""
ACDEK Simple SDK Example.

Demonstrates the basic usage of the ACDEK SDK client.
"""
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdk.acdek_client import Client

def main():
    # Create sample input data
    input_data = {
        "text": "This is a sample text for ACDEK processing",
        "metadata": {
            "source": "example",
            "category": "test"
        }
    }
    
    # Initialize the SDK client
    client = Client(config={"fallback_mode": "relaxed"})
    
    # Run the pipeline
    print("=== Running ACDEK Pipeline ===")
    result = client.full_pipeline(input_data)
    
    # Pretty print the result
    print("\n=== Result ===")
    print(f"Enrichment Success: {result['enrichment']['success']}")
    print(f"Constraints Passed: {result['constraints'].get('passed', False)}")
    print(f"Confidence Score: {result['confidence']['aggregate_score']:.3f}")
    
    # Show detailed confidence factors
    print("\n=== Confidence Factors ===")
    for factor, score in result['confidence']['factors'].items():
        if isinstance(score, float):
            print(f" - {factor}: {score:.3f}")
        else:
            print(f" - {factor}: {score}")

if __name__ == '__main__':
    main()