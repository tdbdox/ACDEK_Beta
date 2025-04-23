#!/usr/bin/env python3
"""
ACDEK Confidence Scoring Example.

Demonstrates the VectorConfidenceScorer usage via SDK.
"""
import numpy as np
import time
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sdk.acdek_client import Client

def main():
    # Create a vector embedding (normally you would get this from a model)
    vector = np.random.rand(768)
    vector = vector / np.linalg.norm(vector)  # Normalize
    
    # Create example input with vector and metadata
    input_data = {
        "vector": vector,
        "model_name": "all-MiniLM-L6-v2",
        "expected_dimension": 768,
        "content_modality": "text",
        "query_vector": vector * 0.9,  # Similar but not identical
        "generation_time": time.time() - 3600,  # 1 hour ago
        "source": "database",
        "category": "product_description"
    }
    
    # Initialize the SDK client
    client = Client()
    
    # Get confidence score
    result = client.score_confidence(input_data)
    print("=== Confidence Score ===")
    print(f"Aggregate Score: {result['aggregate_score']:.3f}")
    print("\nFactor Scores:")
    for factor, score in result['factors'].items():
        if isinstance(score, float):
            print(f" - {factor}: {score:.3f}")
        else:
            print(f" - {factor}: {score}")
    
    print("\n=== Metadata ===")
    for key, value in result['metadata'].items():
        if key != 'vector':  # Skip vector in output for readability
            print(f" - {key}: {value}")

if __name__ == '__main__':
    main()