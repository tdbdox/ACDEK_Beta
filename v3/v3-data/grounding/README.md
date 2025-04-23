# ACDEK v3 Grounding Engine

The Grounding Engine module for ACDEK v3 provides factuality validation and hallucination detection using a hybrid system combining Retrieval-Augmented Generation (RAG) with constraint validation.

## Features

- **Vector-based retrieval** from domain-specific reference datasets
- **Semantic similarity scoring** for factuality validation
- **Efficient caching** of reference data and embeddings
- **Fallback mechanisms** when no high-confidence match is found
- **Integration with constraint validation** pipeline
- **Clean modular design** with well-defined interfaces

## Usage

```python
from v3.grounding import (
    grounded_validation_pipeline,
    retrieve_reference,
    score_factuality,
    build_grounding_context
)

# Basic retrieval and scoring
domain = "medical"
prompt = "What causes high blood pressure?"
response = "High blood pressure can be caused by various factors including diet and genetics."

# Get reference content
reference = retrieve_reference(prompt, domain)
print(f"Reference: {reference['text']}")
print(f"Similarity: {reference['similarity']}")

# Score factuality
score = score_factuality(response, reference['text'])
print(f"Factuality score: {score}")

# Full validation pipeline
result = grounded_validation_pipeline(prompt, response, domain, include_debug=True)
print(f"Hallucination detected: {result['hallucination_flag']}")
print(f"Factuality score: {result['factuality_score']}")
print(f"Constraint pass: {result['constraint_pass']}")
```

## Module Structure

- `grounding_engine.py` - Core implementation of grounding and factuality validation
- `__init__.py` - Package initialization and exports
- `test_grounding.py` - Test script for validating the implementation

## Domain Reference Data

The module works with domain-specific reference data located in the `v3/data/` directory:

- `medical.json` - Medical domain reference data
- `legal.json` - Legal domain reference data
- `finance.json` - Financial domain reference data

Each reference file contains an array of objects with:
- `id` - Unique identifier for the reference
- `text` - Reference text content for factual validation

## Integration Points

- **ACDEK v3 Kernel** - Can be used directly through the kernel for factuality validation
- **Hallucination Benchmarking** - Works with the benchmarking suite for measuring hallucination rates
- **Constraint Validation** - Integrates with the constraint validation system

## Performance Considerations

- The module uses vector embedding caching for improved performance
- Reference data is cached in memory once loaded
- Each domain file is loaded only when needed
- The module supports multiple domains with fallback strategies

## License

This module is part of the ACDEK v3 system and is subject to the same licensing terms.