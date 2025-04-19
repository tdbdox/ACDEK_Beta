# ACDEK v3 Grounding Engine LLM Integration

This documentation explains how to integrate and test the ACDEK v3 grounding engine with real-world LLM responses to validate its effectiveness at identifying hallucinations and improving factuality.

## Overview

The grounding engine has been designed and tested with reference datasets, but real-world usage requires integration with actual LLM outputs which may contain more subtle and complex hallucinations. This integration suite provides:

1. **LLM Integration Testing**: A framework to evaluate the grounding engine against responses from OpenAI's API
2. **Edge Case Analysis**: Targeted testing of edge cases challenging for hallucination detection
3. **Kernel Integration**: End-to-end integration with the ACDEK v3 kernel
4. **Interactive Demo**: A command-line demo for interactive testing

## Getting Started

### Prerequisites

- OpenAI API Key (set as environment variable `OPENAI_API_KEY` or pass via `--api-key`)
- Python 3.7+
- All ACDEK v3 dependencies installed

### Directory Structure

```
v3/v3/grounding/
├── grounding_engine.py        # Core grounding engine
├── llm_integration_test.py    # LLM integration test suite
├── hallucination_edge_cases.py # Edge case analysis
├── kernel_integration.py      # ACDEK kernel integration
└── LLM_INTEGRATION_README.md  # This documentation
```

## Running the Tests

### 1. LLM Integration Test

This test evaluates how well the grounding engine identifies hallucinations in real LLM responses across multiple domains (financial, legal, medical) with different response types (accuracy-focused, creativity-focused, hallucination-prone).

```bash
python -m v3.v3.grounding.llm_integration_test --model gpt-4 --output-dir ./test_output
```

Optional arguments:
- `--api-key APIKEY`: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
- `--model MODEL`: OpenAI model to use (default: gpt-4)
- `--output-dir DIR`: Output directory for results (default: ./test_output)
- `-v, --verbose`: Enable verbose output

### 2. Edge Case Analysis

This test evaluates the grounding engine against specific edge cases that can be challenging for hallucination detection, such as subtle factual errors or partial truths.

```bash
python -m v3.v3.grounding.hallucination_edge_cases --output-dir ./edge_case_results
```

Optional arguments:
- `--api-key APIKEY`: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
- `--include-llm-variants`: Include LLM-generated variants alongside crafted examples
- `--output-dir DIR`: Output directory for results (default: ./edge_case_results)
- `-v, --verbose`: Enable verbose output

### 3. Interactive Kernel Integration

This provides an interactive demo of the grounding engine integrated with the ACDEK kernel:

```bash
python -m v3.v3.grounding.kernel_integration --interactive --accuracy-mode
```

Optional arguments:
- `--api-key APIKEY`: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
- `--model MODEL`: OpenAI model to use (default: gpt-4)
- `--accuracy-mode`: Focus on accuracy in LLM generation (recommended)

### 4. Example Queries

To run a set of predefined example queries:

```bash
python -m v3.v3.grounding.kernel_integration --examples --output-dir ./integration_results
```

Optional arguments:
- `--api-key APIKEY`: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
- `--model MODEL`: OpenAI model to use (default: gpt-4)
- `--output-dir DIR`: Output directory for results (default: ./integration_results)

## Integration Architecture

The integration follows a layered architecture:

1. **LLM Adapter Layer**: `OpenAIAdapter` class provides standardized access to OpenAI API
2. **Grounding Engine Layer**: Core factuality scoring and hallucination detection
3. **Integration Layer**: `GroundingEnrichmentPipeline` class connects LLM generation to grounding 
4. **ACDEK Kernel Layer**: Full kernel integration for production use

## Expected Results

The test suite aims to validate that:

1. The grounding engine correctly identifies hallucinations in LLM responses
2. Different LLM response modes (accuracy vs. creativity) show expected differences in hallucination rates
3. Domain-specific thresholds effectively balance precision and recall
4. Edge cases demonstrate the strengths and limitations of the current implementation

## Metrics and Analysis

The test suite calculates several key metrics:

- **Detection Rate**: Percentage of hallucinations correctly identified
- **Domain-Specific Performance**: How detection varies across domains
- **Type-Specific Performance**: How detection varies across hallucination types
- **LLM Metrics**: Token usage, latency, and other API-related statistics

## Extending the Tests

### Adding New Test Cases

To add new test cases to the integration test, modify the `TEST_CASES` list in `llm_integration_test.py`:

```python
{
    "id": "your-unique-id",
    "domain": "domain-name",  # financial, legal, medical
    "prompt": "Your test prompt",
    "reference_id": "ref-id"  # ID of the reference in domain data file
}
```

### Adding New Edge Cases

To add new edge cases, modify the `EDGE_CASES` list in `hallucination_edge_cases.py`:

```python
{
    "id": "edge-unique-id",
    "domain": "domain-name",
    "prompt": "Your test prompt",
    "hallucination_type": "type-name",  # e.g., minor_factual_error
    "reference_id": "ref-id",
    "crafted_response": "Your crafted response with hallucination"
}
```

## Common Issues

- **ImportError**: Ensure you're running the script from the project root directory
- **OpenAI API Key**: Ensure your API key is set correctly
- **Rate Limits**: Be mindful of OpenAI API rate limits when running tests

## Future Improvements

Planned improvements for the integration test suite:

1. **Multi-model Comparison**: Compare results across different LLM models
2. **Adaptive Testing**: Dynamically adjust test difficulty based on results
3. **Web Interface**: Add a web UI for easier interaction and visualization
4. **Continuous Benchmarking**: Integrate with CI/CD for ongoing performance tracking