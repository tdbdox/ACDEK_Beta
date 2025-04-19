# ACDEK Benchmark Suite

A comprehensive benchmarking framework for evaluating and comparing LLM performance with and without ACDEK enhancement across multiple dimensions.

## Overview

The ACDEK Benchmark Suite provides tools to quantitatively measure the improvements that ACDEK brings to raw LLM outputs across four critical dimensions:

1. **Reliability**: Accuracy, consistency, and repeatability of responses
2. **Hallucination Mitigation**: Reduction in factual errors and fabrications
3. **Uncertainty Handling**: Proper acknowledgment of knowledge boundaries and guessing elimination
4. **Execution Fidelity**: Task completion, instruction following, and constraint compliance

This suite consists of several components:
- General benchmarking for overall ACDEK improvements
- Specialized grounding benchmarking for factuality validation
- Comparative analysis tools for detailed evaluation
- Visualization dashboards for intuitive result interpretation

## Getting Started

### Prerequisites

- Python 3.8+
- ACDEK v3 properly installed
- Required packages (install via `pip install -r requirements.txt`)

### Running Benchmarks

The simplest way to run all benchmarks is using the combined runner script:

```bash
# Run all benchmarks
python run_benchmarks.py --all

# Run specific benchmarks
python run_benchmarks.py --general --grounding

# Generate dashboards from previous runs
python run_benchmarks.py --dashboard --input-files results/benchmark_results_20250418.json
```

### Configuration Options

Common options for all benchmarks:

- `--models`: Space-separated list of LLM models to test (default: "gpt-4")
- `--domains`: Space-separated list of domains to test (default: "medical financial legal")
- `--parallel`: Run benchmarks in parallel (default: sequential)
- `--max-workers`: Maximum number of parallel workers (default: 4)
- `--output-dir`: Directory to store results (default: "./benchmarks/results")

## Benchmark Types

### General Benchmark

Evaluates ACDEK's overall impact on LLM performance, measuring improvements in factuality, constraint adherence, and latency across multiple domains.

```bash
python benchmark_runner.py --models gpt-4 --domains medical financial legal
```

### Grounding Benchmark

Specifically focuses on the effectiveness of ACDEK's grounding engine in reducing hallucinations and improving factuality.

```bash
python llm_grounding_benchmark.py
```

### Visualization Dashboard

Generates interactive visual dashboards from benchmark results:

```bash
python benchmark_dashboard.py --input results/benchmark_results_20250418.json
```

## Test Case Structure

The benchmark suite includes domain-specific test cases with:

1. **Ground Truth**: Facts, constraints, and invalid claims for validation
2. **Difficulty Levels**: 1-5 scale for progressive challenge
3. **Domains**: Medical, financial, legal, scientific, technical, and general

Example:
```python
TestCase(
    id="med-001",
    prompt="What are the primary symptoms of diabetes?",
    ground_truth=GroundTruth(
        facts=[
            "Symptoms include increased thirst, frequent urination, unexplained weight loss, and fatigue.",
            "Some patients may be asymptomatic, especially in early Type 2 diabetes."
        ],
        constraints=[
            {"type": "source_citation", "required": True}
        ],
        invalid_claims=[
            "completely curable", "always fatal"
        ]
    ),
    domain=Domain.MEDICAL,
    difficulty=1,
    tags=["diabetes", "symptoms", "common"]
)
```

## Evaluation Metrics

The benchmark suite calculates various metrics, including:

- **Factuality Score**: Semantic similarity to ground truth
- **Hallucination Reduction Rate**: Decrease in invalid claims
- **Constraint Adherence**: Compliance with required constraints
- **Latency Overhead**: Additional processing time vs. raw LLM
- **Success Rate**: Overall improvement considering all factors

## Extending the Suite

### Adding New Test Cases

Create new test cases in the appropriate domain method:

```python
def _generate_technical_test_cases(self) -> List[BenchmarkTestCase]:
    cases = []
    
    # Add your new test case
    cases.append(BenchmarkTestCase(
        id="tech-002",
        prompt="What is the difference between machine learning and deep learning?",
        ground_truth=GroundTruth(
            facts=[
                "Machine learning is a subset of AI that enables systems to learn from data.",
                "Deep learning is a subset of machine learning using neural networks with multiple layers."
            ],
            constraints=[
                {"type": "source_citation", "required": True}
            ],
            invalid_claims=[
                "machine learning is always more accurate", 
                "deep learning always requires GPUs"
            ]
        ),
        domain=Domain.TECHNICAL,
        difficulty=2,
        tags=["AI", "machine learning", "deep learning"]
    ))
    
    return cases
```

### Creating New Benchmarks

Follow the pattern in existing benchmark files to create specialized benchmarks for additional dimensions.

## License

[License terms here]

## Contact

[Contact information here]