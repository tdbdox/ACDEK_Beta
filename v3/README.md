# ACDEK v3 - Advanced Control Deck for Evaluating Knowledge

ACDEK is an enterprise-grade orchestration protocol for LLM-based pipelines that improves reliability, factuality, and performance of LLM responses.

![ACDEK Logo](https://via.placeholder.com/200x100?text=ACDEK+v3)

## Features

- **Enrichment Pipeline**: Process data through configurable stages with dependency management
- **Constraint Verification**: Validate outputs against formal constraints with multiple validation modes
- **Confidence Scoring**: Evaluate reliability based on multiple weighted factors
- **Hallucination Detection**: Comprehensive metrics for measuring and reducing hallucinations
- **Performance Profiling**: Detailed metrics, bottleneck detection, and visualizations
- **Plugin Architecture**: Extensible system with dynamic component loading

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-organization/acdek.git
cd acdek

# Install the package
pip install -e .
```

### Basic Usage

```python
from sdk.acdek_client import Client

# Initialize client
client = Client()

# Prepare input data
input_data = {
    "text": "Example input for ACDEK processing"
}

# Process with ACDEK
result = client.full_pipeline(input_data)

# Print results
print(f"Confidence Score: {result['confidence']['aggregate_score']:.3f}")
print(f"Constraints Passed: {result['constraints']['passed']}")
```

## Distribution Interfaces

ACDEK provides three interfaces for integration:

### 1. SDK (Python Client)

The SDK provides a Python interface for integrating ACDEK into your applications:

```python
from sdk.acdek_client import Client

client = Client(config={"fallback_mode": "relaxed"})
result = client.full_pipeline(input_data)
```

### 2. CLI Tools

Command-line tools for running ACDEK from scripts or terminal:

```bash
# Process input from a file
python cli/run_pipeline.py -i input.json

# Run in interactive session mode
python cli/run_pipeline.py --session
```

### 3. Web API

FastAPI server for integration with web applications:

```bash
# Start the server
python tests/api_server_script.py

# Make API requests
curl -X POST http://localhost:8000/run \
    -H "Content-Type: application/json" \
    -d '{"data": {"text": "Example input"}}'
```

## Documentation

Explore the comprehensive documentation in the `docs/` directory:

- [Getting Started Guide](docs/getting_started.md)
- [API Reference](docs/api_reference.md)
- [CLI Guide](docs/cli_guide.md)
- [Troubleshooting](docs/troubleshooting.md)
- [SDK Guide](docs/sdk_guide.md)
- [Pipeline Example](docs/pipeline_example.md)
- [Performance Framework](docs/performance_framework.md)
- [Benchmark Integration](docs/benchmark_integration.md)

## Examples

Check out the `examples/` directory for sample implementations:

- Basic pipeline usage
- Comparative analysis
- Continuous monitoring
- Performance measurement
- Integrated benchmarking

## Benchmarks

ACDEK includes a comprehensive benchmarking suite:

```bash
# Run all benchmarks
python benchmarks/run_benchmarks.py --all

# Run specific benchmarks
python benchmarks/run_benchmarks.py --grounding
```

## Architecture

ACDEK follows a modular architecture with these key components:

1. **ACDEKKernel**: Central orchestration component
2. **EnrichmentPipeline**: Multi-stage data processing pipeline
3. **ConstraintVerifier**: Formal constraint validation system
4. **ConfidenceScorer**: Multi-factor confidence evaluation
5. **Registry**: Plugin management system
6. **Benchmarking Suite**: Comparative evaluation framework

## License

[MIT License](LICENSE)

## Version

Current version: 3.0.0-beta

## Contact

For questions or support, please [open an issue](https://github.com/your-organization/acdek/issues) on GitHub.