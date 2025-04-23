# Getting Started with ACDEK v3

This guide will help you get started with ACDEK v3, the Advanced Control Deck for Evaluating Knowledge. ACDEK is an enterprise-grade orchestration protocol for LLM-based pipelines that improves reliability, factuality, and performance of LLM responses.

## Installation

### Requirements

- Python 3.9 or higher
- pip (Python package installer)

### Installation Steps

1. Clone the repository (if not already available):
   ```bash
   git clone <repository-url>
   cd acdek
   ```

2. Install the package and its dependencies:
   ```bash
   pip install -e .
   ```

   This installs ACDEK in development mode, making it available system-wide while allowing changes to the code.

3. Verify installation:
   ```bash
   python -c "from sdk.acdek_client import Client; print('ACDEK installed successfully')"
   ```

## Basic Usage

### Using the SDK

```python
from sdk.acdek_client import Client
import json

# Initialize the client
client = Client()

# Prepare input data
input_data = {
    "text": "This is an example input for ACDEK processing.",
    "metadata": {
        "source": "user_input",
        "category": "example"
    }
}

# Run the full pipeline
result = client.full_pipeline(input_data)

# Print the results
print(f"Enrichment Success: {result['enrichment']['success']}")
print(f"Constraints Passed: {result['constraints']['passed']}")
print(f"Confidence Score: {result['confidence']['aggregate_score']:.2f}")
```

### Using the CLI

ACDEK provides command-line tools for easy integration into scripts and workflows.

#### Running the Pipeline

```bash
# Process input from a file
python cli/run_pipeline.py -i input.json

# Process input from stdin
echo '{"text": "Example input"}' | python cli/run_pipeline.py

# Run with specific fallback mode and output format
python cli/run_pipeline.py -i input.json -f relaxed -o json
```

#### Interactive Session

```bash
# Start an interactive session
python cli/run_pipeline.py --session
```

In the interactive session, you can input JSON data line by line:

```
acdek> {"text": "Example input for interactive processing"}
```

### Using the Web API

ACDEK provides a FastAPI-based web API for integration with web applications.

#### Starting the API Server

```bash
# Start the API server
python tests/api_server_script.py
```

This starts the server at `http://localhost:8000`.

#### Making API Requests

```bash
# Run the pipeline
curl -X POST http://localhost:8000/run \
    -H "Content-Type: application/json" \
    -d '{"data": {"text": "Example input for API"}}'

# Validate constraints
curl -X POST http://localhost:8000/validate \
    -H "Content-Type: application/json" \
    -d '{"data": {"text": "Example input for validation"}}'

# Score confidence
curl -X POST http://localhost:8000/confidence \
    -H "Content-Type: application/json" \
    -d '{"data": {"text": "Example input for confidence scoring"}}'
```

## Configuration Options

ACDEK is highly configurable to adapt to different use cases.

### SDK Configuration

```python
# Configure the client with custom settings
client = Client(config={
    # Fallback mode controls constraint validation behavior
    "fallback_mode": "relaxed",  # Options: strict, relaxed, progressive, deferred, evidence
    
    # Pipeline type selects which pipeline implementation to use
    "pipeline_type": "standard",  # Options: standard, minimal, dummy
    
    # Constraint plugins to use (None = all)
    "constraint_plugins": ["format_validator", "semantic_validator"]
})
```

### CLI Configuration

Command-line options allow you to configure the pipeline's behavior:

```bash
# Get help on available options
python cli/run_pipeline.py --help

# Run with specific configuration
python cli/run_pipeline.py \
    -i input.json \
    -f relaxed \
    -l INFO \
    -o pretty \
    --component confidence
```

### Web API Configuration

The Web API is configured through environment variables or a configuration file:

```bash
# Set configuration via environment variables
export ACDEK_FALLBACK_MODE=relaxed
export ACDEK_MAX_WORKERS=4

# Then start the server
python tests/api_server_script.py
```

## Example Use Cases

### Enriching LLM Outputs

```python
from sdk.acdek_client import Client

client = Client()

# LLM-generated content
llm_output = {
    "text": "The Earth's diameter is approximately 12,742 kilometers.",
    "generated_by": "gpt-4"
}

# Enrich with confidence score and validation
result = client.full_pipeline(llm_output)

# Check confidence and constraints
if result["confidence"]["aggregate_score"] > 0.8 and result["constraints"]["passed"]:
    print("High confidence, validated output")
else:
    print("Low confidence or validation failed")
```

### Batch Processing with CLI

```bash
#!/bin/bash
# Process multiple input files

for file in inputs/*.json; do
    echo "Processing $file..."
    python cli/run_pipeline.py -i "$file" -o json > "outputs/$(basename "$file")"
done
```

### Web Integration

```python
# Example FastAPI application that uses ACDEK
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import json

app = FastAPI()

class UserQuery(BaseModel):
    query: str

@app.post("/analyze")
async def analyze_query(query: UserQuery):
    # Call ACDEK API
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/run",
            json={"data": {"text": query.query}}
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="ACDEK processing failed")
            
        result = response.json()
        
        # Process and return results
        return {
            "confidence": result["result"]["confidence"]["aggregate_score"],
            "valid": result["result"]["constraints"]["passed"],
            "enriched_data": result["result"]["enrichment"]["data"]
        }
```

## Next Steps

- Explore the documentation in the `docs/` directory for more detailed information
- Try the examples in the `examples/` directory for more complex use cases
- Run the benchmarks in the `benchmarks/` directory to evaluate performance
- Check out the troubleshooting guide if you encounter any issues

## Getting Help

If you need help using ACDEK:

1. Check the troubleshooting guide in `docs/troubleshooting.md`
2. Review the API reference in `docs/api_reference.md`
3. Look for examples in the `examples/` directory that match your use case