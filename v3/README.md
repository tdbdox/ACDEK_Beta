# ACDEK v3 Python SDK

Adaptive Contextual-Decision Execution Kernel (ACDEK) v3 provides a modular, enterprise-grade orchestration protocol for LLM-based pipelines.

This SDK exposes a simplified client to:
- Enrich data via plug-in stages
- Validate formal constraints
- Score confidence with evidence-based metrics
- Handle fallback strategies and retries

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-root>/v3
   ```
2. Install dependencies (if any) or use Python 3.8+:
   ```bash
   pip install -r requirements.txt  # if provided
   ```

## Usage
### SDK
```python
from sdk.acdek_client import Client

# Initialize client (strict fallback by default)
client = Client(config={"fallback_mode": "lenient"})

# Input data
input_data = {"text": "Analyze this input text."}

# Run full pipeline
result = client.full_pipeline(input_data, output_format="dict")
print(result)
```

### CLI
```bash
# From project root (v3 folder)
echo '{"text": "Analyze this input."}' \
  | python3 cli/run_pipeline.py \
    --fallback-mode lenient \
    --output-format json
```

## Documentation
- `docs/sdk_guide.md`: Detailed SDK usage and configuration
- `docs/pipeline_example.md`: Pipeline example with sample input/output

## Examples
- `examples/basic_pipeline.py`: End-to-end demo using SDK and CLI

## Contributing
Contributions welcome. Please follow the repository guidelines.