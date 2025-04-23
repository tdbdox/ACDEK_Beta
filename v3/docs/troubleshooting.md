# ACDEK v3 Troubleshooting Guide

This guide provides solutions for common issues you may encounter while using ACDEK v3.

## Common Issues and Solutions

### Installation Problems

#### Missing Dependencies

**Problem**: Error messages about missing dependencies when installing ACDEK.

**Solution**:
```bash
# Install all dependencies explicitly
pip install -r requirements.txt

# For web API dependencies
pip install fastapi uvicorn websockets jinja2 aiofiles python-multipart
```

#### Installation Path Issues

**Problem**: Import errors after installation.

**Solution**:
```bash
# Install in development mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/acdek/v3
```

### SDK Issues

#### Missing Plugins

**Problem**: Errors about missing plugins like "ar_vector_enricher".

**Solution**:
These warnings are expected if you don't have all optional plugins installed. The system will fall back to using available components. You can use the minimal pipeline to avoid these warnings:

```python
from sdk.acdek_client import Client

# Use minimal pipeline that doesn't require optional plugins
client = Client(config={"pipeline_type": "minimal"})
```

#### Abstract Class Instantiation Errors

**Problem**: "Can't instantiate abstract class" errors.

**Solution**:
This typically means an interface implementation is missing required methods. Check that all interface methods are implemented:

```python
# Example: Ensure all methods are implemented
from interfaces.confidence_scorer_interface import ConfidenceScorerInterface

class MyScorer(ConfidenceScorerInterface):
    def score(self, data):
        # Implementation
        pass
    
    def evaluate_with_threshold(self, data, threshold=0.6):
        # Implementation
        pass
```

#### ValidationMode Errors

**Problem**: "X is not a valid ValidationMode" errors.

**Solution**:
The valid validation modes are: `strict`, `relaxed`, `progressive`, `deferred`, and `evidence`. Ensure you're using one of these:

```python
# Correct usage
client = Client(config={"fallback_mode": "relaxed"})
```

### CLI Issues

#### Command Not Found

**Problem**: "Command not found" when running CLI tools.

**Solution**:
Ensure you're calling the CLI from the correct directory and with the correct Python interpreter:

```bash
# From the repository root
python3 cli/run_pipeline.py --help

# Or if installed as a package
acdek --help
```

#### Input Format Errors

**Problem**: "Failed to load input data" errors.

**Solution**:
Ensure your input JSON is valid:

```bash
# Validate your JSON first
python -c 'import json; json.load(open("input.json"))'

# Pass valid JSON
echo '{"text": "Example input"}' | python cli/run_pipeline.py
```

### Web API Issues

#### Server Won't Start

**Problem**: Error when starting the FastAPI server.

**Solution**:
Make sure you have all dependencies installed and you're running the server from the correct directory:

```bash
# Install dependencies
pip install fastapi uvicorn websockets

# Start the server with the correct module path
cd /path/to/acdek/v3
python -m tests.api_server_script
```

#### API Request Failures

**Problem**: API requests return errors.

**Solution**:
Check that your request format matches the expected format:

```bash
# Example correct request
curl -X POST http://localhost:8000/run \
    -H "Content-Type: application/json" \
    -d '{"data": {"text": "Example input"}}'
```

## Performance Troubleshooting

### Slow Execution

**Problem**: Pipeline execution is slower than expected.

**Solutions**:
1. Check for bottlenecks using the performance CLI:
   ```bash
   python cli/performance_cli.py profile --duration 60
   python cli/performance_cli.py analyze --report-format cli
   ```

2. Enable caching for frequently used data:
   ```python
   # In your pipeline configuration
   pipeline = EnrichmentPipeline(
       # Other config...
       enable_caching=True
   )
   ```

3. Use parallel execution where possible:
   ```python
   # Set execution mode to ADAPTIVE for automatic parallelization
   pipeline = EnrichmentPipeline(
       # Other config...
       execution_mode=ExecutionMode.ADAPTIVE,
       max_workers=4
   )
   ```

### Memory Leaks

**Problem**: Memory usage grows over time.

**Solution**:
1. Limit cache size in the pipeline:
   ```python
   # Custom cache size limit
   pipeline.result_cache_limit = 100  # Limit to 100 entries
   ```

2. Call `reset()` periodically to clear pipeline state:
   ```python
   # Reset pipeline after batch processing
   pipeline.reset()
   ```

## Debugging

### Enable Verbose Logging

For more detailed information, increase the logging level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

For CLI usage:
```bash
python cli/run_pipeline.py -l DEBUG
```

### Check Plugin Status

To verify which plugins are available and properly loaded:

```python
from registry.enhanced_plugin_registry import get_plugin_status
from registry.registry_types import PluginType

# Check all plugins
status = get_plugin_status()
print(status)

# Check specific plugin type
enrichment_plugins = get_plugin_status(PluginType.ENRICHMENT)
print(enrichment_plugins)
```

### Test Components Individually

If you're having issues with the full pipeline, test each component individually:

```python
# Test just the confidence scorer
from enrichment.confidence_scorer import VectorConfidenceScorer
scorer = VectorConfidenceScorer()
result = scorer.score({"vector": [0.1, 0.2, 0.3]})
print(result)
```

## Getting Help

If you encounter issues not covered in this guide:

1. Check the documentation in the `docs/` directory
2. Look for similar issues in the project repository
3. Create detailed bug reports with:
   - ACDEK version
   - Full error messages
   - Steps to reproduce
   - Example input that causes the issue

## Recovery Procedures

### Recovering from Critical Failures

If the system encounters a critical failure:

1. **Reset the pipeline**:
   ```python
   pipeline.reset()
   ```

2. **Use dummy implementation**:
   ```python
   from enrichment.standard_pipelines import dummy_execute_pipeline
   result = dummy_execute_pipeline(input_data)
   ```

3. **Restart the server** (for Web API):
   ```bash
   # Stop the current server (Ctrl+C) and restart
   python -m tests.api_server_script
   ```

### Data Recovery

If you need to recover data after a failure:

1. Check for cached results:
   ```python
   # Access cache directly if needed
   cached_data = pipeline.result_cache.get(cache_key)
   ```

2. Use the telemetry data if available:
   ```python
   from observability.telemetry_server import get_telemetry_client
   telemetry = get_telemetry_client()
   recent_data = telemetry.get_recent_events()
   ```