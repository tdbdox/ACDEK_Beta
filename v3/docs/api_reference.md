# ACDEK v3 API Reference

This document provides a comprehensive reference for the ACDEK v3 API, covering the SDK, CLI, and Web API interfaces.

## SDK API Reference

The SDK provides a Python client for interacting with the ACDEK system programmatically.

### Client Class

The `Client` class in `sdk.acdek_client` is the main entry point for the SDK.

```python
from sdk.acdek_client import Client

# Initialize the client
client = Client(config={
    "fallback_mode": "relaxed",
    "pipeline_type": "standard"
})
```

#### Configuration Options

- `fallback_mode`: Controls constraint validation behavior
  - `strict`: Fail on first constraint violation
  - `relaxed`: Continue validation but report all violations
  - `progressive`: Gradually relax constraints based on confidence
  - `deferred`: Record violations but don't fail validation
  - `evidence`: Only collect evidence, don't make pass/fail judgments
- `pipeline_type`: Controls which enrichment pipeline to use
  - `standard`: Default pipeline with all available components
  - `minimal`: Minimal pipeline with only essential components
  - `dummy`: Dummy pipeline that always succeeds
- `constraint_plugins`: List of plugin names for constraint verifier (None = all)

#### Methods

##### run_enrichment

```python
result = client.run_enrichment(
    input_data, 
    context=None, 
    output_format="dict"
)
```

Executes the enrichment pipeline, which processes and enhances the input data.

**Parameters:**
- `input_data`: Dictionary containing the input data
- `context`: Optional dictionary with contextual information
- `output_format`: Output format (`dict` or `json`)

**Returns:**
A dictionary or JSON string containing:
- `success`: Boolean indicating success
- `data`: The enriched data
- `errors`: List of errors (if any)
- `metrics`: Performance metrics
- `stage_results`: Results from each pipeline stage

##### validate_constraints

```python
result = client.validate_constraints(
    input_data, 
    output_format="dict"
)
```

Validates constraints on the given data.

**Parameters:**
- `input_data`: Dictionary containing the data to validate
- `output_format`: Output format (`dict` or `json`)

**Returns:**
A dictionary or JSON string containing:
- `passed`: Boolean indicating if all constraints passed
- `violations`: List of constraint violations (if any)

##### score_confidence

```python
result = client.score_confidence(
    input_data, 
    output_format="dict"
)
```

Computes confidence scores for the given data.

**Parameters:**
- `input_data`: Dictionary containing the data to score
- `output_format`: Output format (`dict` or `json`)

**Returns:**
A dictionary or JSON string containing:
- `aggregate_score`: Overall confidence score (0.0-1.0)
- `factors`: Individual factor scores
- `metadata`: Additional metadata
- `timestamp`: Timestamp when the score was calculated

##### full_pipeline

```python
result = client.full_pipeline(
    input_data, 
    context=None, 
    output_format="dict"
)
```

Executes the complete pipeline, including enrichment, constraint validation, and confidence scoring.

**Parameters:**
- `input_data`: Dictionary containing the input data
- `context`: Optional dictionary with contextual information
- `output_format`: Output format (`dict` or `json`)

**Returns:**
A dictionary or JSON string containing:
- `enrichment`: Results from the enrichment pipeline
- `constraints`: Results from constraint validation
- `confidence`: Confidence scores
- `timing`: Execution timing information

## CLI Reference

The CLI provides command-line tools for running the ACDEK pipeline and analyzing results.

### Basic Command Line Options

```bash
python cli/run_pipeline.py [options]
```

#### Global Options

- `-i, --input FILE`: Input JSON file path (default: stdin)
- `-f, --fallback-mode MODE`: Fallback mode (`strict`, `relaxed`, `progressive`, `deferred`, or `evidence`)
- `-l, --log-level LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `-o, --output-format FORMAT`: Output format (`pretty` or `json`)
- `--session`: Enable interactive session mode (REPL style)
- `--export-session FILE`: Path to JSON file to export session metrics upon exit
- `--component COMPONENT`: Component to run (`pipeline`, `enrichment`, `constraints`, or `confidence`)

### Performance CLI

```bash
python cli/performance_cli.py [command] [options]
```

#### Commands

- `profile`: Profile ACDEK pipeline execution
  - `--duration SECONDS`: Duration to profile in seconds
  - `--interval SECONDS`: Snapshot interval in seconds
  - `--export`: Export profile data after collection
- `analyze`: Analyze performance data for bottlenecks
  - `--input FILE`: Input file with performance data to analyze
  - `--threshold MS`: Warning threshold in milliseconds
  - `--report-format FORMAT`: Output format (`cli`, `html`, or `json`)
- `visualize`: Create visualizations from performance data
  - `--input FILE`: Input file with performance data to visualize
  - `--output FILE`: Output filename for the dashboard
- `monitor`: Continuously monitor ACDEK performance
  - `--interval SECONDS`: Monitoring interval in seconds
  - `--duration SECONDS`: Duration in seconds (0 = indefinite)
  - `--dashboard`: Continuously update a dashboard

## Web API Reference

The Web API provides HTTP endpoints for interacting with the ACDEK system.

### Server Setup

```bash
# Start the API server
python tests/api_server_script.py
```

This starts the server at `http://localhost:8000`.

### Endpoints

#### POST /run

Run the complete ACDEK pipeline.

**Request Body:**
```json
{
  "data": {
    "text": "Example input for pipeline"
  }
}
```

**Response:**
```json
{
  "result": {
    "enrichment": { /* enrichment results */ },
    "constraints": { /* constraint results */ },
    "confidence": { /* confidence results */ },
    "timing": { /* timing information */ }
  }
}
```

#### POST /validate

Validate constraints on the input data.

**Request Body:**
```json
{
  "data": {
    "text": "Example input for constraint validation"
  }
}
```

**Response:**
```json
{
  "passed": true,
  "violations": []
}
```

#### POST /confidence

Calculate confidence scores for the input data.

**Request Body:**
```json
{
  "data": {
    "vector": [0.1, 0.2, 0.3],
    "model_name": "test-model"
  }
}
```

**Response:**
```json
{
  "aggregate_score": 0.75,
  "factors": {
    "vector_quality": 0.8,
    "embedding_model": 0.7
  },
  "metadata": { /* additional metadata */ }
}
```

#### GET /status

Get the current status of the API server and performance monitor.

**Response:**
```json
{
  "running": true,
  "monitor_available": true,
  "uptime": 3600
}
```

#### GET /metrics

Get recent performance metrics.

**Query Parameters:**
- `count`: Number of metrics to return (default: 10)

**Response:**
```json
{
  "metrics": [
    { /* metric data */ },
    { /* metric data */ }
  ]
}
```

#### WebSocket /ws

Connect to the WebSocket endpoint for real-time updates.

**Commands:**
- `get_status`: Get the current status
- `get_metrics`: Get the latest metrics
- `get_alerts`: Get recent alerts
- `command`: Send a command (e.g., `start_monitor`, `stop_monitor`)

## Error Handling

All interfaces follow similar error handling patterns:

1. **SDK**: Returns structured error information in the result object
2. **CLI**: Outputs errors to stderr with appropriate exit codes
3. **Web API**: Returns appropriate HTTP status codes with error details in the response body

When an error occurs, the system attempts to continue operation using fallback strategies, depending on the configuration.

## Plugin System

ACDEK uses a plugin system for extensibility. Plugins are loaded dynamically and can be configured through all three interfaces.

Key plugin types:
- `ENRICHMENT`: Data enrichment components
- `VECTOR_ALIGNMENT`: Vector alignment and embedding components
- `CONSTRAINT`: Constraint validation components

Plugins are automatically discovered and loaded, with proper error handling for missing or incompatible plugins.