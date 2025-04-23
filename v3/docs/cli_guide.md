# ACDEK Command Line Interface Guide

The ACDEK CLI provides a command-line interface to run the ACDEK pipeline, test individual components, and analyze results.

## Installation

```bash
# Install the package
pip install -e .

# Or run directly from the repo
python -m cli.run_pipeline [options]
```

## Basic Usage

```bash
# Process input from a file
python cli/run_pipeline.py -i input.json

# Process input from stdin
cat input.json | python cli/run_pipeline.py

# Run in interactive session mode
python cli/run_pipeline.py --session
```

## Options

- `-i, --input FILE`: Input JSON file path (default: stdin)
- `-f, --fallback-mode MODE`: Fallback mode (`strict`, `relaxed`, `progressive`, `deferred`, or `evidence`)
- `-l, --log-level LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `-o, --output-format FORMAT`: Output format (`pretty` or `json`)
- `--session`: Enable interactive session mode (REPL style)
- `--export-session FILE`: Path to JSON file to export session metrics upon exit
- `--component COMPONENT`: Component to run (`pipeline`, `enrichment`, `constraints`, or `confidence`)

## Examples

### Process a single file with default settings

```bash
python cli/run_pipeline.py -i examples/input.json
```

### Run with lenient fallback mode and JSON output

```bash
python cli/run_pipeline.py -i examples/input.json -f lenient -o json
```

### Run only the confidence scoring component

```bash
python cli/run_pipeline.py -i examples/input.json --component confidence
```

### Interactive Session

```bash
python cli/run_pipeline.py --session --export-session session_metrics.json
```

In interactive mode, you can type JSON input directly at the prompt:

```
acdek> {"text": "Example input text"}
```

Press Ctrl+D to exit the session.

## Performance CLI

The performance CLI provides tools for profiling, analyzing bottlenecks, and creating visualizations:

```bash
# Profile the system
python cli/performance_cli.py profile --duration 60

# Analyze bottlenecks
python cli/performance_cli.py analyze --threshold 100

# Create visualizations
python cli/performance_cli.py visualize --input profile_data.json

# Monitor system in real-time
python cli/performance_cli.py monitor --interval 5
```