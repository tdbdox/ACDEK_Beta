# ACDEK v3 Real-World Test Suite

This document provides instructions for running the comprehensive real-world test suite for ACDEK v3.

## Overview

The real-world test suite provides a complete validation of the ACDEK v3 pipeline, including:

1. **Performance Measurement**: Detailed profiling of execution time, resource usage, and bottlenecks
2. **Hallucination Detection**: Testing factual accuracy across multiple domains
3. **Scaling Behavior**: Testing under various load patterns (constant, step, spike, ramp)
4. **Cross-Domain Validation**: Testing across different knowledge domains

This test suite is designed to validate that ACDEK v3 meets its key objectives:
- Performance overhead below 200ms per request
- Hallucination reduction of at least 80% compared to raw LLM
- Reliable scaling under increasing load
- Resilience under various operating conditions

## Prerequisites

Before running the tests, ensure you have:

1. A working ACDEK v3 installation
2. Python 3.7+ with required dependencies
3. Access to any external services needed by ACDEK
4. At least 2GB of free disk space for test results

## Running the Tests

### Basic Usage

To run the test suite with default settings:

```bash
cd /Users/travispatchett/Solvion/ACDEK_BETA/v3
python tests/real_world_test.py
```

This will:
- Run tests with 10 concurrent requests
- Test for 300 seconds (5 minutes)
- Exercise all available test domains
- Output results to `./test_results/`

### Options

The test suite provides several options for customization:

```
--domains DOMAINS       Comma-separated list of domains to test (default: all)
--duration DURATION     Test duration in seconds (default: 300)
--concurrency LEVEL     Concurrent request level (default: 10)
--output OUTPUT_DIR     Directory for test results (default: ./test_results)
--report-format FORMAT  Output format: json,csv,html,all (default: all)
--debug                 Enable debug logging
```

### Example Usage

Test specific domains with higher concurrency:

```bash
python tests/real_world_test.py --domains medical,financial --concurrency 20
```

Run a shorter test with HTML-only output:

```bash
python tests/real_world_test.py --duration 60 --report-format html
```

## Load Patterns

The test suite runs four different load patterns sequentially:

1. **Constant Load**: Consistent request rate throughout the test
2. **Step Load**: Increasing request rates at specific intervals
3. **Spike Load**: Base load with periodic traffic spikes
4. **Ramp Load**: Gradually increasing load throughout the test

These patterns test ACDEK's ability to handle different types of traffic scenarios.

## Output

The test suite produces several output files:

### JSON Output

- `test_results_full.json`: Complete test results including all individual test cases
- `test_results_summary.json`: Summary metrics only
- `full_test_profile.json`: Detailed performance profile data

### HTML Output

- `test_results_dashboard.html`: Interactive dashboard showing performance metrics and bottlenecks

### CSV Output

- `test_results_summary.csv`: Summary metrics in CSV format
- `test_results_full.csv`: Individual test results in CSV format

### Pattern-Specific Results

- `constant_load_results.json`: Results from the constant load pattern
- `step_load_results.json`: Results from the step load pattern
- `spike_load_results.json`: Results from the spike load pattern
- `ramp_load_results.json`: Results from the ramp load pattern

## Analyzing Results

The HTML dashboard provides a visual overview of:

1. Overall performance metrics
2. Detected bottlenecks with recommendations
3. Hallucination detection results
4. Execution time breakdown across components

Key metrics to monitor:
- **Average Execution Time**: Should be below 200ms for enterprise readiness
- **95th Percentile Execution Time**: Represents worst-case performance
- **Hallucination Rate**: Percentage of responses with detected hallucinations
- **Scaling Efficiency**: How execution time varies with increased load

## Troubleshooting

Common issues:

1. **High Execution Times**: Check the bottleneck report for specific stages causing slowdowns
2. **Memory Errors**: Reduce concurrency level or check for memory leaks
3. **Connection Errors**: Ensure all dependent services are available

For detailed logs:
```bash
python tests/real_world_test.py --debug
```

The debug logs are written to `real_world_test.log` in the current directory.

## Running in CI/CD

For continuous integration environments, modify command to:
```bash
python tests/real_world_test.py --duration 60 --report-format json --output ./test_artifacts
```

Check the success criteria in `test_results_summary.json` to determine pass/fail status.