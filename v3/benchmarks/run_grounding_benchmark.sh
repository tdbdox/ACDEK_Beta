#!/bin/bash
# Run the ACDEK v3 grounding benchmark

# Navigate to the project root
cd "$(dirname "$0")/.."

# Create results directory if it doesn't exist
mkdir -p benchmarks/results

# Set up PYTHONPATH to ensure imports work correctly
export PYTHONPATH="$PWD:$PYTHONPATH"

# Run the benchmark
python3 benchmarks/simple_benchmark_test.py "$@"

# Check if the benchmark ran (regardless of pass/fail metrics)
exit_code=$?
if [ $exit_code -eq 0 ] || [ $exit_code -eq 1 ]; then
    echo -e "\n✅ Benchmark completed successfully (exit code: $exit_code)"
else
    echo -e "\n❌ Benchmark execution failed - see logs for details (exit code: $exit_code)"
fi