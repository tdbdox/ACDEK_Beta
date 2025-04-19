# ACDEK v3 Grounding Benchmark

This benchmark suite evaluates the effectiveness of ACDEK v3's grounding engine by comparing raw LLM outputs against ACDEK-enhanced responses.

## What This Benchmark Measures

1. **Factuality Improvement**: How much ACDEK improves factual accuracy compared to raw LLM responses
2. **Hallucination Detection**: How accurately ACDEK identifies hallucinations
3. **Execution Metrics**: Performance characteristics including processing time

## Test Cases

The benchmark includes test cases from different domains:

- **Financial Domain**: Investment concepts, financial instruments, etc.
- **Medical Domain**: Medical treatments, vaccines, etc.

Each test case includes:
- A prompt
- A raw LLM response (often containing hallucinations)
- An ACDEK-enhanced response
- Ground truth for validation
- Expected hallucination flag

## Running the Benchmark

You can run the benchmark using the provided shell script:

```bash
# From the v3 directory
./benchmarks/run_grounding_benchmark.sh
```

### Command Line Options

```bash
# Specify a custom output directory
./benchmarks/run_grounding_benchmark.sh --output /path/to/custom/results

# Run without debug information (faster, less detailed output)
./benchmarks/run_grounding_benchmark.sh --no-debug
```

## Output

The benchmark generates:

1. A summary in the terminal showing:
   - Overall factuality scores
   - Improvement rates
   - Hallucination detection accuracy
   - Domain-specific metrics

2. JSON result files in the `benchmarks/results` directory:
   - Detailed results for each test case
   - Summary metrics for the entire benchmark

## Implementation Details

The benchmark works by:

1. Running each test case through ACDEK's grounding validation pipeline
2. Comparing factuality scores between raw and enhanced responses
3. Evaluating hallucination detection accuracy against expected flags
4. Calculating improvement metrics and domain-specific performance

## Adding New Test Cases

To add new test cases, modify the `simple_benchmark_test.py` file:

1. Create `TestCase` objects with appropriate prompts, responses, and ground truth
2. Add them to the appropriate domain-specific list or create a new domain list
3. Update the `TEST_CASES` list to include your new test cases