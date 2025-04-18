# Benchmark Results Directory

This folder will contain benchmark result outputs. Expected artifacts include:

- **Hallucination Scores** (JSON):
  - Side-by-side LLM vs ACDEK outputs
  - Detected issues and severity metrics
- **Constraint Reliability** (JSON):
  - Pass rates, violation details per case
- **Confidence Calibration** (JSON/CSV):
  - Correlation between confidence scores and actual outcomes
- **Performance Snapshots** (JSON/CSV):
  - Latency measurements per case
  - Memory usage (if available)
- **Comparative Reports** (JSON/CSV):
  - Precision, recall, F1 scores
  - Retry and fallback rate aggregations

Ensure consistent naming (e.g., `hallucination_results.json`, `performance_metrics.csv`).