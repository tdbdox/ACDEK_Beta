# ACDEK v3 Benchmark Integration Guide

This guide explains how to integrate the enhanced benchmark suite components to create comprehensive benchmark workflows.

## Overview of Enhanced Components

The ACDEK v3 benchmark suite includes several enhanced components:

1. **Comparative Analyzer**: Statistical significance testing and domain-specific analysis
2. **Threshold Configuration**: Customizable thresholds for different domains and metrics
3. **Advanced Visualizations**: Radar charts, heatmaps, ROC curves, and interactive dashboards
4. **Integration Example**: Complete end-to-end workflow demonstration

## Integration Workflow

The following workflow shows how to integrate these components:

```
                                      ┌───────────────────────┐
                                      │ ACDEK Benchmark Suite │
                                      └───────────┬───────────┘
                                                  │
                                                  ▼
┌───────────────────┐          ┌───────────────────────────────┐          ┌───────────────────┐
│ Threshold Config  │◄─────────┤ Comparative Analyzer          │─────────►│ Advanced          │
│                   │          │ (Statistical Analysis)         │          │ Visualizations    │
└───────────────────┘          └───────────────────────────────┘          └───────────────────┘
                                                  │
                                                  ▼
                                      ┌───────────────────────┐
                                      │ Comprehensive Reports │
                                      └───────────────────────┘
```

## Example Usage

```python
# Import components
from benchmarks.acdek_benchmark_suite import ACDEKBenchmarkSuite, BenchmarkSuiteConfig
from benchmarks.comparative_analyzer import ComparativeAnalyzer
from benchmarks.threshold_config import ThresholdConfig
from benchmarks.advanced_visualizations import AdvancedVisualization

# 1. Configure and run benchmark suite
config = BenchmarkSuiteConfig(
    llm_models=["gpt-4", "claude-3"],
    domains=["medical", "legal", "financial"],
    parallel=True
)
benchmark_suite = ACDEKBenchmarkSuite(config)
results = benchmark_suite.run_benchmark()

# 2. Apply threshold configuration
threshold_config = ThresholdConfig()
# Optionally optimize thresholds based on benchmark data
optimized = threshold_config.optimize_threshold(
    domain="medical",
    metric_type="factuality",
    benchmark_data=results.to_dict()
)

# 3. Analyze with statistical testing
analyzer = ComparativeAnalyzer()
v3_metrics = analyzer.extract_version_metrics(results.to_dict(), version="v3")
raw_metrics = analyzer.extract_version_metrics(results.to_dict(), version="raw")
analysis = analyzer.analyze_with_statistical_significance(
    v3_metrics=v3_metrics,
    raw_metrics=raw_metrics,
    significance_level=0.05
)

# 4. Generate visualizations
visualizer = AdvancedVisualization(output_dir="./reports")
radar_chart = visualizer.create_radar_chart(
    data={"ACDEK v3": v3_metrics["aggregated"], "Raw LLM": raw_metrics["aggregated"]},
    title="Performance Comparison"
)

# 5. Export comprehensive report
report_path = analyzer.export_analysis_report(
    data={"analysis": analysis, "charts": {"radar": str(radar_chart)}},
    output_format="html",
    include_visualizations=True
)
```

## Complete Integration Example

For a complete example of how to integrate all components, see `examples/integrated_benchmark_example.py`.

This example demonstrates:

1. Running a complete benchmark suite
2. Configuring domain-specific thresholds 
3. Performing statistical analysis
4. Generating advanced visualizations
5. Exporting comprehensive reports in multiple formats

## Key Features and Benefits

### Statistical Significance Testing

The enhanced comparative analyzer can determine if improvements are statistically significant using:

- T-tests for paired metrics
- Bootstrapping for small sample sizes
- Confidence intervals to quantify uncertainty

### Domain-Specific Analysis

Metrics are now analyzed with domain-specific considerations:

- Different thresholds for medical, legal, financial domains
- Model-specific normalization factors
- Domain-specific weight adjustments

### Advanced Visualizations

The visualization toolkit provides:

- Radar charts for multi-dimensional comparison
- Heatmaps for domain-metric performance
- ROC curves for classification performance
- Interactive dashboards with Plotly

### Threshold Configuration

The threshold system provides:

- Default thresholds by domain and metric
- Automatic threshold optimization
- Easy configuration and overrides

## Best Practices

1. **Use parallel execution** for large benchmark suites
2. **Normalize by provider** when comparing across different LLM providers
3. **Consider domain-specific factors** when analyzing performance
4. **Use statistical significance testing** to validate improvements
5. **Generate multiple visualization types** for comprehensive analysis
6. **Export to multiple formats** (HTML, PDF, JSON) for different needs