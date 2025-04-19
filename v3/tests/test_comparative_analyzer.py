"""
Test suite for ComparativeAnalyzer

Tests the functionality of the ComparativeAnalyzer class that standardizes metrics,
detects regressions, normalizes scores across LLM providers, and generates reports.
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest import mock
import pytest

from benchmarks.comparative_analyzer import (
    ComparativeAnalyzer,
    VersionMetrics,
    ComparisonResult,
    OutputFormat,
    LLM_PROVIDER_SCALING
)

# Test fixture for sample benchmark data
@pytest.fixture
def sample_benchmark_data():
    """Return sample benchmark data for testing."""
    return {
        "summary_metrics": {
            "hallucination": {
                "total_raw_count": 100,
                "total_v3_count": 20,
                "total_v2_count": 30,
                "avg_v3_reduction_rate": 0.8,
                "avg_v2_reduction_rate": 0.7
            },
            "latency": {
                "avg_raw_latency_ms": 100.0,
                "avg_v3_latency_ms": 150.0,
                "avg_v2_latency_ms": 140.0,
                "avg_v3_overhead_ms": 50.0,
                "avg_v2_overhead_ms": 40.0
            },
            "confidence": {
                "avg_v3_confidence": 0.9,
                "avg_v2_confidence": 0.85,
                "avg_raw_confidence": 0.7
            },
            "constraints": {
                "pass_rate": 0.95
            },
            "coverage": {
                "v3_coverage": 0.85
            }
        },
        "comparisons": [
            {
                "case_id": "test_case_1",
                "metrics": {
                    "hallucination": {
                        "raw_count": 10,
                        "v3_count": 2,
                        "v2_count": 3,
                        "v3_reduction_rate": 0.8,
                        "v2_reduction_rate": 0.7
                    },
                    "latency": {
                        "raw_latency_ms": 100.0,
                        "v3_latency_ms": 150.0,
                        "v2_latency_ms": 140.0,
                        "v3_overhead_ms": 50.0,
                        "v2_overhead_ms": 40.0
                    },
                    "confidence": {
                        "v3_confidence": 0.9,
                        "v2_confidence": 0.85
                    }
                },
                "v3_output": {
                    "constraint_results": {
                        "passed": True
                    }
                },
                "v2_output": {
                    "constraint_results": {
                        "passed": True
                    }
                },
                "raw_llm": {
                    "confidence_score": 0.7
                }
            },
            {
                "case_id": "test_case_2",
                "metrics": {
                    "hallucination": {
                        "raw_count": 15,
                        "v3_count": 2,
                        "v3_reduction_rate": 0.87
                    },
                    "latency": {
                        "raw_latency_ms": 90.0,
                        "v3_latency_ms": 120.0,
                        "v3_overhead_ms": 30.0
                    },
                    "confidence": {
                        "v3_confidence": 0.92
                    }
                },
                "v3_output": {
                    "constraint_results": {
                        "passed": True
                    }
                },
                "raw_llm": {
                    "confidence_score": 0.75
                }
            }
        ],
        "test_cases": [
            {
                "id": "test_case_1",
                "domain": "financial",
                "model_info": {
                    "provider": "openai"
                }
            },
            {
                "id": "test_case_2",
                "domain": "medical",
                "model_info": {
                    "provider": "anthropic"
                }
            }
        ]
    }

# Test fixture for output directory
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_initialize_analyzer():
    """Test ComparativeAnalyzer initialization."""
    analyzer = ComparativeAnalyzer()
    assert analyzer.regression_threshold == 0.05  # Default threshold
    
    custom_analyzer = ComparativeAnalyzer(
        results_dir="/custom/results",
        output_dir="/custom/output",
        regression_threshold=0.1
    )
    assert custom_analyzer.regression_threshold == 0.1
    assert str(custom_analyzer.results_dir) == "/custom/results"
    assert str(custom_analyzer.output_dir) == "/custom/output"

def test_standardize_metrics(sample_benchmark_data):
    """Test metrics standardization from benchmark data."""
    analyzer = ComparativeAnalyzer()
    
    # Test summary metrics
    metrics = analyzer.standardize_metrics(sample_benchmark_data)
    assert len(metrics) == 3  # v3, v2, raw
    
    v3_metrics = next(m for m in metrics if m.version == "v3")
    v2_metrics = next(m for m in metrics if m.version == "v2")
    raw_metrics = next(m for m in metrics if m.version == "raw")
    
    assert v3_metrics.hallucination_reduction == 0.8
    assert v3_metrics.latency_ms == 150.0
    assert v3_metrics.overhead_ms == 50.0
    assert v3_metrics.confidence_score == 0.9
    
    assert v2_metrics.hallucination_reduction == 0.7
    assert v2_metrics.latency_ms == 140.0
    assert v2_metrics.confidence_score == 0.85
    
    assert raw_metrics.hallucination_reduction == 0.0
    assert raw_metrics.latency_ms == 100.0
    assert raw_metrics.overhead_ms == 0.0
    
    # Test case-specific metrics
    case_metrics = analyzer.standardize_metrics(sample_benchmark_data, case_id="test_case_1")
    assert len(case_metrics) == 3
    
    case_v3 = next(m for m in case_metrics if m.version == "v3")
    case_v2 = next(m for m in case_metrics if m.version == "v2")
    case_raw = next(m for m in case_metrics if m.version == "raw")
    
    assert case_v3.hallucination_reduction == 0.8
    assert case_v3.latency_ms == 150.0
    assert case_v3.confidence_score == 0.9
    
    assert case_v2.hallucination_reduction == 0.7
    assert case_v2.latency_ms == 140.0
    
    assert case_raw.hallucination_count == 10

def test_normalize_metrics_by_provider():
    """Test normalization of metrics by LLM provider."""
    analyzer = ComparativeAnalyzer()
    
    # Create test metrics
    v3_metrics = VersionMetrics(
        version="v3",
        hallucination_reduction=0.8,
        latency_ms=150.0,
        confidence_score=0.9
    )
    
    # Test normalization for different providers
    openai_normalized = analyzer.normalize_metrics_by_provider(v3_metrics, "openai")
    assert openai_normalized.hallucination_reduction == 0.8  # openai scale factor = 1.0
    assert openai_normalized.latency_ms == 150.0
    assert openai_normalized.metadata["scaling_factor"] == 1.0
    
    # Test with a provider that has a scaling factor < 1.0
    other_normalized = analyzer.normalize_metrics_by_provider(v3_metrics, "other")
    scale_factor = LLM_PROVIDER_SCALING["other"]
    assert other_normalized.hallucination_reduction == 0.8 / scale_factor
    assert other_normalized.latency_ms == 150.0 * scale_factor
    assert other_normalized.metadata["scaling_factor"] == scale_factor
    
    # Raw metrics shouldn't be normalized
    raw_metrics = VersionMetrics(version="raw", latency_ms=100.0)
    raw_normalized = analyzer.normalize_metrics_by_provider(raw_metrics, "openai")
    assert raw_normalized.latency_ms == 100.0  # Should remain unchanged

def test_compare_versions():
    """Test comparison between version metrics."""
    analyzer = ComparativeAnalyzer(regression_threshold=0.1)
    
    # Create test metrics
    v3_metrics = VersionMetrics(
        version="v3",
        hallucination_reduction=0.8,
        hallucination_count=20,
        latency_ms=150.0,
        overhead_ms=50.0,
        confidence_score=0.9
    )
    
    v2_metrics = VersionMetrics(
        version="v2",
        hallucination_reduction=0.7,
        hallucination_count=30,
        latency_ms=140.0,
        overhead_ms=40.0,
        confidence_score=0.85
    )
    
    raw_metrics = VersionMetrics(
        version="raw",
        hallucination_reduction=0.0,
        hallucination_count=100,
        latency_ms=100.0,
        overhead_ms=0.0,
        confidence_score=0.7
    )
    
    # Test comparison with all versions
    comparison = analyzer.compare_versions(
        v3_metrics=v3_metrics,
        v2_metrics=v2_metrics,
        raw_metrics=raw_metrics,
        case_id="test_case",
        domain="financial",
        llm_provider="openai"
    )
    
    # Check v3 vs v2 deltas
    assert comparison.v3_v2_deltas["hallucination_reduction"] == 0.1
    assert comparison.v3_v2_deltas["latency_ms"] == 10.0
    assert comparison.v3_v2_deltas["confidence_score"] == 0.05
    
    # Check regression detection (should be no regressions with these values)
    assert not comparison.v3_v2_regressions["hallucination_reduction"]
    assert not comparison.v3_v2_regressions["confidence_score"]
    
    # Test with a regression case
    v3_worse = VersionMetrics(
        version="v3",
        hallucination_reduction=0.5,  # Worse than v2
        latency_ms=170.0,  # Worse than v2
        confidence_score=0.7  # Worse than v2
    )
    
    regression_comparison = analyzer.compare_versions(
        v3_metrics=v3_worse,
        v2_metrics=v2_metrics
    )
    
    # Should detect regressions
    assert regression_comparison.v3_v2_regressions["hallucination_reduction"]
    assert regression_comparison.v3_v2_regressions["latency_ms"]
    assert regression_comparison.v3_v2_regressions["confidence_score"]

def test_analyze_benchmark_data(sample_benchmark_data):
    """Test full benchmark data analysis."""
    analyzer = ComparativeAnalyzer()
    
    # Run analysis
    comparison = analyzer.analyze_benchmark_data(sample_benchmark_data)
    
    # Check basic metrics
    assert comparison.metadata["case_count"] == 2
    assert len(comparison.comparisons) == 2
    
    # Check domains
    assert "financial" in comparison.domains
    assert "medical" in comparison.domains
    assert comparison.domains["financial"]["case_count"] == 1
    
    # Check providers
    assert "openai" in comparison.llm_providers
    assert "anthropic" in comparison.llm_providers
    
    # Check summary stats
    assert comparison.summary["v3"]["avg_hallucination_reduction"] > 0.8
    assert comparison.summary["v2_comparison"]["has_v2_data"]
    assert comparison.summary["raw_comparison"]["has_raw_data"]

@pytest.mark.parametrize("format_name", [
    "json", "csv", "markdown", "html", "yaml"
])
def test_export_comparison_report(sample_benchmark_data, temp_dir, format_name):
    """Test exporting comparison reports in different formats."""
    analyzer = ComparativeAnalyzer(output_dir=temp_dir)
    
    # Skip chart generation for tests to avoid matplotlib issues
    with mock.patch.object(analyzer, 'generate_comparative_charts', return_value={}):
        comparison = analyzer.analyze_benchmark_data(sample_benchmark_data)
        
        # Export in specified format
        output_format = OutputFormat(format_name)
        output_file = os.path.join(temp_dir, f"test_report.{format_name}")
        
        report_path = analyzer.export_comparison_report(
            comparison,
            output_format=output_format,
            output_file=output_file,
            include_charts=False
        )
        
        # Verify file was created
        assert os.path.exists(report_path)
        
        # Basic verification based on format
        if format_name == "json":
            with open(report_path) as f:
                data = json.load(f)
                assert "summary" in data
                assert "domains" in data
        
        elif format_name == "csv":
            with open(report_path) as f:
                content = f.read()
                assert "Metric,Value" in content
        
        elif format_name == "markdown":
            with open(report_path) as f:
                content = f.read()
                assert "# ACDEK v3 Comparative Analysis Report" in content
        
        elif format_name == "html":
            with open(report_path) as f:
                content = f.read()
                assert "<title>ACDEK v3 Comparative Analysis Report</title>" in content
        
        elif format_name == "yaml":
            with open(report_path) as f:
                content = f.read()
                assert "summary:" in content

def test_calculate_combined_score():
    """Test the combined score calculation."""
    analyzer = ComparativeAnalyzer()
    
    # Test with good values
    good_score = analyzer._calculate_combined_score({
        "hallucination_reduction": 0.9,
        "confidence": 0.9,
        "latency": 100.0
    })
    
    # Test with poor values
    poor_score = analyzer._calculate_combined_score({
        "hallucination_reduction": 0.3,
        "confidence": 0.4,
        "latency": 500.0
    })
    
    # Combined score should reflect the input quality
    assert good_score > 0.8
    assert poor_score < 0.5
    assert good_score > poor_score

# Main function to run the tests directly
if __name__ == "__main__":
    # This allows running the tests with pytest directly
    pytest.main(["-xvs", __file__])