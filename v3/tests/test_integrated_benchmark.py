#!/usr/bin/env python
"""
ACDEK v3 Integrated Benchmark Tests

Tests the integration between the enhanced components of the ACDEK benchmark suite:
- Benchmark Suite execution
- Threshold Configuration
- Comparative Analysis with statistical testing
- Advanced Visualizations
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path to allow importing from project
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import benchmark components
from benchmarks.acdek_benchmark_suite import (
    ACDEKBenchmarkSuite,
    BenchmarkSuiteConfig,
    Domain,
    BenchmarkSuiteResults
)
from benchmarks.comparative_analyzer import ComparativeAnalyzer, OutputFormat
from benchmarks.threshold_config import ThresholdConfig, MetricType
from benchmarks.advanced_visualizations import AdvancedVisualization

# Import example integration
from examples.integrated_benchmark_example import (
    run_benchmark_suite,
    configure_thresholds,
    analyze_results,
    generate_visualizations,
    export_report
)


# Sample test data
MOCK_BENCHMARK_RESULTS = {
    "config": {
        "llm_models": ["test-model"],
        "domains": ["medical", "legal"],
        "parallel": True
    },
    "results": [
        {
            "test_case": {
                "id": "test-case-1",
                "domain": "medical",
                "prompt": "Test prompt"
            },
            "raw_response": {
                "text": "Raw response",
                "model_info": {"name": "test-model", "type": "raw"},
                "latency": 1.0
            },
            "acdek_response": {
                "text": "ACDEK response",
                "model_info": {"name": "test-model", "type": "acdek"},
                "latency": 1.2,
                "confidence_score": 0.85
            },
            "raw_evaluation": {
                "factuality": {"score": 0.7},
                "constraint_adherence": {"score": 0.6},
                "overall_score": 0.65
            },
            "acdek_evaluation": {
                "factuality": {"score": 0.85},
                "constraint_adherence": {"score": 0.8},
                "overall_score": 0.82
            },
            "metrics": {
                "factuality": {
                    "raw_score": 0.7,
                    "acdek_score": 0.85,
                    "improvement": 0.15
                },
                "constraint_adherence": {
                    "raw_score": 0.6,
                    "acdek_score": 0.8,
                    "improvement": 0.2
                },
                "overall": {
                    "raw_score": 0.65,
                    "acdek_score": 0.82,
                    "improvement": 0.17,
                    "improvement_percent": 26.15
                },
                "latency": {
                    "raw_seconds": 1.0,
                    "acdek_seconds": 1.2,
                    "overhead_seconds": 0.2,
                    "overhead_percent": 20.0
                },
                "hallucination": {
                    "raw_invalid_claims": 3,
                    "acdek_invalid_claims": 1
                },
                "success": {
                    "factuality_improved": True,
                    "constraints_improved": True,
                    "overall_improved": True,
                    "acceptable_latency": True,
                    "overall_success": True
                }
            }
        },
        # Legal domain case
        {
            "test_case": {
                "id": "test-case-2",
                "domain": "legal",
                "prompt": "Legal test prompt"
            },
            "raw_response": {
                "text": "Raw legal response",
                "model_info": {"name": "test-model", "type": "raw"},
                "latency": 1.1
            },
            "acdek_response": {
                "text": "ACDEK legal response",
                "model_info": {"name": "test-model", "type": "acdek"},
                "latency": 1.3,
                "confidence_score": 0.82
            },
            "raw_evaluation": {
                "factuality": {"score": 0.65},
                "constraint_adherence": {"score": 0.7},
                "overall_score": 0.68
            },
            "acdek_evaluation": {
                "factuality": {"score": 0.82},
                "constraint_adherence": {"score": 0.88},
                "overall_score": 0.85
            },
            "metrics": {
                "factuality": {
                    "raw_score": 0.65,
                    "acdek_score": 0.82,
                    "improvement": 0.17
                },
                "constraint_adherence": {
                    "raw_score": 0.7,
                    "acdek_score": 0.88,
                    "improvement": 0.18
                },
                "overall": {
                    "raw_score": 0.68,
                    "acdek_score": 0.85,
                    "improvement": 0.17,
                    "improvement_percent": 25.0
                },
                "latency": {
                    "raw_seconds": 1.1,
                    "acdek_seconds": 1.3,
                    "overhead_seconds": 0.2,
                    "overhead_percent": 18.18
                },
                "hallucination": {
                    "raw_invalid_claims": 4,
                    "acdek_invalid_claims": 1
                },
                "success": {
                    "factuality_improved": True,
                    "constraints_improved": True,
                    "overall_improved": True,
                    "acceptable_latency": True,
                    "overall_success": True
                }
            }
        }
    ],
    "summary_metrics": {
        "overall": {
            "case_count": 2,
            "model_count": 1,
            "improvements": {
                "factuality": 0.16,
                "constraints": 0.19,
                "overall": 0.17
            },
            "latency": {
                "avg_overhead_seconds": 0.2
            },
            "success_rates": {
                "factuality_improved": 1.0,
                "constraints_improved": 1.0,
                "overall_improved": 1.0,
                "acceptable_latency": 1.0,
                "overall_success": 1.0
            }
        },
        "by_model": {
            "test-model": {
                "case_count": 2,
                "improvements": {
                    "factuality": 0.16,
                    "constraints": 0.19,
                    "overall": 0.17
                },
                "latency": {
                    "avg_overhead_seconds": 0.2
                },
                "success_rates": {
                    "factuality_improved": 1.0,
                    "constraints_improved": 1.0,
                    "overall_improved": 1.0,
                    "acceptable_latency": 1.0,
                    "overall_success": 1.0
                }
            }
        },
        "by_domain": {
            "medical": {
                "case_count": 1,
                "avg_improvement": 0.17,
                "success_rate": 1.0
            },
            "legal": {
                "case_count": 1,
                "avg_improvement": 0.17,
                "success_rate": 1.0
            }
        }
    },
    "comparative_metrics": {
        "hallucination_reduction_rate": 0.71,
        "avg_latency_overhead": 0.2,
        "raw_hallucination_count": 7,
        "acdek_hallucination_count": 2,
        "improvement_summary": {
            "total_cases": 2,
            "cases_improved": 2,
            "cases_with_acceptable_latency": 2,
            "overall_success_cases": 2
        }
    },
    "started_at": "2025-04-18T20:00:00",
    "completed_at": "2025-04-18T20:01:00",
    "runtime_seconds": 60.0
}


class TestIntegratedBenchmark(unittest.TestCase):
    """Test the integration between enhanced ACDEK benchmark components."""
    
    def setUp(self):
        """Set up test environment."""
        self.output_dir = Path("./test_output")
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = "20250418_200000"
        
    def tearDown(self):
        """Clean up test environment."""
        # Comment out to keep test outputs
        # import shutil
        # if self.output_dir.exists():
        #     shutil.rmtree(self.output_dir)
        pass
    
    @patch('examples.integrated_benchmark_example.ACDEKBenchmarkSuite')
    def test_run_benchmark_suite(self, mock_benchmark_suite):
        """Test running the benchmark suite."""
        # Configure mock
        mock_instance = mock_benchmark_suite.return_value
        mock_results = MagicMock()
        mock_results.to_dict.return_value = MOCK_BENCHMARK_RESULTS
        mock_results.summary_metrics = MOCK_BENCHMARK_RESULTS["summary_metrics"]
        mock_instance.run_benchmark.return_value = mock_results
        
        # Call the function
        models = ["test-model"]
        domains = ["medical", "legal"]
        results = run_benchmark_suite(models, domains, self.output_dir)
        
        # Assertions
        mock_benchmark_suite.assert_called_once()
        mock_instance.run_benchmark.assert_called_once()
        self.assertEqual(results, MOCK_BENCHMARK_RESULTS)
    
    def test_configure_thresholds(self):
        """Test threshold configuration."""
        # Call the function
        config = configure_thresholds(MOCK_BENCHMARK_RESULTS, optimize=False)
        
        # Assertions
        self.assertIsInstance(config, ThresholdConfig)
        
        # Test with optimization
        with patch.object(ThresholdConfig, 'optimize_threshold') as mock_optimize:
            mock_optimize.return_value = MagicMock(threshold=0.75)
            config = configure_thresholds(MOCK_BENCHMARK_RESULTS, optimize=True)
            self.assertTrue(mock_optimize.called)
    
    @patch('examples.integrated_benchmark_example.ComparativeAnalyzer')
    def test_analyze_results(self, mock_analyzer_class):
        """Test results analysis with statistical significance testing."""
        # Configure mock
        mock_analyzer = mock_analyzer_class.return_value
        mock_analyzer.extract_version_metrics.side_effect = lambda results, version: {
            "aggregated": {"factuality": 0.85 if version == "v3" else 0.7}
        }
        mock_analyzer.analyze_with_statistical_significance.return_value = {
            "significance_tests": {
                "factuality": {"is_significant": True, "p_value": 0.03},
                "hallucination": {"is_significant": True, "p_value": 0.01}
            }
        }
        mock_analyzer.perform_domain_specific_analysis.return_value = {
            "medical": {"v3": {"factuality": 0.85}, "raw": {"factuality": 0.7}}
        }
        
        # Call the function
        threshold_config = ThresholdConfig()
        results = analyze_results(MOCK_BENCHMARK_RESULTS, threshold_config)
        
        # Assertions
        self.assertTrue(mock_analyzer.extract_version_metrics.called)
        self.assertTrue(mock_analyzer.analyze_with_statistical_significance.called)
        self.assertTrue(mock_analyzer.perform_domain_specific_analysis.called)
        self.assertIn("statistical_analysis", results)
        self.assertIn("domain_analysis", results)
        self.assertIn("v3_metrics", results)
        self.assertIn("raw_metrics", results)
    
    @patch('examples.integrated_benchmark_example.AdvancedVisualization')
    def test_generate_visualizations(self, mock_visualizer_class):
        """Test visualization generation."""
        # Configure mock
        mock_visualizer = mock_visualizer_class.return_value
        mock_visualizer.create_radar_chart.return_value = self.output_dir / "radar.png"
        mock_visualizer.create_heatmap.return_value = self.output_dir / "heatmap.png"
        mock_visualizer.create_roc_curve.return_value = self.output_dir / "roc.png"
        mock_visualizer.create_bar_with_error.return_value = self.output_dir / "bar.png"
        mock_visualizer.create_html_dashboard.return_value = self.output_dir / "dashboard.html"
        
        # Prepare analysis results
        analysis_results = {
            "v3_metrics": {"aggregated": {"factuality": 0.85}},
            "raw_metrics": {"aggregated": {"factuality": 0.7}},
            "domain_analysis": {"medical": {"v3": {"factuality": 0.85}}},
            "statistical_analysis": {
                "significance_tests": {
                    "factuality": {"is_significant": True, "p_value": 0.03}
                }
            }
        }
        
        # Call the function
        paths = generate_visualizations(analysis_results, self.output_dir, self.timestamp)
        
        # Assertions
        self.assertTrue(mock_visualizer.create_radar_chart.called)
        self.assertTrue(mock_visualizer.create_heatmap.called)
        self.assertTrue(mock_visualizer.create_roc_curve.called)
        self.assertTrue(mock_visualizer.create_bar_with_error.called)
        self.assertIn("radar_chart", paths)
        self.assertIn("domain_heatmap", paths)
        self.assertIn("roc_curve", paths)
        self.assertIn("significance_chart", paths)
        self.assertIn("dashboard", paths)
    
    @patch('examples.integrated_benchmark_example.ComparativeAnalyzer')
    @patch('examples.integrated_benchmark_example.AdvancedVisualization')
    def test_export_report(self, mock_visualizer_class, mock_analyzer_class):
        """Test report export."""
        # Configure mocks
        mock_analyzer = mock_analyzer_class.return_value
        mock_analyzer.export_analysis_report.return_value = self.output_dir / "report.html"
        
        mock_visualizer = mock_visualizer_class.return_value
        mock_visualizer.generate_pdf_report.return_value = self.output_dir / "report.pdf"
        
        # Prepare inputs
        analysis_results = {
            "statistical_analysis": {"factuality": {"is_significant": True}}
        }
        visualization_paths = {
            "radar_chart": self.output_dir / "radar.png",
            "heatmap": self.output_dir / "heatmap.png"
        }
        
        # Call the function
        path = export_report(
            MOCK_BENCHMARK_RESULTS,
            analysis_results,
            visualization_paths,
            self.output_dir,
            self.timestamp,
            format=OutputFormat.HTML
        )
        
        # Assertions
        self.assertTrue(mock_analyzer.export_analysis_report.called)
        self.assertTrue(mock_visualizer.generate_pdf_report.called)
        self.assertEqual(path, self.output_dir / "report.html")
    
    def test_end_to_end_integration(self):
        """Test the full integration between all components."""
        # This test will mock external tools but test the integration points
        # between the different components
        
        # Set up mocks
        with patch('examples.integrated_benchmark_example.run_benchmark_suite') as mock_run, \
             patch('examples.integrated_benchmark_example.configure_thresholds') as mock_config, \
             patch('examples.integrated_benchmark_example.analyze_results') as mock_analyze, \
             patch('examples.integrated_benchmark_example.generate_visualizations') as mock_viz, \
             patch('examples.integrated_benchmark_example.export_report') as mock_export:
            
            # Configure mock returns
            mock_run.return_value = MOCK_BENCHMARK_RESULTS
            mock_config.return_value = ThresholdConfig()
            mock_analyze.return_value = {"statistical_analysis": {"factuality": {"is_significant": True}}}
            mock_viz.return_value = {"radar_chart": self.output_dir / "radar.png"}
            mock_export.return_value = self.output_dir / "report.html"
            
            # Import the main function
            from examples.integrated_benchmark_example import main
            
            # Mock command line arguments
            with patch('sys.argv', ['integrated_benchmark_example.py', 
                                    '--models', 'test-model', 
                                    '--domains', 'medical,legal',
                                    '--output-dir', str(self.output_dir),
                                    '--output-format', 'html']):
                # Call the main function
                with patch('argparse.ArgumentParser.parse_args') as mock_args:
                    # Configure mock args
                    args = Mock()
                    args.models = 'test-model'
                    args.domains = 'medical,legal'
                    args.output_dir = str(self.output_dir)
                    args.output_format = 'html'
                    args.parallel = True
                    args.significance_level = 0.05
                    args.optimize_thresholds = False
                    mock_args.return_value = args
                    
                    # Run the test
                    result = main()
            
            # Assertions to check the integration flow
            mock_run.assert_called_once()
            mock_config.assert_called_once_with(MOCK_BENCHMARK_RESULTS, optimize=False)
            mock_analyze.assert_called_once()
            mock_viz.assert_called_once()
            mock_export.assert_called_once()
            self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()