"""
Integration tests for hallucination metrics system.

Tests the end-to-end functionality of the hallucination metrics system,
including the drift detector, confidence scorer, and metrics collector.
"""

import unittest
import json
import tempfile
import os
from typing import Dict, List, Any

# Import components for testing
from enrichment.hallucination_metrics import (
    HallucinationMetricsCollector,
    ReferenceSource,
    FactualityGrade,
    DriftSeverity,
    ConfidenceCorrelation
)
from enrichment.drift_detector import SemanticDriftDetector
from enrichment.confidence_scorer import VectorConfidenceScorer
from enrichment.hallucination_connector import HallucinationConnector


class TestHallucinationMetrics(unittest.TestCase):
    """Test the hallucination metrics system components."""
    
    def setUp(self):
        """Set up common test resources."""
        # Create test components
        self.drift_detector = SemanticDriftDetector()
        self.confidence_scorer = VectorConfidenceScorer()
        
        # Use temporary directory for test results
        self.test_dir = tempfile.mkdtemp()
        
        # Create metrics collector
        self.metrics_collector = HallucinationMetricsCollector(
            store_results=True,
            results_dir=self.test_dir
        )
        
        # Create connector that links all components
        self.connector = HallucinationConnector(
            drift_detector=self.drift_detector,
            confidence_scorer=self.confidence_scorer,
            metrics_collector=self.metrics_collector
        )
        
        # Test data
        self.test_prompt = "What are the health benefits of regular exercise?"
        self.test_response = """
            Regular exercise has several proven health benefits:
            - Improves cardiovascular health
            - Helps maintain healthy weight
            - Reduces risk of chronic diseases
            - Strengthens muscles and bones
            - Enhances mental health and mood
            
            However, exercise also directly increases IQ by 20 points and guarantees a longer lifespan of exactly 12.5 years for everyone who exercises 30 minutes daily.
        """
        
        self.reference_sources = [
            ReferenceSource(
                id="ref_1",
                content="""
                Regular physical activity can improve your muscle strength and boost your endurance. 
                Exercise delivers oxygen and nutrients to your tissues and helps your cardiovascular system work more efficiently.
                When your heart and lung health improve, you have more energy.
                """,
                source_type="text"
            ),
            ReferenceSource(
                id="ref_2",
                content="""
                Regular exercise can help you control your weight, reduce your risk of heart disease, and strengthen your bones and muscles.
                It can also improve your mental health and mood, as well as your ability to do daily activities.
                Regular physical activity can help maintain thinking, learning, and judgment skills as you age.
                """,
                source_type="text"
            )
        ]
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up test files
        for file in os.listdir(self.test_dir):
            try:
                os.remove(os.path.join(self.test_dir, file))
            except:
                pass
        try:
            os.rmdir(self.test_dir)
        except:
            pass
    
    def test_factuality_scoring(self):
        """Test factuality scoring against reference sources."""
        # Test direct metrics collector use
        metrics_result = self.metrics_collector.collect_metrics(
            prompt=self.test_prompt,
            response=self.test_response,
            reference_sources=self.reference_sources,
            reported_confidence=0.8
        )
        
        # Check result structure
        self.assertIsNotNone(metrics_result.factuality_score)
        self.assertIsNotNone(metrics_result.factuality_grade)
        self.assertTrue(0.0 <= metrics_result.factuality_score <= 1.0)
        
        # The response has some factual content and some fabricated claims
        # so factuality should be moderate to low
        self.assertTrue(
            metrics_result.factuality_grade in [FactualityGrade.MODERATE, FactualityGrade.LOW],
            f"Unexpected factuality grade: {metrics_result.factuality_grade}"
        )
    
    def test_drift_detection(self):
        """Test semantic drift detection."""
        # Test drift detection through connector
        drift_result = self.connector.detect_semantic_drift(
            prompt=self.test_prompt,
            response=self.test_response
        )
        
        # Check result structure
        self.assertIn("severity", drift_result)
        self.assertIn("drift_score", drift_result)
        self.assertIn("original_context_similarity", drift_result)
        
        # The response drifts at the end with fabricated claims
        self.assertIn(
            drift_result["severity"],
            [DriftSeverity.MINOR.value, DriftSeverity.MODERATE.value, DriftSeverity.SEVERE.value],
            f"Expected drift detection, got: {drift_result['severity']}"
        )
    
    def test_confidence_correlation(self):
        """Test confidence-factuality correlation analysis."""
        # Test with various confidence-factuality pairs
        test_cases = [
            # reported_confidence, factuality_score, expected_type
            (0.9, 0.9, ConfidenceCorrelation.CONFIDENTLY_RIGHT),  # High confidence, high factuality
            (0.9, 0.3, ConfidenceCorrelation.CONFIDENTLY_WRONG),  # High confidence, low factuality
            (0.3, 0.9, ConfidenceCorrelation.UNDERCONFIDENT),     # Low confidence, high factuality
            (0.7, 0.7, ConfidenceCorrelation.ACCURATE)            # Matching confidence and factuality
        ]
        
        for confidence, factuality, expected_type in test_cases:
            correlation_result = self.connector.update_confidence_with_factuality(
                confidence_score=confidence,
                factuality_score=factuality
            )
            
            # Check result structure
            self.assertIn("correlation_type", correlation_result)
            self.assertIn("correlation_score", correlation_result)
            self.assertIn("is_reliable", correlation_result)
            
            # Check expected correlation type
            self.assertEqual(
                correlation_result["correlation_type"],
                expected_type.value,
                f"For confidence={confidence}, factuality={factuality}: "
                f"Expected {expected_type.value}, got {correlation_result['correlation_type']}"
            )
    
    def test_end_to_end_integration(self):
        """Test end-to-end integration of all components."""
        # Analyze response using the connector
        result = self.connector.analyze_response(
            prompt=self.test_prompt,
            response=self.test_response,
            reference_sources=self.reference_sources,
            reported_confidence=0.9,  # High confidence for a partially factual response
            model_info={"name": "test_model", "version": "1.0"}
        )
        
        # Check comprehensive result
        self.assertIsNotNone(result.factuality_score)
        self.assertIsNotNone(result.factuality_grade)
        self.assertIsNotNone(result.semantic_drift)
        self.assertIsNotNone(result.confidence_analysis)
        
        # Should detect issues due to fabricated claims
        self.assertGreater(len(result.detected_issues), 0)
        
        # Overconfidence detection
        self.assertEqual(
            result.confidence_analysis.correlation_type,
            ConfidenceCorrelation.OVERCONFIDENT,  # Should detect overconfidence
            f"Expected overconfidence detection, got: {result.confidence_analysis.correlation_type}"
        )
        
        # Check export functionality
        export_path = self.metrics_collector.export_metrics_report(
            format="json",
            file_path=os.path.join(self.test_dir, "test_report.json")
        )
        
        self.assertTrue(os.path.exists(export_path))
        
        # Verify exported data
        with open(export_path, 'r') as f:
            exported_data = json.load(f)
            self.assertIn("factuality_stats", exported_data)
            self.assertIn("drift_stats", exported_data)
            self.assertIn("confidence_stats", exported_data)


if __name__ == "__main__":
    unittest.main()