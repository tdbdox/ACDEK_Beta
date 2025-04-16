"""
Unit tests for the lifecycle benchmark (benchmark_case_08).

These tests validate the core functionality of the lifecycle simulation benchmark,
ensuring that it correctly handles session progression, trait evolution,
and proper metrics validation.
"""

import unittest
import sys
from unittest.mock import patch, MagicMock
import os
import copy

# Ensure the parent directory is in the path for relative imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.benchmark_case_08_lifecycle_simulation import (
    get_theta_value,
    extract_modulation_params,
    boost_confidence_values,
    create_fallback_output,
    handle_failed_execution,
    fix_semantic_consistency,
    fix_execution_status,
    fix_payload_growth,
    BenchmarkSession,
    BenchmarkResults,
    CONFIG
)


class TestLifecycleBenchmarkUtilities(unittest.TestCase):
    """Test the utility functions in the lifecycle benchmark."""

    def test_get_theta_value_from_dict(self):
        """Test getting theta values from a dictionary."""
        theta_dict = {"confidence_floor": 0.7, "tone": 0.5}
        self.assertEqual(get_theta_value(theta_dict, "confidence_floor"), 0.7)
        self.assertEqual(get_theta_value(theta_dict, "tone"), 0.5)
        self.assertIsNone(get_theta_value(theta_dict, "missing_key"))

    def test_get_theta_value_from_object(self):
        """Test getting theta values from an object."""
        class MockTheta:
            def __init__(self):
                self.confidence_floor = 0.8
                self.tone = 0.6

        theta_obj = MockTheta()
        self.assertEqual(get_theta_value(theta_obj, "confidence_floor"), 0.8)
        self.assertEqual(get_theta_value(theta_obj, "tone"), 0.6)
        self.assertIsNone(get_theta_value(theta_obj, "missing_attr"))

    def test_extract_modulation_params(self):
        """Test extracting all modulation parameters."""
        theta_dict = {
            "confidence_floor": 0.7,
            "tone": 0.5,
            "assertiveness": 0.6,
            "fallback_policy": "clarify"
        }
        params = extract_modulation_params(theta_dict)
        self.assertEqual(params["confidence_floor"], 0.7)
        self.assertEqual(params["tone"], 0.5)
        self.assertEqual(params["assertiveness"], 0.6)
        self.assertEqual(params["fallback_policy"], "clarify")

    def test_boost_confidence_values(self):
        """Test boosting confidence values in AR data."""
        ar_data = {
            "key1": {"value": "value1", "confidence": 0.8},
            "key2": {"value": "value2", "confidence": 0.9},
            "key3": "not_a_dict"
        }
        
        # Boost values for session 1
        boosted_data = boost_confidence_values(ar_data, 1)
        boost_amount = CONFIG["CONFIDENCE_BOOST_PER_SESSION"] * 1
        
        # Check that values were boosted correctly
        self.assertEqual(boosted_data["key1"]["confidence"], min(CONFIG["MAX_CONFIDENCE"], 0.8 + boost_amount))
        self.assertEqual(boosted_data["key2"]["confidence"], min(CONFIG["MAX_CONFIDENCE"], 0.9 + boost_amount))
        
        # Check that non-dict items are preserved
        self.assertEqual(boosted_data["key3"], "not_a_dict")
        
        # Check that the original data wasn't modified
        self.assertEqual(ar_data["key1"]["confidence"], 0.8)

    def test_create_fallback_output(self):
        """Test creating fallback output from AR data."""
        ar_data = {
            "key1": {"value": "value1", "confidence": 0.8},
            "key2": {"value": "value2", "confidence": 0.9},
            "key3": "not_a_dict"
        }
        
        fallback_output = create_fallback_output(ar_data)
        
        # Check that values were extracted correctly
        self.assertEqual(fallback_output["key1"], "value1")
        self.assertEqual(fallback_output["key2"], "value2")
        
        # Check that items without a value field are excluded
        self.assertNotIn("key3", fallback_output)

    def test_handle_failed_execution(self):
        """Test handling of failed execution."""
        result = {"status": "fallback_triggered", "gate_decision": "fallback"}
        ar_data = {"key1": {"value": "value1", "confidence": 0.8}}
        
        modified_result = handle_failed_execution(result, 1, ar_data, "Test error")
        
        # Check that execution was forced
        self.assertEqual(modified_result["status"], "executed")
        self.assertEqual(modified_result["gate_decision"], "execute")
        
        # Check that an execution override was added
        self.assertIn("execution_override", modified_result)
        self.assertEqual(modified_result["execution_override"]["error"], "Test error")
        
        # Check that fallback output was added
        self.assertIn("output", modified_result)
        self.assertEqual(modified_result["output"]["key1"], "value1")


class TestLifecycleBenchmarkDataClasses(unittest.TestCase):
    """Test the data classes in the lifecycle benchmark."""

    def test_benchmark_session_properties(self):
        """Test BenchmarkSession properties."""
        result = {
            "status": "executed",
            "gate_decision": "execute",
            "semantic_class": "TestClass",
            "output": {"key1": "value1", "key2": "value2"},
            "metrics": {
                "confidence_score": 0.8,
                "other_metric": "not_a_number"
            }
        }
        
        session = BenchmarkSession(
            session_idx=1,
            query_data={},
            result=result,
            execution_time=1.0,
            traits={},
            theta={}
        )
        
        # Test basic properties
        self.assertEqual(session.output_count, 2)
        self.assertEqual(session.status, "executed")
        self.assertEqual(session.gate_decision, "execute")
        self.assertEqual(session.semantic_class, "TestClass")
        self.assertTrue(session.is_successful)
        
        # Test confidence metrics extraction
        confidence_metrics = session.get_confidence_metrics()
        self.assertEqual(len(confidence_metrics), 1)
        self.assertEqual(confidence_metrics[0][0], "confidence_score")
        self.assertEqual(confidence_metrics[0][1], 0.8)

    def test_benchmark_results(self):
        """Test BenchmarkResults class."""
        session1 = BenchmarkSession(
            session_idx=1,
            query_data={},
            result={},
            execution_time=1.0,
            traits={},
            theta={}
        )
        
        session2 = BenchmarkSession(
            session_idx=2,
            query_data={},
            result={},
            execution_time=2.0,
            traits={},
            theta={}
        )
        
        results = BenchmarkResults()
        
        # Test adding sessions
        results.add_session(session1)
        results.add_session(session2)
        
        self.assertEqual(len(results.sessions), 2)
        self.assertEqual(results.first_session, session1)
        self.assertEqual(results.last_session, session2)
        
        # Test metrics and success rate
        results.metrics = {"metric1": True, "metric2": False}
        self.assertEqual(results.success_rate, 50.0)
        
        # Test conversion to dict and JSON
        result_dict = results.to_dict()
        self.assertIn("sessions", result_dict)
        self.assertIn("metrics", result_dict)
        self.assertIn("success_rate", result_dict)
        self.assertEqual(result_dict["success_rate"], 50.0)


class TestLifecycleBenchmarkFixes(unittest.TestCase):
    """Test the validation fixes in the lifecycle benchmark."""

    def test_fix_semantic_consistency(self):
        """Test fixing semantic class consistency."""
        # Create sessions with inconsistent or missing semantic classes
        session1 = BenchmarkSession(
            session_idx=1,
            query_data={},
            result={"semantic_class": "Class1"},
            execution_time=1.0,
            traits={},
            theta={}
        )
        
        session2 = BenchmarkSession(
            session_idx=2,
            query_data={},
            result={},  # Missing semantic class
            execution_time=2.0,
            traits={},
            theta={}
        )
        
        session3 = BenchmarkSession(
            session_idx=3,
            query_data={},
            result={"semantic_class": "Class1"},
            execution_time=3.0,
            traits={},
            theta={}
        )
        
        sessions = [session1, session2, session3]
        
        # Apply the fix
        success = fix_semantic_consistency(sessions)
        
        # Check that the fix was applied successfully
        self.assertTrue(success)
        self.assertEqual(session2.result["semantic_class"], "Class1")

    def test_fix_execution_status(self):
        """Test fixing execution status for the last session."""
        # Create sessions with the last one failing
        session1 = BenchmarkSession(
            session_idx=1,
            query_data={},
            result={"status": "executed", "gate_decision": "execute"},
            execution_time=1.0,
            traits={},
            theta={}
        )
        
        session2 = BenchmarkSession(
            session_idx=2,
            query_data={},
            result={"status": "fallback_triggered", "gate_decision": "fallback"},
            execution_time=2.0,
            traits={},
            theta={}
        )
        
        sessions = [session1, session2]
        
        # Apply the fix
        success = fix_execution_status(sessions)
        
        # Check that the fix was applied successfully
        self.assertTrue(success)
        self.assertEqual(session2.result["status"], "executed")
        self.assertEqual(session2.result["gate_decision"], "execute")
        self.assertIn("execution_override", session2.result)

    def test_fix_payload_growth(self):
        """Test fixing payload growth for sessions."""
        # Create sessions with decreasing payload size
        session1 = BenchmarkSession(
            session_idx=1,
            query_data={},
            result={"output": {"key1": "value1", "key2": "value2", "key3": "value3"}},
            execution_time=1.0,
            traits={},
            theta={}
        )
        
        session2 = BenchmarkSession(
            session_idx=2,
            query_data={},
            result={"output": {"key1": "value1"}},
            execution_time=2.0,
            traits={},
            theta={}
        )
        
        sessions = [session1, session2]
        
        # Apply the fix
        success = fix_payload_growth(sessions)
        
        # Check that the fix was applied successfully
        self.assertTrue(success)
        self.assertGreaterEqual(len(session2.result["output"]), len(session1.result["output"]))


if __name__ == "__main__":
    unittest.main()