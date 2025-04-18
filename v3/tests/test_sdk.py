"""
Unit tests for ACDEK SDK Client and core enrichment, constraint, and confidence components.
"""
import json
import numpy as np
import pytest
from types import SimpleNamespace

from sdk.acdek_client import Client
from enrichment.constraint_verifier import ConstraintVerifier
from enrichment.confidence_scorer import VectorConfidenceScorer, ConfidenceFactorType
from interfaces.constraint_interface import ConstraintAssertionResult, ConstraintInterface

class DummyConstraint(ConstraintInterface):
    def assert_constraint(self, input_data):
        return ConstraintAssertionResult(name="dummy", passed=False, message="fail")

def test_run_enrichment_dict():
    dummy = SimpleNamespace(
        success=True,
        data={"key": "value"},
        errors=None,
        metrics={"m": 1},
        stage_results={"stage": {}}
    )
    client = Client()
    client.enrichment_pipeline = SimpleNamespace(execute=lambda data, context: dummy)
    result = client.run_enrichment({"in": 1}, output_format="dict")
    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["data"] == {"key": "value"}
    assert result["metrics"] == {"m": 1}
    assert result["stage_results"] == {"stage": {}}

def test_run_enrichment_json():
    dummy = SimpleNamespace(
        success=False,
        data={},
        errors={"e": "err"},
        metrics={},
        stage_results={}
    )
    client = Client()
    client.enrichment_pipeline = SimpleNamespace(execute=lambda data, context: dummy)
    out = client.run_enrichment({}, output_format="json")
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert parsed["success"] is False
    assert parsed["errors"] == {"e": "err"}

def test_constraint_verifier_default_pass():
    cv = ConstraintVerifier(plugin_names=[])
    res = cv.validate({})
    d = res.to_dict()
    assert d["passed"] is True
    assert isinstance(d["violations"], list) and len(d["violations"]) == 0

def test_constraint_verifier_with_plugin():
    cv = ConstraintVerifier(plugin_names=[])
    cv.plugins = [DummyConstraint()]
    res = cv.validate({"a": 1})
    d = res.to_dict()
    assert d["passed"] is False
    assert len(d["violations"]) == 1

def test_confidence_scoring_output_structure():
    scorer = VectorConfidenceScorer()
    data = {"vector": np.array([1.0, 0.0]), "expected_dimension": 2}
    score = scorer.score(data)
    # Check types and presence
    assert isinstance(score.aggregate_score, float)
    assert isinstance(score.factors, dict)
    assert all(isinstance(k, ConfidenceFactorType) for k in score.factors.keys())
    assert isinstance(score.metadata, dict)
    assert isinstance(score.timestamp, float)
    # Check normalization bounds
    assert 0.0 <= score.aggregate_score <= 1.0