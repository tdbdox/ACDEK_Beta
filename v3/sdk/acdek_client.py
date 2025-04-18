"""
Simplified ACDEK Python SDK: Client for enrichment, constraint validation, and confidence scoring.
"""
import json
from typing import Any, Dict, Optional

from enrichment.enrichment_pipeline import create_standard_pipeline
from enrichment.constraint_verifier import ConstraintVerifier
from enrichment.confidence_scorer import VectorConfidenceScorer


class Client:
    """
    ACDEK SDK Client.

    Methods:
        run_enrichment: Execute enrichment pipeline.
        validate_constraints: Run constraint validation.
        score_confidence: Compute confidence score.
        full_pipeline: Perform entire flow and return combined result.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ACDEK client with optional configuration.

        Config options:
            fallback_mode: 'strict', 'lenient', or 'deferred' (controls constraint verifier).
            constraint_plugins: List of plugin names for constraint verifier (None = all).
        """
        self.config = config or {}
        fallback_mode = self.config.get("fallback_mode", "strict")
        # Enrichment pipeline
        self.enrichment_pipeline = create_standard_pipeline()
        # Constraint verifier
        plugin_names = self.config.get("constraint_plugins", None)
        self.constraint_verifier = ConstraintVerifier(plugin_names=plugin_names, mode=fallback_mode)
        # Confidence scorer
        self.confidence_scorer = VectorConfidenceScorer()

    def run_enrichment(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        output_format: str = "dict",
    ) -> Any:
        """
        Execute the enrichment pipeline.

        Returns:
            dict or JSON string of enrichment result.
        """
        result = self.enrichment_pipeline.execute(input_data, context)
        out = {
            "success": result.success,
            "data": result.data,
            "errors": result.errors,
            "metrics": result.metrics,
            "stage_results": result.stage_results,
        }
        if output_format == "json":
            return json.dumps(out, indent=2)
        return out

    def validate_constraints(
        self,
        input_data: Dict[str, Any],
        output_format: str = "dict",
    ) -> Any:
        """
        Validate constraints on the given data.

        Returns:
            dict or JSON string of constraint validation result.
        """
        result = self.constraint_verifier.validate(input_data)
        out = result.to_dict()
        if output_format == "json":
            return json.dumps(out, indent=2)
        return out

    def score_confidence(
        self,
        input_data: Dict[str, Any],
        output_format: str = "dict",
    ) -> Any:
        """
        Compute confidence scores for the given data.

        Returns:
            dict or JSON string of confidence result.
        """
        score = self.confidence_scorer.score(input_data)
        out = {
            "aggregate_score": score.aggregate_score,
            "factors": {f.value if hasattr(f, "value") else str(f): v for f, v in score.factors.items()},
            "metadata": score.metadata,
            "timestamp": score.timestamp,
        }
        if output_format == "json":
            return json.dumps(out, indent=2)
        return out

    def full_pipeline(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        output_format: str = "dict",
    ) -> Any:
        """
        Execute enrichment, constraint validation, and confidence scoring.

        Returns:
            Combined result as dict or JSON string.
        """
        enrichment = self.run_enrichment(input_data, context, output_format="dict")
        constraints = self.validate_constraints(enrichment["data"], output_format="dict")
        confidence = self.score_confidence(enrichment["data"], output_format="dict")
        out = {
            "enrichment": enrichment,
            "constraints": constraints,
            "confidence": confidence,
        }
        if output_format == "json":
            return json.dumps(out, indent=2)
        return out