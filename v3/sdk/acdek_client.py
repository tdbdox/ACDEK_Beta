"""
Simplified ACDEK Python SDK: Client for enrichment, constraint validation, and confidence scoring.
"""
import json
from typing import Any, Dict, Optional

from enrichment.standard_pipelines import create_standard_pipeline, create_minimal_pipeline, dummy_execute_pipeline
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
            fallback_mode: 'strict', 'relaxed', 'progressive', 'deferred', or 'evidence' (controls constraint verifier).
            constraint_plugins: List of plugin names for constraint verifier (None = all).
            pipeline_type: 'standard', 'minimal', or 'dummy' (controls which pipeline to use).
        """
        self.config = config or {}
        fallback_mode = self.config.get("fallback_mode", "strict")
        pipeline_type = self.config.get("pipeline_type", "standard")
        
        # Initialize components with error handling
        try:
            # Enrichment pipeline
            if pipeline_type == "minimal":
                self.enrichment_pipeline = create_minimal_pipeline()
            elif pipeline_type == "dummy":
                self.enrichment_pipeline = None  # Will use dummy_execute_pipeline instead
            else:  # Default to standard
                self.enrichment_pipeline = create_standard_pipeline()
            
            # Constraint verifier
            plugin_names = self.config.get("constraint_plugins", None)
            try:
                self.constraint_verifier = ConstraintVerifier(plugin_names=plugin_names, mode=fallback_mode)
            except ValueError:
                # Fallback to relaxed mode if specified mode is invalid
                self.constraint_verifier = ConstraintVerifier(plugin_names=plugin_names, mode="relaxed")
            
            # Confidence scorer
            try:
                self.confidence_scorer = VectorConfidenceScorer()
            except Exception as e:
                # Log error but continue - we'll handle this in the methods
                import logging
                logging.getLogger(__name__).error(f"Failed to initialize confidence scorer: {e}")
                self.confidence_scorer = None
                
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error initializing ACDEK client: {e}")
            # Set up minimal functionality to avoid breaking
            self.enrichment_pipeline = None
            self.constraint_verifier = None
            self.confidence_scorer = None

    def run_enrichment(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        output_format: str = "dict",
    ) -> Any:
        """
        Execute the enrichment pipeline.
        
        Args:
            input_data: Input data to enrich
            context: Optional context information
            output_format: Output format ('dict' or 'json')

        Returns:
            dict or JSON string of enrichment result.
        """
        # Handle case when pipeline isn't available
        if self.enrichment_pipeline is None:
            # Use dummy implementation
            result = dummy_execute_pipeline(input_data, context)
        else:
            try:
                # Run the actual pipeline
                result = self.enrichment_pipeline.execute(input_data, context)
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Error executing enrichment pipeline: {e}")
                # Fallback to dummy implementation
                result = dummy_execute_pipeline(input_data, context)
                
        # Format the result
        out = {
            "success": result.success,
            "data": result.data,
            "errors": result.errors if hasattr(result, "errors") else [],
            "metrics": result.metrics if hasattr(result, "metrics") else {},
            "stage_results": result.stage_results if hasattr(result, "stage_results") else {},
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
        
        Args:
            input_data: Data to validate
            output_format: Output format ('dict' or 'json')

        Returns:
            dict or JSON string of constraint validation result.
        """
        # Handle case when constraint verifier isn't available
        if self.constraint_verifier is None:
            # Return dummy result
            out = {
                "passed": True,
                "violations": [],
                "warning": "Using dummy constraint verifier"
            }
        else:
            try:
                # Run actual constraint verification
                result = self.constraint_verifier.validate(input_data)
                out = result.to_dict()
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(f"Error validating constraints: {e}")
                # Return dummy result with error
                out = {
                    "passed": False,
                    "violations": [{
                        "name": "internal_error",
                        "message": f"Error validating constraints: {str(e)}",
                        "severity": "ERROR"
                    }],
                    "error": str(e)
                }
                
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
        
        Args:
            input_data: Data to score
            output_format: Output format ('dict' or 'json')

        Returns:
            dict or JSON string of confidence result.
        """
        # Handle case when confidence scorer isn't available
        if self.confidence_scorer is None:
            # Return dummy confidence score
            import time
            out = {
                "aggregate_score": 0.5,  # Neutral score
                "factors": {
                    "dummy_factor": 0.5
                },
                "metadata": {
                    "warning": "Using dummy confidence scorer"
                },
                "timestamp": time.time(),
            }
        else:
            try:
                # Run actual confidence scoring
                score = self.confidence_scorer.score(input_data)
                out = {
                    "aggregate_score": score.aggregate_score,
                    "factors": {f.value if hasattr(f, "value") else str(f): v for f, v in score.factors.items()},
                    "metadata": score.metadata,
                    "timestamp": score.timestamp,
                }
            except Exception as e:
                import logging, time
                logging.getLogger(__name__).error(f"Error scoring confidence: {e}")
                # Return dummy result with error
                out = {
                    "aggregate_score": 0.5,  # Neutral score
                    "factors": {
                        "error": 0.0
                    },
                    "metadata": {
                        "error": str(e),
                        "warning": "Error during confidence scoring"
                    },
                    "timestamp": time.time(),
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
        
        Args:
            input_data: Input data to process
            context: Optional context information
            output_format: Output format ('dict' or 'json')

        Returns:
            Combined result as dict or JSON string.
        """
        import time
        start_time = time.time()
        
        # Run each component with proper error handling
        try:
            enrichment = self.run_enrichment(input_data, context, output_format="dict")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error in enrichment stage: {e}")
            # Create minimal working enrichment result
            enrichment = {
                "success": False,
                "data": input_data,
                "errors": [str(e)],
                "metrics": {},
                "stage_results": {}
            }
        
        # Always use the original input if enrichment failed or didn't provide data
        data_for_validation = enrichment.get("data", input_data) if enrichment.get("success", False) else input_data
        
        try:
            constraints = self.validate_constraints(data_for_validation, output_format="dict")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error in constraint validation stage: {e}")
            # Create minimal working constraint result
            constraints = {
                "passed": False,
                "violations": [{
                    "name": "system_error",
                    "message": str(e),
                    "severity": "ERROR"
                }]
            }
        
        try:
            confidence = self.score_confidence(data_for_validation, output_format="dict")
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error in confidence scoring stage: {e}")
            # Create minimal working confidence result
            confidence = {
                "aggregate_score": 0.5,
                "factors": {"error": 0.0},
                "metadata": {"error": str(e)},
                "timestamp": time.time()
            }
        
        # Combine all results
        out = {
            "enrichment": enrichment,
            "constraints": constraints,
            "confidence": confidence,
            "timing": {
                "total": time.time() - start_time,
                "enrichment": enrichment.get("metrics", {}).get("duration", 0),
                "constraint": constraints.get("metrics", {}).get("duration", 0),
                "confidence": confidence.get("metadata", {}).get("duration", 0)
            }
        }
        
        if output_format == "json":
            return json.dumps(out, indent=2)
        return out