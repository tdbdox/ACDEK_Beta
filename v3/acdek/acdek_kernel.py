# -*- coding: utf-8 -*-
import logging
import time
from typing import Any, Dict, Optional

from enrichment.enrichment_pipeline import create_standard_pipeline
from enrichment.constraint_verifier import ConstraintVerifier
from enrichment.confidence_scorer import VectorConfidenceScorer
from acdek.fallback_chain import FallbackChain, FallbackResult
from observability.resilience_events import log_fallback_event

logger = logging.getLogger(__name__)


class ACDEKKernel:
    """Central orchestration kernel for ACDEK v3."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize core components with optional configuration."""
        self.config = config or {}
        # Kernel-level fallback mode
        self.fallback_mode = self.config.get("fallback_mode", "strict")
        # Constraint plugin list override
        self.constraint_plugins = self.config.get("constraint_plugins", None)
        # Instantiate enrichment pipeline
        self.enrichment_pipeline = create_standard_pipeline()
        # Instantiate constraint verifier
        self.constraint_verifier = ConstraintVerifier(
            plugin_names=self.constraint_plugins,
            mode=self.fallback_mode
        )
        # Instantiate confidence scorer
        self.confidence_scorer = VectorConfidenceScorer()
        # Optional fallback chain configuration
        fc_conf = self.config.get("fallback_chain")
        if fc_conf:
            self.fallback_chain = FallbackChain(
                plugin_type=fc_conf.get("plugin_type"),
                plugin_names=fc_conf.get("plugin_names", []),
                circuit_breaker_config=fc_conf.get("circuit_breaker_config"),
                retry_config=fc_conf.get("retry_config"),
                fallback_mode=fc_conf.get("fallback_mode", self.fallback_mode)
            )
        else:
            self.fallback_chain = None
        logger.info("ACDEKKernel initialized", extra={"fallback_mode": self.fallback_mode})

    def run(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the full ACDEK pipeline: enrichment, constraint verification, and confidence scoring.

        Args:
            input_data: Raw input payload
            context: Optional context information
            config: Runtime overrides for kernel configuration
        Returns:
            Unified result dictionary with all stages and metadata
        """
        # Merge runtime config overrides
        runtime_cfg = {**self.config, **(config or {})}
        # Apply fallback mode override
        fb_mode = runtime_cfg.get("fallback_mode", self.fallback_mode)
        if fb_mode != self.fallback_mode:
            self.fallback_mode = fb_mode
            self.constraint_verifier.mode = fb_mode
            if self.fallback_chain:
                self.fallback_chain.fallback_mode = fb_mode
        # Apply constraint plugin override
        if "constraint_plugins" in runtime_cfg:
            self.constraint_verifier = ConstraintVerifier(
                plugin_names=runtime_cfg.get("constraint_plugins"),
                mode=self.fallback_mode
            )
        # 1) Enrichment stage
        start_enrich = time.time()
        if self.fallback_chain:
            # Wrap enrichment in fallback chain if configured
            fb_result: FallbackResult = self.fallback_chain.execute(
                "execute", input_data, context  # type: ignore
            )
            enrichment_res = fb_result.result
            fallback_depth = fb_result.depth
            fallback_penalty = fb_result.penalty
            # Log fallback event summary
            log_fallback_event(plugin="enrichment_pipeline", depth=fallback_depth,
                               reason="pipeline_fallback")
        else:
            enrichment_res = self.enrichment_pipeline.execute(input_data, context)
            fallback_depth = 0
            fallback_penalty = 0.0
        enrich_time = time.time() - start_enrich

        # 2) Constraint verification
        start_const = time.time()
        constraint_res = self.constraint_verifier.validate(enrichment_res.data)
        const_time = time.time() - start_const

        # 3) Confidence scoring
        start_conf = time.time()
        confidence_res = self.confidence_scorer.score(enrichment_res.data)
        conf_time = time.time() - start_conf

        # 4) Aggregate results
        result = {
            "enrichment": {
                "success": enrichment_res.success,
                "data": enrichment_res.data,
                "errors": enrichment_res.errors,
                "metrics": enrichment_res.metrics,
                "stage_results": enrichment_res.stage_results,
                "fallback_depth": fallback_depth,
                "fallback_penalty": fallback_penalty,
            },
            "constraints": constraint_res.to_dict(),
            "confidence": {
                "aggregate_score": confidence_res.aggregate_score,
                "factors": confidence_res.factors,
                "metadata": confidence_res.metadata,
                "timestamp": confidence_res.timestamp,
            },
            "timing": {
                "enrichment": enrich_time,
                "constraint": const_time,
                "confidence": conf_time,
                "total": enrich_time + const_time + conf_time
            },
            "config": runtime_cfg
        }
        # Structured log
        logger.info("ACDEK run completed", extra={
            "enrichment_success": enrichment_res.success,
            "constraint_passed": constraint_res.passed,
            "confidence_score": confidence_res.aggregate_score,
            "fallback_depth": fallback_depth
        })
        return result