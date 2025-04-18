import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from registry.plugin_registry import create_instance, registry
from registry.registry_types import PluginType
from interfaces.constraint_interface import ConstraintInterface, ConstraintAssertionResult

logger = logging.getLogger(__name__)

@dataclass
class ConstraintValidationResult:
    """Result of constraint verification."""
    passed: bool
    results: List[ConstraintAssertionResult]
    violations: List[ConstraintAssertionResult]
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    mode: str = "strict"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "mode": self.mode,
            "results": [r.to_dict() for r in self.results],
            "violations": [v.to_dict() for v in self.violations],
            "metrics": self.metrics,
            "timestamp": self.timestamp
        }

class ConstraintVerifier:
    """Pipeline for validating constraints on input data."""
    def __init__(self, plugin_names: Optional[List[str]] = None, mode: str = "strict"):
        self.mode = mode
        self.plugins: List[ConstraintInterface] = []
        # Determine which plugins to load
        if plugin_names:
            names = plugin_names
        else:
            # Discover all registered constraint verifier plugins
            names = [meta.name for meta in registry.list_plugins(PluginType.CONSTRAINT_VERIFIER)]
        for name in names:
            instance = create_instance(plugin_type=PluginType.CONSTRAINT_VERIFIER, name=name)
            if instance:
                self.plugins.append(instance)

    def validate(self, input_data: Dict[str, Any], mode: Optional[str] = None) -> ConstraintValidationResult:
        """Validate input data against all registered constraints."""
        effective_mode = mode or self.mode
        start_time = time.time()
        results: List[ConstraintAssertionResult] = []
        violations: List[ConstraintAssertionResult] = []
        for plugin in self.plugins:
            try:
                if isinstance(plugin, ConstraintInterface):
                    assertion = plugin.assert_constraint(input_data)
                else:
                    # Fallback to generic process method
                    assertion = plugin.process(input_data)
                results.append(assertion)
                # Structured logging for each assertion
                logger.info(
                    "Constraint check",
                    extra={
                        "constraint": assertion.name,
                        "passed": assertion.passed,
                        "message": assertion.message,
                        "mode": effective_mode
                    }
                )
                if not assertion.passed:
                    violations.append(assertion)
                    if effective_mode == "strict":
                        break
            except Exception as e:
                logger.error(f"Error in constraint plugin {plugin}: {e}")
                failure = ConstraintAssertionResult(
                    name=getattr(plugin, "name", str(plugin)),
                    passed=False,
                    message=str(e)
                )
                results.append(failure)
                violations.append(failure)
                if effective_mode == "strict":
                    break
        passed = len(violations) == 0
        metrics = {
            "duration": time.time() - start_time,
            "total_checks": len(self.plugins),
            "violations": len(violations)
        }
        validation_result = ConstraintValidationResult(
            passed=passed,
            results=results,
            violations=violations,
            metrics=metrics,
            mode=effective_mode
        )
        # Summary logging for validation
        logger.info(
            "Constraint validation completed",
            extra={
                "passed": validation_result.passed,
                "violations": len(violations),
                "metrics": metrics
            }
        )
        return validation_result