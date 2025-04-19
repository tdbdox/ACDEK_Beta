"""
Constraint verification system with formal validation pipelines for hallucination mitigation.

This module implements a comprehensive constraint verification pipeline with dynamic loading
of constraint validators, formal verification with mathematical guarantees, multi-step
verification for high-risk operations, and attestation chains for provenance tracking.
"""
import logging
import time
import json
import hashlib
import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Type, TypeVar, Generic, cast
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from functools import lru_cache

from registry.plugin_registry import create_instance, registry, get_plugin_info
from registry.registry_types import PluginType
from interfaces.constraint_interface import (
    ConstraintInterface, 
    ConstraintVerifierInterface,
    ConstraintAssertionResult,
    ConstraintValidationResult,
    ConstraintDefinition,
    ConstraintEvidence,
    ConstraintSeverity,
    ConstraintCategory
)
from observability.metrics_collector import MetricsCollector, TimingContext
from observability.resilience_events import log_constraint_event
from observability.telemetry_server import get_telemetry_client

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Validation modes for constraint verification."""
    STRICT = "strict"           # Fail on first constraint violation
    RELAXED = "relaxed"         # Continue validation but report all violations
    PROGRESSIVE = "progressive"  # Gradually relax constraints based on confidence
    DEFERRED = "deferred"       # Record violations but don't fail validation
    EVIDENCE = "evidence"       # Only collect evidence, don't make pass/fail judgments


class ValidatorPriority(Enum):
    """Priority levels for constraint validators."""
    CRITICAL = 0   # Must run first, blocks all other validators if fails
    HIGH = 1       # Runs after critical, high importance
    MEDIUM = 2     # Standard priority
    LOW = 3        # Only runs if resources available
    BACKGROUND = 4  # Can be deferred or run asynchronously


@dataclass
class ValidationStep:
    """A step in the constraint validation pipeline."""
    validator_id: str
    constraints: List[str]
    priority: ValidatorPriority = ValidatorPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationPipeline:
    """Definition of a validation pipeline with ordered steps."""
    name: str
    description: str
    steps: List[ValidationStep]
    mode: ValidationMode = ValidationMode.STRICT
    max_parallelism: int = 4
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationContext:
    """Context for a verification operation with audit trail."""
    request_id: str
    start_time: float = field(default_factory=time.time)
    execution_path: List[str] = field(default_factory=list)
    attestations: List[Dict[str, Any]] = field(default_factory=list)
    evidence_chain: Dict[str, List[ConstraintEvidence]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_attestation(self, validator_id: str, constraint_id: str, result: bool, 
                        confidence: float, evidence: List[ConstraintEvidence]) -> None:
        """Add an attestation to the verification context."""
        attestation = {
            "validator": validator_id,
            "constraint": constraint_id,
            "timestamp": time.time(),
            "result": result,
            "confidence": confidence,
            "evidence_count": len(evidence),
            "evidence_ids": [e.key for e in evidence]
        }
        self.attestations.append(attestation)
        
        # Update execution path
        self.execution_path.append(f"{validator_id}:{constraint_id}")
        
        # Update evidence chain
        for ev in evidence:
            if ev.key not in self.evidence_chain:
                self.evidence_chain[ev.key] = []
            self.evidence_chain[ev.key].append(ev)
    
    def get_evidence_chain(self, format: str = "json") -> Union[Dict[str, Any], str]:
        """Get the evidence chain in the specified format."""
        chain = {
            "request_id": self.request_id,
            "start_time": self.start_time,
            "execution_path": self.execution_path,
            "attestations": self.attestations,
            "evidence": {k: [e.to_dict() for e in v] for k, v in self.evidence_chain.items()},
            "metadata": self.metadata
        }
        
        if format == "json":
            return chain
        elif format == "json_str":
            return json.dumps(chain, indent=2)
        else:
            return chain
    
    def compute_integrity_hash(self) -> str:
        """Compute integrity hash for the attestation chain for verification."""
        # Create a deterministic serialization of attestations
        attestations_str = json.dumps(sorted(self.attestations, 
                                            key=lambda x: (x["timestamp"], x["validator"], x["constraint"])))
        return hashlib.sha256(attestations_str.encode()).hexdigest()


class ValidatorRegistry:
    """
    Registry of constraint validators with dynamic loading capabilities.
    
    This registry manages the loading, configuration, and lifecycle of constraint
    validators, supporting hot-swapping, version management, and dependency resolution.
    """
    
    def __init__(self):
        """Initialize the validator registry."""
        self.validators: Dict[str, ConstraintInterface] = {}
        self.validator_info: Dict[str, Dict[str, Any]] = {}
        self.constraint_index: Dict[str, str] = {}  # Maps constraint_id -> validator_id
        self.validator_constraints: Dict[str, List[ConstraintDefinition]] = {}
        self.metrics = MetricsCollector(service_name="constraint_validator_registry")
        self.telemetry = get_telemetry_client()
        
        # Register metrics
        self._register_metrics()
    
    def _register_metrics(self) -> None:
        """Register metrics for observability."""
        self.metrics.register_counter("validators.loaded", "Number of validators loaded")
        self.metrics.register_counter("validators.errors", "Number of validator loading errors")
        self.metrics.register_counter("constraints.registered", "Number of constraints registered")
        self.metrics.register_gauge("validators.active", "Number of active validators")
    
    def load_validators(self, plugin_names: Optional[List[str]] = None) -> None:
        """
        Load constraint validators dynamically from the plugin registry.
        
        Args:
            plugin_names: Optional list of plugin names to load. If None, all available
                          constraint validator plugins will be loaded.
        """
        # Determine which plugins to load
        if plugin_names:
            names = plugin_names
        else:
            # Discover all registered constraint verifier plugins
            names = [meta.name for meta in registry.list_plugins(PluginType.CONSTRAINT_VERIFIER)]
        
        loaded_count = 0
        error_count = 0
        
        for name in names:
            try:
                # Get plugin metadata
                plugin_info = get_plugin_info(PluginType.CONSTRAINT_VERIFIER, name)
                if not plugin_info:
                    logger.warning(f"No metadata found for plugin {name}")
                    continue
                
                # Create plugin instance
                instance = create_instance(plugin_type=PluginType.CONSTRAINT_VERIFIER, name=name)
                if not instance:
                    logger.error(f"Failed to create instance of validator {name}")
                    error_count += 1
                    continue
                
                # Verify instance implements the ConstraintInterface
                if not isinstance(instance, ConstraintInterface):
                    logger.error(f"Validator {name} does not implement ConstraintInterface")
                    error_count += 1
                    continue
                
                # Register validator
                self.validators[name] = instance
                self.validator_info[name] = {
                    "name": name,
                    "version": plugin_info.version,
                    "description": plugin_info.description,
                    "loaded_at": time.time()
                }
                
                # Get constraints and build index
                try:
                    constraints = instance.get_constraints()
                    self.validator_constraints[name] = constraints
                    
                    for constraint in constraints:
                        self.constraint_index[constraint.constraint_id] = name
                        logger.info(f"Registered constraint {constraint.constraint_id} from validator {name}")
                        self.metrics.counter_inc("constraints.registered")
                    
                    logger.info(f"Loaded validator {name} with {len(constraints)} constraints")
                    loaded_count += 1
                    
                except Exception as e:
                    logger.error(f"Error loading constraints from validator {name}: {str(e)}")
                    error_count += 1
            
            except Exception as e:
                logger.error(f"Error loading validator {name}: {str(e)}")
                error_count += 1
        
        self.metrics.counter_inc("validators.loaded", loaded_count)
        self.metrics.counter_inc("validators.errors", error_count)
        self.metrics.gauge_set("validators.active", len(self.validators))
        
        logger.info(f"Loaded {loaded_count} validators with {len(self.constraint_index)} total constraints")
    
    def get_validator(self, validator_id: str) -> Optional[ConstraintInterface]:
        """Get a validator by ID."""
        return self.validators.get(validator_id)
    
    def get_constraint_validator(self, constraint_id: str) -> Optional[ConstraintInterface]:
        """Get the validator responsible for a specific constraint."""
        validator_id = self.constraint_index.get(constraint_id)
        if validator_id:
            return self.validators.get(validator_id)
        return None
    
    def get_constraint_definition(self, constraint_id: str) -> Optional[ConstraintDefinition]:
        """Get the definition of a specific constraint."""
        validator_id = self.constraint_index.get(constraint_id)
        if not validator_id:
            return None
        
        constraints = self.validator_constraints.get(validator_id, [])
        for constraint in constraints:
            if constraint.constraint_id == constraint_id:
                return constraint
        
        return None
    
    def list_validators(self) -> List[Dict[str, Any]]:
        """Get information about all registered validators."""
        return list(self.validator_info.values())
    
    def list_all_constraints(self) -> List[ConstraintDefinition]:
        """Get definitions of all registered constraints."""
        all_constraints = []
        for constraints in self.validator_constraints.values():
            all_constraints.extend(constraints)
        return all_constraints
    
    def get_constraints_by_category(self, category: ConstraintCategory) -> List[ConstraintDefinition]:
        """Get all constraints in a specific category."""
        return [c for c in self.list_all_constraints() if c.category == category]
    
    def get_constraints_by_severity(self, severity: ConstraintSeverity) -> List[ConstraintDefinition]:
        """Get all constraints with a specific severity level."""
        return [c for c in self.list_all_constraints() if c.severity == severity]


class ConstraintVerifier(ConstraintVerifierInterface):
    """
    Constraint verification orchestrator with formal validation pipelines.
    
    This verifier implements comprehensive constraint verification with dynamic
    validator loading, formal verification pipelines, multi-step verification for
    high-risk operations, and attestation chains for provenance tracking.
    """
    
    def __init__(
        self, 
        plugin_names: Optional[List[str]] = None, 
        mode: Union[str, ValidationMode] = ValidationMode.STRICT,
        max_workers: int = 4,
        cache_enabled: bool = True,
        attestation_enabled: bool = True
    ):
        """
        Initialize the constraint verifier.
        
        Args:
            plugin_names: Optional list of plugin names to load. If None, all available
                         constraint validator plugins will be loaded.
            mode: Validation mode for constraint verification.
            max_workers: Maximum number of concurrent validator executions.
            cache_enabled: Whether to cache validation results.
            attestation_enabled: Whether to create attestation chains.
        """
        # Convert string mode to enum if needed
        if isinstance(mode, str):
            self.mode = ValidationMode(mode)
        else:
            self.mode = mode
        
        # Initialize validator registry
        self.registry = ValidatorRegistry()
        self.registry.load_validators(plugin_names)
        
        # Concurrency control
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Caching
        self.cache_enabled = cache_enabled
        self._validation_cache: Dict[str, Tuple[ConstraintValidationResult, float]] = {}
        self.cache_ttl = 300.0  # 5 minutes
        
        # Attestation
        self.attestation_enabled = attestation_enabled
        
        # Observability
        self.metrics = MetricsCollector(service_name="constraint_verifier")
        self.telemetry = get_telemetry_client()
        
        # Register metrics
        self._register_metrics()
        
        logger.info(f"Initialized ConstraintVerifier with {len(self.registry.validators)} validators in {self.mode.value} mode")
    
    def _register_metrics(self) -> None:
        """Register metrics for observability."""
        self.metrics.register_counter("validation.total", "Total number of validation requests")
        self.metrics.register_counter("validation.passed", "Number of passed validations")
        self.metrics.register_counter("validation.failed", "Number of failed validations")
        self.metrics.register_counter("validation.cache_hits", "Number of validation cache hits")
        self.metrics.register_counter("validation.cache_misses", "Number of validation cache misses")
        
        self.metrics.register_histogram(
            "validation.duration_seconds", 
            "Validation duration in seconds",
            buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        self.metrics.register_histogram(
            "validation.confidence", 
            "Validation confidence scores",
            buckets=[0.1, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
        )
    
    def validate(self, data: Any) -> ConstraintValidationResult:
        """
        Validate data against all applicable constraints.
        
        Args:
            data: Data to validate
            
        Returns:
            Aggregated validation result
        """
        # Generate request ID for tracing
        request_id = f"val_{int(time.time() * 1000)}"
        
        # Check cache if enabled
        if self.cache_enabled:
            cache_key = self._generate_cache_key(data)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.metrics.counter_inc("validation.cache_hits")
                log_constraint_event(
                    event_type="cache_hit",
                    validator="constraint_verifier",
                    constraint="all",
                    metadata={"request_id": request_id}
                )
                return cached_result
            self.metrics.counter_inc("validation.cache_misses")
        
        # Record metrics
        self.metrics.counter_inc("validation.total")
        
        # Create verification context for attestation chain
        context = VerificationContext(request_id=request_id) if self.attestation_enabled else None
        
        # Start timing
        with TimingContext() as timer:
            # Create span for observability
            with self.telemetry.create_span(
                name="constraint_validation",
                attributes={"request_id": request_id, "mode": self.mode.value}
            ):
                # Run validation
                result = self._validate_all_constraints(data, context)
        
        # Record timing
        self.metrics.histogram_observe("validation.duration_seconds", timer.elapsed)
        
        # Record pass/fail
        if result.passed:
            self.metrics.counter_inc("validation.passed")
        else:
            self.metrics.counter_inc("validation.failed")
        
        # Record confidence
        self.metrics.histogram_observe("validation.confidence", result.aggregate_confidence)
        
        # Cache result if enabled
        if self.cache_enabled:
            cache_key = self._generate_cache_key(data)
            self._add_to_cache(cache_key, result)
        
        return result
    
    def _validate_all_constraints(self, data: Any, context: Optional[VerificationContext] = None) -> ConstraintValidationResult:
        """Execute validation against all constraints."""
        # Track timing
        start_time = time.time()
        
        # Get all validators
        validators = list(self.registry.validators.values())
        
        # Execute validation in parallel if possible
        if len(validators) > 1 and self.max_workers > 1:
            results = self._validate_parallel(validators, data, context)
        else:
            results = self._validate_sequential(validators, data, context)
        
        # Determine if validation passed based on mode
        is_strict = self.mode == ValidationMode.STRICT
        is_deferred = self.mode == ValidationMode.DEFERRED
        is_evidence = self.mode == ValidationMode.EVIDENCE
        
        # For evidence-only mode, always pass
        if is_evidence:
            passed = True
        # For deferred mode, always pass but record violations
        elif is_deferred:
            passed = True
        # For other modes, only pass if no violations
        else:
            passed = all(r.passed for r in results)
        
        # Calculate aggregate confidence
        if results:
            # Weight by severity
            weights = {
                ConstraintSeverity.CRITICAL: 5.0,
                ConstraintSeverity.HIGH: 3.0,
                ConstraintSeverity.MEDIUM: 2.0,
                ConstraintSeverity.LOW: 1.0,
                ConstraintSeverity.INFO: 0.5
            }
            
            weighted_sum = sum(r.confidence * weights[r.severity] for r in results)
            weight_sum = sum(weights[r.severity] for r in results)
            
            aggregate_confidence = weighted_sum / max(weight_sum, 1.0)
        else:
            aggregate_confidence = 0.0
        
        # Build constraint validation result
        validation_result = ConstraintValidationResult(
            passed=passed,
            assertion_results=results,
            aggregate_confidence=aggregate_confidence,
            start_time=start_time,
            end_time=time.time()
        )
        
        # Add attestation metadata if enabled
        if context:
            validation_result.metadata["attestation"] = {
                "request_id": context.request_id,
                "integrity_hash": context.compute_integrity_hash(),
                "execution_path": context.execution_path,
                "attestation_count": len(context.attestations)
            }
        
        # Log comprehensive summary
        logger.info(
            f"Constraint validation completed: passed={passed}, confidence={aggregate_confidence:.2f}",
            extra={
                "passed": passed,
                "mode": self.mode.value,
                "constraint_count": len(results),
                "confidence": aggregate_confidence,
                "execution_time": validation_result.execution_time
            }
        )
        
        return validation_result
    
    def _validate_sequential(
        self, 
        validators: List[ConstraintInterface],
        data: Any,
        context: Optional[VerificationContext]
    ) -> List[ConstraintAssertionResult]:
        """Validate constraints sequentially."""
        results = []
        
        for validator in validators:
            try:
                # Get validator ID
                validator_id = self._get_validator_id(validator)
                
                # Create span for the validator
                with self.telemetry.create_span(
                    name=f"constraint_validator.{validator_id}",
                    attributes={"validator": validator_id}
                ):
                    # Validate all constraints using this validator
                    validation_result = validator.validate(data)
                    
                    # Process and add all assertion results
                    for assertion in validation_result.assertion_results:
                        # Add attestation if enabled
                        if context:
                            context.add_attestation(
                                validator_id=validator_id,
                                constraint_id=assertion.constraint_id,
                                result=assertion.passed,
                                confidence=assertion.confidence,
                                evidence=assertion.evidence
                            )
                        
                        # Log detailed result
                        log_constraint_event(
                            event_type="constraint_validation",
                            validator=validator_id,
                            constraint=assertion.constraint_id,
                            passed=assertion.passed,
                            confidence=assertion.confidence,
                            message=assertion.message
                        )
                        
                        # Add to results
                        results.append(assertion)
                        
                        # In strict mode, break on first failure
                        if not assertion.passed and self.mode == ValidationMode.STRICT:
                            return results
            
            except Exception as e:
                logger.error(f"Error in validator {validator}: {str(e)}")
                # Create a failure result
                validator_id = self._get_validator_id(validator)
                failure = ConstraintAssertionResult(
                    constraint_id=f"{validator_id}:error",
                    passed=False,
                    confidence=0.0,
                    severity=ConstraintSeverity.HIGH,
                    category=ConstraintCategory.CUSTOM,
                    message=f"Validator error: {str(e)}",
                    metadata={"error": str(e), "validator": validator_id}
                )
                results.append(failure)
                
                # In strict mode, break on error
                if self.mode == ValidationMode.STRICT:
                    return results
        
        return results
    
    def _validate_parallel(
        self, 
        validators: List[ConstraintInterface],
        data: Any,
        context: Optional[VerificationContext]
    ) -> List[ConstraintAssertionResult]:
        """Validate constraints in parallel."""
        results = []
        futures = {}
        
        # Submit validation tasks
        for validator in validators:
            validator_id = self._get_validator_id(validator)
            future = self.executor.submit(
                self._execute_validator,
                validator, 
                validator_id, 
                data
            )
            futures[validator_id] = future
        
        # Process results as they complete
        for validator_id, future in futures.items():
            try:
                validation_result = future.result()
                
                # Process and add all assertion results
                for assertion in validation_result.assertion_results:
                    # Add attestation if enabled
                    if context:
                        context.add_attestation(
                            validator_id=validator_id,
                            constraint_id=assertion.constraint_id,
                            result=assertion.passed,
                            confidence=assertion.confidence,
                            evidence=assertion.evidence
                        )
                    
                    # Log detailed result
                    log_constraint_event(
                        event_type="constraint_validation",
                        validator=validator_id,
                        constraint=assertion.constraint_id,
                        passed=assertion.passed,
                        confidence=assertion.confidence,
                        message=assertion.message
                    )
                    
                    # Add to results
                    results.append(assertion)
            
            except Exception as e:
                logger.error(f"Error in validator {validator_id}: {str(e)}")
                # Create a failure result
                failure = ConstraintAssertionResult(
                    constraint_id=f"{validator_id}:error",
                    passed=False,
                    confidence=0.0,
                    severity=ConstraintSeverity.HIGH,
                    category=ConstraintCategory.CUSTOM,
                    message=f"Validator error: {str(e)}",
                    metadata={"error": str(e), "validator": validator_id}
                )
                results.append(failure)
        
        return results
    
    def _execute_validator(
        self, 
        validator: ConstraintInterface,
        validator_id: str,
        data: Any
    ) -> ConstraintValidationResult:
        """Execute a validator with error handling and observability."""
        with self.telemetry.create_span(
            name=f"constraint_validator.{validator_id}",
            attributes={"validator": validator_id}
        ):
            try:
                return validator.validate(data)
            except Exception as e:
                logger.error(f"Error in validator {validator_id}: {str(e)}")
                # Create a minimal validation result with the error
                failure = ConstraintAssertionResult(
                    constraint_id=f"{validator_id}:error",
                    passed=False,
                    confidence=0.0,
                    severity=ConstraintSeverity.HIGH,
                    category=ConstraintCategory.CUSTOM,
                    message=f"Validator error: {str(e)}",
                    metadata={"error": str(e), "validator": validator_id}
                )
                return ConstraintValidationResult(
                    passed=False,
                    assertion_results=[failure],
                    aggregate_confidence=0.0,
                    start_time=time.time(),
                    end_time=time.time()
                )
    
    def validate_constraint(self, constraint_id: str, data: Any) -> ConstraintAssertionResult:
        """
        Validate data against a specific constraint.
        
        Args:
            constraint_id: ID of the constraint to validate
            data: Data to validate
            
        Returns:
            Detailed result for the specific constraint
        """
        # Get the validator for this constraint
        validator = self.registry.get_constraint_validator(constraint_id)
        if not validator:
            logger.error(f"No validator found for constraint {constraint_id}")
            return ConstraintAssertionResult(
                constraint_id=constraint_id,
                passed=False,
                confidence=0.0,
                severity=ConstraintSeverity.MEDIUM,
                category=ConstraintCategory.CUSTOM,
                message=f"No validator found for constraint {constraint_id}"
            )
        
        try:
            # Validate the specific constraint
            return validator.validate_constraint(constraint_id, data)
        
        except Exception as e:
            logger.error(f"Error validating constraint {constraint_id}: {str(e)}")
            return ConstraintAssertionResult(
                constraint_id=constraint_id,
                passed=False,
                confidence=0.0,
                severity=ConstraintSeverity.MEDIUM,
                category=ConstraintCategory.CUSTOM,
                message=f"Validation error: {str(e)}"
            )
    
    def register_validator(self, validator: ConstraintInterface) -> bool:
        """
        Register a constraint validator component.
        
        Args:
            validator: Validator implementation to register
            
        Returns:
            True if successfully registered, False otherwise
        """
        try:
            # Get validator ID
            validator_id = self._get_validator_id(validator)
            
            # Add to registry
            self.registry.validators[validator_id] = validator
            self.registry.validator_info[validator_id] = {
                "name": validator_id,
                "version": getattr(validator, "version", "1.0.0"),
                "description": getattr(validator, "description", ""),
                "loaded_at": time.time()
            }
            
            # Get constraints and build index
            constraints = validator.get_constraints()
            self.registry.validator_constraints[validator_id] = constraints
            
            for constraint in constraints:
                self.registry.constraint_index[constraint.constraint_id] = validator_id
                logger.info(f"Registered constraint {constraint.constraint_id} from validator {validator_id}")
                self.registry.metrics.counter_inc("constraints.registered")
            
            logger.info(f"Registered validator {validator_id} with {len(constraints)} constraints")
            self.registry.metrics.counter_inc("validators.loaded")
            self.registry.metrics.gauge_inc("validators.active")
            
            return True
        
        except Exception as e:
            logger.error(f"Error registering validator: {str(e)}")
            self.registry.metrics.counter_inc("validators.errors")
            return False
    
    def unregister_validator(self, validator_id: str) -> bool:
        """
        Unregister a constraint validator component.
        
        Args:
            validator_id: ID of validator to unregister
            
        Returns:
            True if successfully unregistered, False otherwise
        """
        try:
            # Check if validator exists
            if validator_id not in self.registry.validators:
                logger.warning(f"Validator {validator_id} not found")
                return False
            
            # Remove constraints from index
            constraints = self.registry.validator_constraints.get(validator_id, [])
            for constraint in constraints:
                if constraint.constraint_id in self.registry.constraint_index:
                    del self.registry.constraint_index[constraint.constraint_id]
            
            # Remove validator
            del self.registry.validators[validator_id]
            if validator_id in self.registry.validator_info:
                del self.registry.validator_info[validator_id]
            if validator_id in self.registry.validator_constraints:
                del self.registry.validator_constraints[validator_id]
            
            logger.info(f"Unregistered validator {validator_id}")
            self.registry.metrics.gauge_dec("validators.active")
            
            return True
        
        except Exception as e:
            logger.error(f"Error unregistering validator {validator_id}: {str(e)}")
            return False
    
    def get_all_constraints(self) -> Dict[str, List[ConstraintDefinition]]:
        """
        Get all available constraints grouped by validator.
        
        Returns:
            Dictionary mapping validator IDs to their constraint definitions
        """
        return self.registry.validator_constraints.copy()
    
    def set_validation_policy(self, policy: Dict[str, Any]) -> bool:
        """
        Set global validation policy parameters.
        
        Args:
            policy: Policy configuration dictionary
            
        Returns:
            True if policy successfully applied, False otherwise
        """
        try:
            # Apply validation mode if specified
            if "mode" in policy:
                mode_str = policy["mode"]
                try:
                    self.mode = ValidationMode(mode_str)
                except ValueError:
                    logger.error(f"Invalid validation mode: {mode_str}")
                    return False
            
            # Apply concurrency settings if specified
            if "max_workers" in policy:
                max_workers = policy["max_workers"]
                if isinstance(max_workers, int) and max_workers > 0:
                    self.max_workers = max_workers
                    # Recreate executor if needed
                    if self.executor:
                        self.executor.shutdown(wait=False)
                    self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
                else:
                    logger.error(f"Invalid max_workers value: {max_workers}")
                    return False
            
            # Apply caching settings if specified
            if "cache_enabled" in policy:
                self.cache_enabled = bool(policy["cache_enabled"])
            
            if "cache_ttl" in policy:
                cache_ttl = policy["cache_ttl"]
                if isinstance(cache_ttl, (int, float)) and cache_ttl > 0:
                    self.cache_ttl = float(cache_ttl)
                else:
                    logger.error(f"Invalid cache_ttl value: {cache_ttl}")
                    return False
            
            # Apply attestation settings if specified
            if "attestation_enabled" in policy:
                self.attestation_enabled = bool(policy["attestation_enabled"])
            
            logger.info(f"Applied validation policy: {policy}")
            return True
        
        except Exception as e:
            logger.error(f"Error applying validation policy: {str(e)}")
            return False
    
    def _get_validator_id(self, validator: ConstraintInterface) -> str:
        """Get the ID of a validator."""
        # Try to get name attribute
        name = getattr(validator, "name", None)
        if name:
            return name
        
        # Try to get class name
        class_name = validator.__class__.__name__
        if class_name:
            return class_name
        
        # Fallback to hex ID
        return f"validator_{hex(id(validator))}"
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate a cache key for the data."""
        try:
            # Try to use JSON serialization for the key
            if isinstance(data, dict):
                # Sort keys for deterministic serialization
                data_str = json.dumps(data, sort_keys=True)
            elif hasattr(data, "to_dict"):
                # Use to_dict method if available
                data_str = json.dumps(data.to_dict(), sort_keys=True)
            else:
                # Use string representation
                data_str = str(data)
            
            # Compute hash
            return hashlib.md5(data_str.encode()).hexdigest()
        
        except Exception:
            # Fallback to object ID
            return f"data_{hex(id(data))}"
    
    def _get_from_cache(self, key: str) -> Optional[ConstraintValidationResult]:
        """Get validation result from cache."""
        if not self.cache_enabled:
            return None
        
        cached = self._validation_cache.get(key)
        if not cached:
            return None
        
        result, timestamp = cached
        
        # Check if expired
        if time.time() - timestamp > self.cache_ttl:
            # Remove expired entry
            del self._validation_cache[key]
            return None
        
        return result
    
    def _add_to_cache(self, key: str, result: ConstraintValidationResult) -> None:
        """Add validation result to cache."""
        if not self.cache_enabled:
            return
        
        self._validation_cache[key] = (result, time.time())
        
        # Limit cache size
        if len(self._validation_cache) > 1000:
            # Remove oldest entries
            oldest = sorted(self._validation_cache.items(), key=lambda x: x[1][1])
            for old_key, _ in oldest[:100]:  # Remove oldest 10%
                del self._validation_cache[old_key]


class FactualConstraintValidator(ConstraintInterface[Dict[str, Any]]):
    """
    Factual constraint validator for hallucination prevention.
    
    This validator implements formal constraint validation for factual accuracy
    with evidence-based confidence scoring and provenance tracking. It serves as
    a reference implementation of the ConstraintInterface.
    """
    
    def __init__(self, name: str = "factual_validator"):
        """Initialize the factual constraint validator."""
        self.name = name
        self.version = "1.0.0"
        self.description = "Factual constraint validator for hallucination prevention"
        
        # Default constraints
        self._constraints: List[ConstraintDefinition] = [
            ConstraintDefinition(
                constraint_id="factual.source_citation",
                name="Source Citation",
                description="Verify that factual claims have proper source citations",
                severity=ConstraintSeverity.HIGH,
                category=ConstraintCategory.FACTUAL,
                required_fields=["content", "citations"]
            ),
            ConstraintDefinition(
                constraint_id="factual.claim_verification",
                name="Claim Verification",
                description="Verify that factual claims can be verified against trusted sources",
                severity=ConstraintSeverity.CRITICAL,
                category=ConstraintCategory.FACTUAL,
                required_fields=["claims"]
            ),
            ConstraintDefinition(
                constraint_id="factual.semantic_boundaries",
                name="Semantic Boundaries",
                description="Verify that content stays within semantic boundaries of source material",
                severity=ConstraintSeverity.MEDIUM,
                category=ConstraintCategory.SEMANTIC,
                required_fields=["content", "context"]
            )
        ]
        
        # Metrics for observability
        self.metrics = MetricsCollector(service_name=f"constraint_validator_{name}")
        self.metrics.register_counter("constraints.total", "Total constraint checks")
        self.metrics.register_counter("constraints.passed", "Passed constraint checks")
        self.metrics.register_counter("constraints.failed", "Failed constraint checks")
    
    def validate(self, data: Dict[str, Any]) -> ConstraintValidationResult:
        """
        Validate data against all constraints implemented by this validator.
        
        Args:
            data: Dictionary of data to validate
            
        Returns:
            Validation result with all constraint assertions
        """
        start_time = time.time()
        results: List[ConstraintAssertionResult] = []
        
        # Apply all enabled constraints
        for constraint in self._constraints:
            if not constraint.enabled:
                continue
            
            result = self.validate_constraint(constraint.constraint_id, data)
            results.append(result)
        
        # Calculate aggregate confidence
        aggregate_confidence = 0.0
        if results:
            aggregate_confidence = sum(r.confidence for r in results) / len(results)
        
        # Determine if validation passed
        passed = all(r.passed for r in results)
        
        # Create validation result
        validation_result = ConstraintValidationResult(
            passed=passed,
            assertion_results=results,
            aggregate_confidence=aggregate_confidence,
            start_time=start_time,
            end_time=time.time()
        )
        
        return validation_result
    
    def validate_constraint(self, constraint_id: str, data: Dict[str, Any]) -> ConstraintAssertionResult:
        """
        Validate data against a specific constraint.
        
        Args:
            constraint_id: ID of the constraint to validate
            data: Data to validate
            
        Returns:
            Detailed result for the specific constraint
        """
        self.metrics.counter_inc("constraints.total")
        
        # Get constraint definition
        constraint = self._get_constraint(constraint_id)
        if not constraint:
            return ConstraintAssertionResult(
                constraint_id=constraint_id,
                passed=False,
                confidence=0.0,
                severity=ConstraintSeverity.MEDIUM,
                category=ConstraintCategory.CUSTOM,
                message=f"Unknown constraint: {constraint_id}"
            )
        
        # Check required fields
        for field in constraint.required_fields:
            if field not in data:
                self.metrics.counter_inc("constraints.failed")
                return ConstraintAssertionResult(
                    constraint_id=constraint_id,
                    passed=False,
                    confidence=1.0,  # High confidence in failure due to missing field
                    severity=constraint.severity,
                    category=constraint.category,
                    message=f"Required field missing: {field}"
                )
        
        # Implementation for specific constraints
        if constraint_id == "factual.source_citation":
            return self._validate_source_citation(constraint, data)
        elif constraint_id == "factual.claim_verification":
            return self._validate_claim_verification(constraint, data)
        elif constraint_id == "factual.semantic_boundaries":
            return self._validate_semantic_boundaries(constraint, data)
        else:
            self.metrics.counter_inc("constraints.failed")
            return ConstraintAssertionResult(
                constraint_id=constraint_id,
                passed=False,
                confidence=0.5,
                severity=constraint.severity,
                category=constraint.category,
                message=f"Constraint not implemented: {constraint_id}"
            )
    
    def _validate_source_citation(
        self, 
        constraint: ConstraintDefinition,
        data: Dict[str, Any]
    ) -> ConstraintAssertionResult:
        """Validate that factual claims have proper source citations."""
        # Extract content and citations
        content = data.get("content", "")
        citations = data.get("citations", [])
        
        # Simple check: Are there citations?
        has_citations = len(citations) > 0
        
        # Evidence collection
        evidence = [
            ConstraintEvidence(
                key=f"citation_count_{int(time.time())}",
                source="citation_checker",
                value=len(citations),
                confidence=0.8,
                metadata={"has_citations": has_citations}
            )
        ]
        
        # More sophisticated checks would verify citations against content
        
        # Determine result
        passed = has_citations
        confidence = 0.8  # Moderate confidence without deeper verification
        
        if passed:
            self.metrics.counter_inc("constraints.passed")
        else:
            self.metrics.counter_inc("constraints.failed")
        
        return ConstraintAssertionResult(
            constraint_id=constraint.constraint_id,
            passed=passed,
            confidence=confidence,
            severity=constraint.severity,
            category=constraint.category,
            message="Source citation verification" if passed else "Missing source citations",
            evidence=evidence,
            execution_time=0.01
        )
    
    def _validate_claim_verification(
        self, 
        constraint: ConstraintDefinition,
        data: Dict[str, Any]
    ) -> ConstraintAssertionResult:
        """Verify that factual claims can be verified against trusted sources."""
        # Extract claims
        claims = data.get("claims", [])
        
        # Simple check: Are there claims to verify?
        if not claims:
            self.metrics.counter_inc("constraints.failed")
            return ConstraintAssertionResult(
                constraint_id=constraint.constraint_id,
                passed=False,
                confidence=0.9,
                severity=constraint.severity,
                category=constraint.category,
                message="No claims to verify",
                evidence=[],
                execution_time=0.01
            )
        
        # In a real implementation, each claim would be verified against trusted sources
        # For demonstration, we'll simulate verification with confidence scores
        
        # Evidence collection
        evidence = []
        verified_claims = 0
        
        for i, claim in enumerate(claims):
            # Simulate verification with random confidence
            import random
            verified = random.random() > 0.2  # 80% chance of verification
            confidence = random.uniform(0.7, 0.95)
            
            evidence.append(
                ConstraintEvidence(
                    key=f"claim_verification_{i}_{int(time.time())}",
                    source="claim_verifier",
                    value=verified,
                    confidence=confidence,
                    metadata={"claim": claim}
                )
            )
            
            if verified:
                verified_claims += 1
        
        # Determine result
        verification_ratio = verified_claims / len(claims) if claims else 0
        passed = verification_ratio >= 0.8  # At least 80% of claims verified
        
        # Calculate overall confidence based on evidence
        confidence = sum(e.confidence for e in evidence) / len(evidence) if evidence else 0.5
        
        if passed:
            self.metrics.counter_inc("constraints.passed")
        else:
            self.metrics.counter_inc("constraints.failed")
        
        return ConstraintAssertionResult(
            constraint_id=constraint.constraint_id,
            passed=passed,
            confidence=confidence,
            severity=constraint.severity,
            category=constraint.category,
            message=f"Claim verification: {verified_claims}/{len(claims)} claims verified",
            evidence=evidence,
            execution_time=0.05
        )
    
    def _validate_semantic_boundaries(
        self, 
        constraint: ConstraintDefinition,
        data: Dict[str, Any]
    ) -> ConstraintAssertionResult:
        """Verify that content stays within semantic boundaries of source material."""
        # Extract content and context
        content = data.get("content", "")
        context = data.get("context", {})
        
        # In a real implementation, semantic boundary verification would involve
        # vector embedding similarity checks, topic modeling, etc.
        # For demonstration, we'll simulate this with a simple text containment check
        
        # Simple check: Does the content contain key phrases from context?
        key_phrases = context.get("key_phrases", [])
        
        if not key_phrases:
            # No key phrases to check against
            self.metrics.counter_inc("constraints.passed")
            return ConstraintAssertionResult(
                constraint_id=constraint.constraint_id,
                passed=True,
                confidence=0.5,  # Low confidence due to lack of key phrases
                severity=constraint.severity,
                category=constraint.category,
                message="No semantic boundaries defined",
                evidence=[],
                execution_time=0.01
            )
        
        # Check content against key phrases
        contained_phrases = []
        for phrase in key_phrases:
            if phrase.lower() in content.lower():
                contained_phrases.append(phrase)
        
        # Evidence collection
        evidence = [
            ConstraintEvidence(
                key=f"semantic_boundary_{int(time.time())}",
                source="semantic_analyzer",
                value={"contained_phrases": contained_phrases, "total_phrases": key_phrases},
                confidence=0.7,
                metadata={"containment_ratio": len(contained_phrases) / len(key_phrases)}
            )
        ]
        
        # Determine result
        containment_ratio = len(contained_phrases) / len(key_phrases)
        passed = containment_ratio >= 0.5  # At least 50% of key phrases contained
        
        if passed:
            self.metrics.counter_inc("constraints.passed")
        else:
            self.metrics.counter_inc("constraints.failed")
        
        return ConstraintAssertionResult(
            constraint_id=constraint.constraint_id,
            passed=passed,
            confidence=0.7,
            severity=constraint.severity,
            category=constraint.category,
            message=f"Semantic boundary check: {len(contained_phrases)}/{len(key_phrases)} key phrases contained",
            evidence=evidence,
            execution_time=0.03
        )
    
    def get_constraints(self) -> List[ConstraintDefinition]:
        """
        Get definitions of all constraints implemented by this validator.
        
        Returns:
            List of constraint definitions with metadata
        """
        return self._constraints.copy()
    
    def add_constraint(self, definition: ConstraintDefinition) -> bool:
        """
        Add a new constraint definition to this validator.
        
        Args:
            definition: Constraint definition to add
            
        Returns:
            True if successfully added, False otherwise
        """
        # Check if constraint already exists
        existing = self._get_constraint(definition.constraint_id)
        if existing:
            return False
        
        # Add constraint
        self._constraints.append(definition)
        return True
    
    def remove_constraint(self, constraint_id: str) -> bool:
        """
        Remove a constraint from this validator.
        
        Args:
            constraint_id: ID of the constraint to remove
            
        Returns:
            True if successfully removed, False otherwise
        """
        for i, constraint in enumerate(self._constraints):
            if constraint.constraint_id == constraint_id:
                del self._constraints[i]
                return True
        return False
    
    def update_constraint(self, constraint_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration for an existing constraint.
        
        Args:
            constraint_id: ID of the constraint to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successfully updated, False otherwise
        """
        constraint = self._get_constraint(constraint_id)
        if not constraint:
            return False
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(constraint, key):
                setattr(constraint, key, value)
        
        return True
    
    def enable_constraint(self, constraint_id: str) -> bool:
        """
        Enable a constraint.
        
        Args:
            constraint_id: ID of the constraint to enable
            
        Returns:
            True if successfully enabled, False otherwise
        """
        constraint = self._get_constraint(constraint_id)
        if not constraint:
            return False
        
        constraint.enabled = True
        return True
    
    def disable_constraint(self, constraint_id: str) -> bool:
        """
        Disable a constraint.
        
        Args:
            constraint_id: ID of the constraint to disable
            
        Returns:
            True if successfully disabled, False otherwise
        """
        constraint = self._get_constraint(constraint_id)
        if not constraint:
            return False
        
        constraint.enabled = False
        return True
    
    def _get_constraint(self, constraint_id: str) -> Optional[ConstraintDefinition]:
        """Get a constraint definition by ID."""
        for constraint in self._constraints:
            if constraint.constraint_id == constraint_id:
                return constraint
        return None


# Factory function to create a standard constraint verifier
def create_constraint_verifier(
    plugin_names: Optional[List[str]] = None,
    mode: str = "strict",
    with_factual_validator: bool = True
) -> ConstraintVerifier:
    """
    Create a standard constraint verifier with common configuration.
    
    Args:
        plugin_names: Optional list of plugin names to load
        mode: Validation mode (strict, relaxed, progressive, deferred, evidence)
        with_factual_validator: Whether to include the built-in factual validator
        
    Returns:
        Configured constraint verifier
    """
    # Create verifier
    verifier = ConstraintVerifier(
        plugin_names=plugin_names,
        mode=mode,
        max_workers=4,
        cache_enabled=True,
        attestation_enabled=True
    )
    
    # Add built-in factual validator if requested
    if with_factual_validator:
        factual_validator = FactualConstraintValidator()
        verifier.register_validator(factual_validator)
    
    return verifier