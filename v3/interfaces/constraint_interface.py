"""
Constraint verification framework interface definitions.

This module defines the interfaces for the ACDEK constraint verification system,
which provides formal validation of data against defined constraints with
evidence-based confidence scoring and detailed reporting.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Union, TypeVar, Generic
from dataclasses import dataclass, field
import time
import json


class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    CRITICAL = "critical"   # Must be satisfied, no exceptions
    HIGH = "high"           # Strong requirement, fail with high confidence
    MEDIUM = "medium"       # Important, but may have exceptions
    LOW = "low"             # Guideline rather than hard requirement
    INFO = "info"           # Informational only, doesn't affect passing


class ConstraintCategory(Enum):
    """Categories of constraints for organization and filtering."""
    FACTUAL = "factual"                  # Factual accuracy constraints
    SEMANTIC = "semantic"                # Meaning and interpretation constraints
    LOGICAL = "logical"                  # Logical consistency and reasoning constraints
    FORMAT = "format"                    # Data format and structure constraints
    SECURITY = "security"                # Security and safety constraints
    COMPLIANCE = "compliance"            # Regulatory and policy compliance
    BUSINESS_RULE = "business_rule"      # Organization-specific rules
    CUSTOM = "custom"                    # User-defined custom constraints


@dataclass
class ConstraintEvidence:
    """Evidence supporting a constraint validation result."""
    key: str                             # Unique identifier for the evidence
    source: str                          # Source of the evidence (e.g., "vector_db", "rule_engine")
    value: Any                           # The evidence value itself
    confidence: float                    # Confidence score for this piece of evidence (0-1)
    timestamp: float = field(default_factory=time.time)  # When the evidence was collected
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional evidence metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence to dictionary."""
        result = {
            "key": self.key,
            "source": self.source,
            "confidence": self.confidence,
            "timestamp": self.timestamp
        }
        
        # Handle different types of values
        if hasattr(self.value, 'to_dict'):
            result["value"] = self.value.to_dict()
        elif isinstance(self.value, (str, int, float, bool, type(None))):
            result["value"] = self.value
        else:
            try:
                # Try JSON serialization for complex objects
                json.dumps(self.value)
                result["value"] = self.value
            except (TypeError, ValueError):
                # Fall back to string representation
                result["value"] = str(self.value)
                
        # Add metadata if present
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result


@dataclass
class ConstraintAssertionResult:
    """
    Result of a constraint assertion with detailed evidence and metadata.
    
    This class represents the outcome of evaluating a specific constraint,
    including detailed evidence that supports the decision, confidence scoring,
    and contextual information about the validation process.
    """
    constraint_id: str                   # Unique identifier for the constraint
    passed: bool                         # Whether the constraint was satisfied
    confidence: float                    # Confidence in the validation result (0-1)
    severity: ConstraintSeverity         # Severity of this constraint
    category: ConstraintCategory         # Category this constraint belongs to
    message: Optional[str] = None        # Human-readable explanation of the result
    evidence: List[ConstraintEvidence] = field(default_factory=list)  # Supporting evidence
    timestamp: float = field(default_factory=time.time)  # When validation occurred
    execution_time: float = 0.0          # How long validation took in seconds
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all details."""
        return {
            "constraint_id": self.constraint_id,
            "passed": self.passed,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "evidence": [ev.to_dict() for ev in self.evidence],
            "timestamp": self.timestamp,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }
    
    @property
    def evidence_confidence(self) -> float:
        """Calculate the average confidence of all evidence."""
        if not self.evidence:
            return 0.0
        return sum(ev.confidence for ev in self.evidence) / len(self.evidence)
    
    @property
    def has_critical_evidence(self) -> bool:
        """Check if there is evidence with very high confidence."""
        return any(ev.confidence > 0.9 for ev in self.evidence)
    
    def add_evidence(self, evidence: ConstraintEvidence) -> None:
        """Add a piece of evidence to the result."""
        self.evidence.append(evidence)
        
    def get_evidence_by_source(self, source: str) -> List[ConstraintEvidence]:
        """Get all evidence from a specific source."""
        return [ev for ev in self.evidence if ev.source == source]


@dataclass
class ConstraintValidationResult:
    """
    Aggregated result of validating multiple constraints.
    
    This class consolidates the results of multiple constraint assertions
    into a single comprehensive validation result, with overall scoring,
    statistics, and detailed constraint-by-constraint results.
    """
    passed: bool                          # Overall validation success
    assertion_results: List[ConstraintAssertionResult]  # Individual constraint results
    aggregate_confidence: float           # Overall confidence score (0-1)
    start_time: float                     # When validation started
    end_time: float                       # When validation completed
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def __post_init__(self):
        """Compute derived fields after initialization."""
        self.total_constraints = len(self.assertion_results)
        self.passed_constraints = sum(1 for r in self.assertion_results if r.passed)
        self.failed_constraints = self.total_constraints - self.passed_constraints
        self.execution_time = self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with all details."""
        return {
            "passed": self.passed,
            "aggregate_confidence": self.aggregate_confidence,
            "total_constraints": self.total_constraints,
            "passed_constraints": self.passed_constraints,
            "failed_constraints": self.failed_constraints,
            "execution_time": self.execution_time,
            "assertion_results": [r.to_dict() for r in self.assertion_results],
            "metadata": self.metadata
        }
    
    @property
    def critical_failures(self) -> List[ConstraintAssertionResult]:
        """Get all critical constraint failures."""
        return [r for r in self.assertion_results 
                if not r.passed and r.severity == ConstraintSeverity.CRITICAL]
    
    @property
    def high_confidence_failures(self) -> List[ConstraintAssertionResult]:
        """Get high-confidence constraint failures."""
        return [r for r in self.assertion_results 
                if not r.passed and r.confidence > 0.8]
    
    @property
    def by_category(self) -> Dict[ConstraintCategory, List[ConstraintAssertionResult]]:
        """Group constraint results by category."""
        result = {category: [] for category in ConstraintCategory}
        for assertion in self.assertion_results:
            result[assertion.category].append(assertion)
        return result
    
    @property
    def by_severity(self) -> Dict[ConstraintSeverity, List[ConstraintAssertionResult]]:
        """Group constraint results by severity."""
        result = {severity: [] for severity in ConstraintSeverity}
        for assertion in self.assertion_results:
            result[assertion.severity].append(assertion)
        return result


@dataclass
class ConstraintDefinition:
    """
    Definition of a constraint with metadata.
    
    This class defines a constraint, including its validation logic,
    metadata, dependencies, and configuration options.
    """
    constraint_id: str                   # Unique identifier
    name: str                            # Human-readable name
    description: str                     # Detailed description
    severity: ConstraintSeverity         # Severity level
    category: ConstraintCategory         # Category
    version: str = "1.0.0"               # Version of this constraint definition
    enabled: bool = True                 # Whether this constraint is active
    dependencies: List[str] = field(default_factory=list)  # IDs of dependent constraints
    required_fields: List[str] = field(default_factory=list)  # Data fields required for validation
    config: Dict[str, Any] = field(default_factory=dict)  # Configuration parameters
    tags: List[str] = field(default_factory=list)  # Tags for filtering and organization
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "constraint_id": self.constraint_id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "version": self.version,
            "enabled": self.enabled,
            "dependencies": self.dependencies,
            "required_fields": self.required_fields,
            "config": self.config,
            "tags": self.tags
        }


# Type variable for generic input type
T = TypeVar('T')

class ConstraintInterface(Generic[T], ABC):
    """
    Interface for constraint verification plugins with formal validation.
    
    This interface defines the contract for constraint verification components,
    supporting formal validation pipelines, evidence-based confidence scoring,
    and detailed constraint reporting.
    """
    
    @abstractmethod
    def validate(self, data: T) -> ConstraintValidationResult:
        """
        Validate data against all applicable constraints.
        
        Args:
            data: Data to validate, with type specified by implementation
            
        Returns:
            Comprehensive validation result with details for all constraints
        """
        pass
    
    @abstractmethod
    def validate_constraint(self, constraint_id: str, data: T) -> ConstraintAssertionResult:
        """
        Validate data against a specific constraint.
        
        Args:
            constraint_id: ID of the constraint to validate
            data: Data to validate
            
        Returns:
            Detailed result for the specific constraint
        """
        pass
    
    @abstractmethod
    def get_constraints(self) -> List[ConstraintDefinition]:
        """
        Get definitions of all constraints implemented by this validator.
        
        Returns:
            List of constraint definitions with metadata
        """
        pass
    
    @abstractmethod
    def add_constraint(self, definition: ConstraintDefinition) -> bool:
        """
        Add a new constraint definition to this validator.
        
        Args:
            definition: Constraint definition to add
            
        Returns:
            True if successfully added, False otherwise
        """
        pass
    
    @abstractmethod
    def remove_constraint(self, constraint_id: str) -> bool:
        """
        Remove a constraint from this validator.
        
        Args:
            constraint_id: ID of the constraint to remove
            
        Returns:
            True if successfully removed, False otherwise
        """
        pass
    
    @abstractmethod
    def update_constraint(self, constraint_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration for an existing constraint.
        
        Args:
            constraint_id: ID of the constraint to update
            updates: Dictionary of fields to update
            
        Returns:
            True if successfully updated, False otherwise
        """
        pass
    
    @abstractmethod
    def enable_constraint(self, constraint_id: str) -> bool:
        """
        Enable a constraint.
        
        Args:
            constraint_id: ID of the constraint to enable
            
        Returns:
            True if successfully enabled, False otherwise
        """
        pass
    
    @abstractmethod
    def disable_constraint(self, constraint_id: str) -> bool:
        """
        Disable a constraint.
        
        Args:
            constraint_id: ID of the constraint to disable
            
        Returns:
            True if successfully disabled, False otherwise
        """
        pass


class ConstraintVerifierInterface(ABC):
    """
    Interface for constraint verification orchestration.
    
    This interface defines the contract for components that coordinate
    multiple constraint validators, aggregate results, and enforce validation
    policies across the system.
    """
    
    @abstractmethod
    def validate(self, data: Any) -> ConstraintValidationResult:
        """
        Validate data against all applicable constraints.
        
        Args:
            data: Data to validate
            
        Returns:
            Aggregated validation result
        """
        pass
    
    @abstractmethod
    def register_validator(self, validator: ConstraintInterface) -> bool:
        """
        Register a constraint validator component.
        
        Args:
            validator: Validator implementation to register
            
        Returns:
            True if successfully registered, False otherwise
        """
        pass
    
    @abstractmethod
    def unregister_validator(self, validator_id: str) -> bool:
        """
        Unregister a constraint validator component.
        
        Args:
            validator_id: ID of validator to unregister
            
        Returns:
            True if successfully unregistered, False otherwise
        """
        pass
    
    @abstractmethod
    def get_all_constraints(self) -> Dict[str, List[ConstraintDefinition]]:
        """
        Get all available constraints grouped by validator.
        
        Returns:
            Dictionary mapping validator IDs to their constraint definitions
        """
        pass
    
    @abstractmethod
    def set_validation_policy(self, policy: Dict[str, Any]) -> bool:
        """
        Set global validation policy parameters.
        
        Args:
            policy: Policy configuration dictionary
            
        Returns:
            True if policy successfully applied, False otherwise
        """
        pass