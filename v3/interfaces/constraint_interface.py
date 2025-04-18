"""
Constraint assertion framework interface definitions.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class ConstraintAssertionResult:
    """
    Result of a single constraint assertion.
    """
    name: str
    passed: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0  # optional strength value for relaxed mode

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "strength": self.strength,
        }

class ConstraintInterface(ABC):
    """
    Interface for constraint assertion plugins.
    """
    @abstractmethod
    def assert_constraint(self, input_data: Dict[str, Any]) -> ConstraintAssertionResult:
        """
        Evaluate a constraint against the provided input data.

        Args:
            input_data: The data to validate against the constraint.

        Returns:
            A ConstraintAssertionResult indicating pass/fail and details.
        """
        pass