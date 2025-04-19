"""
Observability scaffold for resilience events: circuit breaker, retries, and fallbacks.
"""
import logging
from typing import Optional

try:
    from opentelemetry import trace
    _tracer = trace.get_tracer(__name__)
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

logger = logging.getLogger("resilience.events")

def log_circuit_breaker_state(
    name: str,
    state: str,
    failure_count: int,
    **kwargs
) -> None:
    """Log circuit breaker state changes."""
    event = {
        "event": "circuit_breaker",
        "name": name,
        "state": state,
        "failure_count": failure_count,
    }
    event.update(kwargs)
    if _OTEL_AVAILABLE:
        with _tracer.start_as_current_span("circuit_breaker") as span:
            for k, v in event.items():
                span.set_attribute(k, v)
        logger.info(event)
    else:
        logger.info(event)

def log_retry_event(
    plugin: str,
    event: str,
    attempt: int,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    delay: Optional[float] = None,
    **kwargs
) -> None:
    """Log retry attempts and outcomes."""
    payload = {
        "event": "retry",
        "plugin": plugin,
        "retry_event": event,
        "attempt": attempt,
    }
    if error_type is not None:
        payload["error_type"] = error_type
    if error_message is not None:
        payload["error_message"] = error_message
    if delay is not None:
        payload["delay"] = delay
    payload.update(kwargs)
    if _OTEL_AVAILABLE:
        with _tracer.start_as_current_span("retry") as span:
            for k, v in payload.items():
                span.set_attribute(k, v)
        logger.info(payload)
    else:
        logger.info(payload)

def log_fallback_event(
    plugin: str,
    depth: int,
    reason: str,
    **kwargs
) -> None:
    """Log when a fallback to another plugin occurs."""
    payload = {
        "event": "fallback",
        "plugin": plugin,
        "depth": depth,
        "reason": reason,
    }
    payload.update(kwargs)
    if _OTEL_AVAILABLE:
        with _tracer.start_as_current_span("fallback") as span:
            for k, v in payload.items():
                span.set_attribute(k, v)
        logger.info(payload)
    else:
        logger.info(payload)

def log_constraint_event(
    event_type: str,
    validator: str,
    constraint: str,
    passed: bool = None,
    confidence: float = None,
    message: str = None,
    metadata: dict = None
) -> None:
    """
    Log constraint verification events.
    
    Args:
        event_type: Type of event (e.g., 'constraint_validation', 'cache_hit')
        validator: ID of the validator
        constraint: ID of the constraint
        passed: Whether the constraint check passed (optional)
        confidence: Confidence score for the validation (optional)
        message: Description message (optional)
        metadata: Additional metadata (optional)
    """
    payload = {
        "event": "constraint",
        "constraint_event": event_type,
        "validator": validator,
        "constraint": constraint,
    }
    
    if passed is not None:
        payload["passed"] = passed
    if confidence is not None:
        payload["confidence"] = confidence
    if message is not None:
        payload["message"] = message
    if metadata is not None:
        payload.update(metadata)
        
    if _OTEL_AVAILABLE:
        with _tracer.start_as_current_span("constraint") as span:
            for k, v in payload.items():
                span.set_attribute(k, v)
        logger.info(payload)
    else:
        logger.info(payload)