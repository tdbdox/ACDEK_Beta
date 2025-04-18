# Fallback chain, circuit breaker, and retry logic for pipeline resilience
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from registry.registry_types import PluginType
from registry.plugin_registry import create_instance
from observability.resilience_events import (
    log_circuit_breaker_state,
    log_retry_event,
    log_fallback_event,
)

logger = logging.getLogger(__name__)
# Shared executor for timeout enforcement
_executor = ThreadPoolExecutor(max_workers=10)

class CircuitBreakerOpenError(Exception):
    """Raised when the circuit is open and calls are not permitted."""
    pass

class CircuitBreakerTimeoutError(Exception):
    """Raised when a call times out."""
    pass

class CircuitBreaker:
    """Implements the circuit breaker pattern for external calls."""
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        timeout: float = 10.0,
        mode: str = "strict",
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.timeout = timeout
        self.mode = mode  # strict, lenient, deferred
        self._lock = threading.Lock()
        self._state = "CLOSED"
        self._failure_count = 0
        self._last_failure_time = 0.0

    def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute func with circuit breaker protection."""
        with self._lock:
            if self._state == "OPEN":
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = "HALF_OPEN"
                else:
                    log_circuit_breaker_state(
                        self.name, self._state, self._failure_count
                    )
                    if self.mode == "strict":
                        raise CircuitBreakerOpenError(f"Circuit '{self.name}' is OPEN")
                    return None

        future = _executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=self.timeout)
        except FutureTimeoutError as err:
            self._record_failure(err)
            if self.mode == "strict":
                raise CircuitBreakerTimeoutError(str(err))
            return None
        except Exception as err:
            self._record_failure(err)
            if self.mode == "strict":
                raise
            return None

        self._record_success()
        return result

    def _record_failure(self, error: Exception) -> None:
        """Handle a failed call and update state."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = "OPEN"
            log_circuit_breaker_state(
                self.name,
                self._state,
                self._failure_count,
                error_type=type(error).__name__,
            )

    def _record_success(self) -> None:
        """Reset failure count upon successful call."""
        with self._lock:
            self._failure_count = 0
            if self._state != "CLOSED":
                self._state = "CLOSED"
            log_circuit_breaker_state(self.name, self._state, self._failure_count)

class RetryStrategy:
    """Progressive retry strategy with backoff logic."""
    def __init__(
        self,
        strategy: str = "exponential",
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 2.0,
    ):
        self.strategy = strategy
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def run(
        self,
        func: Callable[[], Any],
        retry_hooks: Optional[List[Callable[..., None]]] = None,
    ) -> Any:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                result = func()
                if retry_hooks:
                    for hook in retry_hooks:
                        hook(event="success", attempt=attempt)
                return result
            except Exception as err:
                last_error = err
                delay = self._compute_delay(attempt)
                if retry_hooks:
                    for hook in retry_hooks:
                        hook(event="retry", attempt=attempt, error=err, delay=delay)
                time.sleep(delay)
        if retry_hooks:
            for hook in retry_hooks:
                hook(event="giveup", attempt=self.max_retries, error=last_error)
        raise last_error

    def _compute_delay(self, attempt: int) -> float:
        if self.strategy == "fixed":
            return self.base_delay
        elif self.strategy == "exponential":
            return self.base_delay * (2 ** attempt)
        elif self.strategy == "capped_exponential":
            return min(self.base_delay * (2 ** attempt), self.max_delay)
        return self.base_delay

@dataclass
class FallbackResult:
    """Wrapper for a plugin result with fallback metadata."""
    plugin_name: str
    result: Any
    depth: int
    penalty: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plugin_name": self.plugin_name,
            "depth": self.depth,
            "penalty": self.penalty,
            "metadata": self.metadata,
            "result": self.result,
        }

class FallbackChain:
    """
    Executes a chain of plugins with circuit breakers and retries.
    """
    def __init__(
        self,
        plugin_type: PluginType,
        plugin_names: List[str],
        circuit_breaker_config: Optional[Dict[str, Dict[str, Any]]] = None,
        retry_config: Optional[Dict[str, Any]] = None,
        fallback_mode: str = "strict",
    ):
        self.plugin_type = plugin_type
        self.fallback_mode = fallback_mode
        self.plugins: List[Dict[str, Any]] = []
        cb_conf = circuit_breaker_config or {}
        for name in plugin_names:
            instance = create_instance(plugin_type=plugin_type, name=name)
            params = cb_conf.get(name, {})
            breaker = CircuitBreaker(name=name, **params)
            retry = RetryStrategy(**retry_config) if retry_config else None
            self.plugins.append({
                "name": name,
                "instance": instance,
                "breaker": breaker,
                "retry": retry,
            })

    def execute(self, method_name: str, *args, **kwargs) -> FallbackResult:
        """Attempt plugins in order, applying fallback and penalties."""
        last_error: Optional[Exception] = None
        depth = 0
        for info in self.plugins:
            name = info["name"]
            inst = info["instance"]
            if inst is None:
                depth += 1
                continue
            func = getattr(inst, method_name, None)
            if not callable(func):
                depth += 1
                continue
            breaker = info["breaker"]
            retry = info["retry"]
            # Define the function to call
            def call_func():
                return func(*args, **kwargs)
            try:
                if retry:
                    hooks = [
                        lambda event, attempt, error=None, delay=None, plugin=name: log_retry_event(
                            plugin=plugin,
                            event=event,
                            attempt=attempt,
                            error_type=type(error).__name__ if error else None,
                            error_message=str(error) if error else None,
                            delay=delay,
                        )
                    ]
                    result = retry.run(lambda: breaker.call(call_func), retry_hooks=hooks)
                else:
                    result = breaker.call(call_func)
                penalty = max(0.0, 1.0 - depth * 0.1)
                metadata = {"plugin": name, "depth": depth}
                return FallbackResult(
                    plugin_name=name,
                    result=result,
                    depth=depth,
                    penalty=penalty,
                    metadata=metadata,
                )
            except (CircuitBreakerOpenError, CircuitBreakerTimeoutError) as err:
                last_error = err
                log_fallback_event(plugin=name, depth=depth, reason=type(err).__name__)
            except Exception as err:
                last_error = err
                log_fallback_event(plugin=name, depth=depth, reason=str(err))
                if self.fallback_mode == "strict":
                    raise
            depth += 1
        # All plugins failed or skipped
        if self.fallback_mode == "strict":
            raise last_error if last_error else RuntimeError("FallbackChain exhausted with no plugin")
        # Lenient or deferred mode: return empty result
        return FallbackResult(
            plugin_name="",
            result=None,
            depth=depth,
            penalty=0.0,
            metadata={"mode": self.fallback_mode},
        )