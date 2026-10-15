"""Per-provider circuit breaker (INT-22).

States: closed → open (after N consecutive failures) → half-open (after cooldown)
→ closed (probe success) or open (probe failure). An open breaker is excluded
from ``ModelRouter.decide()`` with reason ``circuit_open``. State changes can
emit ``CircuitStateChanged`` via an optional callback (gateway / events).

Thread-safe under concurrent requests (one lock per breaker).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from open_aiops.events import CircuitState, CircuitStateChanged

Clock = Callable[[], float]
OnStateChange = Callable[[CircuitStateChanged], None]


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Failure threshold and cooldown for one provider breaker."""

    failure_threshold: int = 5
    cooldown_s: float = 30.0

    def __post_init__(self) -> None:
        if self.failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if self.cooldown_s < 0:
            raise ValueError("cooldown_s must be >= 0")


class CircuitBreaker:
    """Circuit breaker for a single provider name."""

    def __init__(
        self,
        provider: str,
        *,
        config: CircuitBreakerConfig | None = None,
        clock: Clock = time.monotonic,
        on_state_change: OnStateChange | None = None,
    ):
        self.provider = provider
        self._config = config or CircuitBreakerConfig()
        self._clock = clock
        self._on_state_change = on_state_change
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_probe_in_flight = False
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def is_excluded_from_routing(self) -> bool:
        """True when ``decide()`` should reject this provider as ``circuit_open``.

        Open breakers stay excluded until the cooldown elapses. After that they
        become eligible again (half-open probe is taken via ``before_call``).
        """
        with self._lock:
            if self._state != CircuitState.OPEN:
                return False
            assert self._opened_at is not None
            return (self._clock() - self._opened_at) < self._config.cooldown_s

    def before_call(self) -> bool:
        """Reserve a call slot. Returns False if the breaker is open (still cooling down).

        May transition ``open → half_open`` when the cooldown has elapsed. Only one
        half-open probe is allowed at a time.
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                assert self._opened_at is not None
                if (self._clock() - self._opened_at) < self._config.cooldown_s:
                    return False
                self._set_state_locked(CircuitState.HALF_OPEN)
                self._half_open_probe_in_flight = True
                return True
            # HALF_OPEN
            if self._half_open_probe_in_flight:
                return False
            self._half_open_probe_in_flight = True
            return True

    def record_success(self) -> None:
        """Clear failures; half-open probe success closes the breaker."""
        with self._lock:
            self._consecutive_failures = 0
            self._half_open_probe_in_flight = False
            self._opened_at = None
            if self._state != CircuitState.CLOSED:
                self._set_state_locked(CircuitState.CLOSED)

    def record_failure(self) -> None:
        """Count a failure; open the breaker at the threshold or on a failed probe."""
        with self._lock:
            self._half_open_probe_in_flight = False
            if self._state == CircuitState.HALF_OPEN:
                self._opened_at = self._clock()
                self._set_state_locked(CircuitState.OPEN)
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._config.failure_threshold:
                self._opened_at = self._clock()
                self._set_state_locked(CircuitState.OPEN)

    def _set_state_locked(self, new_state: CircuitState) -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        if self._on_state_change is not None:
            self._on_state_change(
                CircuitStateChanged(
                    provider=self.provider,
                    from_state=old,
                    to_state=new_state,
                )
            )


class CircuitBreakerRegistry:
    """Lazy per-provider breakers sharing one config and optional event hook."""

    def __init__(
        self,
        *,
        config: CircuitBreakerConfig | None = None,
        clock: Clock = time.monotonic,
        on_state_change: OnStateChange | None = None,
        providers: Iterable[str] | None = None,
    ):
        self._config = config or CircuitBreakerConfig()
        self._clock = clock
        self._on_state_change = on_state_change
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        if providers is not None:
            for name in providers:
                self.get(name)

    def get(self, provider: str) -> CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(provider)
            if breaker is None:
                breaker = CircuitBreaker(
                    provider,
                    config=self._config,
                    clock=self._clock,
                    on_state_change=self._on_state_change,
                )
                self._breakers[provider] = breaker
            return breaker

    def is_excluded_from_routing(self, provider: str) -> bool:
        return self.get(provider).is_excluded_from_routing()
