"""Per-provider circuit breaker (INT-22).

States: closed → open (after N consecutive **failover-eligible** failures) →
half-open (after cooldown) → closed (probe success) or open (probe failure).
An open breaker is excluded from ``ModelRouter.decide()`` with reason
``circuit_open``.

Gateway usage::

    if not breaker.before_call():
        ... skip provider ...
    try:
        ...
        breaker.record_outcome(None)          # success
    except Exception as exc:
        breaker.record_outcome(exc)           # trips only if failover_eligible
    finally:
        breaker.release_probe()               # always safe; clears a stuck probe

Thread-safe under concurrent requests. ``on_state_change`` runs outside the
lock and never raises into the caller.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from open_aiops.events import CircuitState, CircuitStateChanged

Clock = Callable[[], float]
OnStateChange = Callable[[CircuitStateChanged], None]

logger = logging.getLogger(__name__)


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
        self._probe_started_at: float | None = None
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
        half-open probe is allowed at a time, unless the in-flight probe has been
        held longer than ``cooldown_s`` (stale probe is stolen).
        """
        event: CircuitStateChanged | None = None
        allowed = False
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                assert self._opened_at is not None
                if (self._clock() - self._opened_at) < self._config.cooldown_s:
                    return False
                event = self._transition_locked(CircuitState.HALF_OPEN)
                self._begin_probe_locked()
                allowed = True
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_probe_in_flight and not self._probe_is_stale_locked():
                    allowed = False
                else:
                    self._begin_probe_locked()
                    allowed = True
        if allowed and event is not None:
            self._emit(event)
        return allowed

    def release_probe(self) -> None:
        """Clear an in-flight half-open probe without changing state.

        Call from a ``finally`` block so cancellation or deadline expiry cannot
        leave the breaker stuck in half-open with the probe slot held forever.
        """
        with self._lock:
            self._half_open_probe_in_flight = False
            self._probe_started_at = None

    def record_success(self) -> None:
        """Clear failures; half-open probe success closes the breaker.

        Ignored while the breaker is ``OPEN`` (late success from a call that
        started before the trip must not close without a probe).
        """
        event: CircuitStateChanged | None = None
        with self._lock:
            if self._state == CircuitState.OPEN:
                return
            self._consecutive_failures = 0
            self._clear_probe_locked()
            self._opened_at = None
            if self._state == CircuitState.HALF_OPEN:
                event = self._transition_locked(CircuitState.CLOSED)
        self._emit(event)

    def record_failure(self) -> None:
        """Count a failure; open at the threshold or on a failed half-open probe.

        Ignored while the breaker is ``OPEN`` (late failures must not restart
        the cooldown).
        """
        event: CircuitStateChanged | None = None
        with self._lock:
            if self._state == CircuitState.OPEN:
                return
            self._clear_probe_locked()
            if self._state == CircuitState.HALF_OPEN:
                self._opened_at = self._clock()
                event = self._transition_locked(CircuitState.OPEN)
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._config.failure_threshold:
                    self._opened_at = self._clock()
                    event = self._transition_locked(CircuitState.OPEN)
        self._emit(event)

    def record_outcome(self, exc: BaseException | None = None) -> None:
        """Record a call result. Only ``failover_eligible`` errors trip the breaker.

        Pass ``None`` on success. Auth / bad-request errors release a half-open
        probe (if any) but do not open the breaker.
        """
        if exc is None:
            self.record_success()
            return
        if bool(getattr(exc, "failover_eligible", False)):
            self.record_failure()
        else:
            self.release_probe()

    def _begin_probe_locked(self) -> None:
        self._half_open_probe_in_flight = True
        self._probe_started_at = self._clock()

    def _clear_probe_locked(self) -> None:
        self._half_open_probe_in_flight = False
        self._probe_started_at = None

    def _probe_is_stale_locked(self) -> bool:
        if self._probe_started_at is None:
            return True
        return (self._clock() - self._probe_started_at) >= self._config.cooldown_s

    def _transition_locked(self, new_state: CircuitState) -> CircuitStateChanged | None:
        old = self._state
        if old == new_state:
            return None
        self._state = new_state
        return CircuitStateChanged(
            provider=self.provider,
            from_state=old,
            to_state=new_state,
        )

    def _emit(self, event: CircuitStateChanged | None) -> None:
        if event is None or self._on_state_change is None:
            return
        try:
            self._on_state_change(event)
        except Exception as exc:  # noqa: BLE001 — never break the request path
            logger.warning(
                "circuit breaker on_state_change failed type=%s provider=%s",
                type(exc).__name__,
                self.provider,
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
