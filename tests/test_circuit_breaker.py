import logging
import threading

import pytest

from open_aiops.core.config import ProviderConfig
from open_aiops.events import CircuitState, CircuitStateChanged
from open_aiops.governance.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
)
from open_aiops.governance.router import ModelRouter, RejectionReason, RouteRequest
from open_aiops.providers.errors import AuthError, BadRequest, RateLimitError

REQUEST = RouteRequest(input_tokens=100, max_output_tokens=100)


def provider(name: str, cost: float = 1.0) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        model=f"{name}-model",
        cost_per_1k_input_tokens=cost,
        cost_per_1k_output_tokens=cost,
        expected_latency_ms=100,
    )


def test_config_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        CircuitBreakerConfig(failure_threshold=0)


def test_opens_after_consecutive_failures_and_emits_event():
    events: list[CircuitStateChanged] = []
    clock = {"t": 0.0}
    breaker = CircuitBreaker(
        "primary",
        config=CircuitBreakerConfig(failure_threshold=3, cooldown_s=10.0),
        clock=lambda: clock["t"],
        on_state_change=events.append,
    )

    assert breaker.before_call() is True
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert events[-1].from_state is CircuitState.CLOSED
    assert events[-1].to_state is CircuitState.OPEN
    assert events[-1].provider == "primary"
    assert breaker.before_call() is False
    assert breaker.is_excluded_from_routing() is True


def test_half_open_probe_success_closes():
    clock = {"t": 0.0}
    events: list[CircuitStateChanged] = []
    breaker = CircuitBreaker(
        "primary",
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=5.0),
        clock=lambda: clock["t"],
        on_state_change=events.append,
    )
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN

    clock["t"] = 5.0
    assert breaker.is_excluded_from_routing() is False
    assert breaker.before_call() is True
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.before_call() is False  # only one probe

    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    assert events[-1].to_state is CircuitState.CLOSED


def test_half_open_probe_failure_reopens():
    clock = {"t": 0.0}
    breaker = CircuitBreaker(
        "primary",
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=5.0),
        clock=lambda: clock["t"],
    )
    breaker.record_failure()
    clock["t"] = 5.0
    assert breaker.before_call() is True
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.before_call() is False


def test_release_probe_unblocks_half_open_after_cancel():
    clock = {"t": 0.0}
    breaker = CircuitBreaker(
        "primary",
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=5.0),
        clock=lambda: clock["t"],
    )
    breaker.record_failure()
    clock["t"] = 5.0
    assert breaker.before_call() is True
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.before_call() is False

    breaker.release_probe()
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.before_call() is True


def test_stale_probe_can_be_stolen_after_cooldown():
    clock = {"t": 0.0}
    breaker = CircuitBreaker(
        "primary",
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=5.0),
        clock=lambda: clock["t"],
    )
    breaker.record_failure()
    clock["t"] = 5.0
    assert breaker.before_call() is True
    assert breaker.before_call() is False

    clock["t"] = 10.0  # probe held longer than cooldown
    assert breaker.before_call() is True
    assert breaker.state is CircuitState.HALF_OPEN


def test_open_ignores_late_failure_and_does_not_restart_cooldown():
    clock = {"t": 0.0}
    breaker = CircuitBreaker(
        "primary",
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=10.0),
        clock=lambda: clock["t"],
    )
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.is_excluded_from_routing() is True

    clock["t"] = 9.0
    breaker.record_failure()  # late in-flight failure must not restart cooldown
    clock["t"] = 10.0
    assert breaker.is_excluded_from_routing() is False
    assert breaker.before_call() is True


def test_open_ignores_late_success():
    clock = {"t": 0.0}
    breaker = CircuitBreaker(
        "primary",
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=10.0),
        clock=lambda: clock["t"],
    )
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    breaker.record_success()
    assert breaker.state is CircuitState.OPEN
    assert breaker.is_excluded_from_routing() is True


def test_on_state_change_runs_outside_lock_and_swallows_errors(caplog):
    clock = {"t": 0.0}
    seen_states: list[CircuitState] = []

    def callback(event: CircuitStateChanged) -> None:
        # Reading .state while callback runs would deadlock if we still held the lock.
        seen_states.append(breaker.state)
        raise RuntimeError("callback boom")

    breaker = CircuitBreaker(
        "primary",
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=5.0),
        clock=lambda: clock["t"],
        on_state_change=callback,
    )
    with caplog.at_level(logging.WARNING):
        breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert seen_states == [CircuitState.OPEN]
    assert any("RuntimeError" in r.message for r in caplog.records)
    assert all("callback boom" not in r.message for r in caplog.records)


def test_record_outcome_only_trips_on_failover_eligible():
    breaker = CircuitBreaker("p", config=CircuitBreakerConfig(failure_threshold=1))
    breaker.record_outcome(AuthError())
    breaker.record_outcome(BadRequest())
    assert breaker.state is CircuitState.CLOSED

    breaker.record_outcome(RateLimitError())
    assert breaker.state is CircuitState.OPEN


def test_record_outcome_none_is_success():
    clock = {"t": 0.0}
    breaker = CircuitBreaker(
        "primary",
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=5.0),
        clock=lambda: clock["t"],
    )
    breaker.record_failure()
    clock["t"] = 5.0
    assert breaker.before_call() is True
    breaker.record_outcome(None)
    assert breaker.state is CircuitState.CLOSED


def test_router_excludes_open_breaker_with_circuit_open_reason():
    clock = {"t": 0.0}
    registry = CircuitBreakerRegistry(
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=30.0),
        clock=lambda: clock["t"],
    )
    registry.get("primary").record_failure()
    router = ModelRouter(
        [provider("primary", 1.0), provider("backup", 2.0)],
        circuit_breakers=registry,
    )
    decision = router.decide(REQUEST)
    assert [p.name for p in decision.ranked] == ["backup"]
    assert decision.rejected["primary"] is RejectionReason.CIRCUIT_OPEN


def test_router_includes_provider_again_after_cooldown():
    clock = {"t": 0.0}
    registry = CircuitBreakerRegistry(
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=10.0),
        clock=lambda: clock["t"],
    )
    registry.get("primary").record_failure()
    router = ModelRouter([provider("primary")], circuit_breakers=registry)
    assert router.decide(REQUEST).ranked == []

    clock["t"] = 10.0
    assert [p.name for p in router.decide(REQUEST).ranked] == ["primary"]


def test_success_resets_failure_count():
    breaker = CircuitBreaker("p", config=CircuitBreakerConfig(failure_threshold=3))
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN


def test_concurrent_half_open_allows_single_probe():
    clock = {"t": 0.0}
    breaker = CircuitBreaker(
        "primary",
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=1.0),
        clock=lambda: clock["t"],
    )
    breaker.record_failure()
    clock["t"] = 1.0

    results: list[bool] = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(breaker.before_call())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results.count(True) == 1
    assert results.count(False) == 7
    assert breaker.state is CircuitState.HALF_OPEN
