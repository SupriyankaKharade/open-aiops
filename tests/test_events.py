import logging

import pytest

from open_aiops.events import (
    AttemptFailed,
    AttemptStarted,
    AttemptSucceeded,
    CircuitState,
    CircuitStateChanged,
    EventDispatcher,
    FallbackTriggered,
    MalformedEventSequence,
    RequestCompleted,
    RequestFailed,
    RequestStarted,
    RetryScheduled,
    RoutingDecided,
    StreamCompleted,
    StreamFirstChunk,
    assert_well_formed,
    new_id,
)


REQ = "req-1"
ATT = "att-1"


def golden_ok_path():
    return [
        RequestStarted(request_id=REQ),
        RoutingDecided(request_id=REQ, ranked=("primary", "backup"), rejected={}),
        AttemptStarted(
            request_id=REQ, attempt_id=ATT, attempt_number=1, provider="primary", model="m"
        ),
        AttemptSucceeded(
            request_id=REQ,
            attempt_id=ATT,
            attempt_number=1,
            provider="primary",
            model="m",
            latency_s=0.2,
            input_tokens=10,
            output_tokens=20,
        ),
        RequestCompleted(request_id=REQ, latency_s=0.25, cost=0.01),
    ]


def test_events_are_immutable():
    event = RequestStarted(request_id=REQ)
    with pytest.raises(Exception):
        event.request_id = "other"  # type: ignore[misc]


def test_assert_well_formed_accepts_golden_path():
    assert_well_formed(golden_ok_path())


def test_assert_well_formed_accepts_failover_path():
    events = [
        RequestStarted(request_id=REQ),
        RoutingDecided(request_id=REQ, ranked=("a", "b"), rejected={"c": "overloaded"}),
        AttemptStarted(
            request_id=REQ, attempt_id="a1", attempt_number=1, provider="a", model="m"
        ),
        AttemptFailed(
            request_id=REQ,
            attempt_id="a1",
            attempt_number=1,
            provider="a",
            model="m",
            error_type="RateLimitError",
        ),
        RetryScheduled(
            request_id=REQ,
            attempt_id="a1",
            attempt_number=1,
            provider="a",
            model="m",
            delay_s=0.0,
            error_type="RateLimitError",
        ),
        AttemptStarted(
            request_id=REQ, attempt_id="a2", attempt_number=2, provider="a", model="m"
        ),
        AttemptFailed(
            request_id=REQ,
            attempt_id="a2",
            attempt_number=2,
            provider="a",
            model="m",
            error_type="RateLimitError",
        ),
        FallbackTriggered(
            request_id=REQ,
            from_provider="a",
            to_provider="b",
            error_type="RateLimitError",
            attempt_number=2,
        ),
        AttemptStarted(
            request_id=REQ, attempt_id="b1", attempt_number=3, provider="b", model="m"
        ),
        AttemptSucceeded(
            request_id=REQ,
            attempt_id="b1",
            attempt_number=3,
            provider="b",
            model="m",
            latency_s=0.1,
        ),
        RequestCompleted(request_id=REQ, latency_s=0.5),
    ]
    assert_well_formed(events)


def test_assert_well_formed_rejects_missing_request_terminal():
    with pytest.raises(MalformedEventSequence):
        assert_well_formed([RequestStarted(request_id=REQ)])


def test_assert_well_formed_rejects_attempt_without_start():
    with pytest.raises(MalformedEventSequence):
        assert_well_formed(
            [
                RequestStarted(request_id=REQ),
                AttemptSucceeded(
                    request_id=REQ,
                    attempt_id=ATT,
                    attempt_number=1,
                    provider="p",
                    model="m",
                    latency_s=0.1,
                ),
            ]
        )


def test_assert_well_formed_rejects_open_attempt_at_request_end():
    with pytest.raises(MalformedEventSequence):
        assert_well_formed(
            [
                RequestStarted(request_id=REQ),
                AttemptStarted(
                    request_id=REQ,
                    attempt_id=ATT,
                    attempt_number=1,
                    provider="p",
                    model="m",
                ),
                RequestFailed(request_id=REQ, error_type="AllProvidersFailed"),
            ]
        )


def test_assert_well_formed_streaming_path():
    events = [
        RequestStarted(request_id=REQ),
        AttemptStarted(
            request_id=REQ, attempt_id=ATT, attempt_number=1, provider="p", model="m"
        ),
        StreamFirstChunk(
            request_id=REQ, attempt_id=ATT, attempt_number=1, provider="p", model="m"
        ),
        StreamCompleted(
            request_id=REQ,
            attempt_id=ATT,
            attempt_number=1,
            provider="p",
            model="m",
            latency_s=1.0,
        ),
        RequestCompleted(request_id=REQ, latency_s=1.0),
    ]
    assert_well_formed(events)


def test_assert_well_formed_ignores_circuit_events():
    assert_well_formed(
        [
            CircuitStateChanged(
                provider="p", from_state=CircuitState.CLOSED, to_state=CircuitState.OPEN
            ),
            *golden_ok_path(),
        ]
    )


def test_dispatcher_returns_immediately_with_no_subscribers():
    EventDispatcher().emit(RequestStarted(request_id=REQ))


def test_dispatcher_delivers_in_order():
    seen: list[str] = []
    dispatcher = EventDispatcher(
        [
            lambda e: seen.append(f"a:{e.type}"),
            lambda e: seen.append(f"b:{e.type}"),
        ]
    )
    dispatcher.emit(RequestStarted(request_id=REQ))
    assert seen == ["a:RequestStarted", "b:RequestStarted"]


def test_dispatcher_swallows_subscriber_errors(caplog):
    def boom(_event):
        raise RuntimeError("secret message must not escape")

    received: list[str] = []
    dispatcher = EventDispatcher([boom, lambda e: received.append(e.type)])
    with caplog.at_level(logging.WARNING):
        dispatcher.emit(RequestStarted(request_id=REQ))
    assert received == ["RequestStarted"]
    assert any("RuntimeError" in r.message for r in caplog.records)
    assert all("secret message" not in r.message for r in caplog.records)


def test_dispatcher_warns_when_subscriber_exceeds_budget(caplog):
    ticks = iter([0.0, 1.0])  # 1s elapsed

    def slow(_event):
        return None

    dispatcher = EventDispatcher(
        [slow],
        subscriber_budget_s=0.05,
        clock=lambda: next(ticks),
    )
    with caplog.at_level(logging.WARNING):
        dispatcher.emit(RequestStarted(request_id=REQ))
    assert any("slow" in r.message for r in caplog.records)


def test_new_id_is_opaque_unique():
    assert new_id() != new_id()
    assert len(new_id()) >= 32


def test_failed_events_carry_error_type_only():
    event = AttemptFailed(
        request_id=REQ,
        attempt_id=ATT,
        attempt_number=1,
        provider="p",
        model="m",
        error_type="TimeoutError",
    )
    dumped = event.model_dump()
    assert set(dumped) >= {"type", "error_type", "request_id", "attempt_id"}
    assert "message" not in dumped
    assert dumped["error_type"] == "TimeoutError"
