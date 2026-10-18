import asyncio

import pytest

from open_aiops.providers.errors import RateLimitError
from open_aiops.governance.retry import (
    RetryPolicy,
    call_with_retry,
    compute_delay_s,
    full_jitter_delay,
    is_retryable,
    retry_after_s,
)


class _Retryable(Exception):
    """Local stand-in with an explicit retryable flag."""

    def __init__(self, retry_after_s: float | None = None):
        super().__init__("retryable")
        self.retryable = True
        self.retry_after_s = retry_after_s


class _LegacyRetryAfter(Exception):
    """Covers the legacy ``retry_after`` attribute name without using RateLimitError."""

    def __init__(self, retry_after: float):
        super().__init__("legacy")
        self.retryable = True
        self.retry_after = retry_after


class _Fatal(Exception):
    def __init__(self):
        super().__init__("fatal")
        self.retryable = False


def test_retry_policy_rejects_invalid_max_attempts():
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)


def test_is_retryable_reads_explicit_flag():
    assert is_retryable(_Retryable()) is True
    assert is_retryable(RateLimitError()) is True
    assert is_retryable(_Fatal()) is False
    assert is_retryable(TimeoutError()) is False
    assert is_retryable(PermissionError()) is False


def test_retry_after_reads_retry_after_s_and_legacy_name():
    assert retry_after_s(RateLimitError(retry_after_s=1.5)) == 1.5
    assert retry_after_s(_Retryable(retry_after_s=2.0)) == 2.0
    assert retry_after_s(_LegacyRetryAfter(retry_after=0.75)) == 0.75
    assert retry_after_s(TimeoutError()) is None


def test_full_jitter_delay_is_within_ceiling():
    delays = [full_jitter_delay(0, base_delay_s=0.5, max_delay_s=2.0, rng=lambda: 1.0)]
    delays.append(full_jitter_delay(1, base_delay_s=0.5, max_delay_s=2.0, rng=lambda: 1.0))
    delays.append(full_jitter_delay(3, base_delay_s=0.5, max_delay_s=2.0, rng=lambda: 1.0))
    assert delays == [0.5, 1.0, 2.0]


def test_compute_delay_honors_retry_after_over_jitter():
    delay = compute_delay_s(
        _Retryable(retry_after_s=1.25),
        0,
        policy=RetryPolicy(base_delay_s=0.1, max_delay_s=2.0),
        rng=lambda: 0.0,
    )
    assert delay == 1.25


def test_compute_delay_stops_when_retry_after_exceeds_cap():
    assert (
        compute_delay_s(
            _Retryable(retry_after_s=3600.0),
            0,
            policy=RetryPolicy(base_delay_s=0.05, max_delay_s=2.0),
            rng=lambda: 1.0,
        )
        is None
    )


def test_compute_delay_returns_none_when_past_deadline():
    assert (
        compute_delay_s(
            _Retryable(),
            0,
            policy=RetryPolicy(),
            rng=lambda: 1.0,
            remaining_s=0.0,
        )
        is None
    )


def test_success_on_first_try_does_not_sleep():
    sleeps: list[float] = []

    async def ok():
        return "ok"

    async def record_sleep(delay: float):
        sleeps.append(delay)

    result = asyncio.run(
        call_with_retry(ok, sleep=record_sleep, rng=lambda: 0.0, clock=lambda: 0.0)
    )
    assert result == "ok"
    assert sleeps == []


def test_retries_then_succeeds_with_injected_backoff():
    calls = {"n": 0}
    sleeps: list[float] = []
    hooks: list[tuple[int, float, str]] = []

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Retryable()
        return "done"

    async def record_sleep(delay: float):
        sleeps.append(delay)

    def on_retry(next_attempt: int, delay_s: float, error: BaseException) -> None:
        hooks.append((next_attempt, delay_s, type(error).__name__))

    result = asyncio.run(
        call_with_retry(
            flaky,
            policy=RetryPolicy(max_attempts=3, base_delay_s=0.2, max_delay_s=1.0),
            sleep=record_sleep,
            rng=lambda: 1.0,
            clock=lambda: 0.0,
            on_retry=on_retry,
        )
    )
    assert result == "done"
    assert calls["n"] == 3
    assert sleeps == [0.2, 0.4]
    assert hooks == [(2, 0.2, "_Retryable"), (3, 0.4, "_Retryable")]


def test_non_retryable_raises_immediately():
    calls = {"n": 0}

    async def boom():
        calls["n"] += 1
        raise _Fatal()

    with pytest.raises(_Fatal):
        asyncio.run(call_with_retry(boom, policy=RetryPolicy(max_attempts=5)))
    assert calls["n"] == 1


def test_exhausted_retries_reraises_last_error():
    async def always():
        raise _Retryable(retry_after_s=0.1)

    with pytest.raises(_Retryable):
        asyncio.run(
            call_with_retry(
                always,
                policy=RetryPolicy(max_attempts=2, base_delay_s=0.0, max_delay_s=2.0),
                sleep=lambda _d: asyncio.sleep(0),
                rng=lambda: 0.0,
                clock=lambda: 0.0,
            )
        )


def test_retry_after_above_cap_reraises_without_sleeping():
    sleeps: list[float] = []
    calls = {"n": 0}

    async def always():
        calls["n"] += 1
        raise _Retryable(retry_after_s=3600.0)

    async def record_sleep(delay: float):
        sleeps.append(delay)

    with pytest.raises(_Retryable):
        asyncio.run(
            call_with_retry(
                always,
                policy=RetryPolicy(max_attempts=5, base_delay_s=0.05, max_delay_s=2.0),
                sleep=record_sleep,
                rng=lambda: 1.0,
                clock=lambda: 0.0,
            )
        )
    assert calls["n"] == 1
    assert sleeps == []


def test_deadline_stops_further_retries():
    calls = {"n": 0}
    ticks = iter([0.0, 0.9])

    async def always():
        calls["n"] += 1
        raise _Retryable()

    with pytest.raises(_Retryable):
        asyncio.run(
            call_with_retry(
                always,
                policy=RetryPolicy(max_attempts=5, base_delay_s=0.5, max_delay_s=1.0),
                deadline=1.0,
                clock=lambda: next(ticks),
                sleep=lambda _d: asyncio.sleep(0),
                rng=lambda: 1.0,
            )
        )
    assert calls["n"] == 1


def test_cancelled_error_passes_through_without_retry():
    calls = {"n": 0}

    async def cancelled():
        calls["n"] += 1
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(call_with_retry(cancelled, policy=RetryPolicy(max_attempts=5)))
    assert calls["n"] == 1
