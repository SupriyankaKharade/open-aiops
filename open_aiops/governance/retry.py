"""Same-provider retry with exponential backoff and full jitter (INT-21).

Retries a single provider call up to N times. ``max_attempts`` counts **all**
attempts (for example 3 means one try plus two retries). Delay uses full jitter
(``random * min(cap, base * 2^attempt)``), honors ``Retry-After`` when it is at
most ``max_delay_s``, and never sleeps past an optional deadline. If
``Retry-After`` is above the cap, retrying stops and the error is re-raised so
failover can take over. Clock, sleep, RNG, and an optional ``on_retry`` hook
are injectable so tests stay deterministic and the gateway can emit
``RetryScheduled``.

Whether to fail over to another provider is decided by the caller (INT-6 /
gateway), using ``failover_eligible`` on the final exception.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]
Rng = Callable[[], float]
OnRetry = Callable[[int, float, BaseException], None]


@dataclass(frozen=True)
class RetryPolicy:
    """How many times to try one provider and how to back off between tries.

    ``max_attempts`` is the total number of calls (try + retries). A value of 3
    means one initial attempt and up to two retries.
    """

    max_attempts: int = 3
    base_delay_s: float = 0.05
    max_delay_s: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay_s < 0 or self.max_delay_s < 0:
            raise ValueError("delays must be >= 0")


def is_retryable(exc: BaseException) -> bool:
    """True if the same provider should be tried again.

    Reads the explicit ``retryable`` flag on provider errors (INT-10). Exceptions
    without the flag are not retried.
    """
    return bool(getattr(exc, "retryable", False))


def retry_after_s(exc: BaseException) -> float | None:
    """Seconds suggested by the provider (``retry_after_s``, with legacy ``retry_after``)."""
    for attr in ("retry_after_s", "retry_after"):
        value = getattr(exc, attr, None)
        if value is not None:
            return max(0.0, float(value))
    return None


def full_jitter_delay(
    failure_index: int,
    *,
    base_delay_s: float,
    max_delay_s: float,
    rng: Rng,
) -> float:
    """Full-jitter delay after ``failure_index`` failed attempts (0-based)."""
    if failure_index < 0:
        raise ValueError("failure_index must be >= 0")
    ceiling = min(max_delay_s, base_delay_s * (2**failure_index))
    return rng() * ceiling


def compute_delay_s(
    exc: BaseException,
    failure_index: int,
    *,
    policy: RetryPolicy,
    rng: Rng,
    remaining_s: float | None = None,
) -> float | None:
    """Delay before the next attempt, or ``None`` to stop retrying this provider.

    Uses the larger of full-jitter backoff and ``Retry-After``, then caps by
    remaining deadline. Returns ``None`` when:

    - ``Retry-After`` is above ``max_delay_s`` (fail over instead of waiting),
    - the deadline has already passed, or
    - the wait would consume the entire remainder (no room for another call).
    """
    suggested = retry_after_s(exc)
    if suggested is not None and suggested > policy.max_delay_s:
        return None

    jittered = full_jitter_delay(
        failure_index,
        base_delay_s=policy.base_delay_s,
        max_delay_s=policy.max_delay_s,
        rng=rng,
    )
    delay = max(jittered, suggested) if suggested is not None else jittered

    if remaining_s is None:
        return delay
    if remaining_s <= 0:
        return None
    if delay >= remaining_s:
        return None
    return delay


async def call_with_retry(
    call: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    deadline: float | None = None,
    clock: Clock = time.monotonic,
    sleep: Sleep = asyncio.sleep,
    rng: Rng = random.random,
    on_retry: OnRetry | None = None,
) -> T:
    """Run ``call`` until it succeeds, attempts are exhausted, or time runs out.

    ``policy.max_attempts`` counts every call (try + retries). Non-retryable
    errors and ``CancelledError`` are raised immediately. After the last failed
    retryable attempt — or when ``Retry-After`` exceeds ``max_delay_s`` — the
    last exception is re-raised so failover can handle it.

    ``on_retry(next_attempt, delay_s, error)`` is invoked just before each sleep
    so callers (the gateway) can emit ``RetryScheduled``.
    """
    policy = policy or RetryPolicy()
    last_exc: BaseException | None = None

    for attempt in range(policy.max_attempts):
        if deadline is not None and clock() >= deadline:
            if last_exc is not None:
                raise last_exc
            raise TimeoutError("retry deadline exceeded before first attempt")

        try:
            return await call()
        except Exception as exc:
            if not is_retryable(exc):
                raise
            last_exc = exc
            if attempt + 1 >= policy.max_attempts:
                raise

            remaining = None if deadline is None else deadline - clock()
            delay = compute_delay_s(
                exc,
                attempt,
                policy=policy,
                rng=rng,
                remaining_s=remaining,
            )
            if delay is None:
                raise
            if on_retry is not None:
                on_retry(attempt + 2, delay, exc)
            await sleep(delay)

    assert last_exc is not None
    raise last_exc
