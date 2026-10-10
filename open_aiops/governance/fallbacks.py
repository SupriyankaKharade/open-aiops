"""Automatic failover: retry a request on the next-best provider when one fails transiently.

Rate limits (429), timeouts and dropped connections move the request straight to the
next provider from `ModelRouter.decide()`, with no sleep in between. Any other error
(auth, bad request) is raised immediately, since another provider would fail the same way.

No router state is touched, so concurrent requests can't affect each other.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from open_aiops.core.config import ProviderConfig
from open_aiops.governance.router import ModelRouter, NoProviderAvailable, RouteRequest

T = TypeVar("T")


class RateLimitError(Exception):
    """Raised by a provider call when the provider answers HTTP 429."""

    def __init__(self, retry_after: float | None = None):
        super().__init__("rate limited")
        self.retry_after = retry_after


# asyncio.TimeoutError is only an alias of TimeoutError from Python 3.11.
FAILOVER_ERRORS: tuple[type[BaseException], ...] = (
    RateLimitError,
    TimeoutError,
    asyncio.TimeoutError,
    ConnectionError,
)


class AllProvidersFailed(NoProviderAvailable):
    def __init__(self, attempts: list[tuple[str, str]]):
        super().__init__(f"All providers failed: {attempts}")
        self.attempts = attempts


async def call_with_failover(
    router: ModelRouter,
    request: RouteRequest,
    call: Callable[[ProviderConfig], Awaitable[T]],
) -> T:
    """Run `call` on providers in ranked order until one succeeds.

    Raises `NoProviderAvailable` if no provider is eligible, or `AllProvidersFailed`
    (with `(provider, error_type)` per attempt) if every eligible one failed transiently.
    """
    decision = router.decide(request)
    if not decision.ranked:
        raise NoProviderAvailable(f"No eligible provider (rejected: {sorted(decision.rejected)})")
    attempts: list[tuple[str, str]] = []
    for provider in decision.ranked:
        try:
            return await call(provider)
        except FAILOVER_ERRORS as exc:
            attempts.append((provider.name, type(exc).__name__))
    raise AllProvidersFailed(attempts)
