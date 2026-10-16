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


from open_aiops.providers.errors import (
    ProviderTimeout,
    ProviderUnavailable,
    RateLimitError,
)

# Kept for backward compatibility
FAILOVER_ERRORS: tuple[type[BaseException], ...] = (
    RateLimitError,
    ProviderTimeout,
    ProviderUnavailable,
    TimeoutError,
    asyncio.TimeoutError,
    ConnectionError,
)


def is_failover_eligible(exc: BaseException) -> bool:
    """Determine whether an exception qualifies for automatic failover.

    Decides based on the exception's explicit `failover_eligible` flag if present.
    Standard network/timeout exceptions from the standard library qualify by default.
    """
    eligible = getattr(exc, "failover_eligible", None)
    if eligible is not None:
        return bool(eligible)
    return isinstance(exc, (TimeoutError, ConnectionError))


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
        except Exception as exc:
            if is_failover_eligible(exc):
                attempts.append((provider.name, type(exc).__name__))
            else:
                raise
    raise AllProvidersFailed(attempts)
