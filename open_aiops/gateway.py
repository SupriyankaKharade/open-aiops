"""Gateway orchestrator for OpenAIOps (INT-13).

Orchestrates the entire request lifecycle:
    Client -> Gateway -> Router -> Circuit Breaker & Retry -> Provider Adapter
                                -> Event Dispatcher -> Telemetry & Metrics Subscribers

Invariants:
- Never imports OpenTelemetry or Prometheus code.
- Never mutates the router's shared overload state per request.
- Enforces deadline and timeout constraints.
- Emits well-formed immutable lifecycle events for every request, attempt, and transition.
- Attaches exact Decimal token costs computed via PricingRegistry.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
import logging
import time
from typing import Any, Protocol, runtime_checkable

from open_aiops.core.config import ProviderConfig
from open_aiops.core.pricing import Cost, PricingRegistry
from open_aiops.events import (
    AttemptCancelled,
    AttemptFailed,
    AttemptStarted,
    AttemptSucceeded,
    EventDispatcher,
    FallbackTriggered,
    RequestCompleted,
    RequestFailed,
    RequestStarted,
    RetryScheduled,
    RoutingDecided,
    new_id,
)
from open_aiops.governance.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry
from open_aiops.governance.fallbacks import AllProvidersFailed, is_failover_eligible
from open_aiops.governance.retry import (
    RetryPolicy,
    compute_delay_s,
    is_retryable,
)
from open_aiops.governance.router import (
    ModelRouter,
    NoProviderAvailable,
    RouteRequest,
    RoutingDecision,
)
from open_aiops.providers.base import (
    Provider,
    ProviderRequest,
    ProviderResponse,
    Usage,
)
from open_aiops.providers.errors import (
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
)

logger = logging.getLogger(__name__)

ProviderCallable = Callable[[ProviderRequest], Awaitable[ProviderResponse]]


@dataclass(frozen=True)
class GatewayResult:
    """Result of a successfully completed request through the gateway.

    Attributes:
        response: Standard ProviderResponse from the serving provider.
        served_by: Name of the provider that successfully served the request.
        attempts: Total number of attempts executed across all providers.
        cost: Calculated exact Decimal cost when pricing information is available.
        cost_status: Status string ('priced', 'unknown_model', 'missing_usage', 'free').
    """

    response: ProviderResponse
    served_by: str
    attempts: int
    cost: Decimal | None = None
    cost_status: str | None = None


class Gateway:
    """Central gateway orchestrator executing routing, resilience, and lifecycle telemetry."""

    def __init__(
        self,
        router: ModelRouter,
        *,
        providers: dict[str, Provider | ProviderCallable] | None = None,
        dispatcher: EventDispatcher | None = None,
        pricing: PricingRegistry | None = None,
        circuit_breakers: CircuitBreakerRegistry | None = None,
        retry_policy: RetryPolicy | None = None,
        default_timeout_ms: float = 30000.0,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = lambda: 0.5,
    ) -> None:
        """Initialize the gateway orchestrator.

        Args:
            router: ModelRouter instance configured with available ProviderConfigs.
            providers: Mapping of provider names to Provider adapters or async callables.
            dispatcher: In-process EventDispatcher for lifecycle events.
            pricing: PricingRegistry for token cost attribution.
            circuit_breakers: Optional CircuitBreakerRegistry for provider health checks.
            retry_policy: RetryPolicy for same-provider retries (defaults to max_attempts=3).
            default_timeout_ms: Global request deadline in milliseconds.
            clock: Monotonic clock for latency and deadline tracking.
            sleep: Async sleep function for backoff delays.
            rng: Random number generator for deterministic retry jitter testing.
        """
        self.router = router
        self.providers: dict[str, Provider | ProviderCallable] = dict(providers or {})
        self.dispatcher = dispatcher or EventDispatcher()
        self.pricing = pricing or PricingRegistry(router._providers)
        self.circuit_breakers = circuit_breakers
        self.retry_policy = retry_policy or RetryPolicy()
        self.default_timeout_ms = default_timeout_ms
        self._clock = clock
        self._sleep = sleep
        self._rng = rng

    def register_provider(
        self, name: str, provider: Provider | ProviderCallable
    ) -> None:
        """Register or update a provider adapter."""
        self.providers[name] = provider

    async def complete(
        self,
        request: ProviderRequest | dict[str, Any],
        *,
        routing_request: RouteRequest | None = None,
        max_latency_ms: float | None = None,
        max_cost: float | None = None,
        tenant_id: str | None = None,
        timeout_ms: float | None = None,
    ) -> GatewayResult:
        """Execute a completion request through routing, retries, and automated failover.

        Args:
            request: Standard ProviderRequest or dict with request fields.
            routing_request: Explicit RouteRequest parameters. If omitted, synthesized
                from request parameters and options.
            max_latency_ms: Optional SLA latency budget for router selection.
            max_cost: Optional maximum cost budget for router selection.
            tenant_id: Optional opaque tenant identifier for telemetry segregation.
            timeout_ms: Per-request deadline override (ms).

        Returns:
            GatewayResult containing response, serving provider, attempt count, and cost.

        Raises:
            NoProviderAvailable: If router rejects all providers.
            AllProvidersFailed: If every ranked provider fails transiently.
            ProviderError / Exception: If an unretryable error occurs.
            ProviderTimeout: If request deadline is exceeded.
        """
        if isinstance(request, dict):
            req = ProviderRequest(**request)
        else:
            req = request

        request_id = new_id()
        self.dispatcher.emit(RequestStarted(request_id=request_id, tenant_id=tenant_id))
        request_start_time = self._clock()

        # Build RouteRequest if not explicitly passed
        if routing_request is None:
            # Estimate input tokens from messages if available
            chars = sum(len(str(m.get("content", ""))) for m in req.messages) if req.messages else 100
            estimated_input = max(1, chars // 4)
            max_output = req.max_output_tokens or 1000
            route_req = RouteRequest(
                input_tokens=estimated_input,
                max_output_tokens=max_output,
                max_latency_ms=max_latency_ms,
                max_cost=max_cost,
            )
        else:
            route_req = routing_request

        decision = self.router.decide(route_req)
        ranked_names = tuple(p.name for p in decision.ranked)
        rejected_dict = {
            p: str(reason.value if hasattr(reason, "value") else reason)
            for p, reason in decision.rejected.items()
        }
        self.dispatcher.emit(
            RoutingDecided(
                request_id=request_id,
                ranked=ranked_names,
                rejected=rejected_dict,
                tenant_id=tenant_id,
            )
        )

        if not decision.ranked:
            self.dispatcher.emit(
                RequestFailed(
                    request_id=request_id,
                    error_type="NoProviderAvailable",
                    tenant_id=tenant_id,
                )
            )
            raise NoProviderAvailable(
                f"No eligible provider (rejected: {sorted(decision.rejected)})"
            )

        deadline_ms = timeout_ms if timeout_ms is not None else self.default_timeout_ms
        deadline_s = request_start_time + (deadline_ms / 1000.0)

        attempt_number = 0
        failed_attempts: list[tuple[str, str]] = []
        total_cost = Decimal(0)
        ranked_list = decision.ranked

        for idx, provider_config in enumerate(ranked_list):
            if self._clock() >= deadline_s:
                self.dispatcher.emit(
                    RequestFailed(
                        request_id=request_id,
                        error_type="ProviderTimeout",
                        tenant_id=tenant_id,
                    )
                )
                raise ProviderTimeout(
                    f"Gateway request deadline exceeded after {attempt_number} attempts"
                )

            breaker = (
                self.circuit_breakers.get(provider_config.name)
                if self.circuit_breakers
                else None
            )
            if breaker and not breaker.before_call():
                continue

            max_tries = self.retry_policy.max_attempts
            for try_idx in range(max_tries):
                remaining_s = deadline_s - self._clock()
                if remaining_s <= 0:
                    self.dispatcher.emit(
                        RequestFailed(
                            request_id=request_id,
                            error_type="ProviderTimeout",
                            tenant_id=tenant_id,
                        )
                    )
                    raise ProviderTimeout("Gateway request deadline exceeded")

                attempt_number += 1
                attempt_id = new_id()
                self.dispatcher.emit(
                    AttemptStarted(
                        request_id=request_id,
                        attempt_id=attempt_id,
                        attempt_number=attempt_number,
                        provider=provider_config.name,
                        model=provider_config.model,
                        tenant_id=tenant_id,
                    )
                )
                attempt_start = self._clock()

                adapter = self.providers.get(provider_config.name)
                if adapter is None:
                    exc = ProviderUnavailable(
                        f"No adapter registered for provider: {provider_config.name}",
                        provider=provider_config.name,
                    )
                    attempt_latency = max(0.0, self._clock() - attempt_start)
                    self.dispatcher.emit(
                        AttemptFailed(
                            request_id=request_id,
                            attempt_id=attempt_id,
                            attempt_number=attempt_number,
                            provider=provider_config.name,
                            model=provider_config.model,
                            error_type=type(exc).__name__,
                            retryable=False,
                            failover_eligible=True,
                            tenant_id=tenant_id,
                        )
                    )
                    failed_attempts.append((provider_config.name, type(exc).__name__))
                    break

                try:
                    call_req = ProviderRequest(
                        model=provider_config.model,
                        messages=req.messages,
                        max_output_tokens=req.max_output_tokens,
                        temperature=req.temperature,
                        extra_params=req.extra_params,
                    )
                    if hasattr(adapter, "complete"):
                        resp = await adapter.complete(call_req)
                    elif callable(adapter):
                        resp = await adapter(call_req)
                    else:
                        raise ProviderError(
                            f"Invalid provider adapter type: {type(adapter)}",
                            provider=provider_config.name,
                        )

                    attempt_latency = max(0.0, self._clock() - attempt_start)
                    if breaker:
                        breaker.record_outcome(None)

                    in_tokens = resp.usage.input_tokens if resp.usage else None
                    out_tokens = resp.usage.output_tokens if resp.usage else None
                    cost_obj = self.pricing.cost(
                        provider_config.name, provider_config.model, in_tokens, out_tokens
                    )
                    attempt_cost = cost_obj.amount
                    cost_status_val = (
                        cost_obj.status.value
                        if hasattr(cost_obj.status, "value")
                        else str(cost_obj.status)
                    )

                    if attempt_cost is not None:
                        total_cost += attempt_cost

                    self.dispatcher.emit(
                        AttemptSucceeded(
                            request_id=request_id,
                            attempt_id=attempt_id,
                            attempt_number=attempt_number,
                            provider=provider_config.name,
                            model=provider_config.model,
                            latency_s=attempt_latency,
                            input_tokens=in_tokens,
                            output_tokens=out_tokens,
                            cost=attempt_cost,
                            cost_status=cost_status_val,
                            tenant_id=tenant_id,
                        )
                    )

                    total_latency = max(0.0, self._clock() - request_start_time)
                    cost_is_complete = cost_status_val in ("priced", "free")
                    self.dispatcher.emit(
                        RequestCompleted(
                            request_id=request_id,
                            latency_s=total_latency,
                            total_cost=total_cost if total_cost > 0 or cost_status_val == "free" else None,
                            cost_complete=cost_is_complete,
                            tenant_id=tenant_id,
                        )
                    )

                    return GatewayResult(
                        response=resp,
                        served_by=provider_config.name,
                        attempts=attempt_number,
                        cost=attempt_cost,
                        cost_status=cost_status_val,
                    )

                except Exception as exc:
                    attempt_latency = max(0.0, self._clock() - attempt_start)
                    if breaker:
                        breaker.record_outcome(exc)

                    err_type = type(exc).__name__
                    retryable_flag = is_retryable(exc)
                    failover_flag = is_failover_eligible(exc)

                    self.dispatcher.emit(
                        AttemptFailed(
                            request_id=request_id,
                            attempt_id=attempt_id,
                            attempt_number=attempt_number,
                            provider=provider_config.name,
                            model=provider_config.model,
                            error_type=err_type,
                            retryable=retryable_flag,
                            failover_eligible=failover_flag,
                            tenant_id=tenant_id,
                        )
                    )

                    remaining_s = deadline_s - self._clock()
                    if retryable_flag and (try_idx + 1 < max_tries):
                        delay = compute_delay_s(
                            exc,
                            try_idx,
                            policy=self.retry_policy,
                            rng=self._rng,
                            remaining_s=remaining_s,
                        )
                        if delay is not None and delay < remaining_s:
                            self.dispatcher.emit(
                                RetryScheduled(
                                    request_id=request_id,
                                    attempt_id=attempt_id,
                                    attempt_number=attempt_number,
                                    provider=provider_config.name,
                                    model=provider_config.model,
                                    delay_s=delay,
                                    error_type=err_type,
                                    tenant_id=tenant_id,
                                )
                            )
                            await self._sleep(delay)
                            continue

                    if failover_flag:
                        failed_attempts.append((provider_config.name, err_type))
                        if idx + 1 < len(ranked_list):
                            next_provider = ranked_list[idx + 1]
                            self.dispatcher.emit(
                                FallbackTriggered(
                                    request_id=request_id,
                                    from_provider=provider_config.name,
                                    to_provider=next_provider.name,
                                    error_type=err_type,
                                    attempt_number=attempt_number,
                                    tenant_id=tenant_id,
                                )
                            )
                        break

                    self.dispatcher.emit(
                        RequestFailed(
                            request_id=request_id,
                            error_type=err_type,
                            tenant_id=tenant_id,
                        )
                    )
                    raise exc

        self.dispatcher.emit(
            RequestFailed(
                request_id=request_id,
                error_type="AllProvidersFailed",
                tenant_id=tenant_id,
            )
        )
        raise AllProvidersFailed(failed_attempts)
