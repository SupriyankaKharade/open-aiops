"""Comprehensive test suite for Gateway orchestrator (INT-13)."""

import asyncio
from decimal import Decimal
import pytest

from open_aiops.core.config import ProviderConfig
from open_aiops.core.pricing import PricingRegistry
from open_aiops.events import (
    EventDispatcher,
    LifecycleEvent,
    assert_well_formed,
)
from open_aiops.gateway import Gateway, GatewayResult
from open_aiops.governance.circuit_breaker import CircuitBreakerConfig, CircuitBreakerRegistry
from open_aiops.governance.fallbacks import AllProvidersFailed
from open_aiops.governance.retry import RetryPolicy
from open_aiops.governance.router import ModelRouter, NoProviderAvailable, RouteRequest
from open_aiops.providers.base import ProviderRequest, ProviderResponse, Usage
from open_aiops.providers.errors import (
    AuthError,
    BadRequest,
    ProviderTimeout,
    RateLimitError,
)
from tests.helpers.fake_provider import FakeProvider


def make_router_and_providers():
    p1 = ProviderConfig(
        name="cloud-primary",
        model="gpt-4o",
        cost_per_1k_input_tokens=0.002,
        cost_per_1k_output_tokens=0.006,
        expected_latency_ms=250.0,
    )
    p2 = ProviderConfig(
        name="cloud-fallback",
        model="claude-3-5-sonnet",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.008,
        expected_latency_ms=400.0,
    )
    p3 = ProviderConfig(
        name="local-economy",
        model="llama-3-70b-instruct",
        cost_per_1k_input_tokens=0.0005,
        cost_per_1k_output_tokens=0.0008,
        expected_latency_ms=800.0,
    )
    router = ModelRouter([p1, p2, p3])
    return router, [p1, p2, p3]


def test_gateway_normal_request_success():
    router, configs = make_router_and_providers()
    emitted: list[LifecycleEvent] = []
    dispatcher = EventDispatcher([emitted.append])

    fake_p1 = FakeProvider("cloud-primary", default_text="Hello from GPT-4o")
    gateway = Gateway(
        router,
        providers={"cloud-primary": fake_p1},
        dispatcher=dispatcher,
    )

    req = ProviderRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
        max_output_tokens=50,
    )
    route_req = RouteRequest(input_tokens=10, max_output_tokens=50, max_latency_ms=300.0)
    res = asyncio.run(gateway.complete(req, routing_request=route_req))

    assert isinstance(res, GatewayResult)
    assert res.served_by == "cloud-primary"
    assert res.response.text == "Hello from GPT-4o"
    assert res.attempts == 1
    assert res.cost is not None
    assert res.cost_status == "priced"

    # Invariant: well-formed event sequence
    assert_well_formed(emitted)

    types = [e.type for e in emitted]
    assert types == [
        "RequestStarted",
        "RoutingDecided",
        "AttemptStarted",
        "AttemptSucceeded",
        "RequestCompleted",
    ]


def test_gateway_dynamic_routing_cost_and_latency():
    router, configs = make_router_and_providers()
    gateway = Gateway(
        router,
        providers={
            "local-economy": FakeProvider("local-economy", default_text="Economy resp"),
            "cloud-primary": FakeProvider("cloud-primary", default_text="Fast resp"),
        },
    )

    req = ProviderRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Summary"}],
    )

    # 1. Cost-sensitive -> routes to cheapest: local-economy
    route_cheap = RouteRequest(input_tokens=100, max_output_tokens=100, max_cost=0.001)
    res_cheap = asyncio.run(gateway.complete(req, routing_request=route_cheap))
    assert res_cheap.served_by == "local-economy"

    # 2. Strict latency budget (< 300ms) -> routes to cloud-primary
    route_fast = RouteRequest(input_tokens=100, max_output_tokens=100, max_latency_ms=300.0)
    res_fast = asyncio.run(gateway.complete(req, routing_request=route_fast))
    assert res_fast.served_by == "cloud-primary"


def test_gateway_automated_failover_on_429():
    router, configs = make_router_and_providers()
    emitted: list[LifecycleEvent] = []
    dispatcher = EventDispatcher([emitted.append])

    # cloud-primary fails with HTTP 429
    fake_p1 = FakeProvider(
        "cloud-primary",
        responses=[RateLimitError(retry_after_s=10.0, provider="cloud-primary")],
    )
    fake_p2 = FakeProvider("cloud-fallback", default_text="Fallback success")

    gateway = Gateway(
        router,
        providers={"cloud-primary": fake_p1, "cloud-fallback": fake_p2},
        dispatcher=dispatcher,
        retry_policy=RetryPolicy(max_attempts=1),  # Fail over immediately
    )

    req = ProviderRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Analyze data"}],
    )
    route_req = RouteRequest(input_tokens=100, max_output_tokens=50, max_latency_ms=500.0)
    res = asyncio.run(gateway.complete(req, routing_request=route_req))

    assert res.served_by == "cloud-fallback"
    assert res.attempts == 2
    assert res.response.text == "Fallback success"

    assert_well_formed(emitted)

    types = [e.type for e in emitted]
    assert "FallbackTriggered" in types
    fallback_event = next(e for e in emitted if e.type == "FallbackTriggered")
    assert fallback_event.from_provider == "cloud-primary"
    assert fallback_event.to_provider == "cloud-fallback"


def test_gateway_same_provider_retry_backoff():
    router, configs = make_router_and_providers()
    emitted: list[LifecycleEvent] = []
    dispatcher = EventDispatcher([emitted.append])

    # Fails on attempt 1, succeeds on attempt 2
    fake_p1 = FakeProvider(
        "cloud-primary",
        responses=[
            RateLimitError(retry_after_s=0.01, provider="cloud-primary"),
            ProviderResponse(
                text="Success on retry",
                model="gpt-4o",
                usage=Usage(input_tokens=20, output_tokens=30),
            ),
        ],
    )

    async def immediate_sleep(delay: float) -> None:
        pass

    gateway = Gateway(
        router,
        providers={"cloud-primary": fake_p1},
        dispatcher=dispatcher,
        retry_policy=RetryPolicy(max_attempts=2, base_delay_s=0.01, max_delay_s=0.05),
        sleep=immediate_sleep,
    )

    req = ProviderRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "test"}],
    )
    route_req = RouteRequest(input_tokens=20, max_output_tokens=30, max_latency_ms=300.0)
    res = asyncio.run(gateway.complete(req, routing_request=route_req))

    assert res.served_by == "cloud-primary"
    assert res.attempts == 2
    assert res.response.text == "Success on retry"

    assert_well_formed(emitted)

    types = [e.type for e in emitted]
    assert types.count("AttemptStarted") == 2
    assert types.count("RetryScheduled") == 1
    assert "AttemptSucceeded" in types


def test_gateway_all_providers_failed():
    router, configs = make_router_and_providers()
    emitted: list[LifecycleEvent] = []
    dispatcher = EventDispatcher([emitted.append])

    fake_p1 = FakeProvider(
        "cloud-primary",
        responses=[RateLimitError(provider="cloud-primary")],
    )
    fake_p2 = FakeProvider(
        "cloud-fallback",
        responses=[RateLimitError(provider="cloud-fallback")],
    )

    gateway = Gateway(
        router,
        providers={"cloud-primary": fake_p1, "cloud-fallback": fake_p2},
        dispatcher=dispatcher,
        retry_policy=RetryPolicy(max_attempts=1),
    )

    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Fail"}])
    route_req = RouteRequest(input_tokens=100, max_output_tokens=50, max_latency_ms=500.0)

    with pytest.raises(AllProvidersFailed):
        asyncio.run(gateway.complete(req, routing_request=route_req))

    assert_well_formed(emitted)
    last_event = emitted[-1]
    assert last_event.type == "RequestFailed"
    assert last_event.error_type == "AllProvidersFailed"


def test_gateway_non_failover_error_raises_immediately():
    router, configs = make_router_and_providers()
    emitted: list[LifecycleEvent] = []
    dispatcher = EventDispatcher([emitted.append])

    fake_p1 = FakeProvider(
        "cloud-primary",
        responses=[AuthError(message="Invalid API Key", provider="cloud-primary")],
    )
    fake_p2 = FakeProvider("cloud-fallback", default_text="Should not be called")

    gateway = Gateway(
        router,
        providers={"cloud-primary": fake_p1, "cloud-fallback": fake_p2},
        dispatcher=dispatcher,
    )

    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Auth"}])
    route_req = RouteRequest(input_tokens=100, max_output_tokens=50, max_latency_ms=500.0)

    with pytest.raises(AuthError):
        asyncio.run(gateway.complete(req, routing_request=route_req))

    assert fake_p2.call_count == 0
    assert_well_formed(emitted)
    assert emitted[-1].type == "RequestFailed"
    assert emitted[-1].error_type == "AuthError"


def test_gateway_no_provider_available_rejection():
    router, configs = make_router_and_providers()
    emitted: list[LifecycleEvent] = []
    dispatcher = EventDispatcher([emitted.append])

    gateway = Gateway(router, dispatcher=dispatcher)

    # Impossible latency SLA: 50ms
    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Impossible"}])
    route_req = RouteRequest(input_tokens=100, max_output_tokens=50, max_latency_ms=50.0)

    with pytest.raises(NoProviderAvailable):
        asyncio.run(gateway.complete(req, routing_request=route_req))

    assert_well_formed(emitted)
    assert emitted[-1].type == "RequestFailed"
    assert emitted[-1].error_type == "NoProviderAvailable"


def test_gateway_circuit_breaker_skips_open_provider():
    router, configs = make_router_and_providers()
    cb_reg = CircuitBreakerRegistry(
        config=CircuitBreakerConfig(failure_threshold=1, cooldown_s=100.0)
    )

    # Trip breaker for cloud-primary
    breaker = cb_reg.get("cloud-primary")
    breaker.record_failure()
    assert breaker.is_excluded_from_routing()

    fake_p1 = FakeProvider("cloud-primary", default_text="Should be skipped")
    fake_p2 = FakeProvider("cloud-fallback", default_text="Served by fallback")

    gateway = Gateway(
        router,
        providers={"cloud-primary": fake_p1, "cloud-fallback": fake_p2},
        circuit_breakers=cb_reg,
    )

    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Check CB"}])
    route_req = RouteRequest(input_tokens=100, max_output_tokens=50, max_latency_ms=500.0)

    res = asyncio.run(gateway.complete(req, routing_request=route_req))
    assert res.served_by == "cloud-fallback"
    assert fake_p1.call_count == 0


def test_gateway_deadline_exceeded():
    router, configs = make_router_and_providers()
    emitted: list[LifecycleEvent] = []
    dispatcher = EventDispatcher([emitted.append])

    # Simulated slow clock that jumps past deadline
    current_time = 100.0

    def clock():
        nonlocal current_time
        current_time += 10.0  # 10s jump
        return current_time

    gateway = Gateway(
        router,
        providers={"cloud-primary": FakeProvider("cloud-primary")},
        dispatcher=dispatcher,
        clock=clock,
        default_timeout_ms=1000.0,  # 1s timeout
    )

    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Slow"}])
    route_req = RouteRequest(input_tokens=10, max_output_tokens=10, max_latency_ms=300.0)

    with pytest.raises(ProviderTimeout):
        asyncio.run(gateway.complete(req, routing_request=route_req))

    assert_well_formed(emitted)
    assert emitted[-1].type == "RequestFailed"
    assert emitted[-1].error_type == "ProviderTimeout"


def test_gateway_subscriber_error_does_not_break_request():
    router, configs = make_router_and_providers()

    def bad_subscriber(event):
        raise RuntimeError("Subscriber explosion!")

    dispatcher = EventDispatcher([bad_subscriber])

    gateway = Gateway(
        router,
        providers={"cloud-primary": FakeProvider("cloud-primary", default_text="Resilient")},
        dispatcher=dispatcher,
    )

    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Resilience"}])
    route_req = RouteRequest(input_tokens=10, max_output_tokens=10, max_latency_ms=300.0)

    res = asyncio.run(gateway.complete(req, routing_request=route_req))
    assert res.response.text == "Resilient"


def test_gateway_tenant_id_propagation_and_dict_request():
    router, configs = make_router_and_providers()
    emitted: list[LifecycleEvent] = []
    dispatcher = EventDispatcher([emitted.append])

    gateway = Gateway(
        router,
        providers={"cloud-primary": FakeProvider("cloud-primary")},
        dispatcher=dispatcher,
    )

    dict_req = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Tenant request"}],
        "max_output_tokens": 25,
    }
    route_req = RouteRequest(input_tokens=10, max_output_tokens=25, max_latency_ms=300.0)

    res = asyncio.run(gateway.complete(dict_req, routing_request=route_req, tenant_id="tenant-xyz"))
    assert res.served_by == "cloud-primary"

    assert_well_formed(emitted)
    for event in emitted:
        assert getattr(event, "tenant_id", None) == "tenant-xyz"

