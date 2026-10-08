import pytest

from open_aiops.governance.router import (
    ModelRouter,
    NoProviderAvailable,
    ProviderConfig,
    RouteRequest,
    estimate_cost,
)


def provider(name: str, cost: float, latency_ms: float) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        model=f"{name}-model",
        cost_per_1k_input_tokens=cost,
        cost_per_1k_output_tokens=cost,
        expected_latency_ms=latency_ms,
    )


@pytest.fixture
def router() -> ModelRouter:
    return ModelRouter(
        [
            provider("primary", cost=1.0, latency_ms=800),
            provider("secondary", cost=2.0, latency_ms=300),
            provider("local", cost=0.0, latency_ms=2000),
        ]
    )


REQUEST = RouteRequest(input_tokens=1000, max_output_tokens=1000)


def test_estimate_cost_uses_input_and_max_output_tokens():
    p = ProviderConfig(
        name="p",
        model="m",
        cost_per_1k_input_tokens=0.5,
        cost_per_1k_output_tokens=1.5,
        expected_latency_ms=100,
    )
    assert estimate_cost(p, RouteRequest(input_tokens=2000, max_output_tokens=1000)) == 2.5


def test_picks_cheapest_eligible_provider(router):
    assert router.route(REQUEST).name == "local"


def test_latency_budget_filters_slow_providers(router):
    assert router.route(REQUEST.model_copy(update={"max_latency_ms": 1000})).name == "primary"


def test_cost_budget_filters_expensive_providers(router):
    request = REQUEST.model_copy(update={"max_latency_ms": 500, "max_cost": 10})
    assert router.route(request).name == "secondary"
    with pytest.raises(NoProviderAvailable):
        router.route(request.model_copy(update={"max_cost": 3}))


def test_overloaded_primary_falls_back_to_secondary(router):
    request = REQUEST.model_copy(update={"max_latency_ms": 1000})
    router.mark_overloaded("primary")
    assert router.route(request).name == "secondary"

    router.mark_healthy("primary")
    assert router.route(request).name == "primary"


def test_all_overloaded_raises(router):
    for name in ("primary", "secondary", "local"):
        router.mark_overloaded(name)
    with pytest.raises(NoProviderAvailable):
        router.route(REQUEST)


def test_ties_break_on_latency_then_list_order():
    router = ModelRouter(
        [
            provider("a", cost=1.0, latency_ms=500),
            provider("b", cost=1.0, latency_ms=200),
            provider("c", cost=1.0, latency_ms=200),
        ]
    )
    assert router.route(REQUEST).name == "b"


def test_rejects_empty_and_duplicate_providers():
    with pytest.raises(ValueError):
        ModelRouter([])
    with pytest.raises(ValueError):
        ModelRouter([provider("a", 1, 1), provider("a", 2, 2)])


def test_unknown_provider_name_raises(router):
    with pytest.raises(KeyError):
        router.mark_overloaded("typo")
