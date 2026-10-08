import asyncio

import pytest

from open_aiops.governance.fallbacks import (
    AllProvidersFailed,
    RateLimitError,
    call_with_failover,
)
from open_aiops.governance.router import (
    ModelRouter,
    NoProviderAvailable,
    ProviderConfig,
    RouteRequest,
)

REQUEST = RouteRequest(input_tokens=100, max_output_tokens=100)


def provider(name: str, cost: float) -> ProviderConfig:
    return ProviderConfig(
        name=name,
        model=f"{name}-model",
        cost_per_1k_input_tokens=cost,
        cost_per_1k_output_tokens=cost,
        expected_latency_ms=100,
    )


@pytest.fixture
def router() -> ModelRouter:
    return ModelRouter([provider("primary", 1.0), provider("backup", 2.0), provider("last", 3.0)])


def scripted(outcomes: dict[str, BaseException | str]):
    """A fake provider call: raises or returns the scripted outcome, and records call order."""
    calls: list[str] = []

    async def call(p: ProviderConfig) -> str:
        calls.append(p.name)
        outcome = outcomes[p.name]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    return call, calls


def run(router, call):
    return asyncio.run(call_with_failover(router, REQUEST, call))


def test_primary_success_makes_one_call(router):
    call, calls = scripted({"primary": "ok"})
    assert run(router, call) == "ok"
    assert calls == ["primary"]


def test_rate_limited_primary_fails_over_to_backup(router):
    call, calls = scripted({"primary": RateLimitError(retry_after=30), "backup": "from backup"})
    assert run(router, call) == "from backup"
    assert calls == ["primary", "backup"]


@pytest.mark.parametrize("error", [TimeoutError(), asyncio.TimeoutError(), ConnectionResetError()])
def test_timeouts_and_connection_errors_fail_over(router, error):
    call, calls = scripted({"primary": error, "backup": "ok"})
    assert run(router, call) == "ok"
    assert calls == ["primary", "backup"]


def test_failover_does_not_wait_between_providers(router, monkeypatch):
    async def no_sleep(*_args, **_kwargs):
        raise AssertionError("failover must not sleep")

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    call, _ = scripted({"primary": RateLimitError(), "backup": "ok"})
    assert run(router, call) == "ok"


def test_all_failing_raises_with_attempts_in_order(router):
    call, calls = scripted(
        {"primary": RateLimitError(), "backup": TimeoutError(), "last": ConnectionRefusedError()}
    )
    with pytest.raises(AllProvidersFailed) as exc_info:
        run(router, call)
    assert calls == ["primary", "backup", "last"]
    assert exc_info.value.attempts == [
        ("primary", "RateLimitError"),
        ("backup", "TimeoutError"),
        ("last", "ConnectionRefusedError"),
    ]


def test_all_failed_is_a_no_provider_available(router):
    call, _ = scripted({name: RateLimitError() for name in ("primary", "backup", "last")})
    with pytest.raises(NoProviderAvailable):
        run(router, call)


def test_non_transient_error_is_raised_without_failover(router):
    call, calls = scripted({"primary": PermissionError("bad key"), "backup": "ok"})
    with pytest.raises(PermissionError):
        run(router, call)
    assert calls == ["primary"]


def test_overloaded_provider_is_skipped(router):
    router.mark_overloaded("primary")
    call, calls = scripted({"backup": "ok"})
    assert run(router, call) == "ok"
    assert calls == ["backup"]


def test_failover_does_not_change_router_state(router):
    call, _ = scripted({"primary": RateLimitError(), "backup": "ok"})
    run(router, call)
    assert not router.is_overloaded("primary")


def test_no_eligible_provider_raises_before_calling(router):
    for name in ("primary", "backup", "last"):
        router.mark_overloaded(name)
    call, calls = scripted({})
    with pytest.raises(NoProviderAvailable) as exc_info:
        run(router, call)
    assert not isinstance(exc_info.value, AllProvidersFailed)
    assert calls == []
