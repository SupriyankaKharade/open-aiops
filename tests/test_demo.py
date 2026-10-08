"""Tests for the end-to-end demo script."""

from examples.run_demo import create_demo_router, run_demo, simulate_agent_step
from open_aiops.governance.router import RouteRequest


def test_create_demo_router():
    router = create_demo_router()
    assert len(router._providers) == 3
    names = {p.name for p in router._providers}
    assert names == {"cloud-primary", "cloud-fallback", "local-economy"}


def test_simulate_agent_step():
    router = create_demo_router()
    req = RouteRequest(input_tokens=100, max_output_tokens=50)
    result = simulate_agent_step(router, "test-agent", req, simulated_output_tokens=30)

    assert result["success"] is True
    assert result["agent"] == "test-agent"
    assert result["provider"] == "local-economy"
    assert result["input_tokens"] == 100
    assert result["output_tokens"] == 30


def test_simulate_agent_step_failure():
    router = create_demo_router()
    req = RouteRequest(input_tokens=100, max_output_tokens=50)
    result = simulate_agent_step(
        router, "test-agent", req, simulated_output_tokens=0, fail_provider=True
    )

    assert result["success"] is False
    assert "Rate limit exceeded" in result["error"]


def test_run_demo_executes_successfully():
    summary = run_demo()
    assert summary["status"] == "success"
    assert summary["requests_processed"] == 5
    assert summary["spans_recorded"] == 5
    assert summary["total_tokens"] > 0
