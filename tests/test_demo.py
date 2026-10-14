"""Tests for the end-to-end demo script (INT-7)."""

import asyncio
import pytest
from examples.run_demo import create_demo_router, execute_request_with_failover, run_demo
from open_aiops.governance.router import RouteRequest


def test_create_demo_router():
    router = create_demo_router()
    assert len(router._providers) == 3
    names = {p.name for p in router._providers}
    assert names == {"cloud-primary", "cloud-fallback", "local-economy"}


def test_execute_request_success():
    router = create_demo_router()
    req = RouteRequest(input_tokens=100, max_output_tokens=50)
    result = asyncio.run(
        execute_request_with_failover(router, "test-agent", req, simulated_output_tokens=30)
    )

    assert result["success"] is True
    assert result["agent"] == "test-agent"
    assert result["result"]["provider"] == "local-economy"
    assert result["result"]["input_tokens"] == 100
    assert result["result"]["output_tokens"] == 30
    assert len(result["attempts"]) == 1


def test_execute_request_with_automated_failover():
    router = create_demo_router()
    req = RouteRequest(input_tokens=100, max_output_tokens=50)
    result = asyncio.run(
        execute_request_with_failover(
            router,
            "test-agent",
            req,
            simulated_output_tokens=30,
            failing_providers={"local-economy"},
        )
    )

    assert result["success"] is True
    assert result["agent"] == "test-agent"
    # Fallback to cloud-primary
    assert result["result"]["provider"] == "cloud-primary"
    assert len(result["attempts"]) == 2
    assert "429" in result["attempts"][0]["status"]
    assert "200" in result["attempts"][1]["status"]


def test_execute_request_all_providers_fail():
    router = create_demo_router()
    req = RouteRequest(input_tokens=100, max_output_tokens=50)
    result = asyncio.run(
        execute_request_with_failover(
            router,
            "test-agent",
            req,
            simulated_output_tokens=30,
            failing_providers={"local-economy", "cloud-primary", "cloud-fallback"},
        )
    )

    assert result["success"] is False
    assert "All providers failed" in result["error"]
    assert len(result["attempts"]) == 3


def test_run_demo_executes_successfully():
    summary = run_demo()
    assert summary["status"] == "success"
    assert summary["requests_processed"] == 3
    assert summary["spans_recorded"] == 4  # 1 for agent A, 1 for agent B, 2 for agent C (1 fail + 1 success)
    assert summary["total_tokens"] > 0
