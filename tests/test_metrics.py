"""Unit tests for OpenTelemetry metrics utilities (INT-4)."""

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from open_aiops.observability.metrics import (
    MetricsManager,
    get_metrics_manager,
    init_metrics,
)


@pytest.fixture
def metric_reader():
    """Fixture providing an InMemoryMetricReader with fresh metrics initialization."""
    reader = InMemoryMetricReader()
    init_metrics(metric_readers=[reader], force_reset=True)
    return reader


def _get_metric_data(reader: InMemoryMetricReader):
    """Helper to extract metric objects from InMemoryMetricReader."""
    data = reader.get_metrics_data()
    metrics_by_name = {}
    if data:
        for resource_metrics in data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    metrics_by_name[metric.name] = metric
    return metrics_by_name


def test_record_request_duration(metric_reader):
    manager = get_metrics_manager()
    manager.record_request_duration(
        duration_s=0.45, provider="openai", model="gpt-4o", outcome="success"
    )

    metrics_by_name = _get_metric_data(metric_reader)
    assert "openaiops.request.duration" in metrics_by_name
    metric = metrics_by_name["openaiops.request.duration"]
    data_points = list(metric.data.data_points)
    assert len(data_points) == 1
    point = data_points[0]
    assert point.attributes["provider"] == "openai"
    assert point.attributes["model"] == "gpt-4o"
    assert point.attributes["outcome"] == "success"
    assert point.count == 1
    assert point.sum == pytest.approx(0.45)


def test_record_requests_counter(metric_reader):
    manager = get_metrics_manager()
    manager.record_request(outcome="success")
    manager.record_request(outcome="failed")

    metrics_by_name = _get_metric_data(metric_reader)
    assert "openaiops.requests" in metrics_by_name
    metric = metrics_by_name["openaiops.requests"]
    data_points = list(metric.data.data_points)
    assert len(data_points) == 2
    outcomes = {p.attributes["outcome"]: p.value for p in data_points}
    assert outcomes["success"] == 1
    assert outcomes["failed"] == 1


def test_record_provider_errors(metric_reader):
    manager = get_metrics_manager()
    manager.record_provider_error(
        provider="anthropic", model="claude-3-5-sonnet", error_type="RateLimitError"
    )

    metrics_by_name = _get_metric_data(metric_reader)
    assert "openaiops.provider.errors" in metrics_by_name
    metric = metrics_by_name["openaiops.provider.errors"]
    data_points = list(metric.data.data_points)
    assert len(data_points) == 1
    point = data_points[0]
    assert point.attributes["provider"] == "anthropic"
    assert point.attributes["model"] == "claude-3-5-sonnet"
    assert point.attributes["error_type"] == "RateLimitError"
    assert point.value == 1


def test_record_retries_and_fallbacks(metric_reader):
    manager = get_metrics_manager()
    manager.record_retry(provider="primary-provider")
    manager.record_fallback(provider="primary-provider")

    metrics_by_name = _get_metric_data(metric_reader)
    assert "openaiops.retries" in metrics_by_name
    assert "openaiops.fallbacks" in metrics_by_name

    retry_point = list(metrics_by_name["openaiops.retries"].data.data_points)[0]
    assert retry_point.attributes["provider"] == "primary-provider"
    assert retry_point.value == 1

    fallback_point = list(metrics_by_name["openaiops.fallbacks"].data.data_points)[0]
    assert fallback_point.attributes["provider"] == "primary-provider"
    assert fallback_point.value == 1


def test_record_tokens_and_cost(metric_reader):
    manager = get_metrics_manager()
    manager.record_tokens(
        count=150, provider="openai", model="gpt-4o", token_type="input"
    )
    manager.record_tokens(
        count=80, provider="openai", model="gpt-4o", token_type="output"
    )
    manager.record_cost(amount=0.0035, provider="openai", model="gpt-4o")

    metrics_by_name = _get_metric_data(metric_reader)
    assert "openaiops.tokens" in metrics_by_name
    assert "openaiops.cost" in metrics_by_name

    token_points = list(metrics_by_name["openaiops.tokens"].data.data_points)
    assert len(token_points) == 2

    cost_point = list(metrics_by_name["openaiops.cost"].data.data_points)[0]
    assert cost_point.attributes["provider"] == "openai"
    assert cost_point.attributes["model"] == "gpt-4o"
    assert cost_point.value == pytest.approx(0.0035)


def test_metric_recording_never_raises():
    """Principle 2: Telemetry never breaks a request."""
    manager = MetricsManager()
    # Force all instruments to raise AttributeError inside try/except
    manager.request_duration = None
    manager.requests = None
    manager.provider_errors = None
    manager.retries = None
    manager.fallbacks = None
    manager.tokens = None
    manager.cost = None

    # Should not raise any exception and should safely log warnings
    manager.record_request_duration(0.5, "p", "m", "ok")
    manager.record_request("ok")
    manager.record_provider_error("p", "m", "err")
    manager.record_retry("p")
    manager.record_fallback("p")
    manager.record_tokens(10, "p", "m", "in")
    manager.record_cost(0.01, "p", "m")


def test_default_manager_and_init_without_readers():
    import open_aiops.observability.metrics as m
    from open_aiops.observability.metrics import get_meter

    meter = get_meter()
    assert meter is not None

    m._DEFAULT_MANAGER = None
    default_mgr = m.get_metrics_manager()
    assert default_mgr is not None

    # init without explicit readers
    mgr = init_metrics(force_reset=True)
    assert mgr is not None
    assert get_metrics_manager() is mgr


