"""Unit tests for OpenTelemetry metrics utilities (INT-4)."""

import math
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from open_aiops.observability.metrics import (
    ALLOWED_ATTRIBUTE_KEYS,
    INSTRUMENT_PROVIDER_ERRORS,
    INSTRUMENT_REQUEST_DURATION,
    INSTRUMENT_REQUESTS,
    MetricsCollector,
    MetricsRecorder,
    get_meter,
    get_metrics_recorder,
)


@pytest.fixture
def isolated_meter_setup():
    """Create an isolated MeterProvider and InMemoryMetricReader for tests.

    Avoids setting process-wide global state so tests run in complete isolation.
    """
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("tests.metrics")
    recorder = MetricsRecorder(meter=meter)
    return reader, recorder


def _extract_metrics_by_name(reader: InMemoryMetricReader):
    """Helper to extract collected metrics mapped by their instrument name."""
    data = reader.get_metrics_data()
    metrics_by_name = {}
    if data:
        for resource_metric in data.resource_metrics:
            for scope_metric in resource_metric.scope_metrics:
                for metric in scope_metric.metrics:
                    metrics_by_name[metric.name] = metric
    return metrics_by_name


def test_construct_facade_without_explicit_meter():
    """The facade constructs successfully with no explicitly supplied meter."""
    recorder = MetricsRecorder()
    assert recorder.meter is not None
    assert recorder.duration_histogram is not None
    assert recorder.requests_counter is not None
    assert recorder.errors_counter is not None


def test_noop_recording_without_sdk():
    """Recording through the default no-op setup operates cleanly without an SDK."""
    recorder = MetricsRecorder()
    # No exception should be raised when recording against default no-op meter
    recorder.record_request(
        provider="openai", model="gpt-4o", duration=0.45, outcome="success"
    )
    recorder.record_error(
        provider="anthropic", model="claude-3-opus", error_type="RateLimitError"
    )


def test_supplied_meter_records_request_duration_in_seconds(isolated_meter_setup):
    """A supplied meter records request durations in seconds with proper aggregates."""
    reader, recorder = isolated_meter_setup

    recorder.record_request_duration(
        provider="openai", model="gpt-4o", duration=0.75, outcome="success"
    )
    recorder.record_request_duration(
        provider="openai", model="gpt-4o", duration=1.25, outcome="success"
    )

    metrics_map = _extract_metrics_by_name(reader)
    assert INSTRUMENT_REQUEST_DURATION in metrics_map
    duration_metric = metrics_map[INSTRUMENT_REQUEST_DURATION]
    assert duration_metric.unit == "s"

    data_points = list(duration_metric.data.data_points)
    assert len(data_points) == 1
    point = data_points[0]
    assert point.count == 2
    assert point.sum == pytest.approx(2.0)
    assert point.attributes["provider"] == "openai"
    assert point.attributes["model"] == "gpt-4o"
    assert point.attributes["outcome"] == "success"


def test_requests_counter_increments_by_outcome(isolated_meter_setup):
    """Successful and failed requests increment the request counter with intended outcomes."""
    reader, recorder = isolated_meter_setup

    recorder.record_request(
        provider="openai", model="gpt-4o", duration=0.35, outcome="success"
    )
    recorder.record_request(
        provider="openai", model="gpt-4o", duration=0.40, outcome="success"
    )
    recorder.record_request(
        provider="openai", model="gpt-4o", duration=0.90, outcome="error"
    )

    metrics_map = _extract_metrics_by_name(reader)
    assert INSTRUMENT_REQUESTS in metrics_map
    request_metric = metrics_map[INSTRUMENT_REQUESTS]

    data_points = list(request_metric.data.data_points)
    assert len(data_points) == 2

    counts_by_outcome = {p.attributes["outcome"]: p.value for p in data_points}
    assert counts_by_outcome["success"] == 2
    assert counts_by_outcome["error"] == 1


def test_provider_errors_counter_increments(isolated_meter_setup):
    """Provider error counter increments and records provider, model, and error_type."""
    reader, recorder = isolated_meter_setup

    recorder.record_error(
        provider="anthropic", model="claude-3-5-sonnet", error_type="RateLimitError"
    )
    recorder.record_error(
        provider="anthropic", model="claude-3-5-sonnet", error_type="RateLimitError"
    )
    recorder.record_error(
        provider="anthropic", model="claude-3-5-sonnet", error_type="TimeoutError"
    )

    metrics_map = _extract_metrics_by_name(reader)
    assert INSTRUMENT_PROVIDER_ERRORS in metrics_map
    errors_metric = metrics_map[INSTRUMENT_PROVIDER_ERRORS]

    data_points = list(errors_metric.data.data_points)
    assert len(data_points) == 2

    counts_by_type = {p.attributes["error_type"]: p.value for p in data_points}
    assert counts_by_type["RateLimitError"] == 2
    assert counts_by_type["TimeoutError"] == 1


def test_only_permitted_attribute_keys_emitted(isolated_meter_setup):
    """Only the four permitted attribute keys (provider, model, outcome, error_type) are emitted."""
    reader, recorder = isolated_meter_setup

    recorder.record_request(
        provider="openai", model="gpt-4-turbo", duration=0.5, outcome="success"
    )
    recorder.record_error(
        provider="openai", model="gpt-4-turbo", error_type="ConnectionError"
    )

    metrics_map = _extract_metrics_by_name(reader)
    for metric in metrics_map.values():
        for point in metric.data.data_points:
            keys = set(point.attributes.keys())
            assert keys.issubset(ALLOWED_ATTRIBUTE_KEYS), f"Unpermitted attributes found: {keys}"


def test_no_sensitive_or_free_text_recorded_from_exceptions(isolated_meter_setup):
    """Exception instances extract stable class names and never leak messages or secrets."""
    reader, recorder = isolated_meter_setup

    # Simulate an exception containing sensitive secret/prompt data in message
    sensitive_msg = "Leaked API key sk-live-secret-9999 and user prompt: 'hello world'"
    exc = RuntimeError(sensitive_msg)

    recorder.record_error(provider="cohere", model="command-r", error_type=exc)

    metrics_map = _extract_metrics_by_name(reader)
    errors_metric = metrics_map[INSTRUMENT_PROVIDER_ERRORS]
    point = list(errors_metric.data.data_points)[0]

    # Verify error_type is the class name, and sensitive text is completely absent
    assert point.attributes["error_type"] == "RuntimeError"
    for attr_val in point.attributes.values():
        assert "sk-live-secret" not in str(attr_val)
        assert "hello world" not in str(attr_val)


def test_invalid_duration_inputs_rejected(isolated_meter_setup):
    """Negative and non-finite duration inputs raise ValueError consistently."""
    _, recorder = isolated_meter_setup

    # Negative duration
    with pytest.raises(ValueError, match="negative"):
        recorder.record_request_duration(
            provider="openai", model="gpt-4o", duration=-0.01
        )

    # NaN duration
    with pytest.raises(ValueError, match="finite"):
        recorder.record_request_duration(
            provider="openai", model="gpt-4o", duration=float("nan")
        )

    # Inf duration
    with pytest.raises(ValueError, match="finite"):
        recorder.record_request_duration(
            provider="openai", model="gpt-4o", duration=float("inf")
        )

    # Non-numeric
    with pytest.raises(ValueError, match="numeric"):
        recorder.record_request_duration(
            provider="openai", model="gpt-4o", duration="invalid"  # type: ignore
        )


def test_invalid_outcome_rejected(isolated_meter_setup):
    """Invalid outcome strings raise ValueError."""
    _, recorder = isolated_meter_setup

    with pytest.raises(ValueError, match="Invalid outcome"):
        recorder.record_request(
            provider="openai", model="gpt-4o", duration=0.2, outcome="unrecognized"
        )


def test_empty_identifiers_rejected(isolated_meter_setup):
    """Empty provider or model strings raise ValueError."""
    _, recorder = isolated_meter_setup

    with pytest.raises(ValueError, match="non-empty"):
        recorder.record_request(provider="", model="gpt-4", duration=0.1)

    with pytest.raises(ValueError, match="non-empty"):
        recorder.record_request(provider="openai", model="  ", duration=0.1)

    with pytest.raises(ValueError, match="non-empty"):
        recorder.record_error(provider="openai", model="gpt-4", error_type="")


def test_aliases_and_helper_functions():
    """Verify factory and backwards-compatible aliases work seamlessly."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("tests.aliases")

    rec1 = get_metrics_recorder(meter=meter)
    assert isinstance(rec1, MetricsRecorder)

    rec2 = MetricsCollector(meter=meter)
    assert isinstance(rec2, MetricsRecorder)

    # Verify record_provider_error alias
    rec1.record_provider_error(
        provider="google", model="gemini-1.5-pro", error_type="QuotaExceeded"
    )
    metrics_map = _extract_metrics_by_name(reader)
    assert INSTRUMENT_PROVIDER_ERRORS in metrics_map
