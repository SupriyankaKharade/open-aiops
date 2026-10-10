"""Unit tests for OpenTelemetry metrics utilities (INT-4)."""

import math
import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from open_aiops.observability.metrics import (
    ALLOWED_ATTRIBUTE_KEYS,
    CANONICAL_ERROR_TYPES,
    CANONICAL_PROVIDER_ERRORS,
    DEFAULT_LATENCY_HISTOGRAM_BOUNDARIES,
    INSTRUMENT_PROVIDER_ERRORS,
    INSTRUMENT_REQUEST_DURATION,
    INSTRUMENT_REQUESTS,
    MetricsRecorder,
    VALID_OUTCOMES,
    get_meter,
)


# Canonical test exception hierarchy mirroring INT-10 architecture definitions
class ProviderError(Exception):
    """Base canonical provider error."""


class RateLimitError(ProviderError):
    """Canonical 429 rate limit error."""


class ProviderTimeout(ProviderError):
    """Canonical provider timeout."""


class ProviderUnavailable(ProviderError):
    """Canonical 5xx provider unavailable error."""


class AuthError(ProviderError):
    """Canonical 401/403 auth error."""


class BadRequest(ProviderError):
    """Canonical 400/422 bad request error."""


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
    recorder.record_request_duration(
        provider="openai", model="gpt-4o", duration=1.2, outcome="cancelled"
    )
    recorder.record_request_count(
        provider="openai", model="gpt-4o", outcome="error"
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


def test_histogram_latency_bucket_boundaries_and_distribution(isolated_meter_setup):
    """Histogram advisory bucket boundaries span seconds (~0.1 to 60s) with proper distribution."""
    reader, recorder = isolated_meter_setup

    # Record short, typical, and slow LLM request durations
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=0.05)  # < 0.1
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=0.4)   # 0.25 - 0.5
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=1.5)   # 1.0 - 2.5
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=4.0)   # 2.5 - 5.0
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=12.0)  # 10.0 - 15.0
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=45.0)  # 30.0 - 60.0

    metrics_map = _extract_metrics_by_name(reader)
    assert INSTRUMENT_REQUEST_DURATION in metrics_map
    duration_metric = metrics_map[INSTRUMENT_REQUEST_DURATION]
    point = list(duration_metric.data.data_points)[0]

    # Boundaries must match the seconds-tuned configuration
    assert point.explicit_bounds == DEFAULT_LATENCY_HISTOGRAM_BOUNDARIES
    assert point.count == 6
    assert point.sum == pytest.approx(0.05 + 0.4 + 1.5 + 4.0 + 12.0 + 45.0)

    # Observations should not all accumulate in the first bucket
    # Total buckets = len(bounds) + 1
    assert len(point.bucket_counts) == len(DEFAULT_LATENCY_HISTOGRAM_BOUNDARIES) + 1
    assert point.bucket_counts[0] == 1  # 0.05 is <= 0.1
    assert sum(point.bucket_counts) == 6
    # Non-first buckets must contain the slower observations
    assert sum(point.bucket_counts[1:]) == 5


def test_requests_counter_increments_by_outcome(isolated_meter_setup):
    """Successful, error, and cancelled requests increment the counter with intended outcomes."""
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
    recorder.record_request(
        provider="openai", model="gpt-4o", duration=0.20, outcome="cancelled"
    )

    metrics_map = _extract_metrics_by_name(reader)
    assert INSTRUMENT_REQUESTS in metrics_map
    request_metric = metrics_map[INSTRUMENT_REQUESTS]

    data_points = list(request_metric.data.data_points)
    assert len(data_points) == 3

    counts_by_outcome = {p.attributes["outcome"]: p.value for p in data_points}
    assert counts_by_outcome["success"] == 2
    assert counts_by_outcome["error"] == 1
    assert counts_by_outcome["cancelled"] == 1


def test_outcome_normalization_unrecognized_maps_to_error(isolated_meter_setup):
    """Unrecognized outcomes safely normalize to 'error' without raising exceptions."""
    reader, recorder = isolated_meter_setup

    # Various invalid or unrecognized outcome inputs
    recorder.record_request(provider="openai", model="gpt-4o", duration=0.1, outcome="unrecognized")
    recorder.record_request(provider="openai", model="gpt-4o", duration=0.1, outcome="failed")
    recorder.record_request(provider="openai", model="gpt-4o", duration=0.1, outcome="pending")
    recorder.record_request(provider="openai", model="gpt-4o", duration=0.1, outcome=None)  # type: ignore
    recorder.record_request(provider="openai", model="gpt-4o", duration=0.1, outcome=123)  # type: ignore

    metrics_map = _extract_metrics_by_name(reader)
    assert INSTRUMENT_REQUESTS in metrics_map
    request_metric = metrics_map[INSTRUMENT_REQUESTS]

    data_points = list(request_metric.data.data_points)
    assert len(data_points) == 1
    point = data_points[0]
    # All unrecognized inputs normalized to "error"
    assert point.attributes["outcome"] == "error"
    assert point.value == 5


def test_cancelled_outcome_supported(isolated_meter_setup):
    """Cancelled outcome is explicitly supported and recorded on duration and count."""
    reader, recorder = isolated_meter_setup

    recorder.record_request(
        provider="anthropic", model="claude-3-5-sonnet", duration=2.5, outcome="cancelled"
    )

    metrics_map = _extract_metrics_by_name(reader)
    assert INSTRUMENT_REQUESTS in metrics_map
    assert INSTRUMENT_REQUEST_DURATION in metrics_map

    req_point = list(metrics_map[INSTRUMENT_REQUESTS].data.data_points)[0]
    assert req_point.attributes["outcome"] == "cancelled"
    assert req_point.value == 1

    dur_point = list(metrics_map[INSTRUMENT_REQUEST_DURATION].data.data_points)[0]
    assert dur_point.attributes["outcome"] == "cancelled"
    assert dur_point.count == 1
    assert dur_point.sum == pytest.approx(2.5)


def test_provider_errors_counter_increments(isolated_meter_setup):
    """Provider error counter increments and records provider, model, and canonical error_type."""
    reader, recorder = isolated_meter_setup

    recorder.record_error(
        provider="anthropic", model="claude-3-5-sonnet", error_type="RateLimitError"
    )
    recorder.record_error(
        provider="anthropic", model="claude-3-5-sonnet", error_type="RateLimitError"
    )
    recorder.record_error(
        provider="anthropic", model="claude-3-5-sonnet", error_type="ProviderTimeout"
    )

    metrics_map = _extract_metrics_by_name(reader)
    assert INSTRUMENT_PROVIDER_ERRORS in metrics_map
    errors_metric = metrics_map[INSTRUMENT_PROVIDER_ERRORS]

    data_points = list(errors_metric.data.data_points)
    assert len(data_points) == 2

    counts_by_type = {p.attributes["error_type"]: p.value for p in data_points}
    assert counts_by_type["RateLimitError"] == 2
    assert counts_by_type["ProviderTimeout"] == 1


def test_canonical_error_type_normalization(isolated_meter_setup):
    """Canonical INT-10 error classes, instances, and strings normalize to canonical names."""
    reader, recorder = isolated_meter_setup

    # 1. Canonical strings
    recorder.record_error(provider="openai", model="gpt-4o", error_type="ProviderError")
    recorder.record_error(provider="openai", model="gpt-4o", error_type="RateLimitError")
    recorder.record_error(provider="openai", model="gpt-4o", error_type="ProviderTimeout")
    recorder.record_error(provider="openai", model="gpt-4o", error_type="ProviderUnavailable")
    recorder.record_error(provider="openai", model="gpt-4o", error_type="AuthError")
    recorder.record_error(provider="openai", model="gpt-4o", error_type="BadRequest")

    # 2. Canonical exception classes
    recorder.record_error(provider="openai", model="gpt-4o", error_type=RateLimitError)
    recorder.record_error(provider="openai", model="gpt-4o", error_type=ProviderTimeout)

    # 3. Canonical exception instances
    recorder.record_error(provider="openai", model="gpt-4o", error_type=AuthError("Invalid API key"))
    recorder.record_error(provider="openai", model="gpt-4o", error_type=BadRequest("Invalid params"))

    # 4. Subclass of canonical provider error
    class CustomRateLimit(RateLimitError):
        pass

    recorder.record_error(provider="openai", model="gpt-4o", error_type=CustomRateLimit("Too many tokens"))

    metrics_map = _extract_metrics_by_name(reader)
    errors_metric = metrics_map[INSTRUMENT_PROVIDER_ERRORS]
    recorded_types = {p.attributes["error_type"] for p in errors_metric.data.data_points}

    assert recorded_types == {
        "ProviderError",
        "RateLimitError",
        "ProviderTimeout",
        "ProviderUnavailable",
        "AuthError",
        "BadRequest",
    }


def test_unknown_error_types_map_to_unknown(isolated_meter_setup):
    """Non-canonical errors, old slugs, standard exceptions, and malformed inputs map to 'unknown'."""
    reader, recorder = isolated_meter_setup

    # Old slugs removed per maintainer requirements
    recorder.record_error(provider="openai", model="gpt-4o", error_type="rate_limit")
    recorder.record_error(provider="openai", model="gpt-4o", error_type="timeout")
    recorder.record_error(provider="openai", model="gpt-4o", error_type="service_unavailable")

    # Standard non-provider exceptions
    recorder.record_error(provider="openai", model="gpt-4o", error_type=ValueError)
    recorder.record_error(provider="openai", model="gpt-4o", error_type=RuntimeError("something failed"))
    recorder.record_error(provider="openai", model="gpt-4o", error_type=KeyError("missing_key"))

    # Unrecognized strings and types
    recorder.record_error(provider="openai", model="gpt-4o", error_type="unrecognized_error")
    recorder.record_error(provider="openai", model="gpt-4o", error_type="unknown")
    recorder.record_error(provider="openai", model="gpt-4o", error_type=123)  # type: ignore
    recorder.record_error(provider="openai", model="gpt-4o", error_type=None)  # type: ignore

    metrics_map = _extract_metrics_by_name(reader)
    errors_metric = metrics_map[INSTRUMENT_PROVIDER_ERRORS]
    data_points = list(errors_metric.data.data_points)

    assert len(data_points) == 1
    point = data_points[0]
    assert point.attributes["error_type"] == "unknown"
    assert point.value == 10


def test_sensitive_inputs_and_credentials_never_leak(isolated_meter_setup):
    """Raw error messages, bearer tokens, API keys, and prompts never leak into metric labels."""
    reader, recorder = isolated_meter_setup

    fake_api_key = "sk-live-secret-test-dummy-key"
    bearer_token = "Bearer fake-test-token-not-a-real-jwt"
    user_prompt = "User prompt: summarize confidential financial quarterly earnings"

    # Pass credential strings directly as error_type
    recorder.record_error(provider="openai", model="gpt-4o", error_type=fake_api_key)
    recorder.record_error(provider="openai", model="gpt-4o", error_type=bearer_token)
    recorder.record_error(provider="openai", model="gpt-4o", error_type=user_prompt)

    # Pass exception with sensitive message
    sensitive_exc = RuntimeError(f"Failed with key {fake_api_key} and prompt {user_prompt}")
    recorder.record_error(provider="openai", model="gpt-4o", error_type=sensitive_exc)

    metrics_map = _extract_metrics_by_name(reader)
    errors_metric = metrics_map[INSTRUMENT_PROVIDER_ERRORS]
    data_points = list(errors_metric.data.data_points)

    assert len(data_points) == 1
    point = data_points[0]
    assert point.attributes["error_type"] == "unknown"

    # Assert none of the sensitive strings exist in any metric attribute
    for attr_val in point.attributes.values():
        assert "sk-live-secret" not in str(attr_val)
        assert "Bearer" not in str(attr_val)
        assert "confidential" not in str(attr_val)


def test_invalid_durations_skipped_safely(isolated_meter_setup):
    """Negative, NaN, infinite, or non-numeric durations are skipped without raising ValueError."""
    reader, recorder = isolated_meter_setup

    # record_request_duration with invalid durations must never throw
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=-0.01)
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=-100.0)
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=float("nan"))
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=float("inf"))
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=float("-inf"))
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration="invalid")  # type: ignore
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=None)  # type: ignore
    recorder.record_request_duration(provider="openai", model="gpt-4o", duration=True)  # type: ignore

    # Verify no histogram observations were recorded
    metrics_map = _extract_metrics_by_name(reader)
    assert INSTRUMENT_REQUEST_DURATION not in metrics_map


def test_invalid_duration_preserves_request_count(isolated_meter_setup):
    """In record_request, invalid duration skips the histogram but preserves request count."""
    reader, recorder = isolated_meter_setup

    recorder.record_request(
        provider="openai", model="gpt-4o", duration=-1.5, outcome="success"
    )

    metrics_map = _extract_metrics_by_name(reader)
    # Duration histogram observation must be skipped
    assert INSTRUMENT_REQUEST_DURATION not in metrics_map

    # Request count must be preserved and incremented
    assert INSTRUMENT_REQUESTS in metrics_map
    req_metric = metrics_map[INSTRUMENT_REQUESTS]
    data_points = list(req_metric.data.data_points)
    assert len(data_points) == 1
    point = data_points[0]
    assert point.attributes["provider"] == "openai"
    assert point.attributes["model"] == "gpt-4o"
    assert point.attributes["outcome"] == "success"
    assert point.value == 1


def test_empty_or_malformed_identifiers_handled_safely(isolated_meter_setup):
    """Empty or non-string provider and model identifiers do not raise and record nothing."""
    reader, recorder = isolated_meter_setup

    recorder.record_request(provider="", model="gpt-4", duration=0.1)
    recorder.record_request(provider="openai", model="  ", duration=0.1)
    recorder.record_request(provider=None, model="gpt-4", duration=0.1)  # type: ignore
    recorder.record_request_duration(provider="", model="gpt-4", duration=0.1)
    recorder.record_request_count(provider="openai", model="")
    recorder.record_error(provider="", model="gpt-4", error_type="RateLimitError")

    metrics_map = _extract_metrics_by_name(reader)
    assert len(metrics_map) == 0


def test_only_permitted_attribute_keys_emitted(isolated_meter_setup):
    """Only the four permitted attribute keys (provider, model, outcome, error_type) are emitted."""
    reader, recorder = isolated_meter_setup

    recorder.record_request(
        provider="openai", model="gpt-4-turbo", duration=0.5, outcome="success"
    )
    recorder.record_error(
        provider="openai", model="gpt-4-turbo", error_type="RateLimitError"
    )

    metrics_map = _extract_metrics_by_name(reader)
    for metric in metrics_map.values():
        for point in metric.data.data_points:
            keys = set(point.attributes.keys())
            assert keys.issubset(ALLOWED_ATTRIBUTE_KEYS), f"Unpermitted attributes found: {keys}"


def test_instrument_unexpected_raise_handled_safely(isolated_meter_setup):
    """Underlying instrument failures are handled defensively and never escape to caller."""
    _, recorder = isolated_meter_setup

    # Monkeypatch duration_histogram.record to raise an Exception
    def broken_record(*args, **kwargs):
        raise RuntimeError("Underlying histogram hardware failure")

    recorder.duration_histogram.record = broken_record

    # record_request_duration must not raise
    recorder.record_request_duration(
        provider="openai", model="gpt-4o", duration=1.0, outcome="success"
    )

    # In record_request, failing histogram should not break request count recording
    recorder.record_request(
        provider="openai", model="gpt-4o", duration=1.0, outcome="success"
    )

    # Monkeypatch counters to raise as well
    def broken_add(*args, **kwargs):
        raise RuntimeError("Underlying counter storage failure")

    recorder.requests_counter.add = broken_add
    recorder.record_request_count(provider="openai", model="gpt-4o", outcome="success")
    recorder.record_request(provider="openai", model="gpt-4o", duration=1.0, outcome="success")

    recorder.errors_counter.add = broken_add
    recorder.record_error(provider="openai", model="gpt-4o", error_type="RateLimitError")
