"""Unit tests for MetricsSubscriber (INT-19)."""

from decimal import Decimal
import logging
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
import pytest

from open_aiops.events import (
    AttemptCancelled,
    AttemptFailed,
    AttemptStarted,
    AttemptSucceeded,
    CircuitState,
    CircuitStateChanged,
    EventDispatcher,
    FallbackTriggered,
    RequestCompleted,
    RequestFailed,
    RequestStarted,
    RetryScheduled,
    RoutingDecided,
    StreamCompleted,
    StreamFirstChunk,
    StreamInterrupted,
    assert_well_formed,
    new_id,
)
from open_aiops.observability.metrics import (
    ALLOWED_ATTRIBUTE_KEYS,
    INSTRUMENT_COST,
    INSTRUMENT_FALLBACKS,
    INSTRUMENT_PROVIDER_ERRORS,
    INSTRUMENT_REQUEST_DURATION,
    INSTRUMENT_REQUESTS,
    INSTRUMENT_RETRIES,
    INSTRUMENT_TOKENS_INPUT,
    INSTRUMENT_TOKENS_OUTPUT,
    MetricsRecorder,
)
from open_aiops.observability.metrics_subscriber import (
    DEFAULT_MAX_ACTIVE_REQUESTS,
    MetricsSubscriber,
)


@pytest.fixture
def isolated_subscriber_setup():
    """Create an isolated MeterProvider, InMemoryMetricReader, MetricsRecorder, and MetricsSubscriber."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("tests.metrics_subscriber")
    recorder = MetricsRecorder(meter=meter)
    subscriber = MetricsSubscriber(recorder=recorder)
    return reader, recorder, subscriber


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


def test_normal_successful_request_flow(isolated_subscriber_setup):
    """A normal request records exactly 1 request count, 1 duration, tokens, and cost."""
    reader, _, subscriber = isolated_subscriber_setup
    dispatcher = EventDispatcher(subscribers=[subscriber])

    req_id = new_id()
    att_id = new_id()

    events = [
        RequestStarted(request_id=req_id),
        RoutingDecided(request_id=req_id, ranked=("openai",)),
        AttemptStarted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        ),
        AttemptSucceeded(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            latency_s=0.5,
            input_tokens=100,
            output_tokens=50,
            cost=Decimal("0.003"),
            cost_status="priced",
        ),
        RequestCompleted(
            request_id=req_id,
            latency_s=0.55,
            total_cost=Decimal("0.003"),
        ),
    ]

    assert_well_formed(events)
    for event in events:
        dispatcher.emit(event)

    # In-flight state must be cleaned up
    assert subscriber.active_requests_count == 0

    metrics_map = _extract_metrics_by_name(reader)

    # Request count must be 1 with success outcome
    assert INSTRUMENT_REQUESTS in metrics_map
    req_pts = list(metrics_map[INSTRUMENT_REQUESTS].data.data_points)
    assert len(req_pts) == 1
    assert req_pts[0].attributes == {"provider": "openai", "model": "gpt-4o", "outcome": "success"}
    assert req_pts[0].value == 1

    # Request duration must be recorded once with overall request latency (0.55s)
    assert INSTRUMENT_REQUEST_DURATION in metrics_map
    dur_pts = list(metrics_map[INSTRUMENT_REQUEST_DURATION].data.data_points)
    assert len(dur_pts) == 1
    assert dur_pts[0].attributes == {"provider": "openai", "model": "gpt-4o", "outcome": "success"}
    assert dur_pts[0].count == 1
    assert dur_pts[0].sum == pytest.approx(0.55)

    # Tokens recorded
    assert INSTRUMENT_TOKENS_INPUT in metrics_map
    in_pts = list(metrics_map[INSTRUMENT_TOKENS_INPUT].data.data_points)
    assert in_pts[0].sum == 100

    assert INSTRUMENT_TOKENS_OUTPUT in metrics_map
    out_pts = list(metrics_map[INSTRUMENT_TOKENS_OUTPUT].data.data_points)
    assert out_pts[0].sum == 50

    # Cost recorded
    assert INSTRUMENT_COST in metrics_map
    cost_pts = list(metrics_map[INSTRUMENT_COST].data.data_points)
    assert cost_pts[0].value == pytest.approx(0.003)

    # No errors, retries, or fallbacks
    assert INSTRUMENT_PROVIDER_ERRORS not in metrics_map
    assert INSTRUMENT_RETRIES not in metrics_map
    assert INSTRUMENT_FALLBACKS not in metrics_map


def test_v1_golden_scenario_rate_limit_retry_failover(isolated_subscriber_setup):
    """Golden scenario (§4.2): 2 rate limits, 1 retry, 1 fallback, 1 success on provider B."""
    reader, _, subscriber = isolated_subscriber_setup
    dispatcher = EventDispatcher(subscribers=[subscriber])

    req_id = new_id()
    att1, att2, att3 = new_id(), new_id(), new_id()

    events = [
        RequestStarted(request_id=req_id),
        RoutingDecided(request_id=req_id, ranked=("provider_a", "provider_b")),
        # Attempt 1: provider_a fails with 429
        AttemptStarted(
            request_id=req_id, attempt_id=att1, attempt_number=1, provider="provider_a", model="model_a"
        ),
        AttemptFailed(
            request_id=req_id,
            attempt_id=att1,
            attempt_number=1,
            provider="provider_a",
            model="model_a",
            error_type="RateLimitError",
            retryable=True,
            failover_eligible=True,
        ),
        RetryScheduled(
            request_id=req_id,
            attempt_id=att1,
            attempt_number=1,
            provider="provider_a",
            model="model_a",
            delay_s=0.5,
            error_type="RateLimitError",
        ),
        # Attempt 2 (retry): provider_a fails again with 429
        AttemptStarted(
            request_id=req_id, attempt_id=att2, attempt_number=2, provider="provider_a", model="model_a"
        ),
        AttemptFailed(
            request_id=req_id,
            attempt_id=att2,
            attempt_number=2,
            provider="provider_a",
            model="model_a",
            error_type="RateLimitError",
            retryable=False,
            failover_eligible=True,
        ),
        FallbackTriggered(
            request_id=req_id,
            from_provider="provider_a",
            to_provider="provider_b",
            error_type="RateLimitError",
            attempt_number=2,
        ),
        # Attempt 3: provider_b succeeds
        AttemptStarted(
            request_id=req_id, attempt_id=att3, attempt_number=3, provider="provider_b", model="model_b"
        ),
        AttemptSucceeded(
            request_id=req_id,
            attempt_id=att3,
            attempt_number=3,
            provider="provider_b",
            model="model_b",
            latency_s=0.2,
            input_tokens=120,
            output_tokens=340,
            cost=Decimal("0.0043"),
            cost_status="priced",
        ),
        RequestCompleted(
            request_id=req_id,
            latency_s=0.412,
            total_cost=Decimal("0.0043"),
        ),
    ]

    assert_well_formed(events)
    for event in events:
        dispatcher.emit(event)

    assert subscriber.active_requests_count == 0
    metrics_map = _extract_metrics_by_name(reader)

    # 1. Exactly 1 request completed count for provider_b
    assert INSTRUMENT_REQUESTS in metrics_map
    req_pts = list(metrics_map[INSTRUMENT_REQUESTS].data.data_points)
    assert len(req_pts) == 1
    assert req_pts[0].attributes == {"provider": "provider_b", "model": "model_b", "outcome": "success"}
    assert req_pts[0].value == 1

    # 2. Exactly 1 request duration observation for provider_b with 0.412s
    assert INSTRUMENT_REQUEST_DURATION in metrics_map
    dur_pts = list(metrics_map[INSTRUMENT_REQUEST_DURATION].data.data_points)
    assert len(dur_pts) == 1
    assert dur_pts[0].attributes == {"provider": "provider_b", "model": "model_b", "outcome": "success"}
    assert dur_pts[0].count == 1
    assert dur_pts[0].sum == pytest.approx(0.412)

    # 3. Exactly 2 provider errors for provider_a
    assert INSTRUMENT_PROVIDER_ERRORS in metrics_map
    err_pts = list(metrics_map[INSTRUMENT_PROVIDER_ERRORS].data.data_points)
    assert len(err_pts) == 1
    assert err_pts[0].attributes == {
        "provider": "provider_a",
        "model": "model_a",
        "error_type": "RateLimitError",
    }
    assert err_pts[0].value == 2

    # 4. Exactly 1 retry for provider_a
    assert INSTRUMENT_RETRIES in metrics_map
    retry_pts = list(metrics_map[INSTRUMENT_RETRIES].data.data_points)
    assert len(retry_pts) == 1
    assert retry_pts[0].attributes == {"provider": "provider_a"}
    assert retry_pts[0].value == 1

    # 5. Exactly 1 fallback for provider_a
    assert INSTRUMENT_FALLBACKS in metrics_map
    fallback_pts = list(metrics_map[INSTRUMENT_FALLBACKS].data.data_points)
    assert len(fallback_pts) == 1
    assert fallback_pts[0].attributes == {"provider": "provider_a"}
    assert fallback_pts[0].value == 1

    # 6. Cost only attributed to provider_b
    assert INSTRUMENT_COST in metrics_map
    cost_pts = list(metrics_map[INSTRUMENT_COST].data.data_points)
    assert len(cost_pts) == 1
    assert cost_pts[0].attributes == {"provider": "provider_b", "model": "model_b"}
    assert cost_pts[0].value == pytest.approx(0.0043)


def test_request_failure_does_not_duplicate_provider_errors(isolated_subscriber_setup):
    """RequestFailed records 1 request error outcome and does not duplicate provider errors."""
    reader, _, subscriber = isolated_subscriber_setup
    dispatcher = EventDispatcher(subscribers=[subscriber])

    req_id = new_id()
    att_id = new_id()

    events = [
        RequestStarted(request_id=req_id),
        AttemptStarted(
            request_id=req_id, attempt_id=att_id, attempt_number=1, provider="openai", model="gpt-4"
        ),
        AttemptFailed(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4",
            error_type="ProviderTimeout",
            retryable=False,
            failover_eligible=False,
        ),
        RequestFailed(request_id=req_id, error_type="AllProvidersFailed"),
    ]

    assert_well_formed(events)
    for event in events:
        dispatcher.emit(event)

    assert subscriber.active_requests_count == 0
    metrics_map = _extract_metrics_by_name(reader)

    # Exactly 1 provider error recorded from AttemptFailed
    assert INSTRUMENT_PROVIDER_ERRORS in metrics_map
    err_pts = list(metrics_map[INSTRUMENT_PROVIDER_ERRORS].data.data_points)
    assert len(err_pts) == 1
    assert err_pts[0].attributes == {
        "provider": "openai",
        "model": "gpt-4",
        "error_type": "ProviderTimeout",
    }
    assert err_pts[0].value == 1

    # Request count recorded with outcome='error'
    assert INSTRUMENT_REQUESTS in metrics_map
    req_pts = list(metrics_map[INSTRUMENT_REQUESTS].data.data_points)
    assert len(req_pts) == 1
    assert req_pts[0].attributes == {"provider": "openai", "model": "gpt-4", "outcome": "error"}
    assert req_pts[0].value == 1

    # Request duration must NOT be recorded on failure
    assert INSTRUMENT_REQUEST_DURATION not in metrics_map


def test_cancellation_flow(isolated_subscriber_setup):
    """Cancellation produces a single request outcome with outcome='cancelled'."""
    reader, _, subscriber = isolated_subscriber_setup
    dispatcher = EventDispatcher(subscribers=[subscriber])

    req_id = new_id()
    att_id = new_id()

    events = [
        RequestStarted(request_id=req_id),
        AttemptStarted(
            request_id=req_id, attempt_id=att_id, attempt_number=1, provider="openai", model="gpt-4o"
        ),
        AttemptCancelled(
            request_id=req_id, attempt_id=att_id, attempt_number=1, provider="openai", model="gpt-4o"
        ),
        RequestFailed(request_id=req_id, error_type="cancelled"),
    ]

    assert_well_formed(events)
    for event in events:
        dispatcher.emit(event)

    assert subscriber.active_requests_count == 0
    metrics_map = _extract_metrics_by_name(reader)

    # 1 request count with outcome='cancelled'
    assert INSTRUMENT_REQUESTS in metrics_map
    req_pts = list(metrics_map[INSTRUMENT_REQUESTS].data.data_points)
    assert len(req_pts) == 1
    assert req_pts[0].attributes == {
        "provider": "openai",
        "model": "gpt-4o",
        "outcome": "cancelled",
    }
    assert req_pts[0].value == 1

    # No duration and no provider errors
    assert INSTRUMENT_REQUEST_DURATION not in metrics_map
    assert INSTRUMENT_PROVIDER_ERRORS not in metrics_map


def test_streaming_success_and_interrupted_flows(isolated_subscriber_setup):
    """Streaming operations record tokens, duration, and errors appropriately."""
    reader, _, subscriber = isolated_subscriber_setup
    dispatcher = EventDispatcher(subscribers=[subscriber])

    # 1. Successful stream
    req1 = new_id()
    att1 = new_id()
    stream_events = [
        RequestStarted(request_id=req1),
        AttemptStarted(
            request_id=req1, attempt_id=att1, attempt_number=1, provider="anthropic", model="claude"
        ),
        StreamFirstChunk(
            request_id=req1, attempt_id=att1, attempt_number=1, provider="anthropic", model="claude"
        ),
        StreamCompleted(
            request_id=req1,
            attempt_id=att1,
            attempt_number=1,
            provider="anthropic",
            model="claude",
            latency_s=1.2,
            input_tokens=40,
            output_tokens=180,
            cost=Decimal("0.005"),
            cost_status="priced",
        ),
        RequestCompleted(request_id=req1, latency_s=1.25, total_cost=Decimal("0.005")),
    ]
    assert_well_formed(stream_events)
    for event in stream_events:
        dispatcher.emit(event)

    # 2. Interrupted stream
    req2 = new_id()
    att2 = new_id()
    interrupted_events = [
        RequestStarted(request_id=req2),
        AttemptStarted(
            request_id=req2, attempt_id=att2, attempt_number=1, provider="anthropic", model="claude"
        ),
        StreamFirstChunk(
            request_id=req2, attempt_id=att2, attempt_number=1, provider="anthropic", model="claude"
        ),
        StreamInterrupted(
            request_id=req2,
            attempt_id=att2,
            attempt_number=1,
            provider="anthropic",
            model="claude",
            error_type="ProviderUnavailable",
        ),
        RequestFailed(request_id=req2, error_type="ProviderUnavailable"),
    ]
    assert_well_formed(interrupted_events)
    for event in interrupted_events:
        dispatcher.emit(event)

    assert subscriber.active_requests_count == 0
    metrics_map = _extract_metrics_by_name(reader)

    # 1 success and 1 error request
    req_pts = list(metrics_map[INSTRUMENT_REQUESTS].data.data_points)
    outcomes = {p.attributes["outcome"]: p.value for p in req_pts}
    assert outcomes == {"success": 1, "error": 1}

    # StreamInterrupted recorded as provider error
    err_pts = list(metrics_map[INSTRUMENT_PROVIDER_ERRORS].data.data_points)
    assert len(err_pts) == 1
    assert err_pts[0].attributes == {
        "provider": "anthropic",
        "model": "claude",
        "error_type": "ProviderUnavailable",
    }


def test_missing_usage_unknown_model_and_free_cost_handling(isolated_subscriber_setup):
    """Missing tokens, unknown models, and free models do not raise or corrupt metrics."""
    reader, _, subscriber = isolated_subscriber_setup
    dispatcher = EventDispatcher(subscribers=[subscriber])

    # Request with missing usage and missing cost
    req1 = new_id()
    att1 = new_id()
    dispatcher.emit(RequestStarted(request_id=req1))
    dispatcher.emit(
        AttemptStarted(
            request_id=req1, attempt_id=att1, attempt_number=1, provider="local", model="llama"
        )
    )
    dispatcher.emit(
        AttemptSucceeded(
            request_id=req1,
            attempt_id=att1,
            attempt_number=1,
            provider="local",
            model="llama",
            latency_s=0.1,
            input_tokens=None,
            output_tokens=None,
            cost=None,
            cost_status="missing_usage",
        )
    )
    dispatcher.emit(RequestCompleted(request_id=req1, latency_s=0.12))

    # Request with free model (cost=0)
    req2 = new_id()
    att2 = new_id()
    dispatcher.emit(RequestStarted(request_id=req2))
    dispatcher.emit(
        AttemptStarted(
            request_id=req2, attempt_id=att2, attempt_number=1, provider="local", model="llama"
        )
    )
    dispatcher.emit(
        AttemptSucceeded(
            request_id=req2,
            attempt_id=att2,
            attempt_number=1,
            provider="local",
            model="llama",
            latency_s=0.1,
            input_tokens=50,
            output_tokens=50,
            cost=Decimal("0.0"),
            cost_status="free",
        )
    )
    dispatcher.emit(RequestCompleted(request_id=req2, latency_s=0.11))

    metrics_map = _extract_metrics_by_name(reader)
    # 2 requests recorded
    assert metrics_map[INSTRUMENT_REQUESTS].data.data_points[0].value == 2
    # Cost counter has value 0.0 from free model
    assert metrics_map[INSTRUMENT_COST].data.data_points[0].value == 0.0


def test_model_label_capping_and_cardinality_bound(isolated_subscriber_setup):
    """Model labels are capped by length and distinct counts overflow to 'other'."""
    reader, _, subscriber = isolated_subscriber_setup
    # Configure subscriber with small distinct model cap of 3
    subscriber._max_distinct_models = 3
    subscriber._max_model_len = 20

    # 1. Normal model
    subscriber._cap_model("gpt-4o") == "gpt-4o"
    # 2. Second model
    subscriber._cap_model("claude-3") == "claude-3"
    # 3. Third model
    subscriber._cap_model("gemini-1.5") == "gemini-1.5"
    # 4. Fourth model exceeds max_distinct_models -> must be 'other'
    assert subscriber._cap_model("mistral-large") == "other"

    # 5. Model exceeding max_model_len (20 chars) -> must be 'other'
    long_model = "a" * 25
    assert subscriber._cap_model(long_model) == "other"

    # 6. Model containing newlines -> must be 'other'
    assert subscriber._cap_model("model\nwith\nnewline") == "other"

    # 7. Blank or non-string -> 'unknown'
    assert subscriber._cap_model("   ") == "unknown"
    assert subscriber._cap_model(None) == "unknown"


def test_request_state_is_bounded_and_evicts_oldest(isolated_subscriber_setup):
    """In-flight request state does not grow unbounded if terminal events are dropped."""
    _, _, subscriber = isolated_subscriber_setup
    subscriber._max_active_requests = 2

    req1, req2, req3 = new_id(), new_id(), new_id()
    subscriber(AttemptStarted(request_id=req1, attempt_id=new_id(), attempt_number=1, provider="p1", model="m1"))
    subscriber(AttemptStarted(request_id=req2, attempt_id=new_id(), attempt_number=1, provider="p2", model="m2"))
    assert subscriber.active_requests_count == 2

    # Third request causes oldest (req1) to be evicted
    subscriber(AttemptStarted(request_id=req3, attempt_id=new_id(), attempt_number=1, provider="p3", model="m3"))
    assert subscriber.active_requests_count == 2
    assert req1 not in subscriber._requests
    assert req2 in subscriber._requests
    assert req3 in subscriber._requests


def test_non_throwing_guarantee_on_faulty_instrument(isolated_subscriber_setup, caplog):
    """Telemetry instrument failures do not raise or disrupt subscriber execution."""
    _, recorder, subscriber = isolated_subscriber_setup

    def broken_record(*args, **kwargs):
        raise RuntimeError("Hardware failure in metric instrument")

    recorder.duration_histogram.record = broken_record
    recorder.requests_counter.add = broken_record

    req_id = new_id()
    with caplog.at_level(logging.DEBUG):
        # Must execute without raising any exception
        subscriber(
            AttemptStarted(
                request_id=req_id, attempt_id=new_id(), attempt_number=1, provider="openai", model="m"
            )
        )
        subscriber(
            AttemptSucceeded(
                request_id=req_id,
                attempt_id=new_id(),
                attempt_number=1,
                provider="openai",
                model="m",
                latency_s=0.2,
            )
        )
        subscriber(RequestCompleted(request_id=req_id, latency_s=0.25))


def test_privacy_and_allowed_attributes_invariant(isolated_subscriber_setup):
    """Sensitive inputs, credentials, prompts, and request IDs never appear as metric labels."""
    reader, _, subscriber = isolated_subscriber_setup
    dispatcher = EventDispatcher(subscribers=[subscriber])

    secret_key = "sk-super-secret-key-12345"
    prompt_snippet = "Confidential prompt with user credentials"
    req_id = f"req-{secret_key}"
    att_id = f"att-{secret_key}"
    tenant_id = f"tenant-{secret_key}"

    events = [
        RequestStarted(request_id=req_id, tenant_id=tenant_id),
        AttemptStarted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            tenant_id=tenant_id,
        ),
        AttemptSucceeded(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            latency_s=0.4,
            input_tokens=10,
            output_tokens=10,
            cost=Decimal("0.001"),
            cost_status="priced",
            tenant_id=tenant_id,
        ),
        RequestCompleted(request_id=req_id, latency_s=0.45, total_cost=Decimal("0.001"), tenant_id=tenant_id),
    ]

    for event in events:
        dispatcher.emit(event)

    metrics_map = _extract_metrics_by_name(reader)
    for metric_name, metric in metrics_map.items():
        for point in metric.data.data_points:
            keys = set(point.attributes.keys())
            # All emitted keys must strictly belong to ALLOWED_ATTRIBUTE_KEYS
            assert keys.issubset(ALLOWED_ATTRIBUTE_KEYS), f"Unallowed keys in {metric_name}: {keys}"

            # Check that none of the secret strings leak into attribute values
            for attr_val in point.attributes.values():
                assert secret_key not in str(attr_val)
                assert prompt_snippet not in str(attr_val)
                assert req_id not in str(attr_val)
                assert att_id not in str(attr_val)
                assert tenant_id not in str(attr_val)


def test_irrelevant_lifecycle_events_are_safely_ignored(isolated_subscriber_setup):
    """Events not mapping to metrics (e.g. CircuitStateChanged) are ignored cleanly."""
    reader, _, subscriber = isolated_subscriber_setup

    subscriber(CircuitStateChanged(provider="p", from_state=CircuitState.CLOSED, to_state=CircuitState.OPEN))
    subscriber(RoutingDecided(request_id="r1", ranked=("p1", "p2")))

    metrics_map = _extract_metrics_by_name(reader)
    assert len(metrics_map) == 0


def test_subscriber_with_default_recorder_operates_without_sdk():
    """MetricsSubscriber works cleanly out-of-the-box when no SDK is configured."""
    subscriber = MetricsSubscriber()
    req_id = new_id()

    # Operations must safely execute as no-ops
    subscriber(RequestStarted(request_id=req_id))
    subscriber(AttemptStarted(request_id=req_id, attempt_id=new_id(), attempt_number=1, provider="p", model="m"))
    subscriber(
        AttemptSucceeded(
            request_id=req_id,
            attempt_id=new_id(),
            attempt_number=1,
            provider="p",
            model="m",
            latency_s=0.1,
            input_tokens=10,
            output_tokens=5,
            cost=Decimal("0.001"),
            cost_status="priced",
        )
    )
    subscriber(RequestCompleted(request_id=req_id, latency_s=0.15))
    assert subscriber.active_requests_count == 0
