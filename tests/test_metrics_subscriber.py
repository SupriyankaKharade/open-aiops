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
    """Model labels are capped by length, control chars normalize to 'other', and distinct counts never exceed cap."""
    reader, _, subscriber = isolated_subscriber_setup
    # Configure subscriber with distinct model cap of 5:
    # Reserved capacity: 'unknown' (1) and 'other' (1)
    # Available ordinary model capacity: 5 - 2 = 3
    subscriber._max_distinct_models = 5
    subscriber._max_model_len = 20

    emitted_labels = set()

    # 1. First 3 ordinary models are admitted
    m1 = subscriber._cap_model("gpt-4o")
    assert m1 == "gpt-4o"
    emitted_labels.add(m1)

    m2 = subscriber._cap_model("claude-3")
    assert m2 == "claude-3"
    emitted_labels.add(m2)

    m3 = subscriber._cap_model("gemini-1.5")
    assert m3 == "gemini-1.5"
    emitted_labels.add(m3)

    # Repeating an already known model remains valid
    assert subscriber._cap_model("gpt-4o") == "gpt-4o"

    # 2. 4th and 5th ordinary models exceed capacity (3) and overflow to 'other'
    m4 = subscriber._cap_model("mistral-large")
    assert m4 == "other"
    emitted_labels.add(m4)

    m5 = subscriber._cap_model("llama-3")
    assert m5 == "other"
    emitted_labels.add(m5)

    # 3. Model exceeding max_model_len (20 chars) -> 'other'
    long_model = "a" * 25
    m_long = subscriber._cap_model(long_model)
    assert m_long == "other"
    emitted_labels.add(m_long)

    # 4. Control characters detected before stripping -> 'other'
    assert subscriber._cap_model("model\nwith\nnewline") == "other"
    assert subscriber._cap_model("model\rwith\rcr") == "other"
    assert subscriber._cap_model("model\twith\ttab") == "other"
    assert subscriber._cap_model("\tgpt-4o") == "other"
    assert subscriber._cap_model("\t") == "other"
    assert subscriber._cap_model("model\x00with\x00nul") == "other"
    assert subscriber._cap_model("\x00") == "other"
    assert subscriber._cap_model("model\x1fcontrol") == "other"
    assert subscriber._cap_model("model\x7fdel") == "other"

    # 5. Empty, whitespace-only, or non-string inputs -> 'unknown'
    m_blank = subscriber._cap_model("   ")
    assert m_blank == "unknown"
    emitted_labels.add(m_blank)

    assert subscriber._cap_model("") == "unknown"
    assert subscriber._cap_model(None) == "unknown"
    assert subscriber._cap_model(12345) == "unknown"
    assert subscriber._cap_model(["model"]) == "unknown"

    # 6. Reserved inputs pass through without taking ordinary slots
    assert subscriber._cap_model("other") == "other"
    assert subscriber._cap_model("unknown") == "unknown"

    # Total distinct emitted labels must NOT exceed max_distinct_models (5)
    assert emitted_labels == {"gpt-4o", "claude-3", "gemini-1.5", "unknown", "other"}
    assert len(emitted_labels) == 5
    assert len(emitted_labels) <= subscriber._max_distinct_models


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


def test_model_cardinality_deterministic_small_caps():
    """Deterministic handling when max_distinct_models is <= 2."""
    # 1. Cap == 1: only 'other' is ever emitted across all inputs
    sub_cap1 = MetricsSubscriber(max_distinct_models=1)
    inputs_to_test = [
        "gpt-4o",
        "claude-3",
        "",
        "   ",
        None,
        "\tgpt-4",
        "other",
        "unknown",
        "a" * 100,
    ]
    emitted1 = {sub_cap1._cap_model(val) for val in inputs_to_test}
    assert emitted1 == {"other"}
    assert len(emitted1) <= 1

    # 2. Cap == 2: capacity reserved for 'unknown' and 'other'; ordinary models overflow to 'other'
    sub_cap2 = MetricsSubscriber(max_distinct_models=2)
    emitted2 = set()
    for val in inputs_to_test:
        emitted2.add(sub_cap2._cap_model(val))
    assert emitted2.issubset({"unknown", "other"})
    assert len(emitted2) <= 2


def test_total_distinct_model_labels_strictly_bounded_under_load():
    """Total distinct emitted model attribute values strictly remain within configured cap under load."""
    sub = MetricsSubscriber(max_distinct_models=10)
    all_emitted = set()

    # Feed 100 distinct valid model names
    for i in range(100):
        all_emitted.add(sub._cap_model(f"model-service-variant-{i}"))

    # Feed control-character strings (tabs, NUL, newlines, DEL)
    all_emitted.add(sub._cap_model("tab\tmodel"))
    all_emitted.add(sub._cap_model("\0nul_model"))
    all_emitted.add(sub._cap_model("newline\nmodel"))
    all_emitted.add(sub._cap_model("del\x7fmodel"))

    # Feed overlong strings
    all_emitted.add(sub._cap_model("x" * 100))

    # Feed blanks, non-strings, and whitespace
    all_emitted.add(sub._cap_model("    "))
    all_emitted.add(sub._cap_model(""))
    all_emitted.add(sub._cap_model(None))
    all_emitted.add(sub._cap_model(999))

    # Emitted distinct values must strictly be <= 10
    assert len(all_emitted) <= 10
    # Must include 'unknown' and 'other'
    assert "unknown" in all_emitted
    assert "other" in all_emitted


def test_repeated_request_completed_events_are_idempotent(isolated_subscriber_setup):
    """Repeated RequestCompleted events for the same request ID are deduplicated and do not double count."""
    reader, _, subscriber = isolated_subscriber_setup
    dispatcher = EventDispatcher(subscribers=[subscriber])

    req_id = new_id()
    att_id = new_id()

    events = [
        RequestStarted(request_id=req_id),
        AttemptStarted(request_id=req_id, attempt_id=att_id, attempt_number=1, provider="openai", model="gpt-4o"),
        AttemptSucceeded(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            latency_s=0.5,
        ),
        RequestCompleted(request_id=req_id, latency_s=0.55),
    ]
    for event in events:
        dispatcher.emit(event)

    assert subscriber.active_requests_count == 0
    assert subscriber.terminal_requests_count == 1

    # Send duplicate RequestCompleted events
    dispatcher.emit(RequestCompleted(request_id=req_id, latency_s=0.55))
    dispatcher.emit(RequestCompleted(request_id=req_id, latency_s=0.55))

    metrics_map = _extract_metrics_by_name(reader)

    # Request count must remain exactly 1 (not 3!)
    req_pts = list(metrics_map[INSTRUMENT_REQUESTS].data.data_points)
    assert len(req_pts) == 1
    assert req_pts[0].value == 1

    # Request duration observation count must remain exactly 1 (not 3!)
    dur_pts = list(metrics_map[INSTRUMENT_REQUEST_DURATION].data.data_points)
    assert len(dur_pts) == 1
    assert dur_pts[0].count == 1
    assert dur_pts[0].sum == pytest.approx(0.55)


def test_repeated_request_failed_events_are_idempotent(isolated_subscriber_setup):
    """Repeated RequestFailed events for the same request ID do not record duplicate failure counts."""
    reader, _, subscriber = isolated_subscriber_setup
    dispatcher = EventDispatcher(subscribers=[subscriber])

    req_id = new_id()
    att_id = new_id()

    events = [
        RequestStarted(request_id=req_id),
        AttemptStarted(request_id=req_id, attempt_id=att_id, attempt_number=1, provider="openai", model="gpt-4"),
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
        RequestFailed(request_id=req_id, error_type="ProviderTimeout"),
    ]
    for event in events:
        dispatcher.emit(event)

    assert subscriber.active_requests_count == 0
    assert subscriber.terminal_requests_count == 1

    # Emit duplicate RequestFailed event
    dispatcher.emit(RequestFailed(request_id=req_id, error_type="ProviderTimeout"))

    metrics_map = _extract_metrics_by_name(reader)
    req_pts = list(metrics_map[INSTRUMENT_REQUESTS].data.data_points)
    assert len(req_pts) == 1
    assert req_pts[0].value == 1
    assert req_pts[0].attributes["outcome"] == "error"


def test_mixed_terminal_events_same_request_id_are_idempotent(isolated_subscriber_setup):
    """Mixed terminal event types (Completed then Failed, or Failed then Completed) for same ID are ignored."""
    reader, _, subscriber = isolated_subscriber_setup
    dispatcher = EventDispatcher(subscribers=[subscriber])

    # Case 1: Completed followed by Failed
    req1 = new_id()
    dispatcher.emit(RequestStarted(request_id=req1))
    dispatcher.emit(
        AttemptStarted(request_id=req1, attempt_id=new_id(), attempt_number=1, provider="p1", model="m1")
    )
    dispatcher.emit(
        AttemptSucceeded(
            request_id=req1, attempt_id=new_id(), attempt_number=1, provider="p1", model="m1", latency_s=0.2
        )
    )
    dispatcher.emit(RequestCompleted(request_id=req1, latency_s=0.25))

    # Duplicate terminal event with different type (RequestFailed)
    dispatcher.emit(RequestFailed(request_id=req1, error_type="ProviderTimeout"))

    # Case 2: Failed followed by Completed
    req2 = new_id()
    dispatcher.emit(RequestStarted(request_id=req2))
    dispatcher.emit(
        AttemptStarted(request_id=req2, attempt_id=new_id(), attempt_number=1, provider="p2", model="m2")
    )
    dispatcher.emit(
        AttemptFailed(
            request_id=req2,
            attempt_id=new_id(),
            attempt_number=1,
            provider="p2",
            model="m2",
            error_type="RateLimitError",
            retryable=False,
            failover_eligible=False,
        )
    )
    dispatcher.emit(RequestFailed(request_id=req2, error_type="RateLimitError"))

    # Duplicate terminal event with different type (RequestCompleted)
    dispatcher.emit(RequestCompleted(request_id=req2, latency_s=0.10))

    metrics_map = _extract_metrics_by_name(reader)
    req_pts = list(metrics_map[INSTRUMENT_REQUESTS].data.data_points)
    # Total requests should be 2 (1 success from req1, 1 error from req2)
    assert len(req_pts) == 2
    outcomes = {p.attributes["outcome"]: p.value for p in req_pts}
    assert outcomes == {"success": 1, "error": 1}

    # Duration histogram must contain only 1 observation (from req1)
    dur_pts = list(metrics_map[INSTRUMENT_REQUEST_DURATION].data.data_points)
    assert len(dur_pts) == 1
    assert dur_pts[0].count == 1


def test_terminal_request_deduplication_is_bounded(isolated_subscriber_setup):
    """Terminal deduplication cache is strictly bounded by max_active_requests."""
    reader, _, subscriber = isolated_subscriber_setup
    subscriber._max_active_requests = 2

    req1, req2, req3 = new_id(), new_id(), new_id()

    # Emit completion for req1 and req2 -> both in deduplication cache
    subscriber(RequestCompleted(request_id=req1, latency_s=0.1))
    subscriber(RequestCompleted(request_id=req2, latency_s=0.2))
    assert subscriber.terminal_requests_count == 2
    assert req1 in subscriber._terminal_requests
    assert req2 in subscriber._terminal_requests

    # Duplicate req1 is dropped while still in cache
    subscriber(RequestCompleted(request_id=req1, latency_s=0.1))
    metrics_map = _extract_metrics_by_name(reader)
    assert metrics_map[INSTRUMENT_REQUESTS].data.data_points[0].value == 2

    # Third distinct completion evicts oldest (req1) from bounded cache
    subscriber(RequestCompleted(request_id=req3, latency_s=0.3))
    assert subscriber.terminal_requests_count == 2
    assert req1 not in subscriber._terminal_requests
    assert req2 in subscriber._terminal_requests
    assert req3 in subscriber._terminal_requests

    # Because req1 was evicted from the bounded window, another event for req1 is now processed
    subscriber(RequestCompleted(request_id=req1, latency_s=0.1))
    metrics_map = _extract_metrics_by_name(reader)
    assert metrics_map[INSTRUMENT_REQUESTS].data.data_points[0].value == 4


def test_terminal_event_without_active_attempt_state_is_safe(isolated_subscriber_setup):
    """A valid terminal event with no active attempt metadata records safely with 'unknown' labels."""
    reader, _, subscriber = isolated_subscriber_setup

    req_id = new_id()
    # Emit RequestCompleted without prior AttemptStarted
    subscriber(RequestCompleted(request_id=req_id, latency_s=0.33))

    assert subscriber.active_requests_count == 0
    assert subscriber.terminal_requests_count == 1

    metrics_map = _extract_metrics_by_name(reader)
    req_pts = list(metrics_map[INSTRUMENT_REQUESTS].data.data_points)
    assert len(req_pts) == 1
    assert req_pts[0].attributes == {"provider": "unknown", "model": "unknown", "outcome": "success"}
    assert req_pts[0].value == 1

    dur_pts = list(metrics_map[INSTRUMENT_REQUEST_DURATION].data.data_points)
    assert len(dur_pts) == 1
    assert dur_pts[0].attributes == {"provider": "unknown", "model": "unknown", "outcome": "success"}
    assert dur_pts[0].sum == pytest.approx(0.33)

    # Duplicate is still deduplicated
    subscriber(RequestCompleted(request_id=req_id, latency_s=0.33))
    assert list(metrics_map[INSTRUMENT_REQUESTS].data.data_points)[0].value == 1
