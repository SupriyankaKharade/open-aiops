"""Unit tests for TracingSubscriber and span lifecycle transformations (INT-18)."""

from __future__ import annotations

import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

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
from open_aiops.observability.semconv import (
    ATTR_CIRCUIT_FROM_STATE,
    ATTR_CIRCUIT_PROVIDER,
    ATTR_CIRCUIT_TO_STATE,
    ATTR_ERROR_TYPE,
    ATTR_FALLBACK_ERROR_TYPE,
    ATTR_FALLBACK_FROM,
    ATTR_FALLBACK_TO,
    ATTR_GEN_AI_OPERATION_NAME,
    ATTR_GEN_AI_PROVIDER_NAME,
    ATTR_GEN_AI_REQUEST_MODEL,
    ATTR_GEN_AI_USAGE_INPUT_TOKENS,
    ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
    ATTR_OPENAIOPS_ATTEMPT_ID,
    ATTR_OPENAIOPS_ATTEMPT_NUMBER,
    ATTR_OPENAIOPS_COST_STATUS,
    ATTR_OPENAIOPS_COST_TOTAL,
    ATTR_OPENAIOPS_LATENCY_S,
    ATTR_OPENAIOPS_REQUEST_ID,
    ATTR_OPENAIOPS_TENANT_ID,
    ATTR_OPENAIOPS_USAGE_TOTAL_TOKENS,
    ATTR_RETRY_DELAY_S,
    ATTR_RETRY_ERROR_TYPE,
    ATTR_ROUTING_RANKED,
    ATTR_ROUTING_REJECTED,
    ATTR_STREAM_TS,
    EVENT_CIRCUIT_STATE_CHANGED,
    EVENT_FALLBACK_TRIGGERED,
    EVENT_RETRY_SCHEDULED,
    EVENT_ROUTING_DECIDED,
    EVENT_STREAM_FIRST_CHUNK,
)
from open_aiops.observability.subscribers import SpanRegistry, TracingSubscriber


@pytest.fixture
def memory_exporter():
    return InMemorySpanExporter()


@pytest.fixture
def tracer_and_subscriber(memory_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    tracer = provider.get_tracer("test.tracer")
    subscriber = TracingSubscriber(tracer=tracer)
    return tracer, subscriber, memory_exporter


def test_successful_request_and_attempt_lifecycle(tracer_and_subscriber):
    _, subscriber, exporter = tracer_and_subscriber
    req_id = new_id()
    att_id = new_id()

    events = [
        RequestStarted(request_id=req_id, tenant_id="tenant-123"),
        AttemptStarted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            tenant_id="tenant-123",
        ),
        AttemptSucceeded(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            latency_s=0.85,
            input_tokens=120,
            output_tokens=45,
            cost=0.0025,
        ),
        RequestCompleted(request_id=req_id, latency_s=0.90, cost=0.0025),
    ]

    assert_well_formed(events)
    for ev in events:
        subscriber(ev)

    # All spans must be closed and removed from registry
    assert subscriber.registry.is_empty()

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    # Attempt span finishes first, then request span
    attempt_span = spans[0]
    request_span = spans[1]

    # Check attempt span attributes & SpanKind
    assert attempt_span.name == "open_aiops.attempt"
    assert attempt_span.kind == SpanKind.CLIENT
    assert attempt_span.attributes[ATTR_GEN_AI_OPERATION_NAME] == "chat"
    assert attempt_span.attributes[ATTR_GEN_AI_PROVIDER_NAME] == "openai"
    assert attempt_span.attributes[ATTR_GEN_AI_REQUEST_MODEL] == "gpt-4o"
    assert attempt_span.attributes[ATTR_OPENAIOPS_ATTEMPT_NUMBER] == 1
    assert attempt_span.attributes[ATTR_GEN_AI_USAGE_INPUT_TOKENS] == 120
    assert attempt_span.attributes[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] == 45
    assert attempt_span.attributes[ATTR_OPENAIOPS_USAGE_TOTAL_TOKENS] == 165
    assert attempt_span.attributes[ATTR_OPENAIOPS_LATENCY_S] == 0.85
    assert attempt_span.attributes[ATTR_OPENAIOPS_COST_TOTAL] == 0.0025
    assert attempt_span.attributes[ATTR_OPENAIOPS_REQUEST_ID] == req_id
    assert attempt_span.attributes[ATTR_OPENAIOPS_ATTEMPT_ID] == att_id
    assert attempt_span.attributes[ATTR_OPENAIOPS_TENANT_ID] == "tenant-123"
    assert attempt_span.status.status_code == StatusCode.OK

    # Check request span attributes & parent-child hierarchy
    assert request_span.name == "open_aiops.request"
    assert request_span.attributes[ATTR_OPENAIOPS_REQUEST_ID] == req_id
    assert request_span.attributes[ATTR_OPENAIOPS_TENANT_ID] == "tenant-123"
    assert request_span.attributes[ATTR_OPENAIOPS_LATENCY_S] == 0.90
    assert request_span.attributes[ATTR_OPENAIOPS_COST_TOTAL] == 0.0025
    assert request_span.status.status_code == StatusCode.OK

    # Parent relationship: attempt span's parent context points to request span
    assert attempt_span.parent.span_id == request_span.context.span_id


def test_failover_multi_attempt_flow(tracer_and_subscriber):
    _, subscriber, exporter = tracer_and_subscriber
    req_id = new_id()
    att1_id = new_id()
    att2_id = new_id()

    events = [
        RequestStarted(request_id=req_id),
        RoutingDecided(
            request_id=req_id,
            ranked=("openai", "anthropic"),
            rejected={"mock_provider": "overloaded"},
        ),
        AttemptStarted(
            request_id=req_id,
            attempt_id=att1_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        ),
        AttemptFailed(
            request_id=req_id,
            attempt_id=att1_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            error_type="RateLimitError",
        ),
        FallbackTriggered(
            request_id=req_id,
            from_provider="openai",
            to_provider="anthropic",
            error_type="RateLimitError",
            attempt_number=1,
        ),
        AttemptStarted(
            request_id=req_id,
            attempt_id=att2_id,
            attempt_number=2,
            provider="anthropic",
            model="claude-3-5-sonnet",
        ),
        AttemptSucceeded(
            request_id=req_id,
            attempt_id=att2_id,
            attempt_number=2,
            provider="anthropic",
            model="claude-3-5-sonnet",
            latency_s=0.62,
            input_tokens=100,
            output_tokens=30,
            cost=0.0018,
        ),
        RequestCompleted(request_id=req_id, latency_s=1.20, cost=0.0018),
    ]

    assert_well_formed(events)
    for ev in events:
        subscriber(ev)

    assert subscriber.registry.is_empty()
    spans = exporter.get_finished_spans()
    assert len(spans) == 3

    attempt1_span = spans[0]
    attempt2_span = spans[1]
    request_span = spans[2]

    # Attempt 1 failed with error status
    assert attempt1_span.attributes[ATTR_GEN_AI_PROVIDER_NAME] == "openai"
    assert attempt1_span.attributes[ATTR_ERROR_TYPE] == "RateLimitError"
    assert attempt1_span.status.status_code == StatusCode.ERROR
    assert attempt1_span.parent.span_id == request_span.context.span_id

    # Attempt 2 succeeded with OK status
    assert attempt2_span.attributes[ATTR_GEN_AI_PROVIDER_NAME] == "anthropic"
    assert attempt2_span.status.status_code == StatusCode.OK
    assert attempt2_span.parent.span_id == request_span.context.span_id

    # Request span contains routing and fallback events under openaiops.*
    event_names = [e.name for e in request_span.events]
    assert EVENT_ROUTING_DECIDED in event_names
    assert EVENT_FALLBACK_TRIGGERED in event_names

    fallback_event = next(e for e in request_span.events if e.name == EVENT_FALLBACK_TRIGGERED)
    assert fallback_event.attributes[ATTR_FALLBACK_FROM] == "openai"
    assert fallback_event.attributes[ATTR_FALLBACK_TO] == "anthropic"
    assert fallback_event.attributes[ATTR_FALLBACK_ERROR_TYPE] == "RateLimitError"
    assert fallback_event.attributes[ATTR_OPENAIOPS_ATTEMPT_NUMBER] == 1


def test_retry_event_recording_lands_on_request_span(tracer_and_subscriber):
    """When AttemptFailed terminates attempt 1, subsequent RetryScheduled lands on request span."""
    _, subscriber, exporter = tracer_and_subscriber
    req_id = new_id()
    att1_id = new_id()
    att2_id = new_id()

    events = [
        RequestStarted(request_id=req_id),
        AttemptStarted(
            request_id=req_id,
            attempt_id=att1_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        ),
        AttemptFailed(
            request_id=req_id,
            attempt_id=att1_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            error_type="ProviderTimeout",
        ),
        RetryScheduled(
            request_id=req_id,
            attempt_id=att1_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            delay_s=1.5,
            error_type="ProviderTimeout",
        ),
        AttemptStarted(
            request_id=req_id,
            attempt_id=att2_id,
            attempt_number=2,
            provider="openai",
            model="gpt-4o",
        ),
        AttemptSucceeded(
            request_id=req_id,
            attempt_id=att2_id,
            attempt_number=2,
            provider="openai",
            model="gpt-4o",
            latency_s=0.45,
            input_tokens=50,
            output_tokens=20,
        ),
        RequestCompleted(request_id=req_id, latency_s=2.1),
    ]

    assert_well_formed(events)
    for ev in events:
        subscriber(ev)

    assert subscriber.registry.is_empty()
    spans = exporter.get_finished_spans()
    assert len(spans) == 3

    attempt1_span = spans[0]
    attempt2_span = spans[1]
    request_span = spans[2]

    # Attempt 1 span was already ended before RetryScheduled arrived
    assert len(attempt1_span.events) == 0
    assert len(attempt2_span.events) == 0

    # RetryScheduled landed on request span
    retry_events = [e for e in request_span.events if e.name == EVENT_RETRY_SCHEDULED]
    assert len(retry_events) == 1
    assert retry_events[0].attributes[ATTR_RETRY_DELAY_S] == 1.5
    assert retry_events[0].attributes[ATTR_RETRY_ERROR_TYPE] == "ProviderTimeout"


def test_streaming_lifecycle(tracer_and_subscriber):
    _, subscriber, exporter = tracer_and_subscriber
    req_id = new_id()
    att_id = new_id()

    events = [
        RequestStarted(request_id=req_id),
        AttemptStarted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        ),
        StreamFirstChunk(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        ),
        StreamCompleted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            latency_s=1.2,
            input_tokens=80,
            output_tokens=150,
            cost=0.0035,
        ),
        RequestCompleted(request_id=req_id, latency_s=1.25, cost=0.0035),
    ]

    assert_well_formed(events)
    for ev in events:
        subscriber(ev)

    assert subscriber.registry.is_empty()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    attempt_span = spans[0]
    event_names = [e.name for e in attempt_span.events]
    assert EVENT_STREAM_FIRST_CHUNK in event_names

    first_chunk_event = next(e for e in attempt_span.events if e.name == EVENT_STREAM_FIRST_CHUNK)
    assert ATTR_STREAM_TS in first_chunk_event.attributes

    assert attempt_span.status.status_code == StatusCode.OK
    assert attempt_span.attributes[ATTR_OPENAIOPS_USAGE_TOTAL_TOKENS] == 230


def test_stream_interrupted_and_request_failed(tracer_and_subscriber):
    _, subscriber, exporter = tracer_and_subscriber
    req_id = new_id()
    att_id = new_id()

    events = [
        RequestStarted(request_id=req_id),
        AttemptStarted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        ),
        StreamFirstChunk(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        ),
        StreamInterrupted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            error_type="ProviderTimeout",
        ),
        RequestFailed(request_id=req_id, error_type="ProviderTimeout"),
    ]

    assert_well_formed(events)
    for ev in events:
        subscriber(ev)

    assert subscriber.registry.is_empty()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    attempt_span = spans[0]
    request_span = spans[1]

    assert attempt_span.status.status_code == StatusCode.ERROR
    assert attempt_span.attributes[ATTR_ERROR_TYPE] == "ProviderTimeout"
    assert request_span.status.status_code == StatusCode.ERROR
    assert request_span.attributes[ATTR_ERROR_TYPE] == "ProviderTimeout"


def test_attempt_cancelled(tracer_and_subscriber):
    _, subscriber, exporter = tracer_and_subscriber
    req_id = new_id()
    att_id = new_id()

    events = [
        RequestStarted(request_id=req_id),
        AttemptStarted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        ),
        AttemptCancelled(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        ),
        RequestFailed(request_id=req_id, error_type="cancelled"),
    ]

    assert_well_formed(events)
    for ev in events:
        subscriber(ev)

    assert subscriber.registry.is_empty()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes[ATTR_ERROR_TYPE] == "cancelled"


def test_lost_attempt_span_ended_on_terminal_request(tracer_and_subscriber):
    """An open attempt at request end exports both attempt and request spans with attempt in error status."""
    _, subscriber, exporter = tracer_and_subscriber
    req_id = new_id()
    att_id = new_id()

    # Start request and attempt, but complete request without explicit attempt terminal
    subscriber(RequestStarted(request_id=req_id))
    subscriber(
        AttemptStarted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        )
    )

    assert subscriber.registry.active_request_count() == 1
    assert subscriber.registry.active_attempt_count() == 1

    # Terminal request event cleans up everything and ends open attempt span
    subscriber(RequestCompleted(request_id=req_id, latency_s=1.0))
    assert subscriber.registry.is_empty()

    spans = exporter.get_finished_spans()
    # Both attempt span and request span must be exported!
    assert len(spans) == 2

    attempt_span = spans[0]
    request_span = spans[1]

    # Attempt span ended with error status and error.type = "abandoned"
    assert attempt_span.status.status_code == StatusCode.ERROR
    assert attempt_span.attributes[ATTR_ERROR_TYPE] == "abandoned"
    assert attempt_span.attributes[ATTR_OPENAIOPS_REQUEST_ID] == req_id
    assert attempt_span.attributes[ATTR_OPENAIOPS_ATTEMPT_ID] == att_id

    # Request span succeeded
    assert request_span.status.status_code == StatusCode.OK


def test_lost_attempt_span_ended_on_request_failed(tracer_and_subscriber):
    """An open attempt at request failure also exports both spans with attempt in error status."""
    _, subscriber, exporter = tracer_and_subscriber
    req_id = new_id()
    att_id = new_id()

    subscriber(RequestStarted(request_id=req_id))
    subscriber(
        AttemptStarted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        )
    )
    subscriber(RequestFailed(request_id=req_id, error_type="GatewayTimeout"))

    assert subscriber.registry.is_empty()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    attempt_span = spans[0]
    assert attempt_span.status.status_code == StatusCode.ERROR
    assert attempt_span.attributes[ATTR_ERROR_TYPE] == "abandoned"


def test_registry_empty_after_cancelled_and_interrupted_streams(tracer_and_subscriber):
    """Verify registry is empty after cancelled and interrupted stream flows."""
    _, subscriber, _ = tracer_and_subscriber

    # Case 1: Cancelled stream
    req_1 = new_id()
    att_1 = new_id()
    subscriber(RequestStarted(request_id=req_1))
    subscriber(AttemptStarted(request_id=req_1, attempt_id=att_1, attempt_number=1, provider="openai", model="m"))
    subscriber(StreamFirstChunk(request_id=req_1, attempt_id=att_1, attempt_number=1, provider="openai", model="m"))
    subscriber(AttemptCancelled(request_id=req_1, attempt_id=att_1, attempt_number=1, provider="openai", model="m"))
    subscriber(RequestFailed(request_id=req_1, error_type="cancelled"))
    assert subscriber.registry.is_empty()

    # Case 2: Interrupted stream
    req_2 = new_id()
    att_2 = new_id()
    subscriber(RequestStarted(request_id=req_2))
    subscriber(AttemptStarted(request_id=req_2, attempt_id=att_2, attempt_number=1, provider="openai", model="m"))
    subscriber(StreamFirstChunk(request_id=req_2, attempt_id=att_2, attempt_number=1, provider="openai", model="m"))
    subscriber(StreamInterrupted(request_id=req_2, attempt_id=att_2, attempt_number=1, provider="openai", model="m", error_type="Disconnect"))
    subscriber(RequestFailed(request_id=req_2, error_type="Disconnect"))
    assert subscriber.registry.is_empty()


def test_privacy_no_api_key_or_prompt_text_on_spans(tracer_and_subscriber):
    """Verify fake API key and prompt text never appear on spans or span events."""
    _, subscriber, exporter = tracer_and_subscriber
    req_id = new_id()
    att_id = new_id()
    fake_key = "sk-live-secret-test-key-999"
    prompt_text = "Confidential user prompt text containing sensitive PII"

    subscriber(RequestStarted(request_id=req_id, tenant_id="tenant-xyz"))
    subscriber(
        AttemptStarted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        )
    )
    subscriber(
        AttemptSucceeded(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            latency_s=0.5,
            input_tokens=10,
            output_tokens=10,
        )
    )
    subscriber(RequestCompleted(request_id=req_id, latency_s=0.55))

    spans = exporter.get_finished_spans()
    for s in spans:
        for k, v in s.attributes.items():
            assert fake_key not in str(k) and fake_key not in str(v)
            assert prompt_text not in str(k) and prompt_text not in str(v)
        for ev in s.events:
            for k, v in ev.attributes.items():
                assert fake_key not in str(k) and fake_key not in str(v)
                assert prompt_text not in str(k) and prompt_text not in str(v)


def test_failing_exporter_leaves_request_unaffected(caplog):
    """A failing exporter must never raise or break request execution."""
    class BrokenExporter(InMemorySpanExporter):
        def export(self, spans):
            raise RuntimeError("Backend connection refused")

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(BrokenExporter()))
    tracer = provider.get_tracer("broken.tracer")
    subscriber = TracingSubscriber(tracer=tracer)

    req_id = new_id()
    att_id = new_id()

    # Must complete without throwing any exception
    subscriber(RequestStarted(request_id=req_id))
    subscriber(AttemptStarted(request_id=req_id, attempt_id=att_id, attempt_number=1, provider="openai", model="m"))
    subscriber(AttemptSucceeded(request_id=req_id, attempt_id=att_id, attempt_number=1, provider="openai", model="m", latency_s=0.1))
    subscriber(RequestCompleted(request_id=req_id, latency_s=0.15))


def test_registry_capacity_cap_and_eviction(tracer_and_subscriber):
    """Oldest requests are evicted and ended when registry reaches max_active_requests."""
    _, _, exporter = tracer_and_subscriber
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("cap.tracer")

    # Set cap to 2 active requests
    subscriber = TracingSubscriber(tracer=tracer, max_active_requests=2)

    req1, req2, req3 = new_id(), new_id(), new_id()
    att1 = new_id()

    # Request 1 with an attempt
    subscriber(RequestStarted(request_id=req1))
    subscriber(AttemptStarted(request_id=req1, attempt_id=att1, attempt_number=1, provider="p1", model="m1"))

    # Request 2
    subscriber(RequestStarted(request_id=req2))

    assert subscriber.registry.active_request_count() == 2

    # Request 3 triggers eviction of oldest (req1 and its attempt att1)
    subscriber(RequestStarted(request_id=req3))

    assert subscriber.registry.active_request_count() == 2
    assert subscriber.registry.get_request_span(req1) is None
    assert subscriber.registry.get_attempt_span(att1) is None

    # Check exported spans: req1 and att1 should both be exported with error status
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    evicted_att = next(s for s in spans if s.attributes.get(ATTR_OPENAIOPS_ATTEMPT_ID) == att1)
    evicted_req = next(s for s in spans if s.attributes.get(ATTR_OPENAIOPS_REQUEST_ID) == req1)

    assert evicted_att.status.status_code == StatusCode.ERROR
    assert evicted_att.attributes[ATTR_ERROR_TYPE] == "abandoned"
    assert evicted_req.status.status_code == StatusCode.ERROR
    assert evicted_req.attributes[ATTR_ERROR_TYPE] == "abandoned"


def test_circuit_state_changed_event(tracer_and_subscriber):
    _, subscriber, exporter = tracer_and_subscriber
    req_id = new_id()

    subscriber(RequestStarted(request_id=req_id))
    subscriber(
        CircuitStateChanged(
            provider="openai",
            from_state=CircuitState.CLOSED,
            to_state=CircuitState.OPEN,
            request_id=req_id,
        )
    )
    subscriber(RequestCompleted(request_id=req_id, latency_s=0.1))

    spans = exporter.get_finished_spans()
    request_span = spans[0]
    circuit_events = [e for e in request_span.events if e.name == EVENT_CIRCUIT_STATE_CHANGED]
    assert len(circuit_events) == 1
    assert circuit_events[0].attributes[ATTR_CIRCUIT_PROVIDER] == "openai"
    assert circuit_events[0].attributes[ATTR_CIRCUIT_FROM_STATE] == "CircuitState.CLOSED"
    assert circuit_events[0].attributes[ATTR_CIRCUIT_TO_STATE] == "CircuitState.OPEN"


def test_non_throwing_guarantee_logs_exception_type_only(tracer_and_subscriber, caplog):
    _, subscriber, _ = tracer_and_subscriber

    class MalformedEvent:
        type = "RequestStarted"

        @property
        def request_id(self):
            raise RuntimeError("secret_db_password_12345")

    with caplog.at_level(logging.WARNING):
        subscriber(MalformedEvent())  # type: ignore

    # Must log the exception type only, never the message carrying potential PII/secrets
    assert "TracingSubscriber failed to handle event type=RequestStarted: RuntimeError" in caplog.text
    assert "secret_db_password_12345" not in caplog.text
