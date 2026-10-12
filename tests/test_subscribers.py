"""Unit tests for TracingSubscriber and span lifecycle transformations (INT-18)."""

from __future__ import annotations

import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

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
    ATTR_ERROR_TYPE,
    ATTR_FALLBACK_ERROR_TYPE,
    ATTR_FALLBACK_FROM,
    ATTR_FALLBACK_TO,
    ATTR_GEN_AI_ATTEMPT_NUMBER,
    ATTR_GEN_AI_PROVIDER_NAME,
    ATTR_GEN_AI_REQUEST_MODEL,
    ATTR_GEN_AI_USAGE_INPUT_TOKENS,
    ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
    ATTR_GEN_AI_USAGE_TOTAL_TOKENS,
    ATTR_OPENAIOPS_ATTEMPT_ID,
    ATTR_OPENAIOPS_COST_TOTAL,
    ATTR_OPENAIOPS_LATENCY_S,
    ATTR_OPENAIOPS_REQUEST_ID,
    ATTR_OPENAIOPS_TENANT_ID,
    ATTR_RETRY_DELAY_S,
    ATTR_RETRY_ERROR_TYPE,
    ATTR_ROUTING_RANKED,
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

    # Check attempt span attributes
    assert attempt_span.name == "open_aiops.attempt"
    assert attempt_span.attributes[ATTR_GEN_AI_PROVIDER_NAME] == "openai"
    assert attempt_span.attributes[ATTR_GEN_AI_REQUEST_MODEL] == "gpt-4o"
    assert attempt_span.attributes[ATTR_GEN_AI_ATTEMPT_NUMBER] == 1
    assert attempt_span.attributes[ATTR_GEN_AI_USAGE_INPUT_TOKENS] == 120
    assert attempt_span.attributes[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS] == 45
    assert attempt_span.attributes[ATTR_GEN_AI_USAGE_TOTAL_TOKENS] == 165
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

    # Request span contains routing and fallback events
    event_names = [e.name for e in request_span.events]
    assert EVENT_ROUTING_DECIDED in event_names
    assert EVENT_FALLBACK_TRIGGERED in event_names

    fallback_event = next(e for e in request_span.events if e.name == EVENT_FALLBACK_TRIGGERED)
    assert fallback_event.attributes[ATTR_FALLBACK_FROM] == "openai"
    assert fallback_event.attributes[ATTR_FALLBACK_TO] == "anthropic"
    assert fallback_event.attributes[ATTR_FALLBACK_ERROR_TYPE] == "RateLimitError"


def test_retry_event_recording(tracer_and_subscriber):
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
        RetryScheduled(
            request_id=req_id,
            attempt_id=att1_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            delay_s=1.5,
            error_type="ProviderTimeout",
        ),
        AttemptFailed(
            request_id=req_id,
            attempt_id=att1_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
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
    retry_events = [e for e in attempt1_span.events if e.name == EVENT_RETRY_SCHEDULED]
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
    assert attempt_span.status.status_code == StatusCode.OK
    assert attempt_span.attributes[ATTR_GEN_AI_USAGE_TOTAL_TOKENS] == 230


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


def test_terminal_cleanup_with_dangling_attempt(tracer_and_subscriber):
    """If a request ends abruptly, any dangling attempt spans are cleaned up."""
    _, subscriber, _ = tracer_and_subscriber
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

    # Terminal request event cleans up everything
    subscriber(RequestCompleted(request_id=req_id, latency_s=1.0))
    assert subscriber.registry.is_empty()


def test_event_dispatcher_integration(tracer_and_subscriber):
    _, subscriber, exporter = tracer_and_subscriber
    dispatcher = EventDispatcher()
    dispatcher.subscribe(subscriber)

    req_id = new_id()
    att_id = new_id()

    dispatcher.emit(RequestStarted(request_id=req_id))
    dispatcher.emit(
        AttemptStarted(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
        )
    )
    dispatcher.emit(
        AttemptSucceeded(
            request_id=req_id,
            attempt_id=att_id,
            attempt_number=1,
            provider="openai",
            model="gpt-4o",
            latency_s=0.5,
            input_tokens=10,
            output_tokens=20,
        )
    )
    dispatcher.emit(RequestCompleted(request_id=req_id, latency_s=0.55))

    assert subscriber.registry.is_empty()
    assert len(exporter.get_finished_spans()) == 2


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
    circuit_events = [e for e in request_span.events if e.name == "gen_ai.circuit.state_changed"]
    assert len(circuit_events) == 1
    assert circuit_events[0].attributes["provider"] == "openai"


def test_non_throwing_guarantee_on_malformed_event(tracer_and_subscriber, caplog):
    _, subscriber, _ = tracer_and_subscriber

    class MalformedEvent:
        type = "RequestStarted"

        @property
        def request_id(self):
            raise RuntimeError("unexpected failure")

    with caplog.at_level(logging.WARNING):
        # Must not raise
        subscriber(MalformedEvent())  # type: ignore

    assert "TracingSubscriber failed to handle event" in caplog.text
