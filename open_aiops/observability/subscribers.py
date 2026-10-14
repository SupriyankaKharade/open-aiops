"""Observability event subscribers for OpenAIOps (INT-18).

Builds OpenTelemetry span hierarchies from in-process lifecycle events.
Ensures zero leaks via deterministic terminal span cleanup, thread-safe registries,
bounded registry capacity, and non-throwing execution guarantees.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, StatusCode, Tracer

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
from open_aiops.observability.telemetry import get_tracer

if TYPE_CHECKING:
    from open_aiops.events import LifecycleEvent

logger = logging.getLogger(__name__)

DEFAULT_MAX_ACTIVE_REQUESTS: int = 1000


class SpanRegistry:
    """Thread-safe registry tracking active request and attempt spans.

    Ensures that spans can be looked up by ID across event emissions and
    cleaned deterministically upon terminal events or eviction to prevent memory leaks.
    """

    def __init__(self, max_active_requests: int = DEFAULT_MAX_ACTIVE_REQUESTS) -> None:
        self._lock = threading.RLock()
        self._max_active_requests = max_active_requests
        self._request_spans: dict[str, Span] = {}
        self._attempt_spans: dict[str, Span] = {}
        self._attempt_to_request: dict[str, str] = {}

    def register_request_span(self, request_id: str, span: Span) -> None:
        with self._lock:
            if (
                len(self._request_spans) >= self._max_active_requests
                and request_id not in self._request_spans
            ):
                self._evict_oldest_request()
            self._request_spans[request_id] = span

    def _evict_oldest_request(self) -> None:
        """Evict and end the oldest request and any of its open attempts."""
        if not self._request_spans:
            return
        oldest_id = next(iter(self._request_spans))
        oldest_span = self._request_spans.pop(oldest_id, None)
        self.cleanup_lingering_attempts(oldest_id)
        if oldest_span:
            oldest_span.set_attribute(ATTR_ERROR_TYPE, "abandoned")
            oldest_span.set_status(
                StatusCode.ERROR, description="Request evicted due to registry capacity"
            )
            oldest_span.end()

    def get_request_span(self, request_id: str) -> Optional[Span]:
        with self._lock:
            return self._request_spans.get(request_id)

    def pop_request_span(self, request_id: str) -> Optional[Span]:
        with self._lock:
            return self._request_spans.pop(request_id, None)

    def register_attempt_span(
        self, attempt_id: str, request_id: str, span: Span
    ) -> None:
        with self._lock:
            self._attempt_spans[attempt_id] = span
            self._attempt_to_request[attempt_id] = request_id

    def get_attempt_span(self, attempt_id: str) -> Optional[Span]:
        with self._lock:
            return self._attempt_spans.get(attempt_id)

    def pop_attempt_span(self, attempt_id: str) -> Optional[Span]:
        with self._lock:
            self._attempt_to_request.pop(attempt_id, None)
            return self._attempt_spans.pop(attempt_id, None)

    def cleanup_lingering_attempts(self, request_id: str) -> None:
        """End and drop any lingering attempt spans for request_id to prevent lost spans."""
        with self._lock:
            attempts_to_remove = [
                aid
                for aid, req_id in self._attempt_to_request.items()
                if req_id == request_id
            ]
            for aid in attempts_to_remove:
                span = self._attempt_spans.pop(aid, None)
                self._attempt_to_request.pop(aid, None)
                if span:
                    span.set_attribute(ATTR_ERROR_TYPE, "abandoned")
                    span.set_status(StatusCode.ERROR, description="Attempt abandoned")
                    span.end()

    def cleanup_request(self, request_id: str) -> None:
        """Purge request span and end any lingering attempt spans for request_id."""
        with self._lock:
            self._request_spans.pop(request_id, None)
            self.cleanup_lingering_attempts(request_id)

    def active_request_count(self) -> int:
        with self._lock:
            return len(self._request_spans)

    def active_attempt_count(self) -> int:
        with self._lock:
            return len(self._attempt_spans)

    def is_empty(self) -> bool:
        with self._lock:
            return (
                len(self._request_spans) == 0
                and len(self._attempt_spans) == 0
                and len(self._attempt_to_request) == 0
            )


class TracingSubscriber:
    """Subscriber that transforms lifecycle events into an OpenTelemetry span tree.

    Builds an hierarchical span tree where requests are root spans and attempts
    are nested children. Subscribes to EventDispatcher emissions. Conforms to the
    non-throwing principle: exceptions during span creation or attribute attachment
    are caught and logged without propagating.
    """

    def __init__(
        self,
        tracer: Optional[Tracer] = None,
        *,
        request_span_name: str = "open_aiops.request",
        attempt_span_name: str = "open_aiops.attempt",
        max_active_requests: int = DEFAULT_MAX_ACTIVE_REQUESTS,
        log: Optional[logging.Logger] = None,
    ) -> None:
        self._tracer = tracer or get_tracer()
        self._request_span_name = request_span_name
        self._attempt_span_name = attempt_span_name
        self._log = log or logger
        self._registry = SpanRegistry(max_active_requests=max_active_requests)

    @property
    def registry(self) -> SpanRegistry:
        """Access the underlying span registry for introspection and test assertions."""
        return self._registry

    def __call__(self, event: LifecycleEvent) -> None:
        """Process a lifecycle event and manage corresponding OpenTelemetry spans."""
        try:
            self._dispatch_event(event)
        except Exception as exc:  # noqa: BLE001 — telemetry must never break traffic
            self._log.warning(
                "TracingSubscriber failed to handle event type=%s: %s",
                getattr(event, "type", type(event).__name__),
                type(exc).__name__,
            )

    def _dispatch_event(self, event: LifecycleEvent) -> None:
        et = getattr(event, "type", None)

        if et == "RequestStarted":
            self._on_request_started(event)
        elif et == "AttemptStarted":
            self._on_attempt_started(event)
        elif et == "RoutingDecided":
            self._on_routing_decided(event)
        elif et == "StreamFirstChunk":
            self._on_stream_first_chunk(event)
        elif et == "RetryScheduled":
            self._on_retry_scheduled(event)
        elif et == "FallbackTriggered":
            self._on_fallback_triggered(event)
        elif et == "AttemptSucceeded":
            self._on_attempt_succeeded(event)
        elif et == "StreamCompleted":
            self._on_stream_completed(event)
        elif et == "AttemptFailed":
            self._on_attempt_failed(event)
        elif et == "StreamInterrupted":
            self._on_stream_interrupted(event)
        elif et == "AttemptCancelled":
            self._on_attempt_cancelled(event)
        elif et == "RequestCompleted":
            self._on_request_completed(event)
        elif et == "RequestFailed":
            self._on_request_failed(event)
        elif et == "CircuitStateChanged":
            self._on_circuit_state_changed(event)
        else:
            self._log.debug("TracingSubscriber unhandled event type: %s", et)

    # -------------------------------------------------------------------------
    # Lifecycle Handlers
    # -------------------------------------------------------------------------

    def _on_request_started(self, event: LifecycleEvent) -> None:
        attrs: dict[str, object] = {
            ATTR_OPENAIOPS_REQUEST_ID: event.request_id,
        }
        if event.tenant_id:
            attrs[ATTR_OPENAIOPS_TENANT_ID] = event.tenant_id

        span = self._tracer.start_span(
            name=self._request_span_name,
            attributes=attrs,
        )
        self._registry.register_request_span(event.request_id, span)

    def _on_attempt_started(self, event: LifecycleEvent) -> None:
        parent_span = self._registry.get_request_span(event.request_id)
        context = trace.set_span_in_context(parent_span) if parent_span else None

        attrs: dict[str, object] = {
            ATTR_GEN_AI_OPERATION_NAME: "chat",
            ATTR_GEN_AI_PROVIDER_NAME: event.provider,
            ATTR_GEN_AI_REQUEST_MODEL: event.model,
            ATTR_OPENAIOPS_ATTEMPT_NUMBER: event.attempt_number,
            ATTR_OPENAIOPS_REQUEST_ID: event.request_id,
            ATTR_OPENAIOPS_ATTEMPT_ID: event.attempt_id,
        }
        if event.tenant_id:
            attrs[ATTR_OPENAIOPS_TENANT_ID] = event.tenant_id

        span = self._tracer.start_span(
            name=self._attempt_span_name,
            context=context,
            kind=SpanKind.CLIENT,
            attributes=attrs,
        )
        self._registry.register_attempt_span(event.attempt_id, event.request_id, span)

    def _on_routing_decided(self, event: LifecycleEvent) -> None:
        span = self._registry.get_request_span(event.request_id)
        if span:
            event_attrs: dict[str, object] = {
                ATTR_ROUTING_RANKED: list(event.ranked),
            }
            if event.rejected:
                event_attrs[ATTR_ROUTING_REJECTED] = json.dumps(event.rejected)
            span.add_event(EVENT_ROUTING_DECIDED, attributes=event_attrs)

    def _on_stream_first_chunk(self, event: LifecycleEvent) -> None:
        span = self._registry.get_attempt_span(event.attempt_id)
        if span:
            span.add_event(
                EVENT_STREAM_FIRST_CHUNK,
                attributes={ATTR_STREAM_TS: event.ts.isoformat()},
            )

    def _on_retry_scheduled(self, event: LifecycleEvent) -> None:
        span = self._registry.get_attempt_span(
            event.attempt_id
        ) or self._registry.get_request_span(event.request_id)
        if span:
            span.add_event(
                EVENT_RETRY_SCHEDULED,
                attributes={
                    ATTR_RETRY_DELAY_S: float(event.delay_s),
                    ATTR_RETRY_ERROR_TYPE: event.error_type,
                },
            )

    def _on_fallback_triggered(self, event: LifecycleEvent) -> None:
        span = self._registry.get_request_span(event.request_id)
        if span:
            span.add_event(
                EVENT_FALLBACK_TRIGGERED,
                attributes={
                    ATTR_FALLBACK_FROM: event.from_provider,
                    ATTR_FALLBACK_TO: event.to_provider,
                    ATTR_FALLBACK_ERROR_TYPE: event.error_type,
                    ATTR_OPENAIOPS_ATTEMPT_NUMBER: event.attempt_number,
                },
            )

    def _on_attempt_succeeded(self, event: LifecycleEvent) -> None:
        span = self._registry.pop_attempt_span(event.attempt_id)
        if not span:
            return

        span.set_attribute(ATTR_OPENAIOPS_LATENCY_S, float(event.latency_s))
        if event.input_tokens is not None:
            span.set_attribute(ATTR_GEN_AI_USAGE_INPUT_TOKENS, event.input_tokens)
        if event.output_tokens is not None:
            span.set_attribute(ATTR_GEN_AI_USAGE_OUTPUT_TOKENS, event.output_tokens)
        if event.input_tokens is not None and event.output_tokens is not None:
            span.set_attribute(
                ATTR_OPENAIOPS_USAGE_TOTAL_TOKENS,
                event.input_tokens + event.output_tokens,
            )
        if event.cost is not None:
            span.set_attribute(ATTR_OPENAIOPS_COST_TOTAL, float(event.cost))
        if getattr(event, "cost_status", None) is not None:
            span.set_attribute(ATTR_OPENAIOPS_COST_STATUS, str(event.cost_status))

        span.set_status(StatusCode.OK)
        span.end()

    def _on_stream_completed(self, event: LifecycleEvent) -> None:
        self._on_attempt_succeeded(event)

    def _on_attempt_failed(self, event: LifecycleEvent) -> None:
        span = self._registry.pop_attempt_span(event.attempt_id)
        if not span:
            return

        span.set_attribute(ATTR_ERROR_TYPE, event.error_type)
        span.set_status(StatusCode.ERROR, description=event.error_type)
        span.end()

    def _on_stream_interrupted(self, event: LifecycleEvent) -> None:
        self._on_attempt_failed(event)

    def _on_attempt_cancelled(self, event: LifecycleEvent) -> None:
        span = self._registry.pop_attempt_span(event.attempt_id)
        if not span:
            return

        span.set_attribute(ATTR_ERROR_TYPE, "cancelled")
        span.set_status(StatusCode.ERROR, description="Attempt cancelled")
        span.end()

    def _on_request_completed(self, event: LifecycleEvent) -> None:
        span = self._registry.pop_request_span(event.request_id)
        # End and purge any lingering attempts to ensure they are exported
        self._registry.cleanup_lingering_attempts(event.request_id)

        if not span:
            return

        span.set_attribute(ATTR_OPENAIOPS_LATENCY_S, float(event.latency_s))
        if event.cost is not None:
            span.set_attribute(ATTR_OPENAIOPS_COST_TOTAL, float(event.cost))
        if getattr(event, "cost_status", None) is not None:
            span.set_attribute(ATTR_OPENAIOPS_COST_STATUS, str(event.cost_status))

        span.set_status(StatusCode.OK)
        span.end()

    def _on_request_failed(self, event: LifecycleEvent) -> None:
        span = self._registry.pop_request_span(event.request_id)
        self._registry.cleanup_lingering_attempts(event.request_id)

        if not span:
            return

        span.set_attribute(ATTR_ERROR_TYPE, event.error_type)
        span.set_status(StatusCode.ERROR, description=event.error_type)
        span.end()

    def _on_circuit_state_changed(self, event: LifecycleEvent) -> None:
        if event.request_id:
            span = self._registry.get_request_span(event.request_id)
            if span:
                span.add_event(
                    EVENT_CIRCUIT_STATE_CHANGED,
                    attributes={
                        ATTR_CIRCUIT_PROVIDER: event.provider,
                        ATTR_CIRCUIT_FROM_STATE: str(event.from_state),
                        ATTR_CIRCUIT_TO_STATE: str(event.to_state),
                    },
                )
