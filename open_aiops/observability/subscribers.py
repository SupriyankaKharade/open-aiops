"""Observability event subscribers for OpenAIOps (INT-18).

Builds OpenTelemetry span hierarchies from in-process lifecycle events.
Ensures zero leaks via deterministic terminal span cleanup, thread-safe registries,
and non-throwing execution guarantees.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

from open_aiops.observability.telemetry import get_tracer

if TYPE_CHECKING:
    from open_aiops.events import LifecycleEvent

logger = logging.getLogger(__name__)


class SpanRegistry:
    """Thread-safe registry tracking active request and attempt spans.

    Ensures that spans can be looked up by ID across event emissions and
    cleaned deterministically upon terminal events to prevent memory leaks.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._request_spans: dict[str, Span] = {}
        self._attempt_spans: dict[str, Span] = {}
        self._attempt_to_request: dict[str, str] = {}

    def register_request_span(self, request_id: str, span: Span) -> None:
        with self._lock:
            self._request_spans[request_id] = span

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

    def cleanup_request(self, request_id: str) -> None:
        """Purge request span and any lingering attempt spans for request_id."""
        with self._lock:
            self._request_spans.pop(request_id, None)
            attempts_to_remove = [
                aid
                for aid, req_id in self._attempt_to_request.items()
                if req_id == request_id
            ]
            for aid in attempts_to_remove:
                self._attempt_spans.pop(aid, None)
                self._attempt_to_request.pop(aid, None)

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

    Subscribes to EventDispatcher emissions. Conforms to the non-throwing principle:
    exceptions during span creation or attribute attachment are caught and logged.
    """

    def __init__(
        self,
        tracer: Optional[Tracer] = None,
        *,
        request_span_name: str = "open_aiops.request",
        attempt_span_name: str = "open_aiops.attempt",
        log: Optional[logging.Logger] = None,
    ) -> None:
        self._tracer = tracer or get_tracer()
        self._request_span_name = request_span_name
        self._attempt_span_name = attempt_span_name
        self._log = log or logger
        self._registry = SpanRegistry()

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
                exc,
            )

    def _dispatch_event(self, event: LifecycleEvent) -> None:
        """Dispatch event to its specific handler method."""
        # Initial interface stub; event handlers implemented in Commit 2
        pass
