"""Immutable lifecycle events and a safe in-process dispatcher (INT-14).

Subscribers (tracing, metrics) listen here. The gateway emits events and never
imports OpenTelemetry. Events must not carry prompts, completions, keys,
headers, or exception message text — error *types* only.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1"

logger = logging.getLogger(__name__)

# Soft budget for a single subscriber call; overruns log a warning, not an error.
DEFAULT_SUBSCRIBER_BUDGET_S = 0.05


def new_id() -> str:
    """Opaque UUID string for request_id / attempt_id."""
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EventBase(BaseModel):
    model_config = {"frozen": True}

    schema_version: Literal["1"] = SCHEMA_VERSION
    request_id: str
    ts: datetime = Field(default_factory=utc_now)
    tenant_id: str | None = None


class AttemptEventBase(EventBase):
    attempt_id: str
    attempt_number: int = Field(ge=1)
    provider: str
    model: str


class RequestStarted(EventBase):
    type: Literal["RequestStarted"] = "RequestStarted"


class RequestCompleted(EventBase):
    type: Literal["RequestCompleted"] = "RequestCompleted"
    latency_s: float = Field(ge=0)
    # Cost is filled by a later ticket; keep the slot optional and numeric only.
    cost: float | None = Field(default=None, ge=0)


class RequestFailed(EventBase):
    type: Literal["RequestFailed"] = "RequestFailed"
    error_type: str


class RoutingDecided(EventBase):
    type: Literal["RoutingDecided"] = "RoutingDecided"
    ranked: tuple[str, ...]
    rejected: dict[str, str] = Field(default_factory=dict)


class AttemptStarted(AttemptEventBase):
    type: Literal["AttemptStarted"] = "AttemptStarted"


class AttemptSucceeded(AttemptEventBase):
    type: Literal["AttemptSucceeded"] = "AttemptSucceeded"
    latency_s: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)


class AttemptFailed(AttemptEventBase):
    type: Literal["AttemptFailed"] = "AttemptFailed"
    error_type: str


class AttemptCancelled(AttemptEventBase):
    type: Literal["AttemptCancelled"] = "AttemptCancelled"


class StreamFirstChunk(AttemptEventBase):
    type: Literal["StreamFirstChunk"] = "StreamFirstChunk"


class StreamCompleted(AttemptEventBase):
    type: Literal["StreamCompleted"] = "StreamCompleted"
    latency_s: float = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)


class StreamInterrupted(AttemptEventBase):
    type: Literal["StreamInterrupted"] = "StreamInterrupted"
    error_type: str


class RetryScheduled(AttemptEventBase):
    type: Literal["RetryScheduled"] = "RetryScheduled"
    delay_s: float = Field(ge=0)
    error_type: str


class FallbackTriggered(EventBase):
    type: Literal["FallbackTriggered"] = "FallbackTriggered"
    from_provider: str
    to_provider: str
    error_type: str
    attempt_number: int = Field(ge=1)


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitStateChanged(BaseModel):
    """Provider-level; not tied to a single request lifecycle."""

    model_config = {"frozen": True}

    type: Literal["CircuitStateChanged"] = "CircuitStateChanged"
    schema_version: Literal["1"] = SCHEMA_VERSION
    ts: datetime = Field(default_factory=utc_now)
    provider: str
    from_state: CircuitState
    to_state: CircuitState
    request_id: str | None = None
    tenant_id: str | None = None


LifecycleEvent = Annotated[
    Union[
        RequestStarted,
        RequestCompleted,
        RequestFailed,
        RoutingDecided,
        AttemptStarted,
        AttemptSucceeded,
        AttemptFailed,
        AttemptCancelled,
        StreamFirstChunk,
        StreamCompleted,
        StreamInterrupted,
        RetryScheduled,
        FallbackTriggered,
        CircuitStateChanged,
    ],
    Field(discriminator="type"),
]

Subscriber = Callable[[LifecycleEvent], None]


class EventDispatcher:
    """Ordered, exception-safe delivery to in-process subscribers."""

    def __init__(
        self,
        subscribers: Iterable[Subscriber] | None = None,
        *,
        subscriber_budget_s: float = DEFAULT_SUBSCRIBER_BUDGET_S,
        clock: Callable[[], float] = time.perf_counter,
        log: logging.Logger | None = None,
    ):
        self._subscribers: list[Subscriber] = list(subscribers or ())
        self._budget_s = subscriber_budget_s
        self._clock = clock
        self._log = log or logger

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)

    def emit(self, event: LifecycleEvent) -> None:
        if not self._subscribers:
            return
        for subscriber in self._subscribers:
            started = self._clock()
            try:
                subscriber(event)
            except Exception as exc:  # noqa: BLE001 — must never break the request
                self._log.warning(
                    "event subscriber failed type=%s event=%s",
                    type(exc).__name__,
                    getattr(event, "type", type(event).__name__),
                )
            elapsed = self._clock() - started
            if elapsed > self._budget_s:
                self._log.warning(
                    "event subscriber slow elapsed_s=%.4f budget_s=%.4f event=%s",
                    elapsed,
                    self._budget_s,
                    getattr(event, "type", type(event).__name__),
                )


class MalformedEventSequence(AssertionError):
    pass


_REQUEST_TERMINALS = frozenset({"RequestCompleted", "RequestFailed"})
_ATTEMPT_TERMINALS_NONSTREAM = frozenset(
    {"AttemptSucceeded", "AttemptFailed", "AttemptCancelled"}
)
_ATTEMPT_TERMINALS_STREAM = frozenset(
    {"StreamCompleted", "StreamInterrupted", "AttemptCancelled", "AttemptFailed"}
)
_MARKERS = frozenset(
    {"RoutingDecided", "StreamFirstChunk", "RetryScheduled", "FallbackTriggered"}
)


def assert_well_formed(events: Sequence[LifecycleEvent]) -> None:
    """Validate the lifecycle state table used by gateway and subscriber tests.

    Raises ``MalformedEventSequence`` on the first violation.
    ``CircuitStateChanged`` is ignored for request lifecycle checks.
    """
    request_open = False
    request_done = False
    # attempt_id -> {"streaming": bool|None, "done": bool, "had_first_chunk": bool}
    attempts: dict[str, dict[str, object]] = {}

    for event in events:
        et = event.type  # type: ignore[union-attr]

        if et == "CircuitStateChanged":
            continue

        if et == "RequestStarted":
            if request_open or request_done:
                raise MalformedEventSequence("duplicate or late RequestStarted")
            request_open = True
            continue

        if not request_open:
            raise MalformedEventSequence(f"{et} before RequestStarted")

        if request_done:
            raise MalformedEventSequence(f"{et} after request terminal")

        if et in _REQUEST_TERMINALS:
            for attempt_id, state in attempts.items():
                if not state["done"]:
                    raise MalformedEventSequence(
                        f"request terminal with open attempt {attempt_id}"
                    )
            request_done = True
            request_open = False
            continue

        if et == "AttemptStarted":
            if event.attempt_id in attempts:  # type: ignore[union-attr]
                raise MalformedEventSequence(f"duplicate AttemptStarted {event.attempt_id}")  # type: ignore[union-attr]
            attempts[event.attempt_id] = {  # type: ignore[union-attr]
                "streaming": None,
                "done": False,
                "had_first_chunk": False,
                "number": event.attempt_number,  # type: ignore[union-attr]
            }
            continue

        if et in _MARKERS:
            if et == "StreamFirstChunk":
                state = attempts.get(event.attempt_id)  # type: ignore[union-attr]
                if state is None or state["done"]:
                    raise MalformedEventSequence("StreamFirstChunk without open attempt")
                state["streaming"] = True
                state["had_first_chunk"] = True
            continue

        if et in _ATTEMPT_TERMINALS_NONSTREAM | _ATTEMPT_TERMINALS_STREAM:
            aid = event.attempt_id  # type: ignore[union-attr]
            state = attempts.get(aid)
            if state is None:
                raise MalformedEventSequence(f"{et} without AttemptStarted")
            if state["done"]:
                raise MalformedEventSequence(f"second terminal for attempt {aid}")
            if et in {"StreamCompleted", "StreamInterrupted"}:
                state["streaming"] = True
            elif et == "AttemptSucceeded" and state["had_first_chunk"]:
                raise MalformedEventSequence(
                    "AttemptSucceeded after StreamFirstChunk; use StreamCompleted"
                )
            state["done"] = True
            continue

        raise MalformedEventSequence(f"unknown event type {et}")

    if request_open and not request_done:
        raise MalformedEventSequence("RequestStarted without terminal")
    for attempt_id, state in attempts.items():
        if not state["done"]:
            raise MalformedEventSequence(f"open attempt {attempt_id}")
