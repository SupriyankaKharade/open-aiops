"""Metrics subscriber converting lifecycle events to OpenTelemetry metrics (INT-19).

Subscribes to EventDispatcher emissions and records request duration, request totals,
provider errors, retries, fallbacks, token distributions, and cost attribution
via MetricsRecorder.
"""

from collections import OrderedDict
import logging
import threading
from typing import Optional

from open_aiops.events import LifecycleEvent
from open_aiops.observability.metrics import MetricsRecorder

logger = logging.getLogger(__name__)

DEFAULT_MAX_ACTIVE_REQUESTS = 10000
DEFAULT_MAX_DISTINCT_MODELS = 100
DEFAULT_MAX_MODEL_LABEL_LENGTH = 64


class MetricsSubscriber:
    """Subscriber that translates lifecycle events into OpenTelemetry metrics.

    Attaches to EventDispatcher and updates metrics instruments defined in MetricsRecorder.
    Conforms strictly to the non-throwing principle: telemetry failures or malformed events
    are logged and never propagate or disrupt application traffic.
    """

    def __init__(
        self,
        recorder: Optional[MetricsRecorder] = None,
        *,
        max_active_requests: int = DEFAULT_MAX_ACTIVE_REQUESTS,
        max_distinct_models: int = DEFAULT_MAX_DISTINCT_MODELS,
        max_model_len: int = DEFAULT_MAX_MODEL_LABEL_LENGTH,
        log: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the metrics subscriber.

        Args:
            recorder: Optional MetricsRecorder instance. Defaults to a new MetricsRecorder().
            max_active_requests: Maximum number of in-flight requests tracked in memory.
            max_distinct_models: Maximum distinct model labels permitted before overflow to 'other'.
            max_model_len: Maximum length of a model name before normalization to 'other'.
            log: Optional logger instance.
        """
        self._recorder = recorder if recorder is not None else MetricsRecorder()
        self._max_active_requests = max(1, max_active_requests)
        self._max_distinct_models = max(1, max_distinct_models)
        self._max_model_len = max(1, max_model_len)
        self._log = log or logger
        self._lock = threading.Lock()
        self._requests: OrderedDict[str, dict] = OrderedDict()
        self._known_models: set[str] = set()

    @property
    def recorder(self) -> MetricsRecorder:
        """Return the underlying MetricsRecorder instance."""
        return self._recorder

    @property
    def active_requests_count(self) -> int:
        """Return the number of in-flight requests currently tracked."""
        with self._lock:
            return len(self._requests)

    def _cap_model(self, model: object) -> str:
        """Sanitize and bound model label to prevent cardinality explosions.

        - Empty, non-string, or whitespace-only inputs map to 'unknown'.
        - Strings exceeding max_model_len or containing newlines/CR map to 'other'.
        - Distinct models exceeding max_distinct_models map to 'other'.
        """
        if not isinstance(model, str):
            return "unknown"
        cleaned = model.strip()
        if not cleaned:
            return "unknown"
        if len(cleaned) > self._max_model_len or "\n" in cleaned or "\r" in cleaned:
            return "other"

        with self._lock:
            if cleaned in self._known_models:
                return cleaned
            if len(self._known_models) < self._max_distinct_models:
                self._known_models.add(cleaned)
                return cleaned
            return "other"

    def _set_active_attempt(
        self,
        request_id: str,
        provider: str,
        model: str,
        succeeded: bool = False,
        error_type: Optional[str] = None,
    ) -> None:
        if not request_id:
            return
        with self._lock:
            if request_id in self._requests:
                entry = self._requests[request_id]
                if not entry.get("succeeded") or succeeded:
                    entry["provider"] = provider
                    entry["model"] = model
                    if succeeded:
                        entry["succeeded"] = True
                if error_type:
                    entry["error_type"] = error_type
                self._requests.move_to_end(request_id)
            else:
                if len(self._requests) >= self._max_active_requests:
                    self._requests.popitem(last=False)
                self._requests[request_id] = {
                    "provider": provider,
                    "model": model,
                    "succeeded": succeeded,
                    "error_type": error_type,
                }

    def _pop_request(self, request_id: str) -> Optional[dict]:
        if not request_id:
            return None
        with self._lock:
            return self._requests.pop(request_id, None)

    def __call__(self, event: LifecycleEvent) -> None:
        """Process a lifecycle event and record corresponding metrics."""
        try:
            self._dispatch_event(event)
        except Exception as exc:  # Telemetry must never disrupt traffic
            try:
                self._log.debug(
                    "MetricsSubscriber failed to handle event %s: %s",
                    getattr(event, "type", type(event).__name__),
                    type(exc).__name__,
                )
            except Exception:
                pass

    def _dispatch_event(self, event: LifecycleEvent) -> None:
        et = getattr(event, "type", None)

        if et == "AttemptStarted":
            self._on_attempt_started(event)
        elif et == "AttemptSucceeded":
            self._on_attempt_succeeded(event)
        elif et == "AttemptFailed":
            self._on_attempt_failed(event)
        elif et == "AttemptCancelled":
            self._on_attempt_cancelled(event)
        elif et == "StreamCompleted":
            self._on_stream_completed(event)
        elif et == "StreamInterrupted":
            self._on_stream_interrupted(event)
        elif et == "RetryScheduled":
            self._on_retry_scheduled(event)
        elif et == "FallbackTriggered":
            self._on_fallback_triggered(event)
        elif et == "RequestCompleted":
            self._on_request_completed(event)
        elif et == "RequestFailed":
            self._on_request_failed(event)

    def _on_attempt_started(self, event: LifecycleEvent) -> None:
        self._set_active_attempt(
            request_id=getattr(event, "request_id", ""),
            provider=getattr(event, "provider", ""),
            model=getattr(event, "model", ""),
        )

    def _on_attempt_succeeded(self, event: LifecycleEvent) -> None:
        req_id = getattr(event, "request_id", "")
        provider = getattr(event, "provider", "")
        raw_model = getattr(event, "model", "")
        self._set_active_attempt(
            request_id=req_id,
            provider=provider,
            model=raw_model,
            succeeded=True,
        )

        capped_model = self._cap_model(raw_model)
        # Record attempt token usage if supplied
        input_tokens = getattr(event, "input_tokens", None)
        output_tokens = getattr(event, "output_tokens", None)
        if input_tokens is not None or output_tokens is not None:
            self._recorder.record_tokens(
                provider=provider,
                model=capped_model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Record attempt cost only when event indicates known, usable cost
        cost_status = getattr(event, "cost_status", None)
        cost_val = getattr(event, "cost", None)
        if cost_status in ("priced", "free") and cost_val is not None:
            self._recorder.record_cost(
                provider=provider,
                model=capped_model,
                cost=cost_val,
            )

    def _on_attempt_failed(self, event: LifecycleEvent) -> None:
        req_id = getattr(event, "request_id", "")
        provider = getattr(event, "provider", "")
        raw_model = getattr(event, "model", "")
        error_type = getattr(event, "error_type", "unknown")
        self._set_active_attempt(
            request_id=req_id,
            provider=provider,
            model=raw_model,
            error_type=error_type,
        )

        capped_model = self._cap_model(raw_model)
        self._recorder.record_error(
            provider=provider,
            model=capped_model,
            error_type=error_type,
        )

    def _on_attempt_cancelled(self, event: LifecycleEvent) -> None:
        req_id = getattr(event, "request_id", "")
        provider = getattr(event, "provider", "")
        raw_model = getattr(event, "model", "")
        self._set_active_attempt(
            request_id=req_id,
            provider=provider,
            model=raw_model,
            error_type="cancelled",
        )

    def _on_stream_completed(self, event: LifecycleEvent) -> None:
        # StreamCompleted carries the same attempt-level metrics contract as AttemptSucceeded
        self._on_attempt_succeeded(event)

    def _on_stream_interrupted(self, event: LifecycleEvent) -> None:
        # StreamInterrupted carries the same attempt failure contract as AttemptFailed
        self._on_attempt_failed(event)

    def _on_retry_scheduled(self, event: LifecycleEvent) -> None:
        provider = getattr(event, "provider", "")
        self._recorder.record_retry(provider=provider)

    def _on_fallback_triggered(self, event: LifecycleEvent) -> None:
        # In FallbackTriggered, from_provider is the provider whose failure triggered fallback
        from_provider = getattr(event, "from_provider", "")
        self._recorder.record_fallback(provider=from_provider)

    def _on_request_completed(self, event: LifecycleEvent) -> None:
        req_id = getattr(event, "request_id", "")
        info = self._pop_request(req_id)

        provider = info["provider"] if info and "provider" in info else "unknown"
        raw_model = info["model"] if info and "model" in info else "unknown"
        capped_model = self._cap_model(raw_model)

        latency_s = getattr(event, "latency_s", 0.0)
        self._recorder.record_request(
            provider=provider,
            model=capped_model,
            duration=latency_s,
            outcome="success",
        )

    def _on_request_failed(self, event: LifecycleEvent) -> None:
        req_id = getattr(event, "request_id", "")
        info = self._pop_request(req_id)

        provider = info["provider"] if info and "provider" in info else "unknown"
        raw_model = info["model"] if info and "model" in info else "unknown"
        capped_model = self._cap_model(raw_model)

        err_type = str(getattr(event, "error_type", "")).strip().lower()
        outcome = "cancelled" if err_type == "cancelled" else "error"

        self._recorder.record_request_count(
            provider=provider,
            model=capped_model,
            outcome=outcome,
        )
