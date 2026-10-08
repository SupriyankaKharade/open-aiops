"""OpenTelemetry metrics utilities for Open-AIOps.

Implements INT-4 metrics instruments according to architecture section 3.8.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence
from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader

logger = logging.getLogger(__name__)

_METER_NAME = "open_aiops.metrics"


def get_meter() -> Meter:
    """Get the active OpenTelemetry meter."""
    return metrics.get_meter(_METER_NAME)


class MetricsManager:
    """Manages OpenTelemetry metric instruments and safe recording."""

    def __init__(self, meter: Optional[Meter] = None) -> None:
        self.meter = meter or get_meter()
        self._init_instruments()

    def _init_instruments(self) -> None:
        # Request duration histogram (seconds)
        self.request_duration: Histogram = self.meter.create_histogram(
            name="openaiops.request.duration",
            unit="s",
            description="Duration of LLM requests in seconds",
        )
        # Total requests counter
        self.requests: Counter = self.meter.create_counter(
            name="openaiops.requests",
            unit="{request}",
            description="Total count of completed or failed requests",
        )
        # Provider errors counter
        self.provider_errors: Counter = self.meter.create_counter(
            name="openaiops.provider.errors",
            unit="{error}",
            description="Count of provider errors categorized by type",
        )
        # Retries counter
        self.retries: Counter = self.meter.create_counter(
            name="openaiops.retries",
            unit="{retry}",
            description="Total count of retry attempts scheduled",
        )
        # Fallbacks counter
        self.fallbacks: Counter = self.meter.create_counter(
            name="openaiops.fallbacks",
            unit="{fallback}",
            description="Total count of fallbacks triggered to alternate providers",
        )
        # Tokens histogram
        self.tokens: Histogram = self.meter.create_histogram(
            name="openaiops.tokens",
            unit="{token}",
            description="Token count per request or attempt",
        )
        # Cost counter
        self.cost: Counter = self.meter.create_counter(
            name="openaiops.cost",
            unit="USD",
            description="Estimated financial cost in USD",
        )

    def record_request_duration(
        self, duration_s: float, provider: str, model: str, outcome: str
    ) -> None:
        """Record request duration in seconds with bounded labels."""
        try:
            attributes = {
                "provider": provider,
                "model": model,
                "outcome": outcome,
            }
            self.request_duration.record(duration_s, attributes=attributes)
        except Exception as exc:
            logger.warning("Failed to record request duration metric: %s", type(exc).__name__)

    def record_request(self, outcome: str) -> None:
        """Record completed/failed request count."""
        try:
            self.requests.add(1, attributes={"outcome": outcome})
        except Exception as exc:
            logger.warning("Failed to record request metric: %s", type(exc).__name__)

    def record_provider_error(self, provider: str, model: str, error_type: str) -> None:
        """Record a provider error by error type."""
        try:
            attributes = {
                "provider": provider,
                "model": model,
                "error_type": error_type,
            }
            self.provider_errors.add(1, attributes=attributes)
        except Exception as exc:
            logger.warning("Failed to record provider error metric: %s", type(exc).__name__)

    def record_retry(self, provider: str) -> None:
        """Record a retry scheduled on a provider."""
        try:
            self.retries.add(1, attributes={"provider": provider})
        except Exception as exc:
            logger.warning("Failed to record retry metric: %s", type(exc).__name__)

    def record_fallback(self, provider: str) -> None:
        """Record a fallback triggered away from a failing provider."""
        try:
            self.fallbacks.add(1, attributes={"provider": provider})
        except Exception as exc:
            logger.warning("Failed to record fallback metric: %s", type(exc).__name__)

    def record_tokens(
        self, count: int, provider: str, model: str, token_type: str
    ) -> None:
        """Record token usage (input, output, or total)."""
        try:
            attributes = {
                "provider": provider,
                "model": model,
                "token_type": token_type,
            }
            self.tokens.record(count, attributes=attributes)
        except Exception as exc:
            logger.warning("Failed to record tokens metric: %s", type(exc).__name__)

    def record_cost(self, amount: float, provider: str, model: str) -> None:
        """Record monetary cost in USD."""
        try:
            attributes = {
                "provider": provider,
                "model": model,
            }
            self.cost.add(amount, attributes=attributes)
        except Exception as exc:
            logger.warning("Failed to record cost metric: %s", type(exc).__name__)


_DEFAULT_MANAGER: Optional[MetricsManager] = None


def get_metrics_manager() -> MetricsManager:
    """Get or create the global MetricsManager."""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = MetricsManager()
    return _DEFAULT_MANAGER


def init_metrics(
    metric_readers: Optional[Sequence[MetricReader]] = None,
    force_reset: bool = False,
) -> MetricsManager:
    """Initialize OpenTelemetry MeterProvider and configure instruments.

    Args:
        metric_readers: Optional readers/exporters (e.g. InMemoryMetricReader).
        force_reset: Reset existing MeterProvider and instruments (useful for tests).

    Returns:
        A configured MetricsManager instance.
    """
    global _DEFAULT_MANAGER
    if force_reset:
        try:
            from opentelemetry.metrics import _internal as mi

            mi._METER_PROVIDER = None
            if hasattr(mi, "_METER_PROVIDER_SET_ONCE"):
                mi._METER_PROVIDER_SET_ONCE._done = False
        except Exception:
            pass
        _DEFAULT_MANAGER = None

    provider = MeterProvider(metric_readers=list(metric_readers) if metric_readers else [])
    metrics.set_meter_provider(provider)

    meter = provider.get_meter(_METER_NAME)
    _DEFAULT_MANAGER = MetricsManager(meter=meter)
    return _DEFAULT_MANAGER
