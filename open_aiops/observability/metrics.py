"""OpenTelemetry metrics utilities for Open-AIOps.

Implements INT-4 metrics instruments according to architecture section 3.8.
"""

from __future__ import annotations

import logging
from typing import Optional
from opentelemetry import metrics
from opentelemetry.metrics import Counter, Histogram, Meter

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
