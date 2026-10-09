"""OpenTelemetry metrics utilities for open-aiops.

Provides a lightweight project-level metrics facade for tracking LLM request
durations, request counts, and provider errors using the OpenTelemetry Metrics API.

This module intentionally imports only the OpenTelemetry API (not the SDK)
and configures no exporters or global providers. With no SDK configured, all
instrument recording operations safely operate as no-ops.
"""

import math
from typing import Optional, Union

from opentelemetry import metrics

# Instrument names
INSTRUMENT_REQUEST_DURATION = "openaiops.request.duration"
INSTRUMENT_REQUESTS = "openaiops.requests"
INSTRUMENT_PROVIDER_ERRORS = "openaiops.provider.errors"

# Permitted attribute keys (strictly enforced to avoid high cardinality or sensitive leaks)
ALLOWED_ATTRIBUTE_KEYS = frozenset({"provider", "model", "outcome", "error_type"})

# Bounded outcomes
VALID_OUTCOMES = frozenset({"success", "error"})


def get_meter(name: Optional[str] = None) -> metrics.Meter:
    """Obtain an OpenTelemetry Meter from the registered provider.

    Args:
        name: Optional meter name. Defaults to the module namespace.

    Returns:
        metrics.Meter instance (defaults to NoOpMeter if no SDK is configured).
    """
    return metrics.get_meter(name or __name__)


class MetricsRecorder:
    """Metrics facade wrapping OpenTelemetry instruments for LLM operations.

    Tracks request duration, request totals, and provider errors with strictly
    constrained low-cardinality attributes.
    """

    def __init__(self, meter: Optional[metrics.Meter] = None) -> None:
        """Initialize metrics instruments using the provided or default Meter.

        Args:
            meter: Optional OpenTelemetry Meter. If None, the default meter
                is retrieved via `metrics.get_meter(__name__)`.
        """
        self._meter = meter if meter is not None else get_meter()

        self.duration_histogram = self._meter.create_histogram(
            name=INSTRUMENT_REQUEST_DURATION,
            unit="s",
            description="Duration of completed LLM requests in seconds",
        )
        self.requests_counter = self._meter.create_counter(
            name=INSTRUMENT_REQUESTS,
            unit="{request}",
            description="Total count of completed LLM requests",
        )
        self.errors_counter = self._meter.create_counter(
            name=INSTRUMENT_PROVIDER_ERRORS,
            unit="{error}",
            description="Total count of provider errors",
        )

    @property
    def meter(self) -> metrics.Meter:
        """Return the underlying OpenTelemetry Meter."""
        return self._meter

    def _validate_duration(self, duration: float) -> float:
        """Validate that duration is a finite, non-negative number of seconds.

        Args:
            duration: Request duration in seconds.

        Returns:
            Validated float duration.

        Raises:
            ValueError: If duration is negative, NaN, infinite, or not a number.
        """
        if not isinstance(duration, (int, float)):
            raise ValueError(f"Duration must be a numeric value in seconds, got {type(duration).__name__}")
        if math.isnan(duration) or math.isinf(duration):
            raise ValueError(f"Duration must be a finite number of seconds, got {duration}")
        if duration < 0:
            raise ValueError(f"Duration cannot be negative, got {duration}")
        return float(duration)

    def _validate_outcome(self, outcome: str) -> str:
        """Validate and normalize outcome to a bounded set.

        Args:
            outcome: Outcome string ('success' or 'error').

        Returns:
            Normalized lowercase outcome.

        Raises:
            ValueError: If outcome is not in VALID_OUTCOMES.
        """
        if not isinstance(outcome, str):
            raise ValueError(f"Outcome must be a string, got {type(outcome).__name__}")
        normalized = outcome.strip().lower()
        if normalized == "failed":
            normalized = "error"
        if normalized not in VALID_OUTCOMES:
            raise ValueError(
                f"Invalid outcome '{outcome}'. Must be one of {sorted(VALID_OUTCOMES)}"
            )
        return normalized

    def _normalize_error_type(self, error_type: Union[str, type, BaseException]) -> str:
        """Normalize error_type to a stable category identifier.

        Extracts class name if an exception class or instance is provided,
        and sanitizes against raw exception messages or free-form text.

        Args:
            error_type: Stable category string, exception class, or exception instance.

        Returns:
            Sanitized error type string identifier.

        Raises:
            ValueError: If error_type cannot be resolved to a non-empty string.
        """
        if isinstance(error_type, BaseException):
            return type(error_type).__name__
        if isinstance(error_type, type) and issubclass(error_type, BaseException):
            return error_type.__name__
        if isinstance(error_type, str):
            cleaned = error_type.strip()
            if not cleaned:
                raise ValueError("error_type must be a non-empty string or exception")
            # If a free-text message with newlines was accidentally passed, use first token or sanitize
            if "\n" in cleaned or len(cleaned) > 100:
                # Fall back to first token or generic error
                cleaned = cleaned.split()[0][:50]
            return cleaned
        raise ValueError(f"Unsupported error_type type: {type(error_type).__name__}")

    def record_request_duration(
        self,
        provider: str,
        model: str,
        duration: float,
        outcome: str = "success",
    ) -> None:
        """Record request duration in seconds to the duration histogram.

        Args:
            provider: LLM provider name (e.g., 'openai', 'anthropic').
            model: Model identifier (e.g., 'gpt-4o').
            duration: Elapsed time in seconds (must be >= 0 and finite).
            outcome: Bounded outcome ('success' or 'error'). Defaults to 'success'.
        """
        valid_duration = self._validate_duration(duration)
        valid_outcome = self._validate_outcome(outcome)
        provider_clean = str(provider).strip()
        model_clean = str(model).strip()
        if not provider_clean or not model_clean:
            raise ValueError("provider and model must be non-empty strings")

        attributes = {
            "provider": provider_clean,
            "model": model_clean,
            "outcome": valid_outcome,
        }
        self.duration_histogram.record(valid_duration, attributes=attributes)

    def record_request_count(
        self,
        provider: str,
        model: str,
        outcome: str = "success",
    ) -> None:
        """Increment the completed requests counter.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            outcome: Bounded outcome ('success' or 'error'). Defaults to 'success'.
        """
        valid_outcome = self._validate_outcome(outcome)
        provider_clean = str(provider).strip()
        model_clean = str(model).strip()
        if not provider_clean or not model_clean:
            raise ValueError("provider and model must be non-empty strings")

        attributes = {
            "provider": provider_clean,
            "model": model_clean,
            "outcome": valid_outcome,
        }
        self.requests_counter.add(1, attributes=attributes)

    def record_request(
        self,
        provider: str,
        model: str,
        duration: float,
        outcome: str = "success",
    ) -> None:
        """Record a completed request with its duration and outcome.

        Updates both the request duration histogram (in seconds) and increments
        the total requests counter with consistent attributes.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            duration: Request duration in seconds (must be non-negative and finite).
            outcome: Bounded outcome ('success' or 'error'). Defaults to 'success'.
        """
        self.record_request_duration(
            provider=provider, model=model, duration=duration, outcome=outcome
        )
        self.record_request_count(
            provider=provider, model=model, outcome=outcome
        )

    def record_error(
        self,
        provider: str,
        model: str,
        error_type: Union[str, type, BaseException],
    ) -> None:
        """Record a provider error occurrence.

        Increments the provider errors counter with provider, model, and error_type.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            error_type: Stable category string, exception class, or exception instance.
        """
        provider_clean = str(provider).strip()
        model_clean = str(model).strip()
        if not provider_clean or not model_clean:
            raise ValueError("provider and model must be non-empty strings")

        clean_error_type = self._normalize_error_type(error_type)
        attributes = {
            "provider": provider_clean,
            "model": model_clean,
            "error_type": clean_error_type,
        }
        self.errors_counter.add(1, attributes=attributes)

    # Alias for convenience
    record_provider_error = record_error


# Backwards-compatible / ergonomic aliases
MetricsCollector = MetricsRecorder


def get_metrics_recorder(meter: Optional[metrics.Meter] = None) -> MetricsRecorder:
    """Factory helper to obtain a MetricsRecorder instance."""
    return MetricsRecorder(meter=meter)
