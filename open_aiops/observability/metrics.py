"""OpenTelemetry metrics utilities for open-aiops.

Provides a lightweight project-level metrics facade for tracking LLM request
durations, request counts, and provider errors using the OpenTelemetry Metrics API.

This module intentionally imports only the OpenTelemetry API (not the SDK)
and configures no exporters or global providers. With no SDK configured, all
instrument recording operations safely operate as no-ops.
"""

import math
from typing import Optional, Sequence, Union

from opentelemetry import metrics

# Instrument names
INSTRUMENT_REQUEST_DURATION = "openaiops.request.duration"
INSTRUMENT_REQUESTS = "openaiops.requests"
INSTRUMENT_PROVIDER_ERRORS = "openaiops.provider.errors"

# Permitted attribute keys (strictly enforced to avoid high cardinality or sensitive leaks)
ALLOWED_ATTRIBUTE_KEYS = frozenset({"provider", "model", "outcome", "error_type"})

# Supported bounded request outcomes
VALID_OUTCOMES = frozenset({"success", "error", "cancelled"})

# Canonical INT-10 provider error class names
# The accepted error vocabulary consists strictly of these canonical names plus 'unknown'.
# Free-text error messages or raw credentials are strictly prohibited from metric labels.
CANONICAL_PROVIDER_ERRORS = frozenset({
    "ProviderError",
    "RateLimitError",
    "ProviderTimeout",
    "ProviderUnavailable",
    "AuthError",
    "BadRequest",
})
CANONICAL_ERROR_TYPES = CANONICAL_PROVIDER_ERRORS | {"unknown"}

# Explicit advisory histogram bucket boundaries tailored for LLM request latency in seconds (~0.1 to 60s)
DEFAULT_LATENCY_HISTOGRAM_BOUNDARIES = (
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    15.0,
    30.0,
    60.0,
)


def get_meter(name: Optional[str] = None) -> metrics.Meter:
    """Obtain an OpenTelemetry Meter from the registered provider.

    Args:
        name: Optional meter name. Defaults to the module namespace.

    Returns:
        metrics.Meter instance (defaults to NoOpMeter if no SDK is configured).
    """
    return metrics.get_meter(name or __name__)


def _is_valid_duration(duration: object) -> bool:
    """Validate that duration is a finite, non-negative number of seconds.

    Args:
        duration: Request duration candidate.

    Returns:
        True if duration is a valid non-negative, finite number, False otherwise.
    """
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        return False
    if math.isnan(duration) or math.isinf(duration):
        return False
    if duration < 0:
        return False
    return True


def _normalize_outcome(outcome: object) -> str:
    """Validate and normalize outcome to the supported bounded vocabulary.

    Supported outcomes are 'success', 'error', and 'cancelled'. Any unrecognized
    outcome string or non-string input safely normalizes to 'error'.

    Args:
        outcome: Outcome candidate.

    Returns:
        'success', 'error', or 'cancelled'.
    """
    if isinstance(outcome, str):
        normalized = outcome.strip().lower()
        if normalized in VALID_OUTCOMES:
            return normalized
    return "error"


def _normalize_error_type(error_type: Union[str, type, BaseException, object]) -> str:
    """Normalize error_type to the canonical provider error vocabulary or 'unknown'.

    Maps recognized provider exception classes, instances, and canonical class-name
    strings to their canonical class name. Unrecognized classes, instances, or
    strings map to 'unknown'. Arbitrary free-text, slugs, or credentials never
    become metric attributes.

    Args:
        error_type: Canonical category string, exception class, or exception instance.

    Returns:
        Canonical error type string identifier from CANONICAL_ERROR_TYPES.
    """
    try:
        if isinstance(error_type, BaseException):
            for cls in type(error_type).__mro__:
                if cls.__name__ in CANONICAL_PROVIDER_ERRORS:
                    return cls.__name__
            return "unknown"
        if isinstance(error_type, type) and issubclass(error_type, BaseException):
            for cls in error_type.__mro__:
                if cls.__name__ in CANONICAL_PROVIDER_ERRORS:
                    return cls.__name__
            return "unknown"
        if isinstance(error_type, str):
            cleaned = error_type.strip()
            if cleaned in CANONICAL_PROVIDER_ERRORS or cleaned == "unknown":
                return cleaned
            return "unknown"
    except Exception:
        return "unknown"
    return "unknown"


def _validate_identifier(val: object) -> Optional[str]:
    """Validate that provider or model is a non-empty string.

    Args:
        val: Provider or model identifier candidate.

    Returns:
        Cleaned non-empty string or None.
    """
    if not isinstance(val, str):
        return None
    cleaned = val.strip()
    return cleaned if cleaned else None


class MetricsRecorder:
    """Metrics facade wrapping OpenTelemetry instruments for LLM operations.

    Tracks request duration, request totals, and provider errors with strictly
    constrained low-cardinality attributes. All public recording methods are
    non-throwing to ensure telemetry never interrupts application traffic.
    """

    def __init__(
        self,
        meter: Optional[metrics.Meter] = None,
        histogram_boundaries: Optional[Sequence[float]] = None,
    ) -> None:
        """Initialize metrics instruments using the provided or default Meter.

        Args:
            meter: Optional OpenTelemetry Meter. If None, the default meter
                is retrieved via `metrics.get_meter(__name__)`.
            histogram_boundaries: Optional explicit advisory histogram bucket
                boundaries in seconds. Defaults to `DEFAULT_LATENCY_HISTOGRAM_BOUNDARIES`.
        """
        self._meter = meter if meter is not None else get_meter()
        boundaries = (
            histogram_boundaries
            if histogram_boundaries is not None
            else DEFAULT_LATENCY_HISTOGRAM_BOUNDARIES
        )

        self.duration_histogram = self._meter.create_histogram(
            name=INSTRUMENT_REQUEST_DURATION,
            unit="s",
            description="Duration of completed LLM requests in seconds",
            explicit_bucket_boundaries_advisory=list(boundaries),
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

    def record_request_duration(
        self,
        provider: str,
        model: str,
        duration: float,
        outcome: str = "success",
    ) -> None:
        """Record request duration in seconds to the duration histogram.

        Invalid, negative, NaN, infinite, or non-numeric durations are skipped
        rather than raising ValueError. Unrecognized outcomes safely normalize
        to 'error'. Empty or malformed provider/model identifiers are skipped safely.

        Args:
            provider: LLM provider name (e.g., 'openai', 'anthropic').
            model: Model identifier (e.g., 'gpt-4o').
            duration: Elapsed time in seconds (must be finite and non-negative).
            outcome: Request outcome ('success', 'error', or 'cancelled'). Defaults to 'success'.
        """
        try:
            provider_clean = _validate_identifier(provider)
            model_clean = _validate_identifier(model)
            if provider_clean is None or model_clean is None:
                return

            if not _is_valid_duration(duration):
                return

            valid_outcome = _normalize_outcome(outcome)
            attributes = {
                "provider": provider_clean,
                "model": model_clean,
                "outcome": valid_outcome,
            }
            self.duration_histogram.record(float(duration), attributes=attributes)
        except Exception:
            pass

    def record_request_count(
        self,
        provider: str,
        model: str,
        outcome: str = "success",
    ) -> None:
        """Increment the completed requests counter.

        Unrecognized outcomes safely normalize to 'error'. Empty or malformed
        provider/model identifiers are skipped safely.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            outcome: Request outcome ('success', 'error', or 'cancelled'). Defaults to 'success'.
        """
        try:
            provider_clean = _validate_identifier(provider)
            model_clean = _validate_identifier(model)
            if provider_clean is None or model_clean is None:
                return

            valid_outcome = _normalize_outcome(outcome)
            attributes = {
                "provider": provider_clean,
                "model": model_clean,
                "outcome": valid_outcome,
            }
            self.requests_counter.add(1, attributes=attributes)
        except Exception:
            pass

    def record_request(
        self,
        provider: str,
        model: str,
        duration: float,
        outcome: str = "success",
    ) -> None:
        """Record a completed request with its duration and outcome.

        Updates the request duration histogram (in seconds) and increments
        the total requests counter with consistent attributes.

        If duration is invalid, negative, NaN, or non-finite, the duration
        Histogram observation is skipped, but valid request-count recording
        is preserved.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            duration: Request duration in seconds.
            outcome: Request outcome ('success', 'error', or 'cancelled'). Defaults to 'success'.
        """
        try:
            self.record_request_duration(
                provider=provider, model=model, duration=duration, outcome=outcome
            )
            self.record_request_count(
                provider=provider, model=model, outcome=outcome
            )
        except Exception:
            pass

    def record_error(
        self,
        provider: str,
        model: str,
        error_type: Union[str, type, BaseException, object],
    ) -> None:
        """Record a provider error occurrence.

        Increments the provider errors counter with provider, model, and error_type.
        error_type is normalized to the canonical INT-10 error vocabulary or 'unknown'.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            error_type: Canonical error string, exception class, or exception instance.
        """
        try:
            provider_clean = _validate_identifier(provider)
            model_clean = _validate_identifier(model)
            if provider_clean is None or model_clean is None:
                return

            clean_error_type = _normalize_error_type(error_type)
            attributes = {
                "provider": provider_clean,
                "model": model_clean,
                "error_type": clean_error_type,
            }
            self.errors_counter.add(1, attributes=attributes)
        except Exception:
            pass
