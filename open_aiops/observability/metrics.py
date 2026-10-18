"""OpenTelemetry metrics utilities for open-aiops.

Provides a lightweight project-level metrics facade for tracking LLM request
durations, request counts, and provider errors using the OpenTelemetry Metrics API.

This module intentionally imports only the OpenTelemetry API (not the SDK)
and configures no exporters or global providers. With no SDK configured, all
instrument recording operations safely operate as no-ops.
"""

from decimal import Decimal
import logging
import math
from typing import Optional, Sequence, Union

from opentelemetry import metrics

logger = logging.getLogger(__name__)

# Instrument names
INSTRUMENT_REQUEST_DURATION = "openaiops.request.duration"
INSTRUMENT_REQUESTS = "openaiops.requests"
INSTRUMENT_PROVIDER_ERRORS = "openaiops.provider.errors"
INSTRUMENT_RETRIES = "openaiops.retries"
INSTRUMENT_FALLBACKS = "openaiops.fallbacks"
INSTRUMENT_TOKENS_INPUT = "openaiops.tokens.input"
INSTRUMENT_TOKENS_OUTPUT = "openaiops.tokens.output"
INSTRUMENT_COST = "openaiops.cost"

# Permitted attribute keys (strictly enforced to avoid high cardinality or sensitive leaks)
ALLOWED_ATTRIBUTE_KEYS = frozenset({"provider", "model", "outcome", "error_type"})

# Supported bounded request outcomes
VALID_OUTCOMES = frozenset({"success", "error", "cancelled"})

# Canonical INT-10 provider error class names
# The accepted error vocabulary consists strictly of these canonical names plus 'unknown'.
# Free-text error messages or raw credentials are strictly prohibited from metric labels.
# Error classes are recognized by class name (__name__) on purpose so that
# metrics.py does not import provider implementations or create unnecessary
# dependency coupling.
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
    except Exception as exc:
        try:
            logger.debug(
                "Failed to normalize error_type: %s", type(exc).__name__
            )
        except Exception:
            pass
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
        self.retries_counter = self._meter.create_counter(
            name=INSTRUMENT_RETRIES,
            unit="{retry}",
            description="Total count of request retry attempts",
        )
        self.fallbacks_counter = self._meter.create_counter(
            name=INSTRUMENT_FALLBACKS,
            unit="{fallback}",
            description="Total count of provider fallbacks triggered",
        )
        self.tokens_input_histogram = self._meter.create_histogram(
            name=INSTRUMENT_TOKENS_INPUT,
            unit="{token}",
            description="Distribution of input prompt tokens per attempt",
        )
        self.tokens_output_histogram = self._meter.create_histogram(
            name=INSTRUMENT_TOKENS_OUTPUT,
            unit="{token}",
            description="Distribution of output completion tokens per attempt",
        )
        self.cost_counter = self._meter.create_counter(
            name=INSTRUMENT_COST,
            unit="",
            description="Total estimated cost of LLM operations",
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
        except Exception as exc:
            try:
                logger.debug(
                    "Failed to record request duration: %s", type(exc).__name__
                )
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
        except Exception as exc:
            try:
                logger.debug(
                    "Failed to record request count: %s", type(exc).__name__
                )
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
        except Exception as exc:
            try:
                logger.debug(
                    "Failed to record request: %s", type(exc).__name__
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
        except Exception as exc:
            try:
                logger.debug(
                    "Failed to record provider error: %s", type(exc).__name__
                )
            except Exception:
                pass

    def record_retry(self, provider: str) -> None:
        """Increment the retry counter for a provider.

        Args:
            provider: LLM provider name.
        """
        try:
            provider_clean = _validate_identifier(provider)
            if provider_clean is None:
                return

            attributes = {"provider": provider_clean}
            self.retries_counter.add(1, attributes=attributes)
        except Exception as exc:
            try:
                logger.debug("Failed to record retry: %s", type(exc).__name__)
            except Exception:
                pass

    def record_fallback(self, provider: str) -> None:
        """Increment the fallback counter for a failing provider.

        Args:
            provider: LLM provider name from which fallback was triggered.
        """
        try:
            provider_clean = _validate_identifier(provider)
            if provider_clean is None:
                return

            attributes = {"provider": provider_clean}
            self.fallbacks_counter.add(1, attributes=attributes)
        except Exception as exc:
            try:
                logger.debug("Failed to record fallback: %s", type(exc).__name__)
            except Exception:
                pass

    def record_tokens(
        self,
        provider: str,
        model: str,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:
        """Record input and output token distributions per attempt.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            input_tokens: Optional count of input prompt tokens.
            output_tokens: Optional count of output completion tokens.
        """
        try:
            provider_clean = _validate_identifier(provider)
            model_clean = _validate_identifier(model)
            if provider_clean is None or model_clean is None:
                return

            attributes = {
                "provider": provider_clean,
                "model": model_clean,
            }

            if (
                isinstance(input_tokens, (int, float))
                and not isinstance(input_tokens, bool)
                and not math.isnan(input_tokens)
                and not math.isinf(input_tokens)
                and input_tokens >= 0
            ):
                self.tokens_input_histogram.record(
                    float(input_tokens), attributes=attributes
                )

            if (
                isinstance(output_tokens, (int, float))
                and not isinstance(output_tokens, bool)
                and not math.isnan(output_tokens)
                and not math.isinf(output_tokens)
                and output_tokens >= 0
            ):
                self.tokens_output_histogram.record(
                    float(output_tokens), attributes=attributes
                )
        except Exception as exc:
            try:
                logger.debug("Failed to record tokens: %s", type(exc).__name__)
            except Exception:
                pass

    def record_cost(
        self,
        provider: str,
        model: str,
        cost: Union[Decimal, float, int],
    ) -> None:
        """Increment the total cost counter for a provider and model.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            cost: Cost amount (Decimal, float, or int; must be finite and non-negative).
        """
        try:
            provider_clean = _validate_identifier(provider)
            model_clean = _validate_identifier(model)
            if provider_clean is None or model_clean is None:
                return

            if cost is None or isinstance(cost, bool):
                return

            cost_float = float(cost)
            if math.isnan(cost_float) or math.isinf(cost_float) or cost_float < 0:
                return

            attributes = {
                "provider": provider_clean,
                "model": model_clean,
            }
            self.cost_counter.add(cost_float, attributes=attributes)
        except Exception as exc:
            try:
                logger.debug("Failed to record cost: %s", type(exc).__name__)
            except Exception:
                pass
