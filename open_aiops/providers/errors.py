"""Canonical provider error hierarchy for Open-AIOps.

Defines error classes carrying retryable and failover_eligible flags conforming
to architecture specifications (Section 3.3).
"""

from typing import Any


class ProviderError(Exception):
    """Base exception for all model provider failures.
    
    Attributes:
        message: Human-readable error description without sensitive credentials.
        retryable: True if the same provider should be retried with backoff (INT-14).
        failover_eligible: True if failure qualifies for immediate failover to the next provider (INT-6).
        provider: Name of the provider that raised the error, if known.
        status_code: HTTP status code or internal failure code, if applicable.
    """

    def __init__(
        self,
        message: str = "Provider error occurred",
        *,
        retryable: bool = False,
        failover_eligible: bool = False,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.failover_eligible = failover_eligible
        self.provider = provider
        self.status_code = status_code

    def __str__(self) -> str:
        ctx = f" (provider={self.provider})" if self.provider else ""
        return f"{self.message}{ctx}"

    def __repr__(self) -> str:
        parts = [f"message={self.message!r}"]
        if self.provider:
            parts.append(f"provider={self.provider!r}")
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        parts.append(f"retryable={self.retryable}")
        parts.append(f"failover_eligible={self.failover_eligible}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class RateLimitError(ProviderError):
    """Raised when a provider rejects a request due to rate limiting (HTTP 429).
    
    Transient failure: retryable and failover-eligible.
    """

    def __init__(
        self,
        message: str = "Provider rate limit exceeded",
        *,
        retry_after_s: float | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            retryable=True,
            failover_eligible=True,
            provider=provider,
            status_code=429,
        )
        self.retry_after_s = retry_after_s

    def __repr__(self) -> str:
        parts = [f"message={self.message!r}"]
        if self.provider:
            parts.append(f"provider={self.provider!r}")
        if self.retry_after_s is not None:
            parts.append(f"retry_after_s={self.retry_after_s}")
        parts.append(f"retryable={self.retryable}")
        parts.append(f"failover_eligible={self.failover_eligible}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class ProviderTimeout(ProviderError):
    """Raised when a provider request exceeds the timeout threshold.
    
    Transient failure: retryable and failover-eligible.
    """

    def __init__(
        self,
        message: str = "Provider request timed out",
        *,
        provider: str | None = None,
        status_code: int = 408,
    ) -> None:
        super().__init__(
            message=message,
            retryable=True,
            failover_eligible=True,
            provider=provider,
            status_code=status_code,
        )


class ProviderUnavailable(ProviderError):
    """Raised when a provider returns a server-side error (HTTP 5xx).
    
    Transient failure: retryable and failover-eligible.
    """

    def __init__(
        self,
        message: str = "Provider service unavailable",
        *,
        provider: str | None = None,
        status_code: int = 503,
    ) -> None:
        super().__init__(
            message=message,
            retryable=True,
            failover_eligible=True,
            provider=provider,
            status_code=status_code,
        )


class AuthError(ProviderError):
    """Raised when authentication or authorization fails (HTTP 401, 403).
    
    Permanent failure: neither retryable nor failover-eligible (identical credentials would fail elsewhere).
    """

    def __init__(
        self,
        message: str = "Provider authentication failed",
        *,
        provider: str | None = None,
        status_code: int = 401,
    ) -> None:
        super().__init__(
            message=message,
            retryable=False,
            failover_eligible=False,
            provider=provider,
            status_code=status_code,
        )


class BadRequest(ProviderError):
    """Raised when a request is malformed or parameters are invalid (HTTP 400, 422).
    
    Permanent failure: neither retryable nor failover-eligible.
    """

    def __init__(
        self,
        message: str = "Bad request sent to provider",
        *,
        provider: str | None = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(
            message=message,
            retryable=False,
            failover_eligible=False,
            provider=provider,
            status_code=status_code,
        )
