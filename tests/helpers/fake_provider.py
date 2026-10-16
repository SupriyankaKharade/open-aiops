"""Fake provider implementation for unit and integration testing."""

from collections.abc import Sequence
from typing import Any

from open_aiops.providers.base import (
    Provider,
    ProviderRequest,
    ProviderResponse,
    Usage,
)


class FakeProvider:
    """A configurable test double implementing the Provider protocol.

    Records call history and returns configured responses or raises configured errors.
    """

    def __init__(
        self,
        name: str = "fake-provider",
        *,
        model: str | None = None,
        responses: Sequence[ProviderResponse | Exception] | None = None,
        default_response: ProviderResponse | None = None,
        response: ProviderResponse | None = None,
        default_text: str = "fake response",
    ) -> None:
        self.name = name
        self.model = model
        self.requests: list[ProviderRequest] = []
        self._responses: list[ProviderResponse | Exception] = list(responses) if responses else []
        self._call_count = 0
        self.default_response = response if response is not None else default_response
        self.default_text = default_text

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def calls(self) -> list[ProviderRequest]:
        """Convenience alias for requests."""
        return self.requests

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Record the request and return a response or raise an exception."""
        self.requests.append(request)
        self._call_count += 1

        if self._responses:
            outcome = self._responses.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        if self.default_response is not None:
            return self.default_response

        target_model = self.model or request.model
        return ProviderResponse(
            text=self.default_text,
            model=target_model,
            usage=Usage(input_tokens=10, output_tokens=10),
            latency=0.01,
        )
