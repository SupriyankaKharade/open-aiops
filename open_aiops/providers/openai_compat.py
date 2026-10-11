"""OpenAI-compatible async provider adapter for Open-AIOps.

Translates standard ProviderRequest payloads to OpenAI chat completion wire format,
dispatches asynchronous HTTP requests, and normalizes wire responses into standard
ProviderResponse objects.
"""

import time
from typing import Any
import httpx
from pydantic import SecretStr

from open_aiops.providers.base import (
    Provider,
    ProviderRequest,
    ProviderResponse,
    Usage,
)
from open_aiops.providers.errors import (
    AuthError,
    BadRequest,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimitError,
)


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def resolve_chat_url(base_url: str | None = None) -> str:
    """Normalize base URL and resolve chat completions endpoint."""
    url = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def build_headers(
    api_key: SecretStr | str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Construct HTTP headers for OpenAI-compatible APIs without exposing credentials."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    if api_key is not None:
        raw_key = api_key.get_secret_value() if isinstance(api_key, SecretStr) else str(api_key).strip()
        if raw_key:
            headers["Authorization"] = f"Bearer {raw_key}"

    return headers


def build_chat_payload(request: ProviderRequest) -> dict[str, Any]:
    """Translate standard ProviderRequest into OpenAI /v1/chat/completions JSON payload."""
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": request.messages,
    }

    if request.max_output_tokens is not None:
        payload["max_tokens"] = request.max_output_tokens

    if request.temperature is not None:
        payload["temperature"] = request.temperature

    if request.extra_params:
        payload.update(request.extra_params)

    return payload


def parse_retry_after(header_val: str | None) -> float | None:
    """Parse HTTP Retry-After header value into delay in seconds."""
    if not header_val:
        return None
    try:
        return float(header_val.strip())
    except (ValueError, TypeError):
        return None


def translate_http_response(response: httpx.Response, provider_name: str) -> None:
    """Check response status code and raise appropriate canonical ProviderError subclasses."""
    code = response.status_code
    if 200 <= code < 300:
        return

    if code == 429:
        retry_after = parse_retry_after(response.headers.get("Retry-After"))
        raise RateLimitError(
            message=f"Rate limit exceeded (HTTP 429)",
            retry_after_s=retry_after,
            provider=provider_name,
        )

    if code in (401, 403):
        raise AuthError(
            message=f"Authentication or authorization failed (HTTP {code})",
            provider=provider_name,
            status_code=code,
        )

    if code in (400, 422):
        raise BadRequest(
            message=f"Bad request sent to provider (HTTP {code})",
            provider=provider_name,
            status_code=code,
        )

    if code >= 500:
        raise ProviderUnavailable(
            message=f"Provider server error (HTTP {code})",
            provider=provider_name,
            status_code=code,
        )

    raise ProviderError(
        message=f"Provider returned error status (HTTP {code})",
        provider=provider_name,
        status_code=code,
    )


def parse_chat_response(
    data: dict[str, Any],
    latency: float | None = None,
    fallback_model: str = "",
) -> ProviderResponse:
    """Map raw OpenAI wire completion response to standard ProviderResponse."""
    choices = data.get("choices", [])
    text = ""
    if choices and isinstance(choices, list):
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            msg = first_choice.get("message", {})
            text = msg.get("content") or ""

    model = data.get("model") or fallback_model

    usage: Usage | None = None
    wire_usage = data.get("usage")
    if isinstance(wire_usage, dict):
        usage = Usage(
            input_tokens=wire_usage.get("prompt_tokens"),
            output_tokens=wire_usage.get("completion_tokens"),
        )

    return ProviderResponse(
        text=text,
        model=model,
        usage=usage,
        latency=latency,
        raw_response=data,
    )


class OpenAICompatAdapter:
    """Async adapter connecting to any OpenAI-compatible chat completions endpoint.

    Supports OpenAI, Ollama, Groq, vLLM, and other compliant inference servers.
    Conforms to the Provider protocol (§3.3).
    """

    def __init__(
        self,
        name: str = "openai-compatible",
        base_url: str | None = None,
        api_key: SecretStr | str | None = None,
        timeout_s: float = 30.0,
        extra_headers: dict[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.endpoint_url = resolve_chat_url(base_url)
        self.api_key = api_key if isinstance(api_key, SecretStr) else (SecretStr(api_key) if api_key else None)
        self.timeout_s = timeout_s
        self.extra_headers = extra_headers or {}
        self._custom_client = client

    def __repr__(self) -> str:
        # Never leak api_key in string representation (Privacy by default)
        return (
            f"OpenAICompatAdapter(name={self.name!r}, "
            f"endpoint_url={self.endpoint_url!r}, timeout_s={self.timeout_s})"
        )

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        """Execute chat completion request asynchronously against the provider endpoint."""
        payload = build_chat_payload(request)
        headers = build_headers(self.api_key, self.extra_headers)

        start_time = time.perf_counter()

        try:
            if self._custom_client is not None:
                response = await self._custom_client.post(
                    self.endpoint_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_s,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.endpoint_url,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout_s,
                    )
        except httpx.TimeoutException as exc:
            raise ProviderTimeout(
                message=f"Request to {self.name} timed out after {self.timeout_s}s",
                provider=self.name,
            ) from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise ProviderUnavailable(
                message=f"Failed to connect to {self.name} ({type(exc).__name__})",
                provider=self.name,
                status_code=503,
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(
                message=f"HTTP communication error with {self.name} ({type(exc).__name__})",
                provider=self.name,
            ) from exc

        elapsed = time.perf_counter() - start_time
        translate_http_response(response, self.name)
        response_data = response.json()
        return parse_chat_response(response_data, latency=elapsed, fallback_model=request.model)


