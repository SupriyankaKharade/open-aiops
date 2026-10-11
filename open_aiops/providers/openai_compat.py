"""OpenAI-compatible async provider adapter for Open-AIOps.

Translates standard ProviderRequest payloads to OpenAI chat completion wire format,
dispatches asynchronous HTTP requests, and normalizes wire responses into standard
ProviderResponse objects.
"""

from typing import Any
from pydantic import SecretStr

from open_aiops.providers.base import ProviderRequest


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
