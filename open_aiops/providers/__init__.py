"""Provider contract, adapters, and canonical error types."""

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
from open_aiops.providers.openai_compat import (
    OpenAICompatAdapter,
    build_chat_payload,
    build_headers,
    parse_chat_response,
    parse_retry_after,
    resolve_chat_url,
    translate_http_response,
)

__all__ = [
    "AuthError",
    "BadRequest",
    "OpenAICompatAdapter",
    "Provider",
    "ProviderError",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderTimeout",
    "ProviderUnavailable",
    "RateLimitError",
    "Usage",
    "build_chat_payload",
    "build_headers",
    "parse_chat_response",
    "parse_retry_after",
    "resolve_chat_url",
    "translate_http_response",
]
