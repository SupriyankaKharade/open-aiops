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
    resolve_chat_url,
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
    "resolve_chat_url",
]
