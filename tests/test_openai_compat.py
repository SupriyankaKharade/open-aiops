"""Unit tests for the OpenAI-compatible async adapter (INT-11)."""

import asyncio
import json
import httpx
import pytest
from pydantic import SecretStr

from open_aiops.governance.fallbacks import call_with_failover
from open_aiops.governance.router import ModelRouter, ProviderConfig, RouteRequest
from open_aiops.providers.base import Provider, ProviderRequest, Usage
from open_aiops.providers.errors import (
    AuthError,
    BadRequest,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimitError,
)
from open_aiops.providers.openai_compat import (
    DEFAULT_OPENAI_BASE_URL,
    OpenAICompatAdapter,
    build_chat_payload,
    build_headers,
    parse_chat_response,
    parse_retry_after,
    resolve_chat_url,
    translate_http_response,
)


# ---------------------------------------------------------------------------
# URL and Header Construction Tests
# ---------------------------------------------------------------------------


def test_resolve_chat_url():
    """Verify endpoint URL resolution for default and custom endpoints."""
    assert resolve_chat_url() == f"{DEFAULT_OPENAI_BASE_URL}/chat/completions"
    assert resolve_chat_url("http://localhost:11434/v1") == "http://localhost:11434/v1/chat/completions"
    assert (
        resolve_chat_url("http://localhost:11434/v1/chat/completions")
        == "http://localhost:11434/v1/chat/completions"
    )
    assert resolve_chat_url("https://api.groq.com/openai/v1/") == "https://api.groq.com/openai/v1/chat/completions"


def test_build_headers_with_and_without_credentials():
    """Verify HTTP headers generation and secret handling."""
    # Without credentials
    h_none = build_headers()
    assert h_none["Content-Type"] == "application/json"
    assert "Authorization" not in h_none

    # With SecretStr credentials
    secret = SecretStr("sk-secret-test-key-12345")
    h_secret = build_headers(api_key=secret, extra_headers={"X-Custom": "val"})
    assert h_secret["Authorization"] == "Bearer sk-secret-test-key-12345"
    assert h_secret["X-Custom"] == "val"

    # With plain string credentials
    h_str = build_headers(api_key="sk-plain-key")
    assert h_str["Authorization"] == "Bearer sk-plain-key"


def test_build_chat_payload_translation():
    """Verify ProviderRequest translated to OpenAI wire payload."""
    req = ProviderRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Explain gravity"}],
        max_output_tokens=250,
        temperature=0.4,
        extra_params={"top_p": 0.9, "user": "test-user"},
    )
    payload = build_chat_payload(req)
    assert payload["model"] == "gpt-4o"
    assert payload["messages"] == [{"role": "user", "content": "Explain gravity"}]
    assert payload["max_tokens"] == 250
    assert payload["temperature"] == 0.4
    assert payload["top_p"] == 0.9
    assert payload["user"] == "test-user"


def test_parse_retry_after():
    """Verify Retry-After header parsing for seconds delays."""
    assert parse_retry_after("120") == 120.0
    assert parse_retry_after("15.5") == 15.5
    assert parse_retry_after(None) is None
    assert parse_retry_after("invalid-date-or-string") is None


def test_parse_chat_response_wire_mapping():
    """Verify wire usage and text translation into ProviderResponse."""
    raw = {
        "model": "gpt-4o-2024-05-13",
        "choices": [{"message": {"role": "assistant", "content": "Gravity pulls."}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
    }
    resp = parse_chat_response(raw, latency=0.15)
    assert resp.text == "Gravity pulls."
    assert resp.model == "gpt-4o-2024-05-13"
    assert resp.latency == 0.15
    assert resp.usage is not None
    assert resp.usage.input_tokens == 12
    assert resp.usage.output_tokens == 4
    assert resp.usage.total_tokens == 16

    # Response with missing usage or choices
    empty_raw = {"model": "custom-model"}
    empty_resp = parse_chat_response(empty_raw)
    assert empty_resp.text == ""
    assert empty_resp.usage is None


def test_adapter_protocol_conformance_and_privacy():
    """Verify adapter conforms to Provider Protocol and preserves privacy in repr."""
    adapter = OpenAICompatAdapter(
        name="test-adapter",
        api_key=SecretStr("sk-super-secret-key-999"),
        base_url="http://localhost:11434/v1",
    )
    assert isinstance(adapter, Provider)

    # Privacy check: repr must not contain secret key
    repr_str = repr(adapter)
    assert "sk-super-secret" not in repr_str
    assert "test-adapter" in repr_str


# ---------------------------------------------------------------------------
# Mocked Async HTTP Execution & Error Translation Tests
# ---------------------------------------------------------------------------


def make_mock_client(handler) -> httpx.AsyncClient:
    """Helper creating an AsyncClient backed by MockTransport."""
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


def test_complete_successful_200():
    """Verify successful 200 completion execution through mock transport."""
    def handler(request: httpx.Request) -> httpx.Response:
        data = {
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Hello there!"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        return httpx.Response(200, json=data)

    client = make_mock_client(handler)
    adapter = OpenAICompatAdapter(name="mock-primary", client=client)
    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    resp = asyncio.run(adapter.complete(req))
    assert resp.text == "Hello there!"
    assert resp.model == "gpt-4o"
    assert resp.usage.input_tokens == 10
    assert resp.usage.output_tokens == 5
    assert resp.latency is not None and resp.latency >= 0


def test_complete_429_rate_limit_error():
    """Verify HTTP 429 raises RateLimitError with parsed Retry-After header."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "45"}, text="Rate limit exceeded")

    client = make_mock_client(handler)
    adapter = OpenAICompatAdapter(name="mock-groq", client=client)
    req = ProviderRequest(model="llama-3-70b", messages=[{"role": "user", "content": "Hi"}])

    with pytest.raises(RateLimitError) as exc_info:
        asyncio.run(adapter.complete(req))

    err = exc_info.value
    assert err.status_code == 429
    assert err.retry_after_s == 45.0
    assert not hasattr(err, "retry_after")
    assert err.retryable is True
    assert err.failover_eligible is True
    assert err.provider == "mock-groq"


def test_complete_timeout_error():
    """Verify timeout translates to ProviderTimeout with retryable & failover flags."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out", request=request)

    client = make_mock_client(handler)
    adapter = OpenAICompatAdapter(name="mock-slow", client=client)
    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    with pytest.raises(ProviderTimeout) as exc_info:
        asyncio.run(adapter.complete(req))

    err = exc_info.value
    assert err.retryable is True
    assert err.failover_eligible is True
    assert err.provider == "mock-slow"


def test_complete_connection_failure():
    """Verify connection failure translates to ProviderUnavailable."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Failed to connect", request=request)

    client = make_mock_client(handler)
    adapter = OpenAICompatAdapter(name="mock-down", client=client)
    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    with pytest.raises(ProviderUnavailable) as exc_info:
        asyncio.run(adapter.complete(req))

    err = exc_info.value
    assert err.retryable is True
    assert err.failover_eligible is True
    assert err.status_code == 503
    assert err.provider == "mock-down"


def test_complete_auth_error_401():
    """Verify HTTP 401 raises AuthError (neither retryable nor failover-eligible)."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized: Invalid API Key")

    client = make_mock_client(handler)
    adapter = OpenAICompatAdapter(name="mock-auth", client=client)
    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    with pytest.raises(AuthError) as exc_info:
        asyncio.run(adapter.complete(req))

    err = exc_info.value
    assert err.status_code == 401
    assert err.retryable is False
    assert err.failover_eligible is False
    assert err.provider == "mock-auth"


def test_complete_bad_request_400():
    """Verify HTTP 400 raises BadRequest (neither retryable nor failover-eligible)."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request: context length exceeded")

    client = make_mock_client(handler)
    adapter = OpenAICompatAdapter(name="mock-bad", client=client)
    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    with pytest.raises(BadRequest) as exc_info:
        asyncio.run(adapter.complete(req))

    err = exc_info.value
    assert err.status_code == 400
    assert err.retryable is False
    assert err.failover_eligible is False


def test_complete_server_unavailable_500():
    """Verify HTTP 500 raises ProviderUnavailable (retryable & failover-eligible)."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = make_mock_client(handler)
    adapter = OpenAICompatAdapter(name="mock-500", client=client)
    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    with pytest.raises(ProviderUnavailable) as exc_info:
        asyncio.run(adapter.complete(req))

    err = exc_info.value
    assert err.status_code == 500
    assert err.retryable is True
    assert err.failover_eligible is True


def test_complete_unexpected_status_raises_provider_error():
    """Verify other unmapped HTTP status codes raise generic ProviderError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(418, text="I'm a teapot")

    client = make_mock_client(handler)
    adapter = OpenAICompatAdapter(name="mock-teapot", client=client)
    req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

    with pytest.raises(ProviderError) as exc_info:
        asyncio.run(adapter.complete(req))

    assert exc_info.value.status_code == 418


# ---------------------------------------------------------------------------
# End-to-End Failover Integration with OpenAICompatAdapter
# ---------------------------------------------------------------------------


def test_failover_integration_with_adapter():
    """Verify that an adapter returning 429 fails over transparently to a secondary adapter."""
    # Adapter 1: answers HTTP 429
    def handler_1(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "10"}, text="Rate limited")

    # Adapter 2: answers HTTP 200
    def handler_2(request: httpx.Request) -> httpx.Response:
        data = {
            "model": "backup-model",
            "choices": [{"message": {"content": "Fallback served successfully!"}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 20},
        }
        return httpx.Response(200, json=data)

    client_1 = make_mock_client(handler_1)
    client_2 = make_mock_client(handler_2)

    adapter_primary = OpenAICompatAdapter(name="primary", client=client_1)
    adapter_backup = OpenAICompatAdapter(name="backup", client=client_2)

    adapters = {
        "primary": adapter_primary,
        "backup": adapter_backup,
    }

    p_primary = ProviderConfig(
        name="primary",
        model="gpt-4o",
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.03,
        expected_latency_ms=200.0,
    )
    p_backup = ProviderConfig(
        name="backup",
        model="backup-model",
        cost_per_1k_input_tokens=0.02,
        cost_per_1k_output_tokens=0.04,
        expected_latency_ms=300.0,
    )

    router = ModelRouter([p_primary, p_backup])
    route_req = RouteRequest(input_tokens=50, max_output_tokens=20)
    provider_req = ProviderRequest(model="gpt-4o", messages=[{"role": "user", "content": "hello"}])

    async def _execute_call(p: ProviderConfig):
        return await adapters[p.name].complete(provider_req)

    result = asyncio.run(call_with_failover(router, route_req, _execute_call))
    assert result.text == "Fallback served successfully!"
    assert result.model == "backup-model"
    assert result.usage.total_tokens == 70
