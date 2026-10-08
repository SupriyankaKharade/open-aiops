"""Tests for the core configuration module."""

import json

import pytest
from pydantic import SecretStr, ValidationError

from open_aiops.core.config import ProviderConfig, Settings, get_settings


def test_default_fallback_values_are_properly_populated(monkeypatch):
    """Verify default settings and fallback values across Settings and ProviderConfig."""
    for env_key in [
        "OPEN_AIOPS_APP_NAME",
        "OPEN_AIOPS_DEBUG",
        "OPEN_AIOPS_LOG_LEVEL",
        "OPEN_AIOPS_DEFAULT_TIMEOUT_MS",
        "OPEN_AIOPS_PROVIDERS",
        "OPEN_AIOPS_API_KEYS",
    ]:
        monkeypatch.delenv(env_key, raising=False)

    settings = Settings(_env_file=None)
    assert settings.app_name == "open-aiops"
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.default_timeout_ms == 30000.0
    assert settings.providers == []
    assert settings.api_keys == {}

    provider = ProviderConfig(
        name="test-provider",
        model="test-model",
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.02,
        expected_latency_ms=100.0,
    )
    assert provider.name == "test-provider"
    assert provider.model == "test-model"
    assert provider.cost_per_1k_input_tokens == 0.01
    assert provider.cost_per_1k_output_tokens == 0.02
    assert provider.expected_latency_ms == 100.0


def test_settings_from_environment_variables(monkeypatch):
    """Settings correctly reads values from OPEN_AIOPS_ prefixed environment variables."""
    monkeypatch.setenv("OPEN_AIOPS_APP_NAME", "custom-aiops")
    monkeypatch.setenv("OPEN_AIOPS_DEBUG", "true")
    monkeypatch.setenv("OPEN_AIOPS_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("OPEN_AIOPS_DEFAULT_TIMEOUT_MS", "45000.5")

    settings = Settings(_env_file=None)
    assert settings.app_name == "custom-aiops"
    assert settings.debug is True
    assert settings.log_level == "DEBUG"
    assert settings.default_timeout_ms == 45000.5


def test_unprefixed_env_vars_ignored(monkeypatch):
    """Generic unprefixed environment variables (e.g. DEBUG) do not override settings."""
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("APP_NAME", "unprefixed-name")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)
    assert settings.app_name == "open-aiops"
    assert settings.debug is False
    assert settings.log_level == "INFO"


def test_settings_from_dotenv_file(tmp_path, monkeypatch):
    """Settings loads values from a specified .env file with OPEN_AIOPS_ prefix."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPEN_AIOPS_APP_NAME=dotenv-app\n"
        "OPEN_AIOPS_DEBUG=true\n"
        "OPEN_AIOPS_LOG_LEVEL=WARNING\n"
        "OPEN_AIOPS_DEFAULT_TIMEOUT_MS=15000.0\n"
    )

    settings = Settings(_env_file=env_file)
    assert settings.app_name == "dotenv-app"
    assert settings.debug is True
    assert settings.log_level == "WARNING"
    assert settings.default_timeout_ms == 15000.0


def test_env_var_overrides_dotenv(tmp_path, monkeypatch):
    """Environment variables take precedence over .env file values."""
    env_file = tmp_path / ".env"
    env_file.write_text("OPEN_AIOPS_APP_NAME=from-dotenv\nOPEN_AIOPS_DEBUG=false\n")

    monkeypatch.setenv("OPEN_AIOPS_APP_NAME", "from-env")
    monkeypatch.setenv("OPEN_AIOPS_DEBUG", "true")

    settings = Settings(_env_file=env_file)
    assert settings.app_name == "from-env"
    assert settings.debug is True


def test_invalid_log_level_raises_validation_error(monkeypatch):
    """OPEN_AIOPS_LOG_LEVEL with an invalid level raises ValidationError."""
    monkeypatch.setenv("OPEN_AIOPS_LOG_LEVEL", "INVALID")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_negative_timeout_raises_validation_error(monkeypatch):
    """OPEN_AIOPS_DEFAULT_TIMEOUT_MS <= 0 raises ValidationError."""
    monkeypatch.setenv("OPEN_AIOPS_DEFAULT_TIMEOUT_MS", "-5.0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

    monkeypatch.setenv("OPEN_AIOPS_DEFAULT_TIMEOUT_MS", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_non_numeric_timeout_raises_validation_error(monkeypatch):
    """OPEN_AIOPS_DEFAULT_TIMEOUT_MS set to a non-numeric string raises ValidationError."""
    monkeypatch.setenv("OPEN_AIOPS_DEFAULT_TIMEOUT_MS", "abc")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_provider_config_valid():
    """ProviderConfig instantiates with valid router-compatible data."""
    provider = ProviderConfig(
        name="gpt-4",
        model="gpt-4-turbo",
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.03,
        expected_latency_ms=250.0,
    )
    assert provider.name == "gpt-4"
    assert provider.model == "gpt-4-turbo"
    assert provider.cost_per_1k_input_tokens == 0.01
    assert provider.cost_per_1k_output_tokens == 0.03
    assert provider.expected_latency_ms == 250.0


def test_provider_config_negative_cost_rejected():
    """Negative token costs are rejected."""
    with pytest.raises(ValidationError):
        ProviderConfig(
            name="test-p",
            model="test-m",
            cost_per_1k_input_tokens=-0.01,
            cost_per_1k_output_tokens=0.02,
            expected_latency_ms=100.0,
        )

    with pytest.raises(ValidationError):
        ProviderConfig(
            name="test-p",
            model="test-m",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=-0.02,
            expected_latency_ms=100.0,
        )


def test_provider_config_zero_latency_rejected():
    """expected_latency_ms <= 0 is rejected."""
    with pytest.raises(ValidationError):
        ProviderConfig(
            name="test-p",
            model="test-m",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.02,
            expected_latency_ms=0,
        )

    with pytest.raises(ValidationError):
        ProviderConfig(
            name="test-p",
            model="test-m",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.02,
            expected_latency_ms=-50.0,
        )


def test_provider_config_empty_name_or_model_rejected():
    """Empty or whitespace-only name and model are rejected."""
    with pytest.raises(ValidationError):
        ProviderConfig(
            name="",
            model="valid-model",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.02,
            expected_latency_ms=100.0,
        )

    with pytest.raises(ValidationError):
        ProviderConfig(
            name="   ",
            model="valid-model",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.02,
            expected_latency_ms=100.0,
        )


def test_providers_parsed_from_json_env(monkeypatch):
    """OPEN_AIOPS_PROVIDERS JSON string is parsed into list of ProviderConfig instances."""
    providers_data = [
        {
            "name": "gpt-4",
            "model": "gpt-4-turbo",
            "cost_per_1k_input_tokens": 0.01,
            "cost_per_1k_output_tokens": 0.03,
            "expected_latency_ms": 250.0,
        },
        {
            "name": "claude",
            "model": "claude-3-haiku",
            "cost_per_1k_input_tokens": 0.00025,
            "cost_per_1k_output_tokens": 0.00125,
            "expected_latency_ms": 120.0,
        },
    ]
    monkeypatch.setenv("OPEN_AIOPS_PROVIDERS", json.dumps(providers_data))

    settings = Settings(_env_file=None)
    assert len(settings.providers) == 2
    assert isinstance(settings.providers[0], ProviderConfig)
    assert settings.providers[0].name == "gpt-4"
    assert settings.providers[1].name == "claude"


def test_malformed_providers_json_raises(monkeypatch):
    """OPEN_AIOPS_PROVIDERS set to malformed JSON raises an error."""
    monkeypatch.setenv("OPEN_AIOPS_PROVIDERS", "not-valid-json")
    with pytest.raises((ValidationError, ValueError)):
        Settings(_env_file=None)


def test_api_keys_secret_str_security(monkeypatch):
    """API keys typed as SecretStr do not leak in repr, str, or model_dump_json."""
    secret_val = "sensitive-api-token-xyz-12345"
    monkeypatch.setenv("OPEN_AIOPS_API_KEYS", json.dumps({"openai": secret_val}))

    settings = Settings(_env_file=None)
    assert "openai" in settings.api_keys
    assert isinstance(settings.api_keys["openai"], SecretStr)
    assert settings.api_keys["openai"].get_secret_value() == secret_val

    repr_output = repr(settings)
    str_output = str(settings)
    json_output = settings.model_dump_json()

    assert secret_val not in repr_output
    assert secret_val not in str_output
    assert secret_val not in json_output
    assert "**********" in repr_output


def test_get_settings_returns_cached_instance():
    """get_settings() returns the same cached instance across calls."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
