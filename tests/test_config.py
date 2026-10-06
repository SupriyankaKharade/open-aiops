"""Tests for the core configuration module."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from open_aiops.core.config import ProviderConfig, Settings, get_settings


def test_default_settings_load_without_env(monkeypatch):
    """Settings instantiates with all defaults when no env vars are set."""
    # Ensure relevant environment variables are cleared
    for env_key in ["APP_NAME", "DEBUG", "LOG_LEVEL", "DEFAULT_TIMEOUT_MS", "PROVIDERS", "API_KEYS"]:
        monkeypatch.delenv(env_key, raising=False)

    settings = Settings(_env_file=None)
    assert settings.app_name == "open-aiops"
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.default_timeout_ms == 30000.0
    assert settings.providers == []
    assert settings.api_keys == {}


def test_settings_from_environment_variables(monkeypatch):
    """Settings correctly reads values from environment variables."""
    monkeypatch.setenv("APP_NAME", "custom-aiops")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEFAULT_TIMEOUT_MS", "45000.5")

    settings = Settings(_env_file=None)
    assert settings.app_name == "custom-aiops"
    assert settings.debug is True
    assert settings.log_level == "DEBUG"
    assert settings.default_timeout_ms == 45000.5


def test_settings_from_dotenv_file(tmp_path, monkeypatch):
    """Settings loads values from a specified .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "APP_NAME=dotenv-app\n"
        "DEBUG=true\n"
        "LOG_LEVEL=WARNING\n"
        "DEFAULT_TIMEOUT_MS=15000.0\n"
    )

    for env_key in ["APP_NAME", "DEBUG", "LOG_LEVEL", "DEFAULT_TIMEOUT_MS"]:
        monkeypatch.delenv(env_key, raising=False)

    settings = Settings(_env_file=env_file)
    assert settings.app_name == "dotenv-app"
    assert settings.debug is True
    assert settings.log_level == "WARNING"
    assert settings.default_timeout_ms == 15000.0


def test_env_var_overrides_dotenv(tmp_path, monkeypatch):
    """Environment variables take precedence over .env file values."""
    env_file = tmp_path / ".env"
    env_file.write_text("APP_NAME=from-dotenv\nDEBUG=false\n")

    monkeypatch.setenv("APP_NAME", "from-env")
    monkeypatch.setenv("DEBUG", "true")

    settings = Settings(_env_file=env_file)
    assert settings.app_name == "from-env"
    assert settings.debug is True


def test_default_values_are_correct():
    """Verify default values match the specification."""
    settings = Settings(_env_file=None)
    assert settings.app_name == "open-aiops"
    assert settings.debug is False
    assert settings.log_level == "INFO"
    assert settings.default_timeout_ms == 30000.0
    assert settings.providers == []
    assert settings.api_keys == {}


def test_invalid_log_level_raises_validation_error(monkeypatch):
    """LOG_LEVEL with an invalid level raises ValidationError."""
    monkeypatch.setenv("LOG_LEVEL", "INVALID")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_negative_timeout_raises_validation_error(monkeypatch):
    """DEFAULT_TIMEOUT_MS <= 0 raises ValidationError."""
    monkeypatch.setenv("DEFAULT_TIMEOUT_MS", "-5.0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)

    monkeypatch.setenv("DEFAULT_TIMEOUT_MS", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_non_numeric_timeout_raises_validation_error(monkeypatch):
    """DEFAULT_TIMEOUT_MS set to a non-numeric string raises ValidationError."""
    monkeypatch.setenv("DEFAULT_TIMEOUT_MS", "abc")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_provider_config_valid():
    """ProviderConfig instantiates with valid data."""
    provider = ProviderConfig(
        name="gpt-4",
        model="gpt-4-turbo",
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.03,
        expected_latency_ms=250.0,
        priority=1,
    )
    assert provider.name == "gpt-4"
    assert provider.model == "gpt-4-turbo"
    assert provider.cost_per_1k_input_tokens == 0.01
    assert provider.cost_per_1k_output_tokens == 0.03
    assert provider.expected_latency_ms == 250.0
    assert provider.priority == 1


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


def test_provider_config_priority_default():
    """priority defaults to 0 when not specified, maintaining INT-5 compatibility."""
    provider = ProviderConfig(
        name="claude-3",
        model="claude-3-opus",
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.075,
        expected_latency_ms=350.0,
    )
    assert provider.priority == 0


def test_providers_parsed_from_json_env(monkeypatch):
    """PROVIDERS JSON string is parsed into list of ProviderConfig instances."""
    providers_data = [
        {
            "name": "gpt-4",
            "model": "gpt-4-turbo",
            "cost_per_1k_input_tokens": 0.01,
            "cost_per_1k_output_tokens": 0.03,
            "expected_latency_ms": 250.0,
            "priority": 2,
        },
        {
            "name": "claude",
            "model": "claude-3-haiku",
            "cost_per_1k_input_tokens": 0.00025,
            "cost_per_1k_output_tokens": 0.00125,
            "expected_latency_ms": 120.0,
        },
    ]
    monkeypatch.setenv("PROVIDERS", json.dumps(providers_data))

    settings = Settings(_env_file=None)
    assert len(settings.providers) == 2
    assert isinstance(settings.providers[0], ProviderConfig)
    assert settings.providers[0].name == "gpt-4"
    assert settings.providers[0].priority == 2
    assert settings.providers[1].name == "claude"
    assert settings.providers[1].priority == 0


def test_malformed_providers_json_raises(monkeypatch):
    """PROVIDERS set to malformed JSON raises an error."""
    monkeypatch.setenv("PROVIDERS", "not-valid-json")
    with pytest.raises((ValidationError, ValueError)):
        Settings(_env_file=None)


def test_api_keys_parsed_and_masked(monkeypatch):
    """API_KEYS is parsed from JSON and safely masked in representations."""
    monkeypatch.setenv("API_KEYS", json.dumps({"openai": "sk-abcdef123456789", "short": "abc"}))
    settings = Settings(_env_file=None)

    assert settings.api_keys["openai"] == "sk-abcdef123456789"
    masked = settings.get_masked_api_keys()
    assert masked["openai"] == "sk-...789"
    assert masked["short"] == "***"

    # Representation should not leak unmasked key
    repr_str = repr(settings)
    assert "sk-abcdef123456789" not in repr_str
    assert "sk-...789" in repr_str

    str_str = str(settings)
    assert "sk-abcdef123456789" not in str_str


def test_get_settings_returns_cached_instance():
    """get_settings() returns the same cached instance across calls."""
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
