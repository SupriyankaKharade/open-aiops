"""Tests for the core configuration module."""

import json
import os

import pytest
from pydantic import SecretStr, ValidationError

from open_aiops.core.config import ProviderConfig, Settings, get_settings


@pytest.fixture(autouse=True)
def isolate_config_environment(monkeypatch, tmp_path):
    """Isolate configuration tests from ambient environment variables, .env, and caches.

    Ensures tests are completely deterministic and independent of:
    - Ambient OPEN_AIOPS_* environment variables in the developer's shell
    - Ambient .env file in the workspace or repository root
    - Settings cached in get_settings across test executions
    """
    # 1. Clear get_settings cache before test execution
    get_settings.cache_clear()

    # 2. Delete all ambient OPEN_AIOPS_* environment variables
    for env_key in list(os.environ.keys()):
        if env_key.upper().startswith("OPEN_AIOPS_"):
            monkeypatch.delenv(env_key, raising=False)

    # 3. Switch working directory to an isolated empty directory so any repo .env is not read
    monkeypatch.chdir(tmp_path)

    yield

    # 4. Clear get_settings cache after test execution
    get_settings.cache_clear()


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
    assert provider.vendor is None


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
    assert provider.vendor is None


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


# =========================================================================
# INT-15 Follow-ups: Duplicate Provider Names Validation Tests
# =========================================================================


def test_duplicate_provider_names_rejected_python_data():
    """Settings.providers rejects duplicate provider names from Python data."""
    p1 = ProviderConfig(
        name="gpt-4",
        model="gpt-4-turbo",
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.03,
        expected_latency_ms=250.0,
    )
    p2 = ProviderConfig(
        name="gpt-4",
        model="gpt-4o",
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.015,
        expected_latency_ms=200.0,
    )
    with pytest.raises(ValidationError, match="Duplicate provider names detected"):
        Settings(providers=[p1, p2], _env_file=None)


def test_duplicate_provider_names_rejected_whitespace_normalized():
    """Settings.providers rejects duplicate provider names even with whitespace variance."""
    p1 = ProviderConfig(
        name="claude-3",
        model="claude-3-opus",
        cost_per_1k_input_tokens=0.015,
        cost_per_1k_output_tokens=0.075,
        expected_latency_ms=300.0,
    )
    p2 = ProviderConfig(
        name="  claude-3  ",
        model="claude-3-sonnet",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        expected_latency_ms=150.0,
    )
    with pytest.raises(ValidationError, match="Duplicate provider names detected"):
        Settings(providers=[p1, p2], _env_file=None)


def test_duplicate_provider_names_rejected_from_env_json(monkeypatch):
    """Settings rejects duplicate provider names configured via OPEN_AIOPS_PROVIDERS JSON."""
    duplicate_providers = [
        {
            "name": "gemini-pro",
            "model": "gemini-1.5-pro",
            "cost_per_1k_input_tokens": 0.00125,
            "cost_per_1k_output_tokens": 0.005,
            "expected_latency_ms": 180.0,
        },
        {
            "name": "gemini-pro",
            "model": "gemini-1.5-flash",
            "cost_per_1k_input_tokens": 0.0001,
            "cost_per_1k_output_tokens": 0.0004,
            "expected_latency_ms": 100.0,
        },
    ]
    monkeypatch.setenv("OPEN_AIOPS_PROVIDERS", json.dumps(duplicate_providers))
    with pytest.raises(ValidationError, match="Duplicate provider names detected"):
        Settings(_env_file=None)


def test_distinct_provider_names_accepted():
    """Settings accepts distinct provider names from Python data."""
    p1 = ProviderConfig(
        name="openai-primary",
        model="gpt-4o",
        cost_per_1k_input_tokens=0.005,
        cost_per_1k_output_tokens=0.015,
        expected_latency_ms=200.0,
    )
    p2 = ProviderConfig(
        name="anthropic-fallback",
        model="claude-3-5-sonnet",
        cost_per_1k_input_tokens=0.003,
        cost_per_1k_output_tokens=0.015,
        expected_latency_ms=180.0,
    )
    settings = Settings(providers=[p1, p2], _env_file=None)
    assert len(settings.providers) == 2
    assert settings.providers[0].name == "openai-primary"
    assert settings.providers[1].name == "anthropic-fallback"


def test_distinct_provider_names_accepted_from_env_json(monkeypatch):
    """Settings accepts distinct provider names configured via OPEN_AIOPS_PROVIDERS JSON."""
    distinct_providers = [
        {
            "name": "provider-a",
            "model": "model-a",
            "cost_per_1k_input_tokens": 0.01,
            "cost_per_1k_output_tokens": 0.02,
            "expected_latency_ms": 150.0,
        },
        {
            "name": "provider-b",
            "model": "model-b",
            "cost_per_1k_input_tokens": 0.02,
            "cost_per_1k_output_tokens": 0.04,
            "expected_latency_ms": 250.0,
        },
    ]
    monkeypatch.setenv("OPEN_AIOPS_PROVIDERS", json.dumps(distinct_providers))
    settings = Settings(_env_file=None)
    assert len(settings.providers) == 2
    assert settings.providers[0].name == "provider-a"
    assert settings.providers[1].name == "provider-b"


# =========================================================================
# INT-15 Follow-ups: Optional Vendor Field Tests
# =========================================================================


def test_provider_config_vendor_omitted_defaults_to_none():
    """ProviderConfig vendor field is optional and defaults to None."""
    p = ProviderConfig(
        name="test-p",
        model="test-m",
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.02,
        expected_latency_ms=100.0,
    )
    assert p.vendor is None


def test_provider_config_vendor_provided():
    """ProviderConfig accepts and stores a valid vendor string."""
    p = ProviderConfig(
        name="test-p",
        model="test-m",
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.02,
        expected_latency_ms=100.0,
        vendor="openai",
    )
    assert p.vendor == "openai"


def test_provider_config_vendor_whitespace_normalized():
    """ProviderConfig vendor strips leading and trailing whitespace."""
    p = ProviderConfig(
        name="test-p",
        model="test-m",
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.02,
        expected_latency_ms=100.0,
        vendor="  anthropic  ",
    )
    assert p.vendor == "anthropic"


def test_provider_config_vendor_empty_or_whitespace_rejected():
    """ProviderConfig rejects empty or whitespace-only vendor values."""
    with pytest.raises(ValidationError):
        ProviderConfig(
            name="test-p",
            model="test-m",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.02,
            expected_latency_ms=100.0,
            vendor="",
        )

    with pytest.raises(ValidationError):
        ProviderConfig(
            name="test-p",
            model="test-m",
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.02,
            expected_latency_ms=100.0,
            vendor="   ",
        )


def test_provider_config_vendor_explicit_none_allowed():
    """ProviderConfig explicitly passing vendor=None is valid and remains None."""
    p = ProviderConfig(
        name="test-p",
        model="test-m",
        cost_per_1k_input_tokens=0.01,
        cost_per_1k_output_tokens=0.02,
        expected_latency_ms=100.0,
        vendor=None,
    )
    assert p.vendor is None


def test_provider_config_vendor_from_env_json(monkeypatch):
    """OPEN_AIOPS_PROVIDERS parses vendor when provided and defaults to None when omitted."""
    providers_data = [
        {
            "name": "gpt-4",
            "model": "gpt-4-turbo",
            "cost_per_1k_input_tokens": 0.01,
            "cost_per_1k_output_tokens": 0.03,
            "expected_latency_ms": 250.0,
            "vendor": "openai",
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
    assert settings.providers[0].vendor == "openai"
    assert settings.providers[1].vendor is None


# =========================================================================
# INT-15 Follow-ups: Configuration Test Isolation Tests
# =========================================================================


def test_configuration_test_isolation_from_ambient_environment(monkeypatch):
    """Configuration tests run isolated from ambient OPEN_AIOPS_* environment variables."""
    # When no explicit test environment variables are set, defaults are consistently returned
    settings = Settings(_env_file=None)
    assert settings.app_name == "open-aiops"
    assert settings.debug is False
    assert settings.log_level == "INFO"


def test_get_settings_cache_clearing_and_isolation(monkeypatch):
    """get_settings cache can be cleared and reloads fresh settings without cross-test leakage."""
    get_settings.cache_clear()
    s1 = get_settings()
    assert s1.debug is False

    monkeypatch.setenv("OPEN_AIOPS_DEBUG", "true")
    get_settings.cache_clear()
    s2 = get_settings()
    assert s2.debug is True
    assert s1 is not s2

    get_settings.cache_clear()
