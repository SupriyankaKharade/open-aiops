"""Unit tests for the pricing registry (INT-17)."""

from decimal import Decimal
import pytest

from open_aiops.core.config import ProviderConfig, Settings
from open_aiops.core.pricing import Cost, CostStatus, PricingRegistry


@pytest.fixture
def sample_providers():
    """Sample provider configurations for testing."""
    return [
        ProviderConfig(
            name="openai",
            model="gpt-4o",
            cost_per_1k_input_tokens=0.005,
            cost_per_1k_output_tokens=0.015,
            expected_latency_ms=200.0,
        ),
        ProviderConfig(
            name="openai",
            model="gpt-4o-mini",
            cost_per_1k_input_tokens=0.00015,
            cost_per_1k_output_tokens=0.0006,
            expected_latency_ms=100.0,
        ),
        ProviderConfig(
            name="azure",
            model="gpt-4o",
            cost_per_1k_input_tokens=0.006,
            cost_per_1k_output_tokens=0.018,
            expected_latency_ms=250.0,
        ),
        ProviderConfig(
            name="local-ollama",
            model="llama-3.1-8b",
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=0.0,
            expected_latency_ms=80.0,
        ),
        ProviderConfig(
            name="prompt-caching-provider",
            model="cached-model",
            cost_per_1k_input_tokens=0.0,
            cost_per_1k_output_tokens=0.02,
            expected_latency_ms=120.0,
        ),
    ]


@pytest.fixture
def pricing_registry(sample_providers):
    """Populated pricing registry fixture."""
    return PricingRegistry(sample_providers)


def test_empty_registry_returns_unknown_model():
    """An empty registry returns unknown_model with None amount and never raises."""
    registry = PricingRegistry()
    cost = registry.cost(provider="openai", model="gpt-4o", input_tokens=100, output_tokens=50)
    assert cost.status == CostStatus.UNKNOWN_MODEL
    assert cost.amount is None
    assert cost.cost is None
    assert cost.cost_status == CostStatus.UNKNOWN_MODEL


def test_unknown_model_returns_unknown_status(pricing_registry):
    """Querying an unknown model under a known provider returns unknown_model."""
    cost = pricing_registry.cost(
        provider="openai", model="nonexistent-model", input_tokens=1000, output_tokens=500
    )
    assert cost.status == CostStatus.UNKNOWN_MODEL
    assert cost.amount is None


def test_unknown_provider_returns_unknown_status(pricing_registry):
    """Querying a model under an unknown provider returns unknown_model."""
    cost = pricing_registry.cost(
        provider="anthropic", model="gpt-4o", input_tokens=1000, output_tokens=500
    )
    assert cost.status == CostStatus.UNKNOWN_MODEL
    assert cost.amount is None


def test_accurate_cost_calculation_with_decimal_precision(pricing_registry):
    """Verify calculated cost matches exact Decimal math for input and output rates."""
    # openai/gpt-4o: $0.005/1k input, $0.015/1k output
    # 2,000 input tokens = $0.010
    # 1,000 output tokens = $0.015
    # Total = $0.025
    cost = pricing_registry.cost(
        provider="openai", model="gpt-4o", input_tokens=2000, output_tokens=1000
    )
    assert cost.status == CostStatus.PRICED
    assert isinstance(cost.amount, Decimal)
    assert cost.amount == Decimal("0.025")


def test_decimal_safety_avoids_binary_float_drift(pricing_registry):
    """Verify decimal-safe arithmetic avoids floating point representation artifacts."""
    # gpt-4o-mini: 0.00015 input per 1k, 0.0006 output per 1k
    # 1,234 input tokens: 1234 * 0.00015 / 1000 = 0.0001851
    # 567 output tokens: 567 * 0.0006 / 1000 = 0.0003402
    # Total = 0.0005253
    cost = pricing_registry.cost(
        provider="openai", model="gpt-4o-mini", input_tokens=1234, output_tokens=567
    )
    assert cost.status == CostStatus.PRICED
    expected = Decimal("0.0001851") + Decimal("0.0003402")
    assert cost.amount == expected
    assert cost.amount == Decimal("0.0005253")


def test_zero_token_usage_on_paid_model_is_priced_not_free(pricing_registry):
    """A zero-cost request on a paid model must be status 'priced', not 'free'."""
    cost = pricing_registry.cost(
        provider="openai", model="gpt-4o", input_tokens=0, output_tokens=0
    )
    assert cost.status == CostStatus.PRICED
    assert cost.status != CostStatus.FREE
    assert cost.amount == Decimal("0")


def test_explicitly_free_provider_is_free_status(pricing_registry):
    """An explicitly configured free provider returns status 'free' with zero Decimal amount."""
    # local-ollama is configured with 0.0 input and 0.0 output
    cost = pricing_registry.cost(
        provider="local-ollama", model="llama-3.1-8b", input_tokens=5000, output_tokens=2000
    )
    assert cost.status == CostStatus.FREE
    assert cost.amount == Decimal("0")


def test_explicitly_free_provider_with_missing_or_zero_tokens_is_free(pricing_registry):
    """An explicitly free model remains status 'free' even if token usage is missing or zero."""
    cost_missing = pricing_registry.cost(
        provider="local-ollama", model="llama-3.1-8b", input_tokens=None, output_tokens=None
    )
    assert cost_missing.status == CostStatus.FREE
    assert cost_missing.amount == Decimal("0")

    cost_zero = pricing_registry.cost(
        provider="local-ollama", model="llama-3.1-8b", input_tokens=0, output_tokens=0
    )
    assert cost_zero.status == CostStatus.FREE
    assert cost_zero.amount == Decimal("0")


def test_missing_input_or_output_tokens_returns_missing_usage(pricing_registry):
    """Missing or None token counts on a paid model return missing_usage with None amount."""
    # Both None
    c1 = pricing_registry.cost(
        provider="openai", model="gpt-4o", input_tokens=None, output_tokens=None
    )
    assert c1.status == CostStatus.MISSING_USAGE
    assert c1.amount is None

    # input_tokens missing
    c2 = pricing_registry.cost(
        provider="openai", model="gpt-4o", input_tokens=None, output_tokens=500
    )
    assert c2.status == CostStatus.MISSING_USAGE
    assert c2.amount is None

    # output_tokens missing
    c3 = pricing_registry.cost(
        provider="openai", model="gpt-4o", input_tokens=1000, output_tokens=None
    )
    assert c3.status == CostStatus.MISSING_USAGE
    assert c3.amount is None


def test_negative_or_invalid_tokens_return_missing_usage(pricing_registry):
    """Negative, boolean, or non-integer token counts return missing_usage safely."""
    # Negative tokens
    c_neg = pricing_registry.cost(
        provider="openai", model="gpt-4o", input_tokens=-10, output_tokens=100
    )
    assert c_neg.status == CostStatus.MISSING_USAGE
    assert c_neg.amount is None

    # Boolean values
    c_bool = pricing_registry.cost(
        provider="openai", model="gpt-4o", input_tokens=True, output_tokens=100  # type: ignore
    )
    assert c_bool.status == CostStatus.MISSING_USAGE
    assert c_bool.amount is None

    # Non-integer float
    c_float = pricing_registry.cost(
        provider="openai", model="gpt-4o", input_tokens=100.5, output_tokens=100  # type: ignore
    )
    assert c_float.status == CostStatus.MISSING_USAGE
    assert c_float.amount is None


def test_provider_model_lookup_isolation(pricing_registry):
    """Lookup uses both provider and model so identical model names across providers stay isolated."""
    # Both 'openai' and 'azure' provide 'gpt-4o' with different rates:
    # openai: 0.005 / 0.015 -> for 1000/1000 tokens = 0.020
    # azure:  0.006 / 0.018 -> for 1000/1000 tokens = 0.024
    cost_openai = pricing_registry.cost(
        provider="openai", model="gpt-4o", input_tokens=1000, output_tokens=1000
    )
    cost_azure = pricing_registry.cost(
        provider="azure", model="gpt-4o", input_tokens=1000, output_tokens=1000
    )

    assert cost_openai.status == CostStatus.PRICED
    assert cost_openai.amount == Decimal("0.02")

    assert cost_azure.status == CostStatus.PRICED
    assert cost_azure.amount == Decimal("0.024")


def test_whitespace_normalization_in_lookup(pricing_registry):
    """Leading and trailing whitespace in provider or model names is handled gracefully."""
    cost = pricing_registry.cost(
        provider="  openai  ", model="  gpt-4o  ", input_tokens=1000, output_tokens=1000
    )
    assert cost.status == CostStatus.PRICED
    assert cost.amount == Decimal("0.02")


def test_half_free_model_pricing(pricing_registry):
    """A model with free input tokens but paid output tokens is not considered an explicitly free model."""
    # prompt-caching-provider/cached-model: 0.0 input, 0.02 output
    # When output tokens are reported, it is priced
    cost_priced = pricing_registry.cost(
        provider="prompt-caching-provider",
        model="cached-model",
        input_tokens=10000,
        output_tokens=1000,
    )
    assert cost_priced.status == CostStatus.PRICED
    assert cost_priced.amount == Decimal("0.02")

    # When output tokens are 0, it is priced at $0 (not free)
    cost_zero_out = pricing_registry.cost(
        provider="prompt-caching-provider",
        model="cached-model",
        input_tokens=10000,
        output_tokens=0,
    )
    assert cost_zero_out.status == CostStatus.PRICED
    assert cost_zero_out.amount == Decimal("0")

    # When output tokens are missing, it cannot be priced
    cost_missing_out = pricing_registry.cost(
        provider="prompt-caching-provider",
        model="cached-model",
        input_tokens=10000,
        output_tokens=None,
    )
    assert cost_missing_out.status == CostStatus.MISSING_USAGE
    assert cost_missing_out.amount is None


def test_cost_dataclass_properties_and_string_equality():
    """Cost dataclass supports both attribute names and string equality for status."""
    cost = Cost(amount=Decimal("0.05"), status=CostStatus.PRICED)
    assert cost.amount == Decimal("0.05")
    assert cost.cost == Decimal("0.05")
    assert cost.status == CostStatus.PRICED
    assert cost.cost_status == CostStatus.PRICED
    assert cost.status == "priced"
    assert cost.cost_status == "priced"

    # Initializing with plain string converts to CostStatus enum seamlessly
    cost_str = Cost(amount=None, status="missing_usage")  # type: ignore
    assert cost_str.status == CostStatus.MISSING_USAGE
    assert cost_str.cost_status == "missing_usage"


def test_pricing_registry_from_settings():
    """PricingRegistry.from_settings loads providers from a Settings instance."""
    settings = Settings(
        providers=[
            ProviderConfig(
                name="test-prov",
                model="test-mod",
                cost_per_1k_input_tokens=0.01,
                cost_per_1k_output_tokens=0.02,
                expected_latency_ms=100.0,
            )
        ],
        _env_file=None,
    )
    registry = PricingRegistry.from_settings(settings)
    assert registry.has_model("test-prov", "test-mod")
    cost = registry.cost("test-prov", "test-mod", input_tokens=1000, output_tokens=1000)
    assert cost.status == CostStatus.PRICED
    assert cost.amount == Decimal("0.03")


def test_dynamic_registration_and_has_model():
    """PricingRegistry supports dynamic registration and has_model queries."""
    registry = PricingRegistry()
    assert not registry.has_model("custom", "model-v1")

    p = ProviderConfig(
        name="custom",
        model="model-v1",
        cost_per_1k_input_tokens=0.002,
        cost_per_1k_output_tokens=0.004,
        expected_latency_ms=150.0,
    )
    registry.register(p)
    assert registry.has_model("custom", "model-v1")
    cost = registry.cost("custom", "model-v1", input_tokens=500, output_tokens=500)
    assert cost.status == CostStatus.PRICED
    assert cost.amount == Decimal("0.003")


def test_pricing_never_raises_on_malformed_inputs():
    """PricingRegistry.cost never raises when given invalid types or unexpected values."""
    registry = PricingRegistry()
    c1 = registry.cost(provider=None, model=None, input_tokens=None, output_tokens=None)  # type: ignore
    assert c1.status == CostStatus.UNKNOWN_MODEL
    assert c1.amount is None

    c2 = registry.cost(provider=123, model=[], input_tokens="abc", output_tokens={})  # type: ignore
    assert c2.status == CostStatus.UNKNOWN_MODEL
    assert c2.amount is None
