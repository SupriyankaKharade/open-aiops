"""Core pricing and cost attribution module for open-aiops.

Calculates request cost using exact Decimal arithmetic based on configured
provider prices per 1,000 input and output tokens.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

from open_aiops.core.config import ProviderConfig, Settings, get_settings


class CostStatus(str, Enum):
    """Documented cost calculation statuses."""

    PRICED = "priced"
    UNKNOWN_MODEL = "unknown_model"
    MISSING_USAGE = "missing_usage"
    FREE = "free"


@dataclass(frozen=True)
class Cost:
    """Typed cost result containing calculated Decimal amount and status.

    Attributes:
        amount: Calculated cost as a Decimal when available, or None when missing/unknown.
        status: Cost status ('priced', 'unknown_model', 'missing_usage', 'free').
    """

    amount: Optional[Decimal]
    status: CostStatus

    def __post_init__(self) -> None:
        if isinstance(self.status, str) and not isinstance(self.status, CostStatus):
            object.__setattr__(self, "status", CostStatus(self.status))

    @property
    def cost(self) -> Optional[Decimal]:
        """Alias for amount for compatibility with event fields."""
        return self.amount

    @property
    def cost_status(self) -> CostStatus:
        """Alias for status for compatibility with event fields."""
        return self.status


class PricingRegistry:
    """Registry for calculating request costs from provider configurations.

    Maintains prices per 1,000 input/output tokens from ProviderConfig instances
    and computes exact Decimal request costs without floating-point errors.
    """

    def __init__(self, providers: Optional[Iterable[ProviderConfig]] = None) -> None:
        """Initialize the pricing registry.

        Args:
            providers: Optional iterable of ProviderConfig instances to register.
        """
        self._entries: dict[tuple[str, str], ProviderConfig] = {}
        if providers is not None:
            for p in providers:
                self.register(p)

    def register(self, provider: ProviderConfig) -> None:
        """Register a ProviderConfig in the registry.

        Args:
            provider: ProviderConfig containing pricing and model identifiers.
        """
        provider_name = provider.name.strip() if isinstance(provider.name, str) else str(provider.name)
        model_name = provider.model.strip() if isinstance(provider.model, str) else str(provider.model)
        self._entries[(provider_name, model_name)] = provider

    def has_model(self, provider: str, model: str) -> bool:
        """Check if a provider and model pair is registered.

        Args:
            provider: Provider identifier.
            model: Model identifier.

        Returns:
            True if registered, False otherwise.
        """
        if not isinstance(provider, str) or not isinstance(model, str):
            return False
        return (provider.strip(), model.strip()) in self._entries

    def cost(
        self,
        provider: str,
        model: str,
        input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> Cost:
        """Calculate the cost for a request.

        Evaluation precedence:
        1. unknown_model: If the provider and model pair is not registered.
        2. free: If the configured provider prices are explicitly zero.
        3. missing_usage: If token usage is missing, incomplete, or negative.
        4. priced: Calculated exact Decimal cost per 1,000 input/output tokens.

        Pricing never raises an exception for unknown models or missing token counts.

        Args:
            provider: LLM provider name.
            model: Model identifier.
            input_tokens: Number of prompt/input tokens.
            output_tokens: Number of completion/output tokens.

        Returns:
            Cost object with calculated Decimal amount and documented CostStatus.
        """
        try:
            if not isinstance(provider, str) or not isinstance(model, str):
                return Cost(amount=None, status=CostStatus.UNKNOWN_MODEL)

            key = (provider.strip(), model.strip())
            config = self._entries.get(key)
            if config is None:
                return Cost(amount=None, status=CostStatus.UNKNOWN_MODEL)

            price_in = Decimal(str(config.cost_per_1k_input_tokens))
            price_out = Decimal(str(config.cost_per_1k_output_tokens))

            # 2. Check if the model is explicitly configured as free (prices == 0)
            if price_in == Decimal("0") and price_out == Decimal("0"):
                return Cost(amount=Decimal("0"), status=CostStatus.FREE)

            # 3. Validate token usage for paid models
            if (
                input_tokens is None
                or output_tokens is None
                or isinstance(input_tokens, bool)
                or isinstance(output_tokens, bool)
                or not isinstance(input_tokens, int)
                or not isinstance(output_tokens, int)
                or input_tokens < 0
                or output_tokens < 0
            ):
                return Cost(amount=None, status=CostStatus.MISSING_USAGE)

            # 4. Compute priced cost using exact Decimal arithmetic
            thousand = Decimal("1000")
            input_cost = (Decimal(input_tokens) * price_in) / thousand
            output_cost = (Decimal(output_tokens) * price_out) / thousand
            total_cost = input_cost + output_cost

            return Cost(amount=total_cost, status=CostStatus.PRICED)
        except Exception:
            return Cost(amount=None, status=CostStatus.UNKNOWN_MODEL)

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "PricingRegistry":
        """Factory to create a PricingRegistry populated from application Settings.

        Args:
            settings: Optional Settings instance. Defaults to get_settings().

        Returns:
            Configured PricingRegistry.
        """
        resolved_settings = settings if settings is not None else get_settings()
        return cls(resolved_settings.providers)
