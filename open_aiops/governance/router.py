"""Model router: pick the best available provider for a request.

A provider is eligible when it is not flagged as overloaded and fits the
request's latency and cost budgets. Among eligible providers the cheapest
wins, then the fastest, then the one listed first.
"""

from collections.abc import Iterable
from enum import Enum

from pydantic import BaseModel, Field

from open_aiops.core.config import ProviderConfig


class RouteRequest(BaseModel):
    input_tokens: int = Field(ge=0)
    max_output_tokens: int = Field(ge=0)
    max_latency_ms: float | None = Field(default=None, gt=0)
    max_cost: float | None = Field(default=None, ge=0)


class RejectionReason(str, Enum):
    OVERLOADED = "overloaded"
    LATENCY = "latency"
    COST = "cost"


class RoutingDecision(BaseModel):
    ranked: list[ProviderConfig]
    rejected: dict[str, RejectionReason]


class NoProviderAvailable(Exception):
    pass


def estimate_cost(provider: ProviderConfig, request: RouteRequest) -> float:
    """Worst-case cost in the same currency as the provider's pricing."""
    return (
        request.input_tokens * provider.cost_per_1k_input_tokens
        + request.max_output_tokens * provider.cost_per_1k_output_tokens
    ) / 1000


class ModelRouter:
    def __init__(self, providers: Iterable[ProviderConfig]):
        self._providers = list(providers)
        names = [p.name for p in self._providers]
        if not names:
            raise ValueError("ModelRouter needs at least one provider")
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate provider names: {names}")
        self._overloaded: set[str] = set()

    def mark_overloaded(self, name: str) -> None:
        self._check_known(name)
        self._overloaded.add(name)

    def mark_healthy(self, name: str) -> None:
        self._check_known(name)
        self._overloaded.discard(name)

    def is_overloaded(self, name: str) -> bool:
        return name in self._overloaded

    def decide(self, request: RouteRequest) -> RoutingDecision:
        """Rank eligible providers best-first and record why the rest were rejected."""
        eligible: list[ProviderConfig] = []
        rejected: dict[str, RejectionReason] = {}
        for p in self._providers:
            reason = self._rejection_reason(p, request)
            if reason is None:
                eligible.append(p)
            else:
                rejected[p.name] = reason
        # sorted() is stable, so list order breaks remaining ties.
        ranked = sorted(eligible, key=lambda p: (estimate_cost(p, request), p.expected_latency_ms))
        return RoutingDecision(ranked=ranked, rejected=rejected)

    def route(self, request: RouteRequest) -> ProviderConfig:
        decision = self.decide(request)
        if not decision.ranked:
            reasons = {name: reason.value for name, reason in decision.rejected.items()}
            raise NoProviderAvailable(f"No provider fits the request (rejected: {reasons})")
        return decision.ranked[0]

    def _rejection_reason(self, provider: ProviderConfig, request: RouteRequest) -> RejectionReason | None:
        if provider.name in self._overloaded:
            return RejectionReason.OVERLOADED
        if request.max_latency_ms is not None and provider.expected_latency_ms > request.max_latency_ms:
            return RejectionReason.LATENCY
        if request.max_cost is not None and estimate_cost(provider, request) > request.max_cost:
            return RejectionReason.COST
        return None

    def _check_known(self, name: str) -> None:
        if not any(p.name == name for p in self._providers):
            raise KeyError(f"Unknown provider: {name}")
