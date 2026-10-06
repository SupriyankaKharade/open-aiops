"""Core configuration module for open-aiops.

Loads configuration from environment variables and .env files using
pydantic-settings, providing typed models, defaults, and validation.
"""

from functools import lru_cache

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider.

    Defines routing, performance, and cost parameters for model endpoints.
    Fully compatible with ModelRouter requirements.
    """

    name: str = Field(..., min_length=1, description="Unique provider identifier")
    model: str = Field(..., min_length=1, description="Model identifier")
    cost_per_1k_input_tokens: float = Field(
        ..., ge=0, description="Cost per 1k input tokens in USD"
    )
    cost_per_1k_output_tokens: float = Field(
        ..., ge=0, description="Cost per 1k output tokens in USD"
    )
    expected_latency_ms: float = Field(
        ..., gt=0, description="Expected response latency in milliseconds"
    )
    priority: int = Field(
        default=0,
        ge=0,
        description="Provider selection priority (higher = preferred, default 0)",
    )

    @field_validator("name", "model")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure provider string identifiers are non-empty after stripping whitespace."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return v.strip()


class Settings(BaseSettings):
    """Application-level settings loaded from environment or .env file."""

    app_name: str = Field(default="open-aiops", description="Application identifier")
    debug: bool = Field(default=False, description="Debug mode toggle")
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    default_timeout_ms: float = Field(
        default=30000.0, gt=0, description="Default request timeout in milliseconds"
    )
    providers: list[ProviderConfig] = Field(
        default_factory=list, description="Configured LLM providers"
    )
    api_keys: dict[str, str] = Field(
        default_factory=dict,
        description="Provider API keys mapped by provider identifier",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level against standard Python logging levels."""
        upper_v = v.upper()
        if upper_v not in VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log_level '{v}'. Must be one of {sorted(VALID_LOG_LEVELS)}"
            )
        return upper_v

    def get_masked_api_keys(self) -> dict[str, str]:
        """Return API keys masked for safe logging and display."""
        return {
            k: (v[:3] + "..." + v[-3:] if len(v) > 6 else "***")
            for k, v in self.api_keys.items()
        }

    def __repr__(self) -> str:
        """Safe string representation masking sensitive API keys."""
        data = self.model_dump()
        data["api_keys"] = self.get_masked_api_keys()
        fields_str = ", ".join(f"{k}={v!r}" for k, v in data.items())
        return f"{self.__class__.__name__}({fields_str})"

    def __str__(self) -> str:
        """Safe string representation masking sensitive API keys."""
        return self.__repr__()


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
