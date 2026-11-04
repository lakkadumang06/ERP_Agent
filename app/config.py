"""Application settings, loaded from environment variables and a local ``.env``.

Every other module reads configuration through :func:`get_settings`, so this
is the only place that knows environment variable names. Real environment
variables always win over values in ``.env``.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Known provider names. "openai" and "anthropic" are the paid first-party APIs;
# groq, gemini and openrouter have free tiers and speak the OpenAI protocol.
PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "groq", "gemini", "openrouter")


class Settings(BaseSettings):
    """Runtime configuration. Field names map 1:1 to environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- default provider ------------------------------------------------
    llm_provider: str = Field(default="openai", description="Default provider name.")

    # --- paid first-party APIs -------------------------------------------
    openai_api_key: str | None = None
    openai_model: str = "gpt-6.1-sol"
    openai_base_url: str | None = None  # optional custom OpenAI-compatible endpoint
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"

    # --- free tiers (OpenAI-compatible) ----------------------------------
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.5-flash"
    openrouter_api_key: str | None = None
    openrouter_model: str = "nvidia/nemotron-3.5-super-vl:free"

    # --- database ----------------------------------------------------------
    database_url: str = "sqlite:///./data/erp.db"

    # --- LangSmith (LangChain reads LANGSMITH_* itself; mirrored for logging) ---
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "erp-agent"

    log_level: str = "INFO"

    def api_key_for(self, provider: str) -> str | None:
        """Return the configured API key for ``provider``."""
        return getattr(self, f"{provider}_api_key", None) or None

    def has_api_key(self, provider: str) -> bool:
        """True if ``provider`` has an API key configured."""
        return bool(self.api_key_for(provider))

    def model_for(self, provider: str) -> str:
        """Return the configured model name for ``provider``."""
        return getattr(self, f"{provider}_model")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached :class:`Settings` instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Clear the cached settings (tests that change env vars call this)."""
    get_settings.cache_clear()


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once, in a format that reads well in Docker logs."""
    logging.basicConfig(
        level=level or get_settings().log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
