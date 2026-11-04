"""Model-agnostic LLM factory.

This is the *only* module that imports provider SDK classes. Everything else
calls :func:`get_llm` and works with the generic ``BaseChatModel`` interface.

Three SDKs cover five providers: ``ChatAnthropic`` for Anthropic,
``ChatGoogleGenerativeAI`` for Gemini (its native API carries the "thought
signatures" that Gemini 3 requires on replayed tool calls, which the
OpenAI-compatible endpoint drops), and ``ChatOpenAI`` for everything else
that speaks the OpenAI protocol (OpenAI itself, Groq and OpenRouter).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel

from app.config import PROVIDERS, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderPreset:
    """Static facts about a provider; keys and model names come from Settings."""

    name: str
    label: str
    sdk: str  # "openai", "anthropic" or "google"
    base_url: str | None
    free: bool
    key_hint: str  # where to get a key


PRESETS: dict[str, ProviderPreset] = {
    "openai": ProviderPreset("openai", "OpenAI", "openai", None, False, "platform.openai.com"),
    "anthropic": ProviderPreset(
        "anthropic", "Anthropic Claude", "anthropic", None, False, "console.anthropic.com"
    ),
    "groq": ProviderPreset(
        "groq",
        "Groq (free tier)",
        "openai",
        "https://api.groq.com/openai/v1",
        True,
        "console.groq.com",
    ),
    "gemini": ProviderPreset(
        "gemini", "Google Gemini (free tier)", "google", None, True, "aistudio.google.com"
    ),
    "openrouter": ProviderPreset(
        "openrouter",
        "OpenRouter (free models)",
        "openai",
        "https://openrouter.ai/api/v1",
        True,
        "openrouter.ai/keys",
    ),
}
assert set(PRESETS) == set(PROVIDERS)

# APPROXIMATE list prices in USD per 1M tokens (input, output), snapshot of the
# vendors' public pricing pages. They drift; they exist only so the eval script
# can show a rough cost comparison. Free-tier models are listed at 0.
PRICES: dict[str, dict[str, float]] = {
    # --- OpenAI ---
    "gpt-6-astra": {"input": 10.00, "output": 50.00},
    "gpt-6.1-sol": {"input": 2.00, "output": 10.00},
    "gpt-6.1-terra": {"input": 1.00, "output": 5.00},
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "gpt-5.4": {"input": 2.50, "output": 15.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    # --- Anthropic ---
    "claude-fable-5-1": {"input": 10.00, "output": 50.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    # --- free tiers / local (0 while within the free quota) ---
    "openai/gpt-oss-120b": {"input": 0.0, "output": 0.0},
    "gpt-oss-120b": {"input": 0.0, "output": 0.0},
    "gemini-2.5-flash": {"input": 0.0, "output": 0.0},
    "gemini-3.5-flash": {"input": 0.0, "output": 0.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate the USD cost of one call, or ``None`` if the model is not in PRICES.

    Prefix matching lets dated aliases (``gpt-6.1-sol-2026-07-09``) and tagged
    models (``qwen3:8b``) resolve to their base entry.
    """
    prices = PRICES.get(model)
    if prices is None:
        prices = next((p for name, p in PRICES.items() if model.startswith(name)), None)
    if prices is None:
        return None
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


def resolve_provider(provider: str | None = None) -> str:
    """Normalise a provider name, falling back to ``LLM_PROVIDER``."""
    chosen = (provider or get_settings().llm_provider).strip().lower()
    if chosen not in PRESETS:
        raise ValueError(f"Unknown LLM provider {chosen!r}; choose one of {', '.join(PRESETS)}.")
    return chosen


def model_name_for(provider: str | None = None) -> str:
    """Return the configured model name without creating a client."""
    return get_settings().model_for(resolve_provider(provider))


def available_providers() -> list[dict[str, Any]]:
    """Describe every provider: configured or not, which model, free or paid."""
    settings = get_settings()
    default = resolve_provider()
    return [
        {
            "name": p.name,
            "label": p.label,
            "model": settings.model_for(p.name),
            "free": p.free,
            "configured": settings.has_api_key(p.name),
            "default": p.name == default,
            "key_hint": p.key_hint,
        }
        for p in PRESETS.values()
    ]


def get_llm(
    provider: str | None = None,
    temperature: float | None = None,
    *,
    max_tokens: int = 4096,
    **kwargs: Any,
) -> BaseChatModel:
    """Return a chat model for ``provider`` (defaults to ``LLM_PROVIDER``).

    Args:
        provider: One of :data:`PRESETS` (openai, anthropic, groq, gemini,
            openrouter); ``None`` uses the env default.
        temperature: Forwarded only when set. OpenAI/Anthropic reasoning models
            reject custom sampling parameters, so leave it ``None`` there.
        max_tokens: Output token cap per call.
        **kwargs: Extra constructor arguments for the chat model.

    Raises:
        ValueError: Unknown provider, or the provider's API key is missing.
    """
    settings = get_settings()
    name = resolve_provider(provider)
    preset = PRESETS[name]
    api_key = settings.api_key_for(name)
    if not api_key:
        raise ValueError(
            f"{name.upper()}_API_KEY is not set. Get a key at {preset.key_hint} and add it to .env."
        )

    model_name = settings.model_for(name)
    params: dict[str, Any] = {"max_tokens": max_tokens, **kwargs}
    if temperature is not None:
        params["temperature"] = temperature
    logger.info("Creating %s chat model %s", name, model_name)

    if preset.sdk == "anthropic":
        from langchain_anthropic import ChatAnthropic  # lazy: keeps the other SDKs optional

        return ChatAnthropic(model=model_name, api_key=api_key, **params)

    if preset.sdk == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        params["max_output_tokens"] = params.pop("max_tokens")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, **params)

    from langchain_openai import ChatOpenAI

    base_url = preset.base_url
    if name == "openai" and settings.openai_base_url:
        base_url = settings.openai_base_url  # custom OpenAI-compatible endpoint
    if base_url:
        params["base_url"] = base_url
    return ChatOpenAI(model=model_name, api_key=api_key, **params)
