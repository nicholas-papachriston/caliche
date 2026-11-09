"""Environment helpers for configuring Caliche builders."""

from __future__ import annotations

import os
from typing import Any


def _first_non_empty(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


def openai_base_url() -> str | None:
    return _first_non_empty(os.environ.get("OPENAI_BASE_URL"), os.environ.get("OPENAI_API_BASE"))


def resolve_provider_from_env(*, prefer_openrouter: bool = True) -> Any:
    """Resolve a ProviderType from environment variables."""
    import caliche

    forced = (os.environ.get("CALICHE_PROVIDER") or "").strip().lower()
    if forced == "openrouter":
        return caliche.ProviderType.OpenRouter
    if forced == "openai":
        return caliche.ProviderType.OpenAI
    if forced == "anthropic":
        return caliche.ProviderType.Anthropic
    if forced == "lmstudio":
        return caliche.ProviderType.LMStudio
    if forced == "ollama":
        return caliche.ProviderType.Ollama
    if forced in {"google", "gemini"}:
        return caliche.ProviderType.Google
    if forced == "bedrock":
        return caliche.ProviderType.Bedrock
    if forced == "perplexity":
        return caliche.ProviderType.Perplexity
    if forced == "cohere":
        return caliche.ProviderType.Cohere
    if forced in {"azure-openai", "azure_openai", "azureopenai"}:
        return caliche.ProviderType.AzureOpenAI

    base = (openai_base_url() or "").lower()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY") or ""
    if prefer_openrouter and (
        openrouter_key or openai_key.startswith("sk-or") or "openrouter.ai" in base
    ):
        return caliche.ProviderType.OpenRouter
    if openai_key:
        return caliche.ProviderType.OpenAI
    if os.environ.get("ANTHROPIC_API_KEY"):
        return caliche.ProviderType.Anthropic
    if os.environ.get("LMSTUDIO_BASE_URL") or os.environ.get("LM_STUDIO_BASE_URL"):
        return caliche.ProviderType.LMStudio
    return caliche.ProviderType.OpenAI


def resolve_base_url_from_env(provider: Any | None = None) -> str | None:
    """Resolve a provider base URL when one is configured in the environment.

    Matches JS ``resolveBaseUrlFromEnv``:
    - LM Studio defaults to ``http://localhost:1234/v1``
    - Ollama defaults to ``http://localhost:11434``
    - OpenRouter returns ``None`` (do not force ``OPENAI_BASE_URL``)
    - otherwise ``OPENAI_BASE_URL`` / ``OPENAI_API_BASE``
    """
    import caliche

    resolved = provider if provider is not None else resolve_provider_from_env()
    if resolved == caliche.ProviderType.LMStudio:
        return (
            _first_non_empty(
                os.environ.get("LMSTUDIO_BASE_URL"),
                os.environ.get("LM_STUDIO_BASE_URL"),
                "http://localhost:1234/v1",
            )
            or "http://localhost:1234/v1"
        )
    if resolved == caliche.ProviderType.Ollama:
        return (
            _first_non_empty(os.environ.get("OLLAMA_BASE_URL"), "http://localhost:11434")
            or "http://localhost:11434"
        )
    # OpenRouter uses the dedicated provider path; do not force OPENAI_BASE_URL onto it.
    if resolved == caliche.ProviderType.OpenRouter:
        return None
    return openai_base_url()


def resolve_model_from_env(provider: Any | None = None) -> str:
    """Resolve a model id for the given or env-resolved provider."""
    import caliche

    explicit = _first_non_empty(os.environ.get("CALICHE_MODEL"), os.environ.get("MODEL"))
    if explicit:
        return explicit

    resolved = provider if provider is not None else resolve_provider_from_env()
    if resolved == caliche.ProviderType.OpenRouter:
        return (
            _first_non_empty(os.environ.get("OPENROUTER_MODEL"), "openai/gpt-4o-mini")
            or "openai/gpt-4o-mini"
        )
    if resolved == caliche.ProviderType.Anthropic:
        return (
            _first_non_empty(os.environ.get("ANTHROPIC_MODEL"), "claude-3-haiku-20240307")
            or "claude-3-haiku-20240307"
        )
    if resolved == caliche.ProviderType.Ollama:
        return _first_non_empty(os.environ.get("OLLAMA_MODEL"), "llama3.2") or "llama3.2"
    if resolved == caliche.ProviderType.Google:
        return (
            _first_non_empty(os.environ.get("GEMINI_MODEL"), "gemini-2.0-flash")
            or "gemini-2.0-flash"
        )
    if resolved == caliche.ProviderType.LMStudio:
        return _first_non_empty(os.environ.get("LMSTUDIO_MODEL"), "local-model") or "local-model"
    return _first_non_empty(os.environ.get("OPENAI_MODEL"), "gpt-4o-mini") or "gpt-4o-mini"


def resolve_api_key_from_env(provider: Any | None = None) -> str | None:
    """Resolve an API key for the given or env-resolved provider."""
    import caliche

    resolved = provider if provider is not None else resolve_provider_from_env()
    if resolved == caliche.ProviderType.OpenRouter:
        return _first_non_empty(
            os.environ.get("OPENROUTER_API_KEY"), os.environ.get("OPENAI_API_KEY")
        )
    if resolved == caliche.ProviderType.Anthropic:
        return _first_non_empty(os.environ.get("ANTHROPIC_API_KEY"))
    if resolved == caliche.ProviderType.Perplexity:
        return _first_non_empty(os.environ.get("PERPLEXITY_API_KEY"))
    if resolved == caliche.ProviderType.Cohere:
        return _first_non_empty(os.environ.get("COHERE_API_KEY"))
    if resolved == caliche.ProviderType.Google:
        return _first_non_empty(os.environ.get("GEMINI_API_KEY"), os.environ.get("GOOGLE_API_KEY"))
    return _first_non_empty(os.environ.get("OPENAI_API_KEY"))


def configure_builder_from_env(
    builder: Any | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
    json_mode: bool = False,
    include_context_in_system: bool | None = None,
) -> Any:
    """Return an AgentBuilder configured from environment variables."""
    import caliche

    target = builder if builder is not None else caliche.AgentBuilder()
    provider = resolve_provider_from_env()
    target.provider(provider)
    target.model(resolve_model_from_env(provider))

    api_key = resolve_api_key_from_env(provider)
    if api_key:
        target.api_key(api_key)

    base_url = resolve_base_url_from_env(provider)
    if base_url is not None:
        target.base_url(base_url)

    if temperature is not None:
        target.temperature(temperature)
    if max_tokens is not None:
        target.max_tokens(max_tokens)
    if system_prompt is not None:
        target.system_prompt(system_prompt)
    if json_mode:
        target.json_mode()
    if include_context_in_system is not None:
        target.include_context_in_system(include_context_in_system)
    return target


def agent_from_env(**kwargs: Any) -> Any:
    """Build an Agent from environment configuration."""
    return configure_builder_from_env(**kwargs).build()


__all__ = [
    "agent_from_env",
    "configure_builder_from_env",
    "openai_base_url",
    "resolve_api_key_from_env",
    "resolve_base_url_from_env",
    "resolve_model_from_env",
    "resolve_provider_from_env",
]
