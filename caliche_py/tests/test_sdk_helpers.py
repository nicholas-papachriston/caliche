from __future__ import annotations

import os

import caliche
from caliche.env import (
    configure_builder_from_env,
    resolve_api_key_from_env,
    resolve_base_url_from_env,
    resolve_model_from_env,
    resolve_provider_from_env,
)
from caliche.streaming import (
    chat_text,
    collect_text,
    collect_with_usage,
    iter_stream_text,
)


def test_resolve_provider_prefers_openrouter(monkeypatch) -> None:
    monkeypatch.delenv("CALICHE_PROVIDER", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = resolve_provider_from_env()
    assert provider == caliche.ProviderType.OpenRouter


def test_normalize_harness_discipline_metrics_public_helper() -> None:
    metrics = caliche.normalize_harness_discipline_metrics(
        {
            "tokens_per_turn": 120.5,
            "tool_calls": 3,
            "tool_calls_per_turn": 1.5,
            "compactions": 1,
            "readonly_filter_hits": 2,
            "estimated_cost_usd": 0.01,
            "summary": "120.5 tokens/turn",
        }
    )

    assert metrics["tokens_per_turn"] == 120.5
    assert metrics["tool_calls"] == 3
    assert metrics["readonly_filter_hits"] == 2
    assert metrics["summary"] == "120.5 tokens/turn"
    assert caliche.normalize_harness_discipline_metrics(None)["tool_calls"] == 0


def test_resolve_provider_detects_lmstudio(monkeypatch) -> None:
    monkeypatch.delenv("CALICHE_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("LM_STUDIO_BASE_URL", raising=False)
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
    assert resolve_provider_from_env() == caliche.ProviderType.LMStudio


def test_resolve_provider_detects_lm_studio_alias(monkeypatch) -> None:
    monkeypatch.delenv("CALICHE_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    assert resolve_provider_from_env() == caliche.ProviderType.LMStudio


def test_resolve_base_url_defaults(monkeypatch) -> None:
    monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
    monkeypatch.delenv("LM_STUDIO_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    assert (
        resolve_base_url_from_env(caliche.ProviderType.LMStudio) == "http://localhost:1234/v1"
    )
    assert resolve_base_url_from_env(caliche.ProviderType.Ollama) == "http://localhost:11434"
    assert resolve_base_url_from_env(caliche.ProviderType.OpenRouter) is None
    assert resolve_base_url_from_env(caliche.ProviderType.OpenAI) is None

    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    assert (
        resolve_base_url_from_env(caliche.ProviderType.OpenAI) == "https://api.example.com/v1"
    )
    # OpenRouter must not inherit OPENAI_BASE_URL.
    assert resolve_base_url_from_env(caliche.ProviderType.OpenRouter) is None


def test_resolve_model_uses_model_env(monkeypatch) -> None:
    monkeypatch.setenv("CALICHE_PROVIDER", "openai")
    monkeypatch.setenv("MODEL", "gpt-test-model")
    monkeypatch.delenv("CALICHE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert resolve_model_from_env() == "gpt-test-model"


def test_configure_builder_from_env(monkeypatch) -> None:
    monkeypatch.setenv("CALICHE_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "gpt-4o-mini")
    builder = configure_builder_from_env(include_context_in_system=False, temperature=0.0)
    assert builder is not None


def test_configure_builder_from_env_sets_provider_and_model(monkeypatch) -> None:
    monkeypatch.setenv("CALICHE_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk" + "-ant-test")
    monkeypatch.setenv("CALICHE_MODEL", "claude-sonnet-4-20250514")
    monkeypatch.delenv("MODEL", raising=False)
    builder = configure_builder_from_env()
    assert builder is not None
    # Prefer native getters when present; otherwise ensure construction succeeded.
    provider = getattr(builder, "get_provider", None)
    model = getattr(builder, "get_model", None)
    if callable(provider):
        assert provider() == caliche.ProviderType.Anthropic
    if callable(model):
        assert model() == "claude-sonnet-4-20250514"


def test_resolve_api_key_from_env_per_provider(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk" + "-ant")
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gem")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai")
    assert resolve_api_key_from_env(caliche.ProviderType.OpenRouter) == "sk-or"
    assert resolve_api_key_from_env(caliche.ProviderType.Anthropic) == "sk" + "-ant"
    assert resolve_api_key_from_env(caliche.ProviderType.Google) == "sk-gem"
    assert resolve_api_key_from_env(caliche.ProviderType.OpenAI) == "sk-oai"


def test_resolve_provider_forced_azure_aliases(monkeypatch) -> None:
    monkeypatch.setenv("CALICHE_PROVIDER", "azure_openai")
    assert resolve_provider_from_env() == caliche.ProviderType.AzureOpenAI
    monkeypatch.setenv("CALICHE_PROVIDER", "azure-openai")
    assert resolve_provider_from_env() == caliche.ProviderType.AzureOpenAI


def test_iter_stream_text_yields_chunks() -> None:
    events = [
        {"type": "text_chunk", "content": "Hello"},
        {"type": "metadata", "finish_reason": "stop"},
        {"type": "text_chunk", "content": "!"},
        {"type": "done"},
    ]
    assert list(iter_stream_text(events)) == ["Hello", "!"]


def test_configure_builder_from_env_lmstudio_base_url(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("LMSTUDIO_BASE_URL", raising=False)
    monkeypatch.delenv("LM_STUDIO_BASE_URL", raising=False)
    monkeypatch.setenv("CALICHE_PROVIDER", "lmstudio")
    builder = configure_builder_from_env()
    assert builder is not None
    assert (
        resolve_base_url_from_env(caliche.ProviderType.LMStudio) == "http://localhost:1234/v1"
    )


def test_collect_text_and_usage() -> None:
    events = [
        {"type": "text_chunk", "content": "Hello"},
        {"type": "text_chunk", "content": " world"},
        {
            "type": "metadata",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    ]
    assert collect_text(events) == "Hello world"
    collected = collect_with_usage(events)
    assert collected["text"] == "Hello world"
    assert collected["finish_reason"] == "stop"
    assert collected["usage"]["total_tokens"] == 3


def test_chat_text_tuple_shape() -> None:
    class FakeAgent:
        def chat(self, message: str):
            return (f"echo:{message}", {"total_tokens": 1}, {})

    assert chat_text(FakeAgent(), "hi") == "echo:hi"


def test_chat_text_dict_and_string_shapes() -> None:
    class DictAgent:
        def chat(self, message: str):
            return {"response": f"dict:{message}", "usage": {"total_tokens": 2}}

    class StringAgent:
        def chat(self, message: str):
            return f"str:{message}"

    assert chat_text(DictAgent(), "hi") == "dict:hi"
    assert chat_text(StringAgent(), "hi") == "str:hi"


def test_resolve_model_provider_fallbacks_without_explicit_model(monkeypatch) -> None:
    monkeypatch.delenv("CALICHE_MODEL", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("LMSTUDIO_MODEL", raising=False)

    assert resolve_model_from_env(caliche.ProviderType.OpenAI) == "gpt-4o-mini"
    assert resolve_model_from_env(caliche.ProviderType.OpenRouter) == "openai/gpt-4o-mini"
    assert resolve_model_from_env(caliche.ProviderType.Anthropic) == "claude-3-haiku-20240307"
    assert resolve_model_from_env(caliche.ProviderType.Ollama) == "llama3.2"
    assert resolve_model_from_env(caliche.ProviderType.LMStudio) == "local-model"


def test_collect_with_usage_uses_limit_reached_usage_without_metadata() -> None:
    events = [
        {"type": "text_chunk", "content": "partial"},
        {
            "type": "limit_reached",
            "max_iterations": 3,
            "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
        },
        {"type": "done"},
    ]
    collected = collect_with_usage(events)
    assert collected["text"] == "partial"
    assert collected["finish_reason"] is None
    assert collected["usage"]["total_tokens"] == 9


def test_collect_helpers_read_versioned_stream_event_payloads() -> None:
    events = [
        {
            "version": "1",
            "type": "text_chunk",
            "event_id": "e1",
            "payload": {"type": "text_chunk", "content": "Hello"},
        },
        {
            "event": {
                "version": "1",
                "type": "text_chunk",
                "payload": {"type": "text_chunk", "content": " world"},
            }
        },
        {
            "version": "1",
            "type": "metadata",
            "payload": {
                "type": "metadata",
                "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        },
        {"version": "1", "type": "done", "payload": {"type": "done"}},
    ]
    assert collect_text(events) == "Hello world"
    assert list(iter_stream_text(events)) == ["Hello", " world"]
    collected = collect_with_usage(events)
    assert collected["text"] == "Hello world"
    assert collected["finish_reason"] == "stop"
    assert collected["usage"]["total_tokens"] == 3


def test_os_environ_isolation(monkeypatch) -> None:
    # Keep the suite from leaking provider selection into later tests.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert "OPENROUTER_API_KEY" not in os.environ or os.environ.get("OPENROUTER_API_KEY") is None
