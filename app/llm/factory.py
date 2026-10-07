"""
Provider factory.

The single place that knows how to turn a string like ``"opencode"`` into a
live provider object. Everything downstream takes an `LLMProvider`, so swapping
models is a config change, never a code change.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.llm.base import LLMError, LLMProvider, LLMResponse
from app.llm.mock import CallbackLLM, MockLLMProvider, RuleBasedLLM, ScriptedLLM
from app.llm.ollama import OllamaProvider
from app.llm.opencode import OpenCodeProvider
from app.llm.api import APIProvider
from app.llm.gemini import GeminiProvider

# Imported eagerly so `from app.llm import ...` keeps working everywhere.
__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "MockLLMProvider",
    "ScriptedLLM",
    "CallbackLLM",
    "RuleBasedLLM",
    "OllamaProvider",
    "OpenCodeProvider",
    "APIProvider",
    "GeminiProvider",
    "create_provider",
    "list_providers",
    "PROVIDER_REGISTRY",
]

PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "opencode": OpenCodeProvider,
    "ollama": OllamaProvider,
    "api": APIProvider,
    "mock": MockLLMProvider,
    "scripted": ScriptedLLM,
    "rule-based": RuleBasedLLM,
    "callback": CallbackLLM,
}


def _resolve_model(provider_name: str, settings: Settings, override: str | None) -> str:
    if override:
        return override
    if provider_name == "gemini":
        return settings.llm_model or settings.gemini_model
    if provider_name == "opencode":
        return settings.llm_model or settings.opencode_model
    if provider_name == "ollama":
        # Falling through to the provider is deliberate: an unset model must
        # resolve to the cloud default or the local default, not to one guess.
        return settings.llm_model or settings.ollama_model or ""
    if provider_name == "api":
        return settings.llm_model or settings.api_model
    return settings.llm_model or "mock-model"


def create_provider(
    provider_name: str | None = None,
    model: str | None = None,
    settings: Settings | None = None,
    **kwargs,
) -> LLMProvider:
    """
    Build a provider. Falls back to the mock provider when the requested one
    is unknown, so a typo in config degrades instead of crashing a run.
    """
    settings = settings or get_settings()
    name = (provider_name or settings.llm_provider or "mock").strip().lower()
    provider_cls = PROVIDER_REGISTRY.get(name)
    if provider_cls is None:
        from app.logging_config import get_logger

        get_logger("app.llm").warning(
            "Unknown LLM provider '%s', falling back to mock", extra={"provider": name}
        )
        provider_cls = MockLLMProvider

    resolved_model = _resolve_model(name, settings, model)
    common = {
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
        "model": resolved_model,
    }
    common.update(kwargs)

    if name == "gemini":
        return GeminiProvider(
            base_url=settings.gemini_base_url,
            api_key=settings.gemini_api_key,
            model=resolved_model,
            timeout=common["timeout"],
            max_retries=common["max_retries"],
        )
    if name == "api":
        return APIProvider(
            base_url=settings.api_base_url,
            api_key=settings.api_key,
            model=resolved_model,
            timeout=common["timeout"],
            max_retries=common["max_retries"],
        )
    if name == "ollama":
        # One provider serves a local server and Ollama Cloud; the base URL
        # decides which, and the API key is simply unused by a local server.
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
            model=resolved_model or None,
            num_predict=settings.ollama_num_predict,
            timeout=common["timeout"],
            max_retries=common["max_retries"],
        )
    if name == "opencode":
        return OpenCodeProvider(
            model=resolved_model, timeout=common["timeout"], max_retries=common["max_retries"]
        )
    return provider_cls(**common)


def list_providers() -> list[dict[str, str]]:
    return [
        {"id": "gemini", "label": "Google Gemini (Gemini 2.5 Flash, Pro)"},
        {"id": "api", "label": "OpenAI / Compatible API (GPT-4o, Groq, DeepSeek)"},
        {"id": "ollama", "label": "Ollama (local server or Ollama Cloud)"},
        {"id": "opencode", "label": "OpenCode CLI"},
        {"id": "rule-based", "label": "Rule-based (offline demo planner)"},
        {"id": "mock", "label": "Mock (offline canned data)"},
    ]

