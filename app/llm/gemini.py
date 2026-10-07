"""
Google Gemini provider -- uses Google's OpenAI-compatible endpoint or REST API.

Configure with:
    GEMINI_API_KEY   Google AI Studio / Gemini API key
    GEMINI_MODEL     e.g. gemini-2.5-flash, gemini-2.5-pro, gemini-2.0-flash
    GEMINI_BASE_URL  default: https://generativelanguage.googleapis.com/v1beta/openai
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from app.llm.base import LLMError, LLMProvider, LLMResponse


class GeminiProvider(LLMProvider):
    name = "gemini"

    DEFAULT_MODEL = "gemini-2.5-flash"
    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"

    POPULAR_MODELS = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        resolved = model or os.getenv("GEMINI_MODEL") or self.DEFAULT_MODEL
        super().__init__(model=resolved, **kwargs)
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("GEMINI_API_KEY", "")
        )
        self.base_url = (
            base_url
            or os.getenv("GEMINI_BASE_URL", self.DEFAULT_BASE_URL)
        ).rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict:
        if not self.api_key:
            raise LLMError(
                "Gemini API key is missing. Set GEMINI_API_KEY in .env or Settings.",
                retryable=False,
            )
        url = f"{self.base_url}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=self._headers())
                if response.status_code != 200:
                    detail = response.text[:300]
                    try:
                        err_json = response.json()
                        if "error" in err_json:
                            err_info = err_json["error"]
                            msg = err_info.get("message") or err_info.get("status") or str(err_info)
                            detail = f"{response.status_code}: {msg}"
                    except Exception:
                        pass
                    raise LLMError(f"Gemini API error ({detail})", retryable=(response.status_code in {429, 500, 502, 503}))
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"Gemini HTTP {exc.response.status_code}: {exc.response.text[:200]}")
        except httpx.HTTPError as exc:
            raise LLMError(f"Gemini API unreachable at {self.base_url}: {exc}")

    async def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._complete_messages(messages)

    async def _complete_messages(self, messages: list[dict[str, str]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        data = await self._post("/chat/completions", payload)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(
                f"Unexpected Gemini response structure: {str(data)[:200]}",
                retryable=False,
            ) from exc
        if not text.strip():
            raise LLMError("Gemini returned an empty response")
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=text.strip(),
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            raw=data,
        )

    async def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        return await self._complete_messages(messages)

    async def list_available_models(self) -> list[str]:
        if not self.api_key:
            return self.POPULAR_MODELS
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self.base_url}/models", headers=self._headers())
                if resp.status_code == 200:
                    data = resp.json()
                    models = [
                        m["id"]
                        for m in data.get("data", [])
                        if isinstance(m, dict) and "id" in m
                    ]
                    if models:
                        return models
        except Exception:
            pass
        return self.POPULAR_MODELS

