"""
Generic REST API provider -- works with any OpenAI-compatible endpoint.

Configure with:
    API_BASE_URL   e.g. https://api.openai.com/v1
    API_KEY        bearer token
    API_MODEL      model name
"""

from __future__ import annotations

import os

import httpx

from app.llm.base import LLMError, LLMProvider, LLMResponse


class APIProvider(LLMProvider):
    name = "api"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        resolved = model or os.getenv("API_MODEL") or "gpt-4o-mini"
        super().__init__(model=resolved, **kwargs)
        self.base_url = (
            base_url or os.getenv("API_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("API_KEY", "")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=self._headers())
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise LLMError(
                f"API HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"API unreachable at {self.base_url}: {exc}")

    async def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self._complete_messages(messages)

    async def _complete_messages(self, messages: list[dict[str, str]]) -> LLMResponse:
        data = await self._post(
            "/chat/completions", {"model": self.model, "messages": messages}
        )
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(
                f"unexpected API response shape: {str(data)[:200]}", retryable=False
            ) from exc
        if not text.strip():
            raise LLMError("API returned an empty response")
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
        popular = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet", "deepseek-chat", "llama-3.3-70b-versatile"]
        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            return [self.model]
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/models", headers=self._headers())
                if response.status_code == 200:
                    data = response.json()
                    models = [
                        m["id"]
                        for m in data.get("data", [])
                        if isinstance(m, dict) and "id" in m
                    ]
                    if models:
                        return models
        except Exception:
            pass
        return [self.model] if self.model not in popular else popular
