"""
Ollama provider -- one HTTP API, two deployments.

    local server      OLLAMA_BASE_URL=http://localhost:11434  (the default)
    Ollama Cloud      OLLAMA_BASE_URL=https://ollama.com      + OLLAMA_API_KEY

Configure with:
    OLLAMA_BASE_URL   local server, or https://ollama.com for the cloud
    OLLAMA_API_KEY    required by ollama.com; ignored by a local server
    OLLAMA_MODEL      optional. Empty means "the default for this base URL"

Note on cloud model names: the API takes the names `/api/tags` returns
(`gpt-oss:120b`), while the Ollama app and CLI show the same models with a
`:cloud` suffix. Asking the API for `gpt-oss:120b-cloud` is a 404.

Distinguishing the failure modes is most of the value here. "The API key is
rejected", "this model is not in your plan", "the model is not pulled" and
"the network blipped" need four different reactions from the operator, and
three of them will not improve by being retried.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from app.llm.base import LLMError, LLMProvider, LLMResponse

#: Statuses that mean "fix the configuration", not "try again". Retrying these
#: spends attempts and tokens to arrive at the same error.
_FATAL_STATUSES = frozenset({400, 401, 402, 403, 404, 405, 422})

#: What the operator should actually do about each of them.
_STATUS_HINTS: dict[int, str] = {
    400: "the request was rejected as malformed",
    401: "the API key is missing or was rejected -- set OLLAMA_API_KEY",
    402: "this model is not included in your plan -- add usage credits, or pick a model your plan covers",
    403: "access to this model or endpoint was refused",
    404: "no such model: on a local server it is not pulled (ollama pull <model>), on ollama.com the name is wrong -- use the exact name from /api/tags, without any ':cloud' suffix",
    405: "the endpoint does not support this method",
    422: "the request was rejected as unprocessable",
}

#: Cloud models are addressed by different names than local ones, so an empty
#: OLLAMA_MODEL has to resolve differently per deployment.
DEFAULT_CLOUD_MODEL = "gpt-oss:120b"
DEFAULT_LOCAL_MODEL = "qwen2.5-coder:7b"


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        num_predict: int | None = None,
        temperature: float = 0.1,
        **kwargs,
    ):
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        # Containment matters more than breadth here: the agents write whole
        # files, and a cut-off answer looks exactly like a model that decided
        # not to write anything.
        self.num_predict = int(
            num_predict
            if num_predict is not None
            else os.getenv("OLLAMA_NUM_PREDICT", "8192")
        )
        self.temperature = temperature
        resolved = (
            model
            or os.getenv("OLLAMA_MODEL")
            or (DEFAULT_CLOUD_MODEL if self.is_cloud else DEFAULT_LOCAL_MODEL)
        )
        super().__init__(model=resolved, **kwargs)
        self.api_key = (
            api_key if api_key is not None else os.getenv("OLLAMA_API_KEY", "")
        )

    # ── request plumbing ────────────────────────────────────────────────────

    @property
    def is_cloud(self) -> bool:
        """Whether this points at ollama.com rather than a local server."""
        return "ollama.com" in self.base_url

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _options(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "num_predict": self.num_predict}

    @staticmethod
    def _server_message(response: httpx.Response) -> str:
        """The provider's own error text, which is usually the useful part."""
        try:
            body = response.json()
        except ValueError:
            return (response.text or "").strip()[:300]
        if isinstance(body, dict):
            for key in ("error", "message", "detail"):
                value = body.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:300]
        return str(body)[:300]

    def _error_for(self, response: httpx.Response) -> LLMError:
        status = response.status_code
        detail = self._server_message(response) or "no response body"
        where = "ollama cloud" if self.is_cloud else "ollama"
        hint = _STATUS_HINTS.get(status)
        message = f"{where} HTTP {status}"
        if hint:
            message += f" -- {hint}"
        message += f": {detail}"
        if status in _FATAL_STATUSES:
            return LLMError(message, retryable=False)
        if status == 429 or status >= 500:
            return LLMError(message, retryable=True)
        return LLMError(message, retryable=False)

    async def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=self._headers())
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"ollama timed out after {self.timeout}s at {self.base_url}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama unreachable at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            raise self._error_for(response)
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(
                f"ollama returned a non-JSON body from {endpoint}: "
                f"{response.text[:200]}",
                retryable=False,
            ) from exc
        if not isinstance(body, dict):
            raise LLMError(
                f"ollama returned {type(body).__name__} where an object was expected",
                retryable=False,
            )
        return body

    # ── reply interpretation ────────────────────────────────────────────────

    def _check_finish(self, data: dict, text: str, thinking: str) -> None:
        """
        Fail loudly when the answer stopped for a reason other than finishing.

        `done_reason: length` means the budget ran out mid-answer. On a coding
        task that is a half-written file, which must never be mistaken for
        "the model chose to change nothing".
        """
        if data.get("done_reason") != "length":
            return
        budget = f"OLLAMA_NUM_PREDICT (currently {self.num_predict})"
        spent = "the model spent its whole budget reasoning" if thinking else "no reasoning was recorded"
        detail = (
            f" -- raise {budget}" if not thinking else f" -- raise {budget}; {spent}"
        )
        raise LLMError(
            f"ollama stopped at the output limit for '{self.model}'"
            f"{'' if text else ' before producing an answer'}{detail}",
            retryable=False,
        )

    def _build_response(self, data: dict, text: str, thinking: str = "") -> LLMResponse:
        self._check_finish(data, text, thinking)
        if not text:
            if thinking:
                raise LLMError(
                    f"ollama returned reasoning but no answer text for '{self.model}'"
                    f" -- raise OLLAMA_NUM_PREDICT (currently {self.num_predict}) or use "
                    "a model that answers directly",
                    retryable=False,
                )
            raise LLMError("ollama returned an empty response")
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
            raw=data,
        )

    # ── LLMProvider surface ─────────────────────────────────────────────────

    async def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        data = await self._post(
            "/api/generate",
            {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": self._options(),
            },
        )
        # Reasoning models put their working in `thinking`; only the answer is
        # the answer.
        return self._build_response(
            data,
            (data.get("response") or "").strip(),
            (data.get("thinking") or "").strip(),
        )

    async def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        data = await self._post(
            "/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": self._options(),
            },
        )
        message = data.get("message") or {}
        return self._build_response(
            data,
            (message.get("content") or "").strip(),
            (message.get("thinking") or "").strip(),
        )

    # ── discovery & diagnosis ───────────────────────────────────────────────

    async def list_available_models(self) -> list[str]:
        try:
            return await self._fetch_models()
        except LLMError:
            return [self.model]

    async def _fetch_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama unreachable at {self.base_url}: {exc}") from exc
        if response.status_code != 200:
            raise self._error_for(response)
        try:
            body = response.json()
        except ValueError as exc:
            raise LLMError(
                f"ollama returned a non-JSON model list: {response.text[:200]}",
                retryable=False,
            ) from exc
        models = body.get("models", []) if isinstance(body, dict) else body
        names: list[str] = []
        if isinstance(models, list):
            for entry in models:
                if isinstance(entry, dict):
                    name = entry.get("name") or entry.get("model")
                    if isinstance(name, str) and name:
                        names.append(name)
                elif isinstance(entry, str):
                    names.append(entry)
        return names

    async def probe(self) -> dict[str, Any]:
        """
        Can the configured endpoint be reached, and is the model there?

        Never raises: being unable to answer is exactly what this reports. The
        API caches the result at startup and shows it on /health, so a wrong
        model name is a line in the log rather than four retries per task.
        """
        info: dict[str, Any] = {
            "provider": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "cloud": self.is_cloud,
            "has_key": bool(self.api_key),
            "reachable": False,
            "model_available": None,
            "models": [],
            "error": None,
        }
        try:
            models = await self._fetch_models()
        except LLMError as exc:
            info["error"] = str(exc)
            return info

        info["reachable"] = True
        info["models"] = models
        if not models:
            info["model_available"] = None
            info["error"] = (
                f"{self.base_url} offered no models; nothing can be run against it yet"
            )
            return info

        info["model_available"] = self.model in models
        if not info["model_available"]:
            limit = 25
            shown = ", ".join(models[:limit])
            more = "" if len(models) <= limit else f" (+{len(models) - limit} more)"
            extra = (
                " On ollama.com a model can also exist but sit outside your plan, "
                "which fails with HTTP 402 at run time."
                if self.is_cloud
                else ""
            )
            info["error"] = (
                f"model '{self.model}' is not available from {self.base_url}. "
                f"Available: {shown}{more}.{extra}"
            )
        return info
