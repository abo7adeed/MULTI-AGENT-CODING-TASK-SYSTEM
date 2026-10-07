"""
Ollama provider: the cloud deployment.

The provider is the seam between this system and a hosted model, so what is
pinned here is what costs real money or real debugging time:

  * the bearer token actually being sent (without it every call is a 401)
  * a rejected key, an unentitled model and a missing model failing *once*,
    with a message naming the fix, instead of being retried per task
  * an answer cut off at the output limit never being mistaken for a model
    that chose not to write anything

Every test runs offline against `httpx.MockTransport`; nothing here needs a
network, a server or a key.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.llm.base import LLMError
from app.llm.factory import create_provider
from app.llm.ollama import OllamaProvider

CLOUD = "https://ollama.com"


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Retry backoff is real; waiting for it in a test is not useful."""

    async def instant(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", instant, raising=False)


@pytest.fixture(autouse=True)
def _no_ambient_ollama_env(monkeypatch):
    """A developer's own OLLAMA_* variables must not leak into these tests."""
    for name in (
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "OLLAMA_API_KEY",
        "OLLAMA_NUM_PREDICT",
    ):
        monkeypatch.delenv(name, raising=False)


def _transport(monkeypatch, handler) -> list[httpx.Request]:
    """Route every `httpx.AsyncClient` through `handler`, recording the calls."""
    calls: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    def wrapped(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request)

    def build(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(wrapped)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", build)
    return calls


def _provider(**kwargs) -> OllamaProvider:
    params = {"base_url": CLOUD, "api_key": "k-1", "model": "gpt-oss:20b"}
    params.update(kwargs)
    return OllamaProvider(**params)


# ── authentication ──────────────────────────────────────────────────────────


class TestAuthentication:
    @pytest.mark.asyncio
    async def test_the_bearer_token_is_sent(self, monkeypatch):
        def handler(request):
            assert request.headers.get("authorization") == "Bearer k-1"
            assert str(request.url) == "https://ollama.com/api/generate"
            return httpx.Response(200, json={"response": "ok", "model": "gpt-oss:20b"})

        calls = _transport(monkeypatch, handler)
        response = await _provider().complete("p", "sys")
        assert response.text == "ok"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_a_local_server_is_called_without_a_key(self, monkeypatch):
        def handler(request):
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"response": "ok"})

        _transport(monkeypatch, handler)
        provider = OllamaProvider(base_url="http://localhost:11434", model="m")
        assert (await provider.complete("p")).text == "ok"

    @pytest.mark.asyncio
    async def test_the_model_list_is_authenticated_too(self, monkeypatch):
        def handler(request):
            assert str(request.url).endswith("/api/tags")
            assert request.headers.get("authorization") == "Bearer k-1"
            return httpx.Response(200, json={"models": [{"name": "gpt-oss:20b"}]})

        _transport(monkeypatch, handler)
        assert await _provider().list_available_models() == ["gpt-oss:20b"]

    def test_is_cloud_is_decided_by_the_host(self):
        assert _provider().is_cloud is True
        assert OllamaProvider(base_url="http://localhost:11434").is_cloud is False


# ── error taxonomy ──────────────────────────────────────────────────────────


class TestErrorTaxonomy:
    """A status code means one of two things: fix the config, or wait."""

    @pytest.mark.asyncio
    async def test_a_rejected_key_fails_once(self, monkeypatch):
        def handler(request):
            return httpx.Response(401, json={"error": "unauthorized"})

        calls = _transport(monkeypatch, handler)
        with pytest.raises(LLMError) as excinfo:
            await _provider(max_retries=3).complete("p", "sys")
        assert excinfo.value.retryable is False
        assert "OLLAMA_API_KEY" in str(excinfo.value)
        assert len(calls) == 1, "a rejected key must not be retried"

    @pytest.mark.asyncio
    async def test_an_unentitled_model_keeps_the_servers_own_reason(self, monkeypatch):
        server_text = (
            "this model is not included in your free usage, add usage credits to pay as you go"
        )

        def handler(request):
            return httpx.Response(402, json={"error": server_text})

        calls = _transport(monkeypatch, handler)
        with pytest.raises(LLMError) as excinfo:
            await _provider(max_retries=3).complete("p", "sys")
        message = str(excinfo.value)
        assert excinfo.value.retryable is False
        assert server_text in message
        assert "ollama cloud HTTP 402" in message
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_a_wrong_model_name_says_how_to_fix_it(self, monkeypatch):
        def handler(request):
            return httpx.Response(404, json={"error": "model not found"})

        _transport(monkeypatch, handler)
        with pytest.raises(LLMError) as excinfo:
            await _provider().complete("p", "sys")
        message = str(excinfo.value)
        assert excinfo.value.retryable is False
        assert "ollama pull" in message
        assert ":cloud" in message  # the CLI/API naming trap

    @pytest.mark.asyncio
    async def test_a_rate_limit_is_retried(self, monkeypatch):
        attempts = {"n": 0}

        def handler(request):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(429, json={"error": "slow down"})
            return httpx.Response(200, json={"response": "ok"})

        calls = _transport(monkeypatch, handler)
        assert (await _provider(max_retries=2).complete("p")).text == "ok"
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_a_server_error_is_retried(self, monkeypatch):
        attempts = {"n": 0}

        def handler(request):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(503, text="upstream busy")
            return httpx.Response(200, json={"response": "ok"})

        calls = _transport(monkeypatch, handler)
        assert (await _provider(max_retries=2).complete("p")).text == "ok"
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_a_connection_failure_is_retryable(self, monkeypatch):
        def handler(request):
            raise httpx.ConnectError("no route to host")

        _transport(monkeypatch, handler)
        with pytest.raises(LLMError) as excinfo:
            await _provider(max_retries=0).complete("p")
        assert excinfo.value.retryable is True
        assert "unreachable" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_a_non_json_body_is_reported_as_such(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, text="<html>gateway</html>")

        calls = _transport(monkeypatch, handler)
        with pytest.raises(LLMError) as excinfo:
            await _provider(max_retries=3).complete("p")
        assert excinfo.value.retryable is False
        assert "non-JSON" in str(excinfo.value)
        assert len(calls) == 1


# ── truncation ──────────────────────────────────────────────────────────────


class TestTruncation:
    """
    `done_reason: length` means the budget ran out mid-answer. On a coding task
    that is a half-written file, and reporting it as "the agent returned no file
    changes" sends the operator looking in the wrong place entirely.
    """

    @pytest.mark.asyncio
    async def test_an_answer_cut_off_at_the_limit_is_not_a_refusal(self, monkeypatch):
        def handler(request):
            return httpx.Response(
                200,
                json={"response": "", "done_reason": "length", "thinking": "thinking"},
            )

        calls = _transport(monkeypatch, handler)
        with pytest.raises(LLMError) as excinfo:
            await _provider(max_retries=3).complete("p", "sys")
        message = str(excinfo.value)
        assert excinfo.value.retryable is False
        assert "output limit" in message
        assert "OLLAMA_NUM_PREDICT" in message
        assert len(calls) == 1, "a truncated answer must not be retried blindly"

    @pytest.mark.asyncio
    async def test_a_half_written_file_is_not_accepted_either(self, monkeypatch):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "response": '<file path="a.py">\ndef f(',
                    "done_reason": "length",
                },
            )

        _transport(monkeypatch, handler)
        with pytest.raises(LLMError, match="output limit"):
            await _provider().complete("p")

    @pytest.mark.asyncio
    async def test_reasoning_without_an_answer_is_named(self, monkeypatch):
        def handler(request):
            assert str(request.url).endswith("/api/chat")
            return httpx.Response(
                200,
                json={
                    "message": {"content": "", "thinking": "let me think"},
                    "done_reason": "stop",
                },
            )

        _transport(monkeypatch, handler)
        with pytest.raises(LLMError, match="reasoning but no answer") as excinfo:
            await _provider(max_retries=3).chat([{"role": "user", "content": "hi"}])
        assert excinfo.value.retryable is False

    @pytest.mark.asyncio
    async def test_an_empty_answer_is_still_retryable(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, json={"response": ""})

        calls = _transport(monkeypatch, handler)
        with pytest.raises(LLMError, match="empty response") as excinfo:
            await _provider(max_retries=1).complete("p")
        assert excinfo.value.retryable is True
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_reasoning_text_is_not_returned_as_the_answer(self, monkeypatch):
        def handler(request):
            return httpx.Response(
                200, json={"response": "the answer", "thinking": "the reasoning"}
            )

        _transport(monkeypatch, handler)
        response = await _provider().complete("p")
        assert response.text == "the answer"


# ── request shape ───────────────────────────────────────────────────────────


class TestRequestShape:
    @pytest.mark.asyncio
    async def test_the_output_budget_is_sent_explicitly(self, monkeypatch):
        seen = {}

        def handler(request):
            import json

            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"response": "ok"})

        _transport(monkeypatch, handler)
        await _provider().complete("p", "sys")
        assert seen["options"]["num_predict"] == 8192
        assert seen["options"]["temperature"] == 0.1
        assert seen["stream"] is False
        assert seen["system"] == "sys"

    @pytest.mark.asyncio
    async def test_the_budget_is_configurable(self, monkeypatch):
        seen = {}

        def handler(request):
            import json

            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"response": "ok"})

        _transport(monkeypatch, handler)
        await _provider(num_predict=64).complete("p")
        assert seen["options"]["num_predict"] == 64


# ── model defaults ──────────────────────────────────────────────────────────


class TestModelDefaults:
    def test_a_cloud_base_url_gets_a_cloud_model(self):
        assert _provider(model=None).model == "gpt-oss:120b"

    def test_a_local_base_url_keeps_the_local_default(self):
        assert OllamaProvider(base_url="http://localhost:11434").model == "qwen2.5-coder:7b"

    def test_an_explicit_model_always_wins(self):
        assert _provider(model="gpt-oss:20b").model == "gpt-oss:20b"

    def test_the_env_var_beats_the_deployment_default(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "from-env")
        assert _provider(model=None).model == "from-env"

    def test_the_factory_wires_the_key_the_budget_and_the_model(self, settings):
        settings.llm_model = ""
        settings.ollama_model = ""
        settings.ollama_base_url = CLOUD
        settings.ollama_api_key = "k-2"
        settings.ollama_num_predict = 512

        provider = create_provider("ollama", settings=settings)
        assert provider.name == "ollama"
        assert provider.is_cloud is True
        assert provider.api_key == "k-2"
        assert provider.num_predict == 512
        assert provider.model == "gpt-oss:120b"

    def test_the_factory_still_honours_a_configured_model(self, settings):
        settings.llm_model = ""
        settings.ollama_model = "gpt-oss:20b"
        settings.ollama_base_url = CLOUD
        assert create_provider("ollama", settings=settings).model == "gpt-oss:20b"


# ── the startup probe ───────────────────────────────────────────────────────


class TestProbe:
    """A wrong model name should be a line in the log, not four retries a task."""

    @pytest.mark.asyncio
    async def test_a_reachable_model_is_reported_ready(self, monkeypatch):
        def handler(request):
            return httpx.Response(
                200,
                json={"models": [{"name": "gpt-oss:20b"}, {"name": "gemma4:31b"}]},
            )

        _transport(monkeypatch, handler)
        probe = await _provider().probe()
        assert probe["reachable"] is True
        assert probe["model_available"] is True
        assert probe["error"] is None
        assert probe["cloud"] is True
        assert probe["models"] == ["gpt-oss:20b", "gemma4:31b"]

    @pytest.mark.asyncio
    async def test_a_model_that_is_not_offered_is_reported_with_the_list(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, json={"models": [{"name": "gemma4:31b"}]})

        _transport(monkeypatch, handler)
        probe = await _provider(model="nope:1b").probe()
        assert probe["reachable"] is True
        assert probe["model_available"] is False
        assert "nope:1b" in probe["error"]
        assert "gemma4:31b" in probe["error"]

    @pytest.mark.asyncio
    async def test_a_model_gap_on_the_cloud_mentions_the_paid_plan(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, json={"models": [{"name": "gemma4:31b"}]})

        _transport(monkeypatch, handler)
        probe = await _provider(model="kimi-k2.7-code").probe()
        assert "402" in probe["error"]

    @pytest.mark.asyncio
    async def test_an_unreachable_server_is_reported_not_raised(self, monkeypatch):
        def handler(request):
            raise httpx.ConnectError("no route to host")

        _transport(monkeypatch, handler)
        probe = await _provider().probe()
        assert probe["reachable"] is False
        assert probe["model_available"] is None
        assert "unreachable" in probe["error"]

    @pytest.mark.asyncio
    async def test_a_rejected_key_is_reported_not_raised(self, monkeypatch):
        def handler(request):
            return httpx.Response(401, json={"error": "unauthorized"})

        _transport(monkeypatch, handler)
        probe = await _provider(api_key="").probe()
        assert probe["reachable"] is False
        assert "OLLAMA_API_KEY" in probe["error"]

    @pytest.mark.asyncio
    async def test_an_empty_catalogue_is_reported(self, monkeypatch):
        def handler(request):
            return httpx.Response(200, json={"models": []})

        _transport(monkeypatch, handler)
        probe = await _provider().probe()
        assert probe["reachable"] is True
        assert probe["model_available"] is None
        assert "no models" in probe["error"]
