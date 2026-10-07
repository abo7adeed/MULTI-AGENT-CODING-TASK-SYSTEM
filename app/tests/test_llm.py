"""
Provider abstraction tests.

Nothing here touches the network: the HTTP providers are exercised by
monkeypatching `_post`, and the CLI provider by pointing it at a non-existent
binary. What is being tested is the *contract* the rest of the system relies
on -- retries, timeouts, error classification, model resolution and stats.
"""

from __future__ import annotations

import asyncio
import json
import re

import pytest

from app.llm.api import APIProvider
from app.llm.base import LLMError, LLMProvider, LLMResponse
from app.llm.factory import create_provider, list_providers
from app.llm.mock import CallbackLLM, MockLLMProvider, RuleBasedLLM, ScriptedLLM
from app.llm.ollama import OllamaProvider
from app.llm.opencode import OpenCodeProvider, parse_json_response


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch):
    """Retry backoff is real, but not in the test suite."""
    sleeps: list[float] = []

    async def instant(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", instant, raising=False)
    return sleeps


class _Boom(LLMProvider):
    """Provider that fails a fixed number of times before succeeding."""

    name = "boom"

    def __init__(self, failures: int, exc: Exception | None = None, **kwargs):
        super().__init__(**kwargs)
        self.failures = failures
        self.attempts = 0
        self._exc = exc

    async def _call(self, prompt: str, system_prompt: str) -> LLMResponse:
        self.attempts += 1
        if self.attempts <= self.failures:
            raise (self._exc or ConnectionError("socket hang up"))
        return LLMResponse(text="ok", model=self.model)


class TestLLMResponse:
    def test_str_is_the_text(self):
        assert str(LLMResponse(text="hello")) == "hello"


class TestComplete:
    @pytest.mark.asyncio
    async def test_stamps_provider_and_model(self):
        provider = MockLLMProvider(response="hi", model="m1")
        response = await provider.complete("p")
        assert response.text == "hi"
        assert response.provider == "mock"
        assert response.model == "m1"

    @pytest.mark.asyncio
    async def test_server_model_wins_over_configured_default(self):
        class Server(_Boom):
            async def _call(self, prompt, system_prompt):
                return LLMResponse(text="x", model="from-server")

        response = await Server(failures=0, model="configured").complete("p")
        assert response.model == "from-server"

    @pytest.mark.asyncio
    async def test_records_stats(self):
        provider = MockLLMProvider(model="m")
        await provider.complete("p")
        await provider.complete("p")
        stats = provider.stats
        assert stats["provider"] == "mock"
        assert stats["model"] == "m"
        assert stats["calls"] == 2
        assert stats["total_seconds"] >= 0.0

    @pytest.mark.asyncio
    async def test_token_counts_accumulate(self):
        class Token(_Boom):
            async def _call(self, prompt, system_prompt):
                return LLMResponse(text="x", prompt_tokens=10, completion_tokens=5)

        provider = Token(failures=0)
        await provider.complete("p")
        await provider.complete("p")
        assert provider.stats["tokens"] == 30

    @pytest.mark.asyncio
    async def test_transient_failure_is_retried_then_succeeds(self):
        provider = _Boom(failures=2, max_retries=3)
        response = await provider.complete("p")
        assert response.text == "ok"
        assert provider.attempts == 3

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self):
        provider = _Boom(failures=99, max_retries=2)
        with pytest.raises(LLMError, match="socket hang up"):
            await provider.complete("p")
        assert provider.attempts == 3  # initial + 2 retries

    @pytest.mark.asyncio
    async def test_zero_retries_means_one_attempt(self):
        provider = _Boom(failures=1, max_retries=0)
        with pytest.raises(LLMError):
            await provider.complete("p")
        assert provider.attempts == 1

    @pytest.mark.asyncio
    async def test_backoff_grows_and_is_capped(self, _no_backoff_sleep):
        provider = _Boom(failures=99, max_retries=10)
        with pytest.raises(LLMError):
            await provider.complete("p")
        assert _no_backoff_sleep[:3] == [0.5, 1.0, 2.0]
        assert max(_no_backoff_sleep) <= 8.0

    @pytest.mark.asyncio
    async def test_timeout_becomes_an_llm_error(self):
        class Slow(_Boom):
            async def _call(self, prompt, system_prompt):
                await asyncio.get_event_loop().run_in_executor(None, lambda: None)
                await _real_sleep(0.5)
                return LLMResponse(text="late")

        provider = Slow(failures=0, timeout=0.01, max_retries=0)
        with pytest.raises(LLMError, match="timed out"):
            await provider.complete("p")

    @pytest.mark.asyncio
    async def test_non_retryable_llm_error_fails_immediately(self):
        class Fatal(_Boom):
            async def _call(self, prompt, system_prompt):
                self.attempts += 1
                raise LLMError("cli not installed", retryable=False)

        provider = Fatal(failures=0, max_retries=5)
        with pytest.raises(LLMError, match="not installed"):
            await provider.complete("p")
        assert provider.attempts == 1

    @pytest.mark.asyncio
    async def test_retryable_llm_error_is_retried(self):
        class Flaky(_Boom):
            async def _call(self, prompt, system_prompt):
                self.attempts += 1
                raise LLMError("HTTP 503", retryable=True)

        provider = Flaky(failures=0, max_retries=2)
        with pytest.raises(LLMError, match="503"):
            await provider.complete("p")
        assert provider.attempts == 3

    @pytest.mark.asyncio
    async def test_cause_is_preserved_for_debugging(self):
        boom = ConnectionError("dns")
        provider = _Boom(failures=1, exc=boom, max_retries=0)
        with pytest.raises(LLMError) as excinfo:
            await provider.complete("p")
        assert excinfo.value.__cause__ is boom


# `asyncio.sleep` is patched away in this module, so keep a real handle for the
# timeout test which genuinely needs the event loop to keep turning.
import asyncio as _asyncio  # noqa: E402

_real_sleep = _asyncio.sleep


class TestChat:
    @pytest.mark.asyncio
    async def test_messages_are_flattened_with_roles(self):
        provider = MockLLMProvider()
        await provider.chat(
            [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        assert provider.prompts[-1] == "USER: hello\n\nASSISTANT: hi"
        assert provider.system_prompts[-1] == "be terse"

    @pytest.mark.asyncio
    async def test_system_only_message_leaves_prompt_empty(self):
        provider = MockLLMProvider()
        await provider.chat([{"role": "system", "content": "rule"}])
        assert provider.prompts[-1] == ""
        assert provider.system_prompts[-1] == "rule"

    @pytest.mark.asyncio
    async def test_complete_text_returns_only_text(self):
        assert await MockLLMProvider(response="body").complete_text("p") == "body"


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_true_when_the_provider_answers(self):
        assert await MockLLMProvider().health_check() is True

    @pytest.mark.asyncio
    async def test_false_when_the_provider_raises(self):
        provider = MockLLMProvider()
        provider.fail_times = 99
        provider.max_retries = 0
        assert await provider.health_check() is False

    @pytest.mark.asyncio
    async def test_never_raises(self):
        class Hostile(LLMProvider):
            async def _call(self, prompt, system_prompt):
                raise KeyboardInterrupt if False else LLMError("nope", retryable=False)

        assert await Hostile().health_check() is False


class TestMockProviders:
    @pytest.mark.asyncio
    async def test_mock_records_prompts(self):
        provider = MockLLMProvider(response="canned")
        await provider.complete("a", "sys")
        await provider.complete("b", "sys2")
        assert provider.prompts == ["a", "b"]
        assert provider.system_prompts == ["sys", "sys2"]

    @pytest.mark.asyncio
    async def test_fail_times_decrements(self):
        provider = MockLLMProvider(max_retries=3)
        provider.fail_times = 2
        assert (await provider.complete("p")).text == "MOCK RESPONSE"
        assert provider.fail_times == 0

    @pytest.mark.asyncio
    async def test_scripted_consumes_in_order_then_falls_back(self):
        provider = ScriptedLLM(responses=["one", "two"], default="three")
        assert (await provider.complete("p")).text == "one"
        assert (await provider.complete("p")).text == "two"
        assert (await provider.complete("p")).text == "three"

    @pytest.mark.asyncio
    async def test_scripted_with_no_default_returns_empty_text(self):
        assert (await ScriptedLLM(responses=[]).complete("p")).text == ""

    @pytest.mark.asyncio
    async def test_callback_accepts_sync_and_async_callables(self):
        assert (await CallbackLLM(lambda p, s: "sync").complete("p")).text == "sync"

        async def coro(prompt, system):
            return "async"

        assert (await CallbackLLM(coro).complete("p")).text == "async"

    @pytest.mark.asyncio
    async def test_rule_based_picks_the_first_matching_rule(self):
        class R(RuleBasedLLM):
            RULES = [(re.compile(r"deploy"), "use CI"), (re.compile(r"cach"), "use redis")]

        provider = R(default="fallback")
        assert (await provider.complete("add caching to the api")).text == "use redis"
        assert (await provider.complete("set up deploy pipeline")).text == "use CI"
        assert (await provider.complete("write poetry")).text == "fallback"
        assert provider._hits[provider.RULES[0][0].pattern] == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("prompt", "key"),
        [
            ('Reply with JSON: {"tasks": [{"id": "x"}]}', "tasks"),
            ('Reply with JSON: {"findings": [{"severity": "major"}]}', "findings"),
            ('Reply with JSON: {"acceptance_criteria": ["..."]}', "acceptance_criteria"),
            ('Answer with JSON: {"architecture": "..."}', "architecture"),
        ],
    )
    async def test_shipped_rules_answer_every_prompt_the_app_sends(self, prompt, key):
        """The provider is offered in the UI, so it must not answer nothing."""
        response = await RuleBasedLLM().complete(prompt)
        parsed = json.loads(response.text)
        assert key in parsed, f"rule for {key} did not answer its own prompt"

    @pytest.mark.asyncio
    async def test_unknown_prompt_falls_back_to_the_default(self):
        assert (await RuleBasedLLM(default="{}").complete("hello")).text == "{}"


class TestOfflineCoderContract:
    """
    An implementation task that emits no `<file>` block fails permanently, so an
    offline provider advertised in the UI must satisfy that contract or every
    demo run dies at the first non-analysis task.
    """

    WRITER = "You are the architect.\nYou may ONLY touch these paths: app/\n"
    CODER_PROMPT = (
        "## Original request\nadd multiply\n\n"
        "## Your task\n**Design the architecture and interfaces**\n\nbody\n\n"
        "## Relevant existing files\n### app/calculator.py\n```\ndef add(a, b):\n    return a + b\n```\n"
    )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", [MockLLMProvider(), RuleBasedLLM()])
    async def test_a_coder_prompt_gets_a_parseable_in_scope_write(self, provider):
        from app.agents.patch import parse_patch

        response = await provider.complete(self.CODER_PROMPT, self.WRITER)
        patch = parse_patch(response.text)
        assert not patch.is_empty
        assert [w.path for w in patch.writes] == ["app/design-the-architecture-and-interfaces.md"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", [MockLLMProvider(), RuleBasedLLM()])
    async def test_an_existing_file_is_never_rewritten_from_a_prompt_excerpt(
        self, provider
    ):
        """
        `<file path>` carries whole-file content and prompt excerpts are capped,
        so rewriting one from the prompt truncates the user's real code.
        """
        from app.agents.patch import parse_patch

        response = await provider.complete(self.CODER_PROMPT, self.WRITER)
        written = {w.path for w in parse_patch(response.text).writes}
        assert "app/calculator.py" not in written

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", [MockLLMProvider(), RuleBasedLLM()])
    async def test_a_read_only_role_is_answered_without_a_write(self, provider):
        from app.agents.patch import parse_patch

        prompt = self.CODER_PROMPT.replace(
            "Design the architecture and interfaces", "Final code review"
        )
        system = "You are the reviewer.\nYou do not have write access for this task.\n"
        response = await provider.complete(prompt, system)
        assert parse_patch(response.text).is_empty

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider", [MockLLMProvider(), RuleBasedLLM()])
    async def test_a_review_prompt_answers_with_empty_findings(self, provider):
        """
        Otherwise the provider's own canned text becomes a phantom review
        finding, which renders in the report as though it were a real result.
        """
        prompt = (
            "Diff under review:\n\n@@ -1 +1 @@\n"
            'Reply with JSON: {"findings": [{"severity": "critical"}]}'
        )
        response = await provider.complete(prompt, "You are a staff engineer.")
        assert json.loads(response.text) == {"findings": []}

    @pytest.mark.asyncio
    async def test_a_scripted_response_is_never_intercepted(self):
        """An explicitly configured mock stays scriptable."""
        provider = MockLLMProvider(response='<file path="app/a.py">x</file>')
        response = await provider.complete(self.CODER_PROMPT, self.WRITER)
        assert response.text == '<file path="app/a.py">x</file>'


class TestParseJSONResponse:
    def test_plain_object(self):
        assert parse_json_response('{"a": 1}') == {"a": 1}

    def test_fenced_block(self):
        assert parse_json_response('```\n{"a": 1}\n```') == {"a": 1}

    def test_fenced_json_tag(self):
        assert parse_json_response('```json\n{"a": 1}\n```') == {"a": 1}

    def test_prose_around_the_object(self):
        assert parse_json_response('Sure!\n{"a": [1, 2]}\nHope that helps.') == {"a": [1, 2]}

    def test_array(self):
        assert parse_json_response('```json\n[{"id": "t1"}]\n```') == [{"id": "t1"}]

    def test_array_of_arrays_is_not_mangled_into_one_object(self):
        assert parse_json_response('[[1, 2], [3, 4]]') == [[1, 2], [3, 4]]

    def test_object_containing_a_list_still_parses_as_an_object(self):
        assert parse_json_response('{"tasks": ["a", "b"]}') == {"tasks": ["a", "b"]}

    def test_nested_braces_do_not_break_extraction(self):
        assert parse_json_response('noise {"a": {"b": 1}} noise') == {"a": {"b": 1}}

    def test_strips_ansi(self):
        assert parse_json_response('\x1b[32m{"ok": true}\x1b[0m') == {"ok": True}

    def test_garbage_returns_none(self):
        assert parse_json_response("no json here at all") is None

    def test_malformed_json_returns_none(self):
        assert parse_json_response('{"a": ') is None

    def test_empty_string_returns_none(self):
        assert parse_json_response("") is None


class TestOpenCodeProvider:
    def test_default_model_is_not_hard_coded_into_behaviour(self, monkeypatch):
        monkeypatch.delenv("OPENCODE_MODEL", raising=False)
        assert OpenCodeProvider().model == OpenCodeProvider.DEFAULT_MODEL
        assert OpenCodeProvider(model="explicit").model == "explicit"

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_MODEL", "from-env")
        assert OpenCodeProvider().model == "from-env"
        assert OpenCodeProvider(model="explicit").model == "explicit"

    def test_is_available_is_false_for_a_missing_binary(self):
        assert OpenCodeProvider(binary="definitely-not-a-real-cli").is_available() is False

    @pytest.mark.asyncio
    async def test_missing_binary_is_a_non_retryable_error(self):
        provider = OpenCodeProvider(binary="definitely-not-a-real-cli", max_retries=5)
        with pytest.raises(LLMError, match="not found on PATH") as excinfo:
            await provider.complete("p")
        assert excinfo.value.retryable is False

    @pytest.mark.asyncio
    async def test_strip_ansi_removes_colour_codes(self):
        from app.llm.opencode import _strip_ansi

        assert _strip_ansi("\x1b[1;32mgreen\x1b[0m") == "green"

    @pytest.mark.asyncio
    async def test_cli_output_is_passed_through(self, monkeypatch):
        provider = OpenCodeProvider(model="m")
        monkeypatch.setattr(provider, "is_available", lambda: True)

        async def fake_cli(args):
            return "  the answer  "

        monkeypatch.setattr(provider, "_run_cli", fake_cli)
        assert (await provider.complete("p")).text == "the answer"

    @pytest.mark.asyncio
    async def test_empty_cli_output_is_an_error(self, monkeypatch):
        provider = OpenCodeProvider(model="m")
        monkeypatch.setattr(provider, "is_available", lambda: True)

        async def fake_cli(args):
            return "   \n  "

        monkeypatch.setattr(provider, "_run_cli", fake_cli)
        with pytest.raises(LLMError, match="empty response"):
            await provider.complete("p")

    @pytest.mark.asyncio
    async def test_model_listing_falls_back_to_configured_model(self, monkeypatch):
        provider = OpenCodeProvider(model="fallback-model", binary="nope-not-here")
        assert await provider.list_available_models() == ["fallback-model"]


class TestOllamaProvider:
    def test_base_url_and_model_resolution(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
        monkeypatch.delenv("OLLAMA_MODEL", raising=False)
        provider = OllamaProvider()
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "qwen2.5-coder:7b"
        assert OllamaProvider(base_url="http://x:1/").base_url == "http://x:1"

    def test_env_is_used_when_no_argument(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://env-host:1234")
        monkeypatch.setenv("OLLAMA_MODEL", "env-model")
        provider = OllamaProvider()
        assert provider.base_url == "http://env-host:1234"
        assert provider.model == "env-model"

    @pytest.mark.asyncio
    async def test_generate_maps_the_response(self, monkeypatch):
        provider = OllamaProvider(model="m")
        seen = {}

        async def fake_post(endpoint, payload):
            seen["endpoint"] = endpoint
            seen["payload"] = payload
            return {
                "response": "  done  ",
                "model": "m",
                "prompt_eval_count": 7,
                "eval_count": 3,
            }

        monkeypatch.setattr(provider, "_post", fake_post)
        response = await provider.complete("p", "sys")
        assert response.text == "done"
        assert (response.prompt_tokens, response.completion_tokens) == (7, 3)
        assert seen["endpoint"] == "/api/generate"
        assert seen["payload"]["system"] == "sys"
        assert seen["payload"]["stream"] is False

    @pytest.mark.asyncio
    async def test_chat_uses_the_chat_endpoint(self, monkeypatch):
        provider = OllamaProvider(model="m")

        async def fake_post(endpoint, payload):
            assert endpoint == "/api/chat"
            return {"message": {"content": "hi"}, "model": "m"}

        monkeypatch.setattr(provider, "_post", fake_post)
        assert (await provider.chat([{"role": "user", "content": "hey"}])).text == "hi"

    @pytest.mark.asyncio
    async def test_empty_response_is_an_error(self, monkeypatch):
        provider = OllamaProvider(model="m")

        async def fake_post(endpoint, payload):
            return {"response": "   "}

        monkeypatch.setattr(provider, "_post", fake_post)
        with pytest.raises(LLMError, match="empty response"):
            await provider.complete("p")

    @pytest.mark.asyncio
    async def test_unreachable_server_becomes_a_retryable_llm_error(self, monkeypatch):
        provider = OllamaProvider(model="m", base_url="http://127.0.0.1:1")

        async def boom(endpoint, payload):
            raise LLMError("ollama unreachable")

        monkeypatch.setattr(provider, "_post", boom)
        with pytest.raises(LLMError) as excinfo:
            await provider.complete("p")
        assert excinfo.value.retryable is True

    @pytest.mark.asyncio
    async def test_model_listing_falls_back_when_the_server_is_down(self, monkeypatch):
        provider = OllamaProvider(model="only", base_url="http://127.0.0.1:1")
        assert await provider.list_available_models() == ["only"]


class TestAPIProvider:
    def test_headers_include_bearer_token_when_present(self):
        assert APIProvider(api_key="sk-1")._headers()["Authorization"] == "Bearer sk-1"
        assert "Authorization" not in APIProvider(api_key="")._headers()

    def test_base_url_trailing_slash_is_trimmed(self):
        assert APIProvider(base_url="https://x/v1/").base_url == "https://x/v1"

    def test_model_resolution(self, monkeypatch):
        monkeypatch.delenv("API_MODEL", raising=False)
        assert APIProvider().model == "gpt-4o-mini"
        assert APIProvider(model="custom").model == "custom"
        monkeypatch.setenv("API_MODEL", "env-model")
        assert APIProvider().model == "env-model"

    @pytest.mark.asyncio
    async def test_completion_shape_is_unpacked(self, monkeypatch):
        provider = APIProvider(model="m", base_url="http://x/v1")
        captured = {}

        async def fake_post(endpoint, payload):
            captured.update(endpoint=endpoint, payload=payload)
            return {
                "model": "m",
                "choices": [{"message": {"content": "  hello  "}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6},
            }

        monkeypatch.setattr(provider, "_post", fake_post)
        response = await provider.complete("p", "be nice")
        assert response.text == "hello"
        assert (response.prompt_tokens, response.completion_tokens) == (4, 6)
        assert captured["endpoint"] == "/chat/completions"
        assert captured["payload"]["messages"][0]["role"] == "system"
        assert captured["payload"]["messages"][1]["content"] == "p"

    @pytest.mark.asyncio
    async def test_system_prompt_is_omitted_when_empty(self, monkeypatch):
        provider = APIProvider(model="m")
        captured = {}

        async def fake_post(endpoint, payload):
            captured.update(payload=payload)
            return {"choices": [{"message": {"content": "x"}}]}

        monkeypatch.setattr(provider, "_post", fake_post)
        await provider.complete("p")
        assert [m["role"] for m in captured["payload"]["messages"]] == ["user"]

    @pytest.mark.asyncio
    async def test_wrong_response_shape_is_not_retried(self, monkeypatch):
        provider = APIProvider(model="m")

        async def fake_post(endpoint, payload):
            return {"unexpected": "shape"}

        monkeypatch.setattr(provider, "_post", fake_post)
        with pytest.raises(LLMError, match="response shape") as excinfo:
            await provider.complete("p")
        assert excinfo.value.retryable is False

    @pytest.mark.asyncio
    async def test_empty_choices_is_not_retried(self, monkeypatch):
        provider = APIProvider(model="m")

        async def fake_post(endpoint, payload):
            return {"choices": []}

        monkeypatch.setattr(provider, "_post", fake_post)
        with pytest.raises(LLMError):
            await provider.complete("p")

    @pytest.mark.asyncio
    async def test_blank_content_is_an_error(self, monkeypatch):
        provider = APIProvider(model="m")

        async def fake_post(endpoint, payload):
            return {"choices": [{"message": {"content": "   "}}]}

        monkeypatch.setattr(provider, "_post", fake_post)
        with pytest.raises(LLMError, match="empty response"):
            await provider.complete("p")


class TestFactory:
    def test_every_registered_name_builds(self, settings):
        for name in ("mock", "scripted", "rule-based", "ollama", "api", "opencode"):
            provider = create_provider(name, settings=settings)
            assert provider.name in {name, "callback"}

    def test_unknown_name_degrades_to_mock(self, settings):
        assert create_provider("not-a-provider", settings=settings).name == "mock"

    def test_name_is_case_insensitive_and_trimmed(self, settings):
        assert create_provider("  MOCK  ", settings=settings).name == "mock"

    def test_settings_supply_the_default_name(self, settings, monkeypatch):
        settings.llm_provider = "ollama"
        assert create_provider(settings=settings).name == "ollama"

    def test_no_arguments_never_raises(self, settings):
        assert create_provider(settings=settings) is not None

    def test_model_resolution_prefers_the_override(self, settings):
        settings.ollama_model = "from-settings"
        assert create_provider("ollama", model="explicit", settings=settings).model == "explicit"

    def test_model_falls_back_to_the_provider_default(self, settings):
        settings.llm_model = ""
        settings.ollama_model = "ollama-default"
        assert create_provider("ollama", settings=settings).model == "ollama-default"

    def test_generic_llm_model_beats_the_provider_default(self, settings):
        settings.llm_model = "universal"
        assert create_provider("ollama", settings=settings).model == "universal"
        assert create_provider("mock", settings=settings).model == "universal"

    def test_api_provider_receives_url_and_key(self, settings):
        settings.api_base_url = "http://host/v1"
        settings.api_key = "sk-test"
        provider = create_provider("api", settings=settings)
        assert provider.base_url == "http://host/v1"
        assert provider.api_key == "sk-test"

    def test_timeouts_and_retries_come_from_settings(self, settings):
        settings.llm_timeout_seconds = 12.5
        settings.llm_max_retries = 5
        provider = create_provider("mock", settings=settings)
        assert provider.timeout == 12.5
        assert provider.max_retries == 5

    def test_extra_kwargs_are_forwarded(self, settings):
        provider = create_provider("mock", settings=settings, response="custom")
        assert provider._response == "custom"

    def test_list_providers_is_for_the_ui(self):
        entries = list_providers()
        ids = {e["id"] for e in entries}
        assert {"mock", "opencode", "ollama", "api"} <= ids
        assert all(e["label"] for e in entries)
