"""
Application configuration.

All tunables are environment-driven so the system can run in dev, CI and
production without code changes. Nothing here hard-codes an LLM model: the
provider and model are selected at runtime.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="", env_file=".env", extra="ignore", case_sensitive=False
    )

    # ── LLM provider ────────────────────────────────────────────────────────
    llm_provider: str = Field(
        default="mock",
        description=(
            "mock | gemini | api | ollama | opencode | rule-based. "
            "'ollama' covers both a local server and Ollama Cloud."
        ),
    )
    llm_model: str = Field(
        default="",
        description="Provider-specific model id. Empty means provider default.",
    )
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    opencode_model: str = "nemotron-3.5-lightning-free"
    # Empty means "the default for the base URL": a local server and
    # ollama.com do not share a vocabulary of model names.
    ollama_model: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str = Field(
        default="",
        description="Required by https://ollama.com, ignored by a local server.",
    )
    ollama_num_predict: int = Field(
        default=8192,
        description=(
            "Output ceiling per completion. Agents rewrite whole files, and a "
            "truncated answer is indistinguishable from a refusal."
        ),
    )
    api_base_url: str = "https://api.openai.com/v1"
    api_model: str = "gpt-4o-mini"
    api_key: str = ""
    llm_timeout_seconds: float = 300.0
    llm_max_retries: int = 2

    # ── Scheduling ──────────────────────────────────────────────────────────
    max_parallel_tasks: int = 4
    max_task_retries: int = 3
    retry_backoff_seconds: float = 1.0
    retry_backoff_multiplier: float = 2.0
    task_timeout_seconds: float = 1800.0
    orchestrator_timeout_seconds: float = 7200.0

    # ── Storage ─────────────────────────────────────────────────────────────
    state_db_path: str = "./state/orchestrations.db"
    workspace_root: str = "./workspaces"

    # ── Integrator ──────────────────────────────────────────────────────────
    test_command: str = ""          # empty = auto-detect from repo
    lint_command: str = ""          # empty = auto-detect from repo
    typecheck_command: str = ""     # empty = auto-detect from repo
    test_timeout_seconds: float = 900.0
    max_conflict_resolution_rounds: int = 2

    # ── Sandbox ─────────────────────────────────────────────────────────────
    sandbox_enabled: bool = False
    sandbox_image: str = "python:3.11-slim"
    sandbox_memory: str = "2g"
    sandbox_cpus: str = "2.0"
    sandbox_pids_limit: int = 256
    sandbox_network: str = "none"
    sandbox_timeout_seconds: float = 600.0

    # ── API ─────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"
    log_level: str = "INFO"
    log_json: bool = False

    @property
    def base_branch(self) -> str:
        return os.getenv("FREEBUFF_BASE_BRANCH", "main")

    @property
    def db_path(self) -> Path:
        p = Path(self.state_db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def workspaces_dir(self) -> Path:
        p = Path(self.workspace_root)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
