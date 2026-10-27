"""Runtime configuration, read from environment variables with sane defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # storage
    database_url: str = field(default_factory=lambda: _env("MF_DATABASE_URL", ""))
    sqlite_path: str = field(default_factory=lambda: _env("MF_SQLITE_PATH", "runs/memoryforge.sqlite3"))
    runs_dir: Path = field(default_factory=lambda: Path(_env("MF_RUNS_DIR", "runs")))

    # llm
    llm_backend: str = field(default_factory=lambda: _env("MF_LLM_BACKEND", "simulated"))
    llm_base_url: str = field(default_factory=lambda: _env("MF_LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("MF_LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("MF_LLM_MODEL", "qwen2.5-coder:7b"))
    hf_model: str = field(default_factory=lambda: _env("MF_HF_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct"))

    # embeddings
    embedder: str = field(default_factory=lambda: _env("MF_EMBEDDER", "auto"))  # auto | minilm | hash

    # agent
    max_trials: int = field(default_factory=lambda: int(_env("MF_MAX_TRIALS", "3")))
    top_k: int = field(default_factory=lambda: int(_env("MF_TOP_K", "3")))
    sandbox_timeout: float = field(default_factory=lambda: float(_env("MF_SANDBOX_TIMEOUT", "6")))


settings = Settings()
