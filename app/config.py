"""Environment-driven configuration. No secret values live here or anywhere
in the repo — only variable names and safe defaults."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
LLM_TIMEOUT_SECONDS: float = _float_env("LLM_TIMEOUT_SECONDS", 10.0)
REQUEST_TIMEOUT_SECONDS: float = _float_env("REQUEST_TIMEOUT_SECONDS", 25.0)
PORT: int = int(os.getenv("PORT", "8000"))

# Numeric tolerance used throughout guardrails/optimizer/replay validation.
# Matches the Problem Statement's stated judge tolerance (Section 11.5).
NUMERIC_TOLERANCE: float = 0.01
