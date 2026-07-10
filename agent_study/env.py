"""Minimal .env loader so ANTHROPIC_API_KEY doesn't need to be typed into a
chat transcript or a shell session that won't persist across tool calls.

Copy agent_study/.env.example to agent_study/.env and fill in your key --
that file is gitignored.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_PATH = Path(__file__).parent / ".env"


def load_env() -> None:
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value
