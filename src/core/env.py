"""Dependency-free .env loading (kept separate so light scripts do not import the whole pipeline)."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_if_present(env_path: str | Path = ".env") -> None:
    """Populate os.environ from a local .env file without overwriting existing env vars."""
    path = Path(env_path)
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
