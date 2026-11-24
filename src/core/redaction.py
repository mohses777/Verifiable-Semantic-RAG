"""Remove credentials from text before it is written to experiment records."""

from __future__ import annotations

import os
import re

_KEY_LIKE = re.compile(r"sk-[A-Za-z0-9_\-\*\.]{4,}|Bearer\s+[A-Za-z0-9_\-\.\*]{8,}")
_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET")


def redact_secrets(text: str) -> str:
    """Mask key-like patterns and the literal values of secret environment variables."""
    for name, value in os.environ.items():
        if any(marker in name.upper() for marker in _SECRET_ENV_MARKERS) and len(value) >= 8:
            text = text.replace(value, "[REDACTED]")
    return _KEY_LIKE.sub("[REDACTED]", text)
