#!/usr/bin/env python
"""Executable CLI script for end-to-end verifiable question answering."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.qa import _qa_cli

if __name__ == "__main__":
    _qa_cli()
