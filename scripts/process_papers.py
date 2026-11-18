#!/usr/bin/env python
"""Executable script to process documents through the end-to-end pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline.runner import _cli

if __name__ == "__main__":
    _cli()

