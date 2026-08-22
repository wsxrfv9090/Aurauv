#!/usr/bin/env python3
"""One-file launcher for the sibling aurauv.pyz runtime."""

from __future__ import annotations

from pathlib import Path
import sys

PYZ = Path(__file__).resolve().with_name("aurauv.pyz")
if not PYZ.is_file():
    raise SystemExit(f"[aurauv] ERROR: missing runtime: {PYZ}")
sys.path.insert(0, str(PYZ))
from aurauv.cli import main  # noqa: E402

raise SystemExit(main())
