#!/usr/bin/env python3
"""Vendored Aurauv entry point for a target project.

The installed ``aurauv`` CLI intentionally mirrors uv.  This project-local
launcher also keeps four ergonomic setup aliases for personal deployment:
``setup.py`` -> ``aurauv sync`` and ``setup.py cpu|mps|cuda`` -> a routed sync.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

DEPLOYMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOYMENT_DIR.parent
PYZ = DEPLOYMENT_DIR / "aurauv.pyz"
RUNTIME = PYZ if PYZ.is_file() else DEPLOYMENT_DIR
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

try:
    from aurauv.cli import main
    from aurauv.invocation import UV_COMMANDS
except ImportError as exc:  # pragma: no cover - launcher diagnostic
    raise SystemExit(
        "[aurauv] ERROR: deployment/aurauv.pyz or deployment/aurauv/ is missing or invalid."
    ) from exc


_DEVICE_ALIASES = {"cpu", "mps", "cuda"}


def _normalize_args(arguments: Sequence[str]) -> list[str]:
    """Translate only the project-local setup aliases; preserve all other uv args."""

    raw = list(arguments)
    if not raw:
        return ["sync"]
    if raw[0] not in _DEVICE_ALIASES:
        return raw

    device = raw.pop(0)
    aura: list[str] = ["--aura-device", device]
    forwarded: list[str] = []
    index = 0
    while index < len(raw):
        token = raw[index]
        if token == "--fallback":
            index += 1
            if index >= len(raw):
                raise SystemExit("[aurauv] ERROR: --fallback requires cpu, mps, or cuda.")
            aura.extend(("--aura-fallback", raw[index]))
        elif token.startswith("--fallback="):
            aura.extend(("--aura-fallback", token.split("=", 1)[1]))
        else:
            forwarded.append(token)
        index += 1

    if not any(token in UV_COMMANDS or token == "aura" for token in forwarded):
        forwarded.insert(0, "sync")
    return [*aura, *forwarded]


if __name__ == "__main__":
    # Project-local launchers always start discovery from their own project.
    # A later explicit global --project still wins; child-command --project
    # arguments remain after `uv run` and are not inspected by Aurauv.
    raise SystemExit(
        main(["--project", str(PROJECT_ROOT), *_normalize_args(sys.argv[1:])])
    )
