"""Small Python API for notebook/bootstrap scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .cli import main


def bootstrap_current(
    *,
    project: str | Path | None = None,
    routes: Mapping[str, str] | None = None,
    fallbacks: Mapping[str, str] | None = None,
    device: str | None = None,
    fallback: str | None = None,
    assume_yes: bool = False,
    no_input: bool = False,
) -> int:
    """Install the locked routed environment into the running interpreter.

    This is intentionally non-exact: it does not delete unrelated packages from a
    notebook host. The current interpreter must already satisfy ``requires-python``.

    ``device`` and ``fallback`` are convenience aliases for the common
    ``accelerator`` route. Generic projects should use ``routes`` and ``fallbacks``.
    """

    selected_routes = dict(routes or {})
    selected_fallbacks = dict(fallbacks or {})
    if device is not None:
        selected_routes["accelerator"] = device
    if fallback is not None:
        selected_fallbacks["accelerator"] = fallback

    arguments: list[str] = ["--aura-target", "current"]
    for route, option in selected_routes.items():
        arguments.extend(("--aura-route", f"{route}={option}"))
    for route, option in selected_fallbacks.items():
        arguments.extend(("--aura-fallback", f"{route}={option}"))
    if assume_yes:
        arguments.append("--aura-yes")
    if no_input:
        arguments.append("--aura-no-input")
    if project is not None:
        arguments.extend(("--project", str(Path(project).resolve())))
    arguments.extend(("aura", "bootstrap"))
    return main(arguments)
