"""Project-vendored Colab/current-interpreter entry for Aurauv."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping


def bootstrap(
    *,
    device: str | None = None,
    fallback: str | None = None,
    routes: Mapping[str, str] | None = None,
    fallbacks: Mapping[str, str] | None = None,
    assume_yes: bool = False,
    no_input: bool = True,
) -> int:
    project_root = Path(__file__).resolve().parents[1]
    deployment = project_root / "deployment"
    archive = deployment / "aurauv.pyz"
    source_runtime = deployment / "aurauv"
    runtime = archive if archive.is_file() else deployment
    if not archive.is_file() and not source_runtime.is_dir():
        raise FileNotFoundError(
            f"Missing Aurauv runtime: expected {archive} or {source_runtime}"
        )
    sys.path.insert(0, str(runtime))
    from aurauv.bootstrap_api import bootstrap_current

    return bootstrap_current(
        project=project_root,
        routes=routes,
        fallbacks=fallbacks,
        device=device,
        fallback=fallback,
        assume_yes=assume_yes,
        no_input=no_input,
    )


if __name__ == "__main__":
    raise SystemExit(bootstrap())
