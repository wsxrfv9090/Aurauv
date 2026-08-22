"""uv and Python availability checks with explicit user authorization."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Mapping

from .errors import AurauvError, CommandError
from .models import AuraOptions, AurauvConfig, PythonSelection
from .process import CommandRunner
from .utils import info, parse_version, version_at_least, warn


_LOWER_BOUND = re.compile(r">=\s*(\d+(?:\.\d+){0,2})")


def find_uv() -> Path:
    configured = os.environ.get("AURAUV_UV")
    candidate = configured or shutil.which("uv")
    if not candidate:
        raise AurauvError(
            "uv was not found. Aurauv did not modify the project or environment. "
            "Install uv with Astral's official standalone installer, then retry."
        )
    path = Path(candidate).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        source = "AURAUV_UV" if configured else "PATH"
        raise AurauvError(
            f"The uv executable selected from {source} is not runnable: {path}. "
            "Aurauv did not modify the project or environment."
        )
    return path


def uv_version(uv: Path, runner: CommandRunner) -> tuple[str, tuple[int, ...]]:
    completed = runner.run(
        [str(uv), "--version"],
        cwd=Path.cwd(),
        capture_output=True,
        announce=False,
    )
    output = (completed.stdout or completed.stderr or "").strip()
    return output, parse_version(output)


def _interactive(aura: AuraOptions) -> bool:
    return not aura.no_input and sys.stdin.isatty()


def _ask(message: str, aura: AuraOptions) -> bool:
    if aura.assume_yes:
        return True
    if not _interactive(aura):
        return False
    try:
        answer = input(f"[aurauv] {message} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _policy_decision(
    *,
    explicit: bool | None,
    policy: str,
    prompt: str,
    aura: AuraOptions,
) -> bool:
    if explicit is not None:
        return explicit
    if policy == "always":
        return True
    if policy == "never":
        return False
    return _ask(prompt, aura)


def ensure_uv_version(
    uv: Path,
    config: AurauvConfig,
    aura: AuraOptions,
    runner: CommandRunner,
) -> str:
    text, actual = uv_version(uv, runner)
    if version_at_least(actual, config.minimum_uv):
        return text
    approved = _policy_decision(
        explicit=aura.update_uv,
        policy=config.uv_update_policy,
        prompt=(
            f"uv {text!r} is older than required {config.minimum_uv_text}. "
            "Run `uv self update` now?"
        ),
        aura=aura,
    )
    if not approved:
        raise AurauvError(
            f"uv {text!r} does not satisfy minimum {config.minimum_uv_text}. "
            "No project environment changes were made."
        )
    try:
        runner.run([str(uv), "self", "update"], cwd=config.owner_root)
    except CommandError as exc:
        raise AurauvError(
            "uv could not update itself. This commonly means uv is managed by a package "
            f"manager; update it through that installation method.\n{exc}"
        ) from exc
    updated_text, updated = uv_version(uv, runner)
    if not version_at_least(updated, config.minimum_uv):
        raise AurauvError(
            f"uv remains at {updated_text!r}, below required {config.minimum_uv_text}."
        )
    return updated_text


def _project_requires_python(config: AurauvConfig) -> str | None:
    value = config.pyproject.get("project", {}).get("requires-python")
    return value if isinstance(value, str) and value.strip() else None


def _install_request(config: AurauvConfig) -> str:
    if config.python_request:
        return config.python_request
    requirement = _project_requires_python(config)
    if requirement:
        match = _LOWER_BOUND.search(requirement)
        if match:
            parts = match.group(1).split(".")
            return ".".join(parts[:2])
        return requirement
    return "3"


def _python_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable), "-c", "import platform; print(platform.python_version())"],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise AurauvError(f"Could not run selected Python {executable}: {detail}")
    return completed.stdout.strip()


def _find_project_python(
    uv: Path,
    config: AurauvConfig,
    runner: CommandRunner,
) -> Path | None:
    command = [str(uv), "--no-python-downloads", "python", "find"]
    if config.python_request:
        command.append(config.python_request)
    command.extend(("--project", str(config.owner_root)))
    try:
        completed = runner.run(
            command,
            cwd=config.owner_root,
            capture_output=True,
            announce=False,
        )
    except CommandError:
        return None
    output = (completed.stdout or "").strip()
    return Path(output.splitlines()[-1]).resolve() if output else None


def ensure_python(
    uv: Path,
    config: AurauvConfig,
    aura: AuraOptions,
    runner: CommandRunner,
    *,
    target: str,
) -> PythonSelection:
    if target == "current":
        current = Path(sys.executable).absolute()
        command = [
            str(uv),
            "--no-python-downloads",
            "python",
            "find",
            str(current),
            "--project",
            str(config.owner_root),
        ]
        try:
            runner.run(
                command,
                cwd=config.owner_root,
                capture_output=True,
                announce=False,
            )
        except CommandError as exc:
            raise AurauvError(
                "The current interpreter does not satisfy the project Python contract. "
                "A running notebook kernel cannot be replaced in place; select a compatible "
                f"runtime first.\n{exc}"
            ) from exc
        return PythonSelection(current, _python_version(current), "current interpreter")

    found = _find_project_python(uv, config, runner)
    if found is not None:
        return PythonSelection(found, _python_version(found), "existing compatible Python")

    request = _install_request(config)
    approved = _policy_decision(
        explicit=aura.install_python,
        policy=config.python_install_policy,
        prompt=(
            f"No installed Python satisfies this project. Allow uv to install {request!r}?"
        ),
        aura=aura,
    )
    if not approved:
        raise AurauvError(
            "No compatible Python was found, and installation was not authorized. "
            "No project environment changes were made."
        )
    runner.run([str(uv), "python", "install", request], cwd=config.owner_root)
    found = _find_project_python(uv, config, runner)
    if found is None:
        raise AurauvError(
            f"uv installed Python request {request!r}, but no compatible interpreter was found."
        )
    return PythonSelection(found, _python_version(found), f"uv-installed Python {request}")


def child_environment(python: PythonSelection) -> Mapping[str, str]:
    return {
        "UV_PYTHON": str(python.executable),
        "UV_PYTHON_DOWNLOADS": "never",
    }


def confirm(message: str, aura: AuraOptions) -> bool:
    """Request an explicit confirmation, respecting --aura-yes/no-input."""

    return _ask(message, aura)
