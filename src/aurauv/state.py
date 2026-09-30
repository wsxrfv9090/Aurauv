"""Per-project, per-environment route state."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import os
from typing import Any, Sequence

from . import __version__
from .models import (
    AurauvConfig,
    AurauvState,
    EnvironmentIdentity,
    RouteSelection,
    Topology,
)
from .utils import atomic_write_json, json_sha256, read_json, text_sha256, warn


STATE_SCHEMA = 1


def environment_identity(
    config: AurauvConfig,
    *,
    target: str,
    uv_args: Sequence[str],
    current_prefix: Path,
) -> EnvironmentIdentity:
    use_active = next(
        (arg == "--active" for arg in reversed(uv_args)
         if arg in {"--active", "--no-active"}),
        False,
    )
    if target == "current":
        environment_path = current_prefix.resolve()
    elif use_active and os.environ.get("VIRTUAL_ENV"):
        environment_path = Path(os.environ["VIRTUAL_ENV"]).expanduser().resolve()
    elif os.environ.get("UV_PROJECT_ENVIRONMENT"):
        raw = Path(os.environ["UV_PROJECT_ENVIRONMENT"]).expanduser()
        environment_path = (
            raw.resolve() if raw.is_absolute() else (config.owner_root / raw).resolve()
        )
    else:
        environment_path = (config.owner_root / ".venv").resolve()
    identity = text_sha256(f"{target}\0{environment_path}")[:20]
    state_path = config.owner_root / config.state_directory / "state" / f"{identity}.json"
    return EnvironmentIdentity(target=target, path=environment_path, state_path=state_path)


def config_digest(config: AurauvConfig) -> str:
    return json_sha256(config.pyproject.get("tool", {}).get("aurauv", {}))


def read_route_state(
    config: AurauvConfig,
    topology: Topology,
    environment: EnvironmentIdentity,
    *,
    refresh: bool,
    machine_signature: dict[str, Any],
) -> dict[str, str]:
    if refresh:
        return {}
    if environment.target == "project" and not environment.path.exists():
        return {}
    payload = read_json(environment.state_path)
    if payload is None:
        return {}
    if payload.get("schema_version") != STATE_SCHEMA:
        warn(f"Ignoring state with unsupported schema in {environment.state_path}.")
        return {}
    if payload.get("owner_root") != str(config.owner_root):
        warn(f"Ignoring state belonging to another environment owner.")
        return {}
    if payload.get("environment_path") != str(environment.path):
        warn(f"Ignoring state belonging to another environment path.")
        return {}
    if payload.get("config_sha256") != config_digest(config):
        warn("Aurauv configuration changed; persisted routes will be detected again.")
        return {}
    stored_machine = payload.get("machine_fingerprint")
    if not isinstance(stored_machine, dict) or stored_machine != machine_signature:
        warn("Machine fingerprint changed; persisted routes will be detected again.")
        return {}
    selections = payload.get("route_selections")
    if not isinstance(selections, dict):
        return {}
    result: dict[str, str] = {}
    for route_name, value in selections.items():
        if route_name not in config.routes or not isinstance(value, dict):
            continue
        selected = value.get("selected")
        if isinstance(selected, str) and selected in config.routes[route_name].options:
            result[route_name] = selected
    return result


def write_state(
    config: AurauvConfig,
    topology: Topology,
    environment: EnvironmentIdentity,
    selections: dict[str, RouteSelection],
    provider_results: dict[str, dict[str, Any]],
    *,
    uv_version: str,
    python_executable: Path,
    python_version: str,
    pyproject_sha256: str,
    lock_sha256: str | None,
    machine_signature: dict[str, Any],
) -> Path:
    state = AurauvState(
        schema_version=STATE_SCHEMA,
        aurauv_version=__version__,
        owner_root=str(config.owner_root),
        invocation_root=str(topology.invocation_root),
        topology_kind=topology.kind,
        environment_target=environment.target,
        environment_path=str(environment.path),
        machine_fingerprint=machine_signature,
        config_sha256=config_digest(config),
        route_selections={name: asdict(selection) for name, selection in selections.items()},
        provider_results=provider_results,
        uv_version=uv_version,
        python_executable=str(python_executable),
        python_version=python_version,
        pyproject_sha256=pyproject_sha256,
        lock_sha256=lock_sha256,
    )
    payload = asdict(state)
    atomic_write_json(environment.state_path, payload)
    return environment.state_path


def environment_python(environment: EnvironmentIdentity) -> Path:
    if environment.target == "current":
        import sys

        return Path(sys.executable).absolute()
    root = environment.path
    if root.is_file() and not root.is_symlink():
        try:
            target = Path(root.read_text(encoding="utf-8").strip()).expanduser()
        except OSError:
            target = root
        else:
            if target:
                root = target.resolve()
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"
