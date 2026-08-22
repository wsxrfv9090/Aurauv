"""Immutable configuration and runtime models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RouteOptionSpec:
    name: str
    extras: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True, slots=True)
class RouteSpec:
    name: str
    default: str
    detector: str | None
    providers: tuple[str, ...]
    fallbacks: dict[str, str]
    options: dict[str, RouteOptionSpec]


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    name: str
    kind: str
    project: Path
    route: str
    options: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MemberSpec:
    name: str
    path: Path
    distribution: str
    standalone_lock: str
    metadata_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AurauvConfig:
    schema_version: int
    project_label: str
    minimum_uv: tuple[int, ...]
    minimum_uv_text: str
    python_request: str | None
    python_install_policy: str
    uv_update_policy: str
    state_directory: str
    verify_imports: tuple[str, ...]
    allow_unmanaged_submodule: bool
    routes: dict[str, RouteSpec]
    providers: dict[str, ProviderSpec]
    members: dict[str, MemberSpec]
    owner_root: Path
    pyproject: dict[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True)
class Topology:
    invocation_root: Path
    environment_owner: Path
    kind: str
    is_workspace_member: bool
    is_workspace_root: bool
    is_git_submodule: bool
    superproject_root: Path | None
    git_submodule_paths: tuple[Path, ...]
    uv_workspace_members: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class AuraOptions:
    route_overrides: dict[str, str]
    fallback_overrides: dict[str, str]
    assume_yes: bool
    no_input: bool
    install_python: bool | None
    update_uv: bool | None
    refresh: bool
    target: str
    json_output: bool
    no_route: bool


@dataclass(frozen=True, slots=True)
class ParsedInvocation:
    aura: AuraOptions
    uv_args: tuple[str, ...]
    command: str | None
    command_index: int | None
    special_command: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class DetectionResult:
    option: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RouteSelection:
    route: str
    requested: str
    selected: str
    origin: str
    fallback_from: str | None = None
    fallback_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    provider: str
    route: str
    option: str
    details: dict[str, Any]


@dataclass(frozen=True, slots=True)
class EnvironmentIdentity:
    target: str
    path: Path
    state_path: Path


@dataclass(frozen=True, slots=True)
class PythonSelection:
    executable: Path
    version: str
    origin: str


@dataclass(frozen=True, slots=True)
class AurauvState:
    schema_version: int
    aurauv_version: str
    owner_root: str
    invocation_root: str
    topology_kind: str
    environment_target: str
    environment_path: str
    machine_fingerprint: dict[str, Any]
    config_sha256: str
    route_selections: dict[str, dict[str, Any]]
    provider_results: dict[str, dict[str, Any]]
    uv_version: str
    python_executable: str
    python_version: str
    pyproject_sha256: str
    lock_sha256: str | None
