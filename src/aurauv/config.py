"""Load and validate ``[tool.aurauv]`` from the environment owner."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from .errors import AurauvError, ensure
from .models import (
    AurauvConfig,
    MemberSpec,
    ProviderSpec,
    RouteOptionSpec,
    RouteSpec,
)
from .utils import load_toml, normalized_name, parse_version, requirement_name_and_extras


SUPPORTED_SCHEMA = 1
_VALID_POLICY = {"ask", "always", "never"}
_VALID_MEMBER_LOCK = {"none", "metadata"}


def _string(value: Any, *, field: str, default: str | None = None) -> str:
    if value is None:
        value = default
    if not isinstance(value, str) or not value.strip():
        raise AurauvError(f"{field} must be a non-empty string.")
    return value.strip()


def _string_list(
    value: Any, *, field: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if value is None and allow_empty:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise AurauvError(f"{field} must be an array of non-empty strings.")
    if not allow_empty and not value:
        raise AurauvError(f"{field} must not be empty.")
    return tuple(item.strip() for item in value)


def _string_map(value: Any, *, field: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise AurauvError(f"{field} must be a string-to-string table.")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, str) or not item:
            raise AurauvError(f"{field} must be a string-to-string table.")
        result[key] = item
    return result


def _workspace_includes(owner: Path, member: Path, pyproject: dict[str, Any]) -> bool:
    workspace = pyproject.get("tool", {}).get("uv", {}).get("workspace")
    if not isinstance(workspace, dict):
        return False
    patterns = workspace.get("members", [])
    excludes = workspace.get("exclude", [])
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        return False
    if not isinstance(excludes, list) or not all(isinstance(item, str) for item in excludes):
        return False
    relative = PurePosixPath(member.relative_to(owner).as_posix())
    return any(relative.match(pattern) for pattern in patterns) and not any(
        relative.match(pattern) for pattern in excludes
    )


def _project_distribution(path: Path) -> str:
    pyproject = load_toml(path / "pyproject.toml")
    name = pyproject.get("project", {}).get("name")
    if not isinstance(name, str) or not name:
        raise AurauvError(f"{path}/pyproject.toml has no project.name.")
    return name


def load_config(owner_root: Path) -> AurauvConfig:
    owner_root = owner_root.resolve()
    pyproject_path = owner_root / "pyproject.toml"
    pyproject = load_toml(pyproject_path)
    raw = pyproject.get("tool", {}).get("aurauv")
    if not isinstance(raw, dict):
        raise AurauvError(
            f"{pyproject_path} has no [tool.aurauv] table. "
            "Aurauv only manages projects that declare an explicit routing contract."
        )
    schema = raw.get("schema-version")
    if schema != SUPPORTED_SCHEMA:
        raise AurauvError(
            f"Unsupported tool.aurauv schema-version {schema!r}; "
            f"this build supports {SUPPORTED_SCHEMA}."
        )

    project_name = pyproject.get("project", {}).get("name", owner_root.name)
    project_label = _string(raw.get("project"), field="tool.aurauv.project", default=str(project_name))
    minimum_uv_text = _string(
        raw.get("minimum-uv"), field="tool.aurauv.minimum-uv", default="0.10.0"
    )
    python_request_raw = raw.get("python-request")
    python_request = None if python_request_raw is None else _string(
        python_request_raw, field="tool.aurauv.python-request"
    )
    python_policy = _string(
        raw.get("python-install-policy"),
        field="tool.aurauv.python-install-policy",
        default="ask",
    )
    uv_policy = _string(
        raw.get("uv-update-policy"), field="tool.aurauv.uv-update-policy", default="ask"
    )
    if python_policy not in _VALID_POLICY:
        raise AurauvError("python-install-policy must be ask, always, or never.")
    if uv_policy not in _VALID_POLICY:
        raise AurauvError("uv-update-policy must be ask, always, or never.")
    state_directory = _string(
        raw.get("state-directory"), field="tool.aurauv.state-directory", default=".aurauv"
    )
    state_path = Path(state_directory)
    ensure(
        not state_path.is_absolute() and ".." not in state_path.parts,
        "state-directory must be a relative path contained by the environment owner.",
    )
    allow_unmanaged_submodule = raw.get("allow-unmanaged-submodule", False)
    if not isinstance(allow_unmanaged_submodule, bool):
        raise AurauvError("allow-unmanaged-submodule must be boolean.")

    verify_raw = raw.get("verify", {})
    if not isinstance(verify_raw, dict):
        raise AurauvError("tool.aurauv.verify must be a table.")
    verify_imports = _string_list(
        verify_raw.get("imports", []), field="tool.aurauv.verify.imports", allow_empty=True
    )

    raw_routes = raw.get("routes")
    if not isinstance(raw_routes, dict) or not raw_routes:
        raise AurauvError("tool.aurauv.routes must be a non-empty table.")
    routes: dict[str, RouteSpec] = {}
    owner_optional = pyproject.get("project", {}).get("optional-dependencies", {})
    if not isinstance(owner_optional, dict):
        raise AurauvError("project.optional-dependencies must be a table.")
    all_routed_extras: set[str] = set()
    for route_name, route_value in raw_routes.items():
        if not isinstance(route_name, str) or not isinstance(route_value, dict):
            raise AurauvError("Each route must be a named table.")
        default = _string(
            route_value.get("default"), field=f"routes.{route_name}.default", default="auto"
        )
        detector_value = route_value.get("detector")
        detector = None if detector_value is None else _string(
            detector_value, field=f"routes.{route_name}.detector"
        )
        providers = _string_list(
            route_value.get("providers", []),
            field=f"routes.{route_name}.providers",
            allow_empty=True,
        )
        fallbacks = _string_map(
            route_value.get("fallbacks"), field=f"routes.{route_name}.fallbacks"
        )
        raw_options = route_value.get("options")
        if not isinstance(raw_options, dict) or not raw_options:
            raise AurauvError(f"routes.{route_name}.options must be a non-empty table.")
        options: dict[str, RouteOptionSpec] = {}
        for option_name, option_value in raw_options.items():
            if not isinstance(option_name, str) or not isinstance(option_value, dict):
                raise AurauvError(f"Each option in route {route_name!r} must be a table.")
            extras = _string_list(
                option_value.get("extras", []),
                field=f"routes.{route_name}.options.{option_name}.extras",
                allow_empty=True,
            )
            description = option_value.get("description", "")
            if not isinstance(description, str):
                raise AurauvError(
                    f"routes.{route_name}.options.{option_name}.description must be a string."
                )
            for extra in extras:
                if extra not in owner_optional:
                    raise AurauvError(
                        f"Route {route_name}={option_name} refers to missing root extra {extra!r}."
                    )
                all_routed_extras.add(extra)
            options[option_name] = RouteOptionSpec(option_name, extras, description)
        if default != "auto" and default not in options:
            raise AurauvError(
                f"Route {route_name!r} default must be 'auto' or one of {', '.join(options)}."
            )
        if default == "auto" and detector is None:
            raise AurauvError(f"Route {route_name!r} uses auto but declares no detector.")
        for source, target in fallbacks.items():
            if source not in options or target not in options:
                raise AurauvError(
                    f"Route {route_name!r} fallback {source}={target} uses an unknown option."
                )
        routes[route_name] = RouteSpec(
            name=route_name,
            default=default,
            detector=detector,
            providers=providers,
            fallbacks=fallbacks,
            options=options,
        )

    conflicts = pyproject.get("tool", {}).get("uv", {}).get("conflicts", [])
    conflict_sets: list[set[str]] = []
    if isinstance(conflicts, list):
        for value in conflicts:
            if isinstance(value, list):
                conflict_sets.append(
                    {
                        item.get("extra")
                        for item in value
                        if isinstance(item, dict) and isinstance(item.get("extra"), str)
                    }
                )
    for route in routes.values():
        route_extras = {extra for option in route.options.values() for extra in option.extras}
        if len(route.options) > 1 and route_extras and not any(
            route_extras.issubset(conflict) for conflict in conflict_sets
        ):
            raise AurauvError(
                f"All extras controlled by route {route.name!r} must occur in one "
                "tool.uv.conflicts set."
            )

    raw_members = raw.get("members", {})
    if not isinstance(raw_members, dict):
        raise AurauvError("tool.aurauv.members must be a table.")
    members: dict[str, MemberSpec] = {}
    owner_dependencies = pyproject.get("project", {}).get("dependencies", [])
    if not isinstance(owner_dependencies, list):
        raise AurauvError("project.dependencies must be an array.")
    owner_dependency_names = {
        requirement_name_and_extras(item)[0]
        for item in owner_dependencies
        if isinstance(item, str)
    }
    for member_name, member_value in raw_members.items():
        if not isinstance(member_name, str) or not isinstance(member_value, dict):
            raise AurauvError("Each member must be a named table.")
        relative = _string(member_value.get("path"), field=f"members.{member_name}.path")
        member_path = (owner_root / relative).resolve()
        try:
            member_path.relative_to(owner_root)
        except ValueError as exc:
            raise AurauvError(f"Member {member_name!r} must stay inside the owner root.") from exc
        if not (member_path / "pyproject.toml").is_file():
            raise AurauvError(f"Configured member has no pyproject.toml: {member_path}")
        distribution = _string(
            member_value.get("distribution"),
            field=f"members.{member_name}.distribution",
            default=_project_distribution(member_path),
        )
        actual = _project_distribution(member_path)
        if normalized_name(distribution) != normalized_name(actual):
            raise AurauvError(
                f"Member {member_name!r} expects distribution {distribution!r}, "
                f"but declares {actual!r}."
            )
        if not _workspace_includes(owner_root, member_path, pyproject):
            raise AurauvError(
                f"Configured member {relative!r} is not included by tool.uv.workspace.members."
            )
        if normalized_name(distribution) not in owner_dependency_names:
            raise AurauvError(
                f"Workspace member {distribution!r} must be a root project dependency so "
                "the shared environment contains it."
            )
        lock_mode = _string(
            member_value.get("standalone-lock"),
            field=f"members.{member_name}.standalone-lock",
            default="none",
        )
        if lock_mode not in _VALID_MEMBER_LOCK:
            raise AurauvError(
                f"members.{member_name}.standalone-lock must be none or metadata."
            )
        metadata_files = _string_list(
            member_value.get("metadata-files", ["pyproject.toml", "uv.lock", "README.md"]),
            field=f"members.{member_name}.metadata-files",
            allow_empty=False,
        )
        members[member_name] = MemberSpec(
            name=member_name,
            path=member_path,
            distribution=distribution,
            standalone_lock=lock_mode,
            metadata_files=metadata_files,
        )

    raw_providers = raw.get("providers", {})
    if not isinstance(raw_providers, dict):
        raise AurauvError("tool.aurauv.providers must be a table.")
    providers: dict[str, ProviderSpec] = {}
    member_paths = {member.path for member in members.values()}
    for provider_name, provider_value in raw_providers.items():
        if not isinstance(provider_name, str) or not isinstance(provider_value, dict):
            raise AurauvError("Each provider must be a named table.")
        kind = _string(provider_value.get("type"), field=f"providers.{provider_name}.type")
        route = _string(provider_value.get("route"), field=f"providers.{provider_name}.route")
        if route not in routes:
            raise AurauvError(f"Provider {provider_name!r} refers to unknown route {route!r}.")
        project_relative = _string(
            provider_value.get("project"), field=f"providers.{provider_name}.project", default="."
        )
        provider_project = (owner_root / project_relative).resolve()
        if provider_project != owner_root and provider_project not in member_paths:
            raise AurauvError(
                f"Provider {provider_name!r} project must be the owner root or a configured member."
            )
        options = dict(provider_value)
        for key in ("type", "route", "project"):
            options.pop(key, None)
        providers[provider_name] = ProviderSpec(
            name=provider_name,
            kind=kind,
            project=provider_project,
            route=route,
            options=options,
        )

    for route in routes.values():
        for provider in route.providers:
            if provider not in providers:
                raise AurauvError(
                    f"Route {route.name!r} refers to unknown provider {provider!r}."
                )
            if providers[provider].route != route.name:
                raise AurauvError(
                    f"Provider {provider!r} is bound to route {providers[provider].route!r}, "
                    f"not {route.name!r}."
                )
        if route.detector is not None and route.detector not in route.providers:
            raise AurauvError(
                f"Route {route.name!r} detector must also occur in its providers list."
            )

    return AurauvConfig(
        schema_version=schema,
        project_label=project_label,
        minimum_uv=parse_version(minimum_uv_text),
        minimum_uv_text=minimum_uv_text,
        python_request=python_request,
        python_install_policy=python_policy,
        uv_update_policy=uv_policy,
        state_directory=state_directory,
        verify_imports=verify_imports,
        allow_unmanaged_submodule=allow_unmanaged_submodule,
        routes=routes,
        providers=providers,
        members=members,
        owner_root=owner_root,
        pyproject=pyproject,
    )
