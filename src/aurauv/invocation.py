"""Parse Aurauv's namespaced options while preserving uv's argument stream."""

from __future__ import annotations

from typing import Sequence

from .errors import AurauvError
from .models import AuraOptions, ParsedInvocation


UV_COMMANDS = {
    "auth",
    "workspace",
    "run",
    "init",
    "add",
    "remove",
    "version",
    "sync",
    "lock",
    "export",
    "tree",
    "check",
    "audit",
    "format",
    "tool",
    "python",
    "pip",
    "venv",
    "build",
    "publish",
    "cache",
    "self",
    "help",
}

_VALUE_OPTIONS = {
    "--aura-route",
    "--aura-device",
    "--aura-fallback",
    "--aura-target",
}

_UV_GLOBAL_VALUE_OPTIONS = {
    "--allow-insecure-host",
    "--trusted-host",
    "--cache-dir",
    "--color",
    "--config-file",
    "--directory",
    "--project",
    "--python-preference",
    "--python-fetch",
    "--preview-features",
    "--preview-feature",
}


_FLAG_OPTIONS = {
    "--aura-yes",
    "--aura-no-input",
    "--aura-install-python",
    "--aura-no-install-python",
    "--aura-update-uv",
    "--aura-no-update-uv",
    "--aura-refresh",
    "--aura-json",
    "--aura-no-route",
}


def _assignment(value: str, *, option: str) -> tuple[str, str]:
    if "=" not in value:
        raise AurauvError(f"{option} expects ROUTE=OPTION, got {value!r}.")
    route, selected = (part.strip() for part in value.split("=", 1))
    if not route or not selected:
        raise AurauvError(f"{option} expects non-empty ROUTE=OPTION.")
    return route, selected


def parse_invocation(argv: Sequence[str]) -> ParsedInvocation:
    raw = list(argv)
    uv_args: list[str] = []
    route_overrides: dict[str, str] = {}
    fallback_overrides: dict[str, str] = {}
    assume_yes = False
    no_input = False
    install_python: bool | None = None
    update_uv: bool | None = None
    refresh = False
    target = "project"
    json_output = False
    no_route = False

    command: str | None = None
    command_index: int | None = None
    special: tuple[str, ...] | None = None
    index = 0
    while index < len(raw):
        token = raw[index]
        if command is not None:
            uv_args.extend(raw[index:])
            break

        if token == "--":
            # Do not reinterpret uv's separator or anything after it. Let uv
            # validate a separator before the command without project effects.
            uv_args.extend(raw[index:])
            break

        name, equals, inline_value = token.partition("=")
        if name in _VALUE_OPTIONS:
            if equals:
                value = inline_value
            else:
                index += 1
                if index >= len(raw):
                    raise AurauvError(f"{name} requires a value.")
                value = raw[index]
            if name == "--aura-route":
                route, selected = _assignment(value, option=name)
                route_overrides[route] = selected
            elif name == "--aura-device":
                if not value.strip():
                    raise AurauvError("--aura-device requires a non-empty option.")
                route_overrides["accelerator"] = value.strip()
            elif name == "--aura-fallback":
                if "=" in value:
                    route, selected = _assignment(value, option=name)
                else:
                    route, selected = "accelerator", value.strip()
                    if not selected:
                        raise AurauvError("--aura-fallback requires an option.")
                fallback_overrides[route] = selected
            elif name == "--aura-target":
                if value not in {"project", "current"}:
                    raise AurauvError("--aura-target must be 'project' or 'current'.")
                target = value
            index += 1
            continue

        if token in _FLAG_OPTIONS:
            if token == "--aura-yes":
                assume_yes = True
            elif token == "--aura-no-input":
                no_input = True
            elif token == "--aura-install-python":
                install_python = True
            elif token == "--aura-no-install-python":
                install_python = False
            elif token == "--aura-update-uv":
                update_uv = True
            elif token == "--aura-no-update-uv":
                update_uv = False
            elif token == "--aura-refresh":
                refresh = True
            elif token == "--aura-json":
                json_output = True
            elif token == "--aura-no-route":
                no_route = True
            index += 1
            continue

        if token == "aura":
            special = tuple(raw[index + 1 :])
            break

        # Preserve uv global option/value pairs before looking for a subcommand.
        # Without this guard a directory literally named "sync" or "run" would
        # be misclassified as the uv command.
        if token in _UV_GLOBAL_VALUE_OPTIONS:
            uv_args.append(token)
            index += 1
            if index >= len(raw):
                # Let uv render its native missing-value diagnostic.
                break
            uv_args.append(raw[index])
            index += 1
            continue

        uv_args.append(token)
        # The first positional token is the command boundary, even for future
        # uv commands. Never consume Aura flags from their argument stream.
        if not token.startswith("-"):
            command = token
            command_index = len(uv_args) - 1
        index += 1

    aura = AuraOptions(
        route_overrides=route_overrides,
        fallback_overrides=fallback_overrides,
        assume_yes=assume_yes,
        no_input=no_input,
        install_python=install_python,
        update_uv=update_uv,
        refresh=refresh,
        target=target,
        json_output=json_output,
        no_route=no_route,
    )
    return ParsedInvocation(
        aura=aura,
        uv_args=tuple(uv_args),
        command=command,
        command_index=command_index,
        special_command=special,
    )
