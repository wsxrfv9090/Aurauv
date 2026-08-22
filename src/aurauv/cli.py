"""Aurauv command-line entry point."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Sequence

from . import __version__
from .engine import AurauvEngine
from .errors import AurauvError, CommandError
from .invocation import parse_invocation
from .utils import error


_HELP = """\
Aurauv — uv 外面的机器路由与 workspace 环境所有权层

日常命令保持 uv 语义：
  aurauv sync
  aurauv add polars
  aurauv remove polars
  aurauv run pytest
  aurauv lock --check

Aurauv 参数必须放在 uv 子命令之前：
  aurauv --aura-device cpu sync
  aurauv --aura-device cuda --aura-fallback cpu sync
  aurauv --aura-route accelerator=cuda sync
  aurauv --aura-yes sync
  aurauv --aura-no-input sync
  aurauv --aura-refresh sync
  aurauv --aura-no-route sync       # 明确绕过路由，原样调用 uv

只读/管理命令：
  aurauv aura status
  aurauv --aura-json aura status
  aurauv aura doctor
  aurauv aura lock-members [--check]
  aurauv --aura-route accelerator=cuda aura upgrade pytorch
  aurauv --aura-target current aura bootstrap
  aurauv aura version

未知或非项目型 uv 命令会原样转发给 uv。
"""


def _print_status(payload: dict[str, Any]) -> None:
    aura = payload["aurauv"]
    topology = payload["topology"]
    print(f"Aurauv project: {aura['project']}")
    print(f"Topology: {topology['kind']}")
    print(f"Invocation root: {topology['invocation_root']}")
    print(f"Environment owner: {topology['environment_owner']}")
    print(
        f"Environment: {aura['environment']['target']} -> "
        f"{aura['environment']['path']}"
    )
    print(
        f"Python: {aura['python']['version']} ({aura['python']['executable']}; "
        f"{aura['python']['origin']})"
    )
    print(f"uv: {aura['uv']}")
    for name, route in payload["routes"].items():
        fallback = (
            f"; fallback from {route['fallback_from']}" if route["fallback_from"] else ""
        )
        print(
            f"Route {name}: {route['selected']} ({route['origin']}{fallback}); "
            f"extras={route['extras']}"
        )
    if payload["members"]:
        print("Managed members:")
        for name, member in payload["members"].items():
            print(
                f"  {name}: {member['path']} "
                f"[standalone-lock={member['standalone_lock']}]"
            )


def _special(engine: AurauvEngine, command: tuple[str, ...], json_output: bool) -> int:
    if not command or command[0] in {"help", "-h", "--help"}:
        print(_HELP)
        return 0
    name, *rest = command
    if name == "version":
        if rest:
            raise AurauvError("aurauv aura version accepts no arguments.")
        print(f"aurauv {__version__}")
        return 0
    if name in {"status", "routes"}:
        if rest:
            raise AurauvError(f"aurauv aura {name} accepts no arguments.")
        payload = engine.status()
        if json_output:
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif name == "routes":
            for route_name, route in payload["routes"].items():
                print(
                    f"{route_name}={route['selected']} origin={route['origin']} "
                    f"extras={','.join(route['extras'])}"
                )
        else:
            _print_status(payload)
        return 0
    if name == "doctor":
        if rest:
            raise AurauvError("aurauv aura doctor accepts no arguments.")
        return engine.doctor()
    if name == "lock-members":
        unknown = [item for item in rest if item != "--check"]
        if unknown:
            raise AurauvError(
                "aurauv aura lock-members only accepts --check; unknown: "
                + " ".join(unknown)
            )
        return engine.lock_members(check_only="--check" in rest)
    if name == "upgrade":
        if len(rest) != 1:
            raise AurauvError(
                "aurauv aura upgrade expects exactly one provider name, for example `pytorch`."
            )
        return engine.upgrade_provider(rest[0])
    if name == "bootstrap":
        if rest:
            raise AurauvError(
                "Place Aurauv options before `aura`; bootstrap accepts no positional arguments."
            )
        return engine.bootstrap_current()
    raise AurauvError(f"Unknown Aurauv management command {name!r}.\n\n{_HELP}")


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    try:
        parsed = parse_invocation(raw)
        if parsed.special_command is not None:
            simple = parsed.special_command[:1]
            if not simple or simple[0] in {"help", "-h", "--help"}:
                print(_HELP)
                return 0
            if simple[0] == "version":
                if len(parsed.special_command) != 1:
                    raise AurauvError("aurauv aura version accepts no arguments.")
                print(f"aurauv {__version__}")
                return 0
        engine = AurauvEngine(parsed, cwd=Path.cwd())
        if parsed.special_command is not None:
            return _special(engine, parsed.special_command, parsed.aura.json_output)
        return engine.execute()
    except CommandError as exc:
        error(str(exc))
        return exc.returncode if 0 < exc.returncode < 256 else 1
    except AurauvError as exc:
        error(str(exc))
        return 1
    except KeyboardInterrupt:
        error("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
