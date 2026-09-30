from __future__ import annotations

import pytest

from aurauv.errors import AurauvError
from aurauv.invocation import parse_invocation


def test_namespaced_options_are_removed_before_uv_command() -> None:
    parsed = parse_invocation(
        [
            "--aura-route",
            "accelerator=cuda",
            "--aura-fallback=accelerator=cpu",
            "--offline",
            "sync",
            "--locked",
        ]
    )
    assert parsed.command == "sync"
    assert parsed.uv_args == ("--offline", "sync", "--locked")
    assert parsed.aura.route_overrides == {"accelerator": "cuda"}
    assert parsed.aura.fallback_overrides == {"accelerator": "cpu"}


def test_aura_option_after_uv_command_is_preserved_for_uv() -> None:
    parsed = parse_invocation(["run", "python", "--aura-route", "x=y"])
    assert parsed.aura.route_overrides == {}
    assert parsed.uv_args == ("run", "python", "--aura-route", "x=y")


def test_invalid_assignment_fails() -> None:
    with pytest.raises(AurauvError, match="ROUTE=OPTION"):
        parse_invocation(["--aura-route", "cuda", "sync"])


def test_uv_global_option_value_named_like_command_is_not_misparsed() -> None:
    parsed = parse_invocation(["--project", "run", "sync"])
    assert parsed.uv_args == ("--project", "run", "sync")
    assert parsed.command == "sync"
    assert parsed.command_index == 2


def test_friendly_accelerator_aliases_are_namespaced() -> None:
    parsed = parse_invocation(
        [
            "--aura-device",
            "cuda",
            "--aura-fallback",
            "cpu",
            "--aura-no-route",
            "sync",
        ]
    )
    assert parsed.aura.route_overrides == {"accelerator": "cuda"}
    assert parsed.aura.fallback_overrides == {"accelerator": "cpu"}
    assert parsed.aura.no_route is True
    assert parsed.uv_args == ("sync",)


def test_child_project_and_aura_flags_after_run_are_not_wrapper_options() -> None:
    parsed = parse_invocation(
        ["--aura-device", "cpu", "run", "python", "--project", "child", "--aura-no-route"]
    )
    assert parsed.aura.route_overrides == {"accelerator": "cpu"}
    assert parsed.aura.no_route is False
    assert parsed.uv_args == (
        "run",
        "python",
        "--project",
        "child",
        "--aura-no-route",
    )


@pytest.mark.parametrize('command', ['workspace', 'future-command'])
def test_unknown_command_preserves_its_entire_argument_stream(command: str) -> None:
    args = [command, 'run', '--aura-no-route', 'aura', 'version']
    parsed = parse_invocation(args)
    assert parsed.command == command
    assert parsed.command_index == 0
    assert parsed.uv_args == tuple(args)
    assert parsed.special_command is None
    assert parsed.aura.no_route is False


def test_global_separator_is_forwarded_without_consuming_aura_options() -> None:
    args = ['--', 'sync', '--aura-no-route']
    parsed = parse_invocation(args)
    assert parsed.uv_args == tuple(args)
    assert parsed.command is None
    assert parsed.aura.no_route is False
