from __future__ import annotations

from pathlib import Path

import pytest

from aurauv.config import load_config
from aurauv.errors import AurauvError
from aurauv.models import EnvironmentIdentity, RouteSelection, Topology
from aurauv.runtime import find_uv
from aurauv.state import read_route_state, write_state


def _configured_project(tmp_path: Path) -> tuple[Path, object]:
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '''[project]
name = "state-demo"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
a = []
b = []

[tool.uv]
conflicts = [[{ extra = "a" }, { extra = "b" }]]

[tool.aurauv]
schema-version = 1
project = "State demo"
minimum-uv = "0.10.0"
python-install-policy = "never"
uv-update-policy = "never"

[tool.aurauv.routes.runtime]
default = "a"
providers = []

[tool.aurauv.routes.runtime.options.a]
extras = ["a"]

[tool.aurauv.routes.runtime.options.b]
extras = ["b"]
''',
        encoding="utf-8",
    )
    return root, load_config(root)


def test_find_uv_rejects_invalid_configured_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing-uv"
    monkeypatch.setenv("AURAUV_UV", str(missing))
    with pytest.raises(AurauvError, match="not runnable"):
        find_uv()


def test_state_is_bound_to_existing_environment_and_machine_signature(
    tmp_path: Path,
) -> None:
    root, config = _configured_project(tmp_path)
    topology = Topology(
        invocation_root=root,
        environment_owner=root,
        kind="standalone",
        is_workspace_member=False,
        is_workspace_root=False,
        is_git_submodule=False,
        superproject_root=None,
        git_submodule_paths=(),
        uv_workspace_members=(root,),
    )
    environment = EnvironmentIdentity(
        target="project",
        path=root / ".venv",
        state_path=root / ".aurauv/state/demo.json",
    )
    environment.path.mkdir()
    selection = RouteSelection(
        route="runtime",
        requested="a",
        selected="a",
        origin="test",
    )
    signature = {"base": {"machine": "one"}, "providers": {}}
    write_state(
        config,
        topology,
        environment,
        {"runtime": selection},
        {},
        uv_version="uv 0.10.0",
        python_executable=Path("/python"),
        python_version="3.13",
        pyproject_sha256="x",
        lock_sha256=None,
        machine_signature=signature,
    )

    assert read_route_state(
        config,
        topology,
        environment,
        refresh=False,
        machine_signature=signature,
    ) == {"runtime": "a"}
    assert read_route_state(
        config,
        topology,
        environment,
        refresh=False,
        machine_signature={"base": {"machine": "two"}, "providers": {}},
    ) == {}

    environment.path.rmdir()
    assert read_route_state(
        config,
        topology,
        environment,
        refresh=False,
        machine_signature=signature,
    ) == {}
