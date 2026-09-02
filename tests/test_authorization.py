from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from aurauv.config import load_config
from aurauv.errors import AurauvError, CommandError
from aurauv.models import AuraOptions
from aurauv.runtime import ensure_python, ensure_uv_version


def _config(tmp_path: Path, *, uv_policy: str = "ask", python_policy: str = "ask"):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f'''[project]
name = "authorization-demo"
version = "0.1.0"
requires-python = ">=3.13"

[project.optional-dependencies]
local = []

[tool.aurauv]
schema-version = 1
minimum-uv = "0.12.0"
python-request = "3.13"
python-install-policy = "{python_policy}"
uv-update-policy = "{uv_policy}"

[tool.aurauv.routes.runtime]
default = "local"
providers = []

[tool.aurauv.routes.runtime.options.local]
extras = ["local"]
''',
        encoding="utf-8",
    )
    return load_config(root)


def _aura(*, update_uv=None, install_python=None, assume_yes=False, no_input=True):
    return AuraOptions(
        route_overrides={},
        fallback_overrides={},
        assume_yes=assume_yes,
        no_input=no_input,
        install_python=install_python,
        update_uv=update_uv,
        refresh=False,
        target="project",
        json_output=False,
        no_route=False,
    )


class VersionRunner:
    def __init__(self) -> None:
        self.updated = False
        self.commands: list[tuple[str, ...]] = []

    def run(self, command, **kwargs):
        normalized = tuple(str(item) for item in command)
        self.commands.append(normalized)
        if normalized[-2:] == ("self", "update"):
            self.updated = True
            return subprocess.CompletedProcess(normalized, 0, "", "")
        version = "uv 0.12.5\n" if self.updated else "uv 0.10.0\n"
        return subprocess.CompletedProcess(normalized, 0, version, "")


def test_old_uv_is_not_updated_without_authorization(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner = VersionRunner()
    with pytest.raises(AurauvError, match="No project environment changes"):
        ensure_uv_version(Path("/fake/uv"), config, _aura(), runner)  # type: ignore[arg-type]
    assert not runner.updated


def test_explicit_uv_update_is_verified_after_update(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner = VersionRunner()
    result = ensure_uv_version(
        Path("/fake/uv"),
        config,
        _aura(update_uv=True),
        runner,  # type: ignore[arg-type]
    )
    assert result == "uv 0.12.5"
    assert runner.updated


class PythonRunner:
    def __init__(self) -> None:
        self.installed = False
        self.commands: list[tuple[str, ...]] = []

    def run(self, command, **kwargs):
        normalized = tuple(str(item) for item in command)
        self.commands.append(normalized)
        if "find" in normalized:
            if not self.installed:
                raise CommandError(normalized, 2, "", "not found")
            return subprocess.CompletedProcess(normalized, 0, str(Path(sys.executable).absolute()) + "\n", "")
        if "install" in normalized:
            self.installed = True
            return subprocess.CompletedProcess(normalized, 0, "", "")
        raise AssertionError(normalized)


def test_python_install_requires_authorization(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner = PythonRunner()
    with pytest.raises(AurauvError, match="installation was not authorized"):
        ensure_python(
            Path("/fake/uv"),
            config,
            _aura(),
            runner,  # type: ignore[arg-type]
            target="project",
        )
    assert not runner.installed


def test_explicit_python_install_uses_uv_then_rechecks(tmp_path: Path) -> None:
    config = _config(tmp_path)
    runner = PythonRunner()
    selected = ensure_python(
        Path("/fake/uv"),
        config,
        _aura(install_python=True),
        runner,  # type: ignore[arg-type]
        target="project",
    )
    assert runner.installed
    assert selected.executable == Path(sys.executable).resolve()
    assert selected.origin == "uv-installed Python 3.13"
