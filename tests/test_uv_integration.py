from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from helpers import make_wheel


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(project: Path, uv_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "AURAUV_UV": str(uv_path),
    }
    return subprocess.run(
        [sys.executable, "-m", "aurauv", *args],
        cwd=project,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def test_offline_sync_add_run_remove_preserves_route(tmp_path: Path, uv_path: Path) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    a = make_wheel(wheels, "route-a", "route_a")
    b = make_wheel(wheels, "route-b", "route_b")
    c = make_wheel(wheels, "ordinary-c", "ordinary_c")
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        f'''[project]
name = "demo"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
a = ["route-a @ {a.as_uri()}"]
b = ["route-b @ {b.as_uri()}"]

[tool.uv]
conflicts = [[{{ extra = "a" }}, {{ extra = "b" }}]]

[tool.aurauv]
schema-version = 1
project = "Demo"
minimum-uv = "0.10.0"
python-request = "3.13"
python-install-policy = "never"
uv-update-policy = "never"

[tool.aurauv.verify]
imports = ["route_a"]

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

    sync = _run(project, uv_path, "--offline", "sync")
    assert sync.returncode == 0, sync.stderr + sync.stdout
    python = project / ".venv/bin/python"
    assert python.is_file()
    assert subprocess.run([python, "-c", "import route_a"], check=False).returncode == 0
    assert subprocess.run([python, "-c", "import route_b"], check=False).returncode != 0

    add = _run(project, uv_path, "--offline", "add", f"ordinary-c @ {c.as_uri()}")
    assert add.returncode == 0, add.stderr + add.stdout
    assert subprocess.run(
        [python, "-c", "import route_a, ordinary_c"], check=False
    ).returncode == 0

    run = _run(
        project,
        uv_path,
        "--offline",
        "run",
        "python",
        "-c",
        "import route_a, ordinary_c; print('ok')",
    )
    assert run.returncode == 0, run.stderr + run.stdout
    assert "ok" in run.stdout

    remove = _run(project, uv_path, "--offline", "remove", "ordinary-c")
    assert remove.returncode == 0, remove.stderr + remove.stdout
    assert subprocess.run([python, "-c", "import route_a"], check=False).returncode == 0
    assert subprocess.run([python, "-c", "import ordinary_c"], check=False).returncode != 0

    states = list((project / ".aurauv/state").glob("*.json"))
    assert len(states) == 1
    payload = json.loads(states[0].read_text(encoding="utf-8"))
    assert payload["route_selections"]["runtime"]["selected"] == "a"


def _exclusive_distribution_project(tmp_path: Path) -> Path:
    wheels = tmp_path / "exclusive-wheels"
    wheels.mkdir()
    gui = make_wheel(wheels, "vision-gui", "shared_cv")
    headless = make_wheel(wheels, "vision-headless", "shared_cv")
    project = tmp_path / "exclusive-project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        f'''[project]
name = "exclusive-integration"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
vision-gui = ["vision-gui @ {gui.as_uri()}"]
vision-headless = ["vision-headless @ {headless.as_uri()}"]

[tool.uv]
package = false
conflicts = [[{{ extra = "vision-gui" }}, {{ extra = "vision-headless" }}]]

[tool.aurauv]
schema-version = 1
minimum-uv = "0.10.0"
python-install-policy = "never"
uv-update-policy = "never"

[tool.aurauv.routes.vision-runtime]
default = "headless"
providers = ["vision"]

[tool.aurauv.routes.vision-runtime.options.gui]
extras = ["vision-gui"]

[tool.aurauv.routes.vision-runtime.options.headless]
extras = ["vision-headless"]

[tool.aurauv.providers.vision]
type = "exclusive-distribution"
route = "vision-runtime"
family = ["vision-gui", "vision-headless"]
selections = {{ gui = "vision-gui", headless = "vision-headless" }}
module = "shared_cv"
required-attributes = {{ gui = ["VALUE"], headless = ["VALUE"] }}
''',
        encoding="utf-8",
    )
    return project


def test_exclusive_distribution_route_switches_shared_import_wheels(
    tmp_path: Path, uv_path: Path
) -> None:
    project = _exclusive_distribution_project(tmp_path)

    sync = _run(project, uv_path, "--offline", "sync")
    assert sync.returncode == 0, sync.stderr + sync.stdout
    python = project / ".venv/bin/python"
    headless_probe = subprocess.run(
        [
            python,
            "-c",
            "import importlib.metadata as m, shared_cv; "
            "print(m.version('vision-headless'), shared_cv.VALUE)",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert headless_probe.returncode == 0, headless_probe.stderr
    assert "shared_cv" in headless_probe.stdout
    assert subprocess.run(
        [python, "-c", "import importlib.metadata as m; m.version('vision-gui')"],
        check=False,
    ).returncode != 0

    switched = _run(
        project,
        uv_path,
        "--offline",
        "--aura-route",
        "vision-runtime=gui",
        "sync",
    )
    assert switched.returncode == 0, switched.stderr + switched.stdout
    assert subprocess.run(
        [python, "-c", "import importlib.metadata as m; m.version('vision-gui')"],
        check=False,
    ).returncode == 0
    assert subprocess.run(
        [python, "-c", "import importlib.metadata as m; m.version('vision-headless')"],
        check=False,
    ).returncode != 0


def test_exclusive_distribution_provider_verifies_current_interpreter_target(
    tmp_path: Path, uv_path: Path
) -> None:
    project = _exclusive_distribution_project(tmp_path)
    locked = subprocess.run(
        [str(uv_path), "--offline", "lock", "--project", str(project)],
        cwd=project,
        check=False,
        text=True,
        capture_output=True,
    )
    assert locked.returncode == 0, locked.stderr + locked.stdout
    kernel = tmp_path / "exclusive-kernel"
    created = subprocess.run(
        [str(uv_path), "venv", "--python", sys.executable, str(kernel)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )
    assert created.returncode == 0, created.stderr + created.stdout
    kernel_python = kernel / "bin/python"
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "AURAUV_UV": str(uv_path),
    }
    result = subprocess.run(
        [
            str(kernel_python),
            "-m",
            "aurauv",
            "--offline",
            "--aura-target",
            "current",
            "aura",
            "bootstrap",
        ],
        cwd=project,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    probe = subprocess.run(
        [
            str(kernel_python),
            "-c",
            "import importlib.metadata as m, shared_cv; "
            "print(m.version('vision-headless'), shared_cv.VALUE)",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stderr
    states = list((project / ".aurauv/state").glob("*.json"))
    assert len(states) == 1
    payload = json.loads(states[0].read_text(encoding="utf-8"))
    assert payload["environment_target"] == "current"
    assert payload["provider_results"]["vision"]["selected_distribution"] == (
        "vision-headless"
    )


def test_uv_version_passthrough_does_not_require_project(tmp_path: Path, uv_path: Path) -> None:
    result = _run(tmp_path, uv_path, "--version")
    assert result.returncode == 0
    assert "uv " in result.stdout


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Aurauv Test")


def test_real_git_submodule_uses_only_workspace_root_venv(
    tmp_path: Path, uv_path: Path
) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    routed = make_wheel(wheels, "root-route", "root_route")

    member_source = tmp_path / "member-source"
    _init_repo(member_source)
    (member_source / "pyproject.toml").write_text(
        '''[project]
name = "member"
version = "0.1.0"
requires-python = ">=3.11"

[tool.uv]
package = false
''',
        encoding="utf-8",
    )
    _git(member_source, "add", "pyproject.toml")
    _git(member_source, "commit", "-qm", "member")

    root = tmp_path / "root"
    _init_repo(root)
    (root / "pyproject.toml").write_text(
        f'''[project]
name = "root"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["member"]

[project.optional-dependencies]
local = ["root-route @ {routed.as_uri()}"]

[tool.uv]
package = false

[tool.uv.sources]
member = {{ workspace = true }}

[tool.uv.workspace]
members = ["modules/member"]

[tool.aurauv]
schema-version = 1
project = "Root"
minimum-uv = "0.10.0"
python-request = "3.13"
python-install-policy = "never"
uv-update-policy = "never"

[tool.aurauv.verify]
imports = ["root_route"]

[tool.aurauv.routes.runtime]
default = "local"
providers = []

[tool.aurauv.routes.runtime.options.local]
extras = ["local"]

[tool.aurauv.members.member]
path = "modules/member"
distribution = "member"
standalone-lock = "metadata"
metadata-files = ["pyproject.toml", "uv.lock"]
''',
        encoding="utf-8",
    )
    _git(
        root,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "-q",
        str(member_source),
        "modules/member",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "root")

    result = _run(root / "modules/member", uv_path, "--offline", "sync")
    assert result.returncode == 0, result.stderr + result.stdout
    assert (root / ".venv/bin/python").is_file()
    assert not (root / "modules/member/.venv").exists()
    assert (root / "modules/member/uv.lock").is_file()
    assert list((root / ".aurauv/state").glob("*.json"))
    assert subprocess.run(
        [root / ".venv/bin/python", "-c", "import root_route"], check=False
    ).returncode == 0


def test_sync_dry_run_does_not_create_environment_or_state(
    tmp_path: Path, uv_path: Path
) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = make_wheel(wheels, "dry-route", "dry_route")
    project = tmp_path / "dry-project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        f'''[project]
name = "dry-demo"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
local = ["dry-route @ {wheel.as_uri()}"]

[tool.aurauv]
schema-version = 1
minimum-uv = "0.10.0"
python-request = "3.13"
python-install-policy = "never"
uv-update-policy = "never"

[tool.aurauv.routes.runtime]
default = "local"
providers = []

[tool.aurauv.routes.runtime.options.local]
extras = ["local"]
''',
        encoding="utf-8",
    )
    result = _run(project, uv_path, "--offline", "sync", "--dry-run")
    assert result.returncode == 0, result.stderr + result.stdout
    assert not (project / ".venv").exists()
    assert not (project / ".aurauv").exists()


def test_missing_uv_fails_before_any_project_change(tmp_path: Path) -> None:
    project = tmp_path / "missing-uv-project"
    project.mkdir()
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        '''[project]
name = "missing-uv"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
local = []

[tool.aurauv]
schema-version = 1

[tool.aurauv.routes.runtime]
default = "local"
providers = []

[tool.aurauv.routes.runtime.options.local]
extras = ["local"]
''',
        encoding="utf-8",
    )
    before = pyproject.read_bytes()
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "AURAUV_UV": str(tmp_path / "does-not-exist"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "aurauv", "sync"],
        cwd=project,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "not runnable" in result.stderr
    assert pyproject.read_bytes() == before
    assert not (project / "uv.lock").exists()
    assert not (project / ".venv").exists()
    assert not (project / ".aurauv").exists()


def test_current_interpreter_bootstrap_uses_locked_route_without_exact_sync(
    tmp_path: Path, uv_path: Path
) -> None:
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    wheel = make_wheel(wheels, "kernel-route", "kernel_route")
    project = tmp_path / "bootstrap-project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        f'''[project]
name = "bootstrap-demo"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
local = ["kernel-route @ {wheel.as_uri()}"]

[tool.uv]
package = false

[tool.aurauv]
schema-version = 1
minimum-uv = "0.10.0"
python-install-policy = "never"
uv-update-policy = "never"

[tool.aurauv.verify]
imports = ["kernel_route"]

[tool.aurauv.routes.runtime]
default = "local"
providers = []

[tool.aurauv.routes.runtime.options.local]
extras = ["local"]
''',
        encoding="utf-8",
    )
    locked = subprocess.run(
        [str(uv_path), "--offline", "lock", "--project", str(project)],
        cwd=project,
        check=False,
        text=True,
        capture_output=True,
    )
    assert locked.returncode == 0, locked.stderr + locked.stdout

    kernel = tmp_path / "kernel-venv"
    created = subprocess.run(
        [str(uv_path), "venv", "--python", sys.executable, str(kernel)],
        cwd=tmp_path,
        check=False,
        text=True,
        capture_output=True,
    )
    assert created.returncode == 0, created.stderr + created.stdout
    kernel_python = kernel / "bin/python"
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "AURAUV_UV": str(uv_path),
    }
    result = subprocess.run(
        [
            str(kernel_python),
            "-m",
            "aurauv",
            "--offline",
            "--aura-target",
            "current",
            "aura",
            "bootstrap",
        ],
        cwd=project,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    imported = subprocess.run(
        [str(kernel_python), "-c", "import kernel_route; print(kernel_route.VALUE)"],
        check=False,
        text=True,
        capture_output=True,
    )
    assert imported.returncode == 0, imported.stderr
    assert "kernel_route" in imported.stdout
    assert not (project / ".venv").exists()
    states = list((project / ".aurauv/state").glob("*.json"))
    assert len(states) == 1
    payload = json.loads(states[0].read_text(encoding="utf-8"))
    assert payload["environment_target"] == "current"
    assert payload["environment_path"] == str(kernel.resolve())


def test_run_keeps_extraneous_packages_and_ignores_child_extra_flag_for_routing(
    tmp_path: Path, uv_path: Path
) -> None:
    wheels = tmp_path / "wheels-run"
    wheels.mkdir()
    routed_a = make_wheel(wheels, "run-route-a", "run_route_a")
    routed_b = make_wheel(wheels, "run-route-b", "run_route_b")
    extraneous = make_wheel(wheels, "run-extraneous", "run_extraneous")
    project = tmp_path / "run-project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        f'''[project]
name = "run-demo"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
a = ["run-route-a @ {routed_a.as_uri()}"]
b = ["run-route-b @ {routed_b.as_uri()}"]

[tool.uv]
package = false
conflicts = [[{{ extra = "a" }}, {{ extra = "b" }}]]

[tool.aurauv]
schema-version = 1
minimum-uv = "0.10.0"
python-request = "3.13"
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
    sync = _run(project, uv_path, "--offline", "sync")
    assert sync.returncode == 0, sync.stderr + sync.stdout
    python = project / ".venv/bin/python"
    installed = subprocess.run(
        [str(uv_path), "pip", "install", "--python", str(python), str(extraneous)],
        cwd=project,
        check=False,
        text=True,
        capture_output=True,
    )
    assert installed.returncode == 0, installed.stderr + installed.stdout

    # The child command's `--extra b` must remain a child argument, not become an
    # Aurauv/uv route selection. `uv run` is inexact by default, so the unrelated
    # package must remain installed.
    result = _run(
        project,
        uv_path,
        "--offline",
        "run",
        "python",
        "-c",
        "import run_route_a, run_extraneous, sys; print(sys.argv[1:])",
        "--extra",
        "b",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "--extra" in result.stdout and "b" in result.stdout
    assert subprocess.run(
        [python, "-c", "import run_route_a, run_extraneous"], check=False
    ).returncode == 0
    assert subprocess.run([python, "-c", "import run_route_b"], check=False).returncode != 0


def test_run_forwards_dependency_group_to_its_inexact_presync(
    tmp_path: Path, uv_path: Path
) -> None:
    wheels = tmp_path / "wheels-group"
    wheels.mkdir()
    routed = make_wheel(wheels, "group-route", "group_route")
    grouped = make_wheel(wheels, "group-only", "group_only")
    project = tmp_path / "group-project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        f'''[project]
name = "group-demo"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
local = ["group-route @ {routed.as_uri()}"]

[dependency-groups]
experiment = ["group-only @ {grouped.as_uri()}"]

[tool.uv]
package = false

[tool.aurauv]
schema-version = 1
minimum-uv = "0.10.0"
python-request = "3.13"
python-install-policy = "never"
uv-update-policy = "never"

[tool.aurauv.routes.runtime]
default = "local"
providers = []

[tool.aurauv.routes.runtime.options.local]
extras = ["local"]
''',
        encoding="utf-8",
    )
    result = _run(
        project,
        uv_path,
        "--offline",
        "run",
        "--group",
        "experiment",
        "python",
        "-c",
        "import group_route, group_only; print('group-ok')",
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "group-ok" in result.stdout


def test_aura_version_does_not_require_uv(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "AURAUV_UV": str(tmp_path / "missing-uv"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "aurauv", "aura", "version"],
        cwd=tmp_path,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip().startswith("aurauv ")
