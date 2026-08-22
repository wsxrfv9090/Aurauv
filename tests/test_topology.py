from __future__ import annotations

from pathlib import Path
import subprocess

from aurauv.process import CommandRunner
from aurauv.topology import discover_topology


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.invalid")
    _git(path, "config", "user.name", "Aurauv Test")


def test_git_submodule_workspace_member_and_inner_root(tmp_path: Path, uv_path: Path) -> None:
    member_repo = tmp_path / "member-source"
    _init_repo(member_repo)
    (member_repo / "pyproject.toml").write_text(
        """[project]
name = "member"
version = "0.1.0"
requires-python = ">=3.11"

[tool.uv.workspace]
members = []
""",
        encoding="utf-8",
    )
    _git(member_repo, "add", "pyproject.toml")
    _git(member_repo, "commit", "-qm", "init")

    root = tmp_path / "root"
    _init_repo(root)
    (root / "pyproject.toml").write_text(
        """[project]
name = "root"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["member"]

[tool.uv.sources]
member = { workspace = true }

[tool.uv.workspace]
members = ["modules/member"]
""",
        encoding="utf-8",
    )
    _git(root, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(member_repo), "modules/member")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "root")

    topology = discover_topology(
        uv_path,
        (),
        cwd=root / "modules/member",
        runner=CommandRunner(),
    )
    assert topology.kind == "workspace-member-root"
    assert topology.environment_owner == root.resolve()
    assert topology.is_git_submodule is True


def test_unmanaged_git_submodule_is_distinguished(tmp_path: Path, uv_path: Path) -> None:
    member_repo = tmp_path / "standalone-source"
    _init_repo(member_repo)
    (member_repo / "pyproject.toml").write_text(
        "[project]\nname='member'\nversion='0.1.0'\nrequires-python='>=3.11'\n",
        encoding="utf-8",
    )
    _git(member_repo, "add", "pyproject.toml")
    _git(member_repo, "commit", "-qm", "init")
    root = tmp_path / "plain-root"
    _init_repo(root)
    (root / "pyproject.toml").write_text(
        "[project]\nname='root'\nversion='0.1.0'\nrequires-python='>=3.11'\n",
        encoding="utf-8",
    )
    _git(root, "-c", "protocol.file.allow=always", "submodule", "add", "-q", str(member_repo), "modules/member")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "root")
    topology = discover_topology(
        uv_path, (), cwd=root / "modules/member", runner=CommandRunner()
    )
    assert topology.kind == "unmanaged-submodule"


def test_manual_workspace_owner_follows_nested_membership_to_outermost(
    tmp_path: Path,
) -> None:
    from aurauv.topology import manual_workspace_owner

    outer = tmp_path / "outer"
    inner = outer / "modules" / "inner"
    child = inner / "packages" / "child"
    child.mkdir(parents=True)
    (outer / "pyproject.toml").write_text(
        '''[project]\nname="outer"\nversion="0.1.0"\n'''
        '''[tool.uv.workspace]\nmembers=["modules/inner"]\n''',
        encoding="utf-8",
    )
    (inner / "pyproject.toml").write_text(
        '''[project]\nname="inner"\nversion="0.1.0"\n'''
        '''[tool.uv.workspace]\nmembers=["packages/child"]\n''',
        encoding="utf-8",
    )
    (child / "pyproject.toml").write_text(
        '[project]\nname="child"\nversion="0.1.0"\n',
        encoding="utf-8",
    )

    assert manual_workspace_owner(child) == outer.resolve()
