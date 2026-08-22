"""Project, uv workspace, and Git submodule topology discovery."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Sequence

from .errors import AurauvError
from .models import Topology
from .process import CommandRunner
from .utils import discover_nearest_pyproject, load_toml, option_value, resolve_relative, warn


def effective_directory(args: Sequence[str], cwd: Path) -> Path:
    value = option_value(args, "--directory")
    return resolve_relative(value, cwd) if value else cwd.resolve()


def requested_project_root(args: Sequence[str], cwd: Path) -> Path:
    effective = effective_directory(args, cwd)
    value = option_value(args, "--project")
    if value:
        candidate = resolve_relative(value, effective)
        if candidate.is_file() and candidate.name == "pyproject.toml":
            candidate = candidate.parent
        if not (candidate / "pyproject.toml").is_file():
            raise AurauvError(f"--project does not identify a project: {candidate}")
        return candidate
    return discover_nearest_pyproject(effective)


def _workspace_matches(owner: Path, member: Path, pyproject: dict[str, Any]) -> bool:
    workspace = pyproject.get("tool", {}).get("uv", {}).get("workspace")
    if not isinstance(workspace, dict):
        return False
    members = workspace.get("members", [])
    excludes = workspace.get("exclude", [])
    if not isinstance(members, list) or not all(isinstance(item, str) for item in members):
        return False
    if not isinstance(excludes, list) or not all(isinstance(item, str) for item in excludes):
        return False
    try:
        relative = PurePosixPath(member.resolve().relative_to(owner.resolve()).as_posix())
    except ValueError:
        return False
    included = any(relative.match(pattern) for pattern in members)
    excluded = any(relative.match(pattern) for pattern in excludes)
    return included and not excluded


def manual_workspace_owner(project_root: Path) -> Path:
    """Follow explicit workspace membership transitively to the outer owner.

    A nested workspace root can itself be a member of a larger workspace.  The
    closest owner is therefore not necessarily the environment owner.  Walk one
    explicit membership edge at a time until no ancestor includes the current
    root; this enforces the outer-member-wins rule at every nesting depth.
    """

    owner = project_root.resolve()
    visited: set[Path] = set()
    while owner not in visited:
        visited.add(owner)
        parent_owner: Path | None = None
        for ancestor in owner.parents:
            pyproject_path = ancestor / "pyproject.toml"
            if not pyproject_path.is_file():
                continue
            try:
                pyproject = load_toml(pyproject_path)
            except AurauvError:
                continue
            if _workspace_matches(ancestor, owner, pyproject):
                parent_owner = ancestor.resolve()
                break
        if parent_owner is None:
            return owner
        owner = parent_owner
    raise AurauvError(f"Workspace ownership cycle detected from {project_root}.")


def find_workspace_owner(uv: Path, project_root: Path, runner: CommandRunner) -> Path:
    """Return the outermost explicit workspace that owns ``project_root``.

    ``uv workspace dir --project`` deliberately treats a nested workspace root as
    its own root.  For Aurauv, environment ownership is stricter: when that nested
    root is also explicitly included by an ancestor workspace, the ancestor owns
    the single shared environment.  This is what makes "member and inner root"
    behave like a member instead of silently creating a second ``.venv``.
    """

    project_root = project_root.resolve()
    ancestor_owner = manual_workspace_owner(project_root)
    try:
        completed = runner.run(
            [str(uv), "workspace", "dir", "--project", str(project_root)],
            cwd=project_root,
            capture_output=True,
            announce=False,
        )
    except Exception as exc:
        if ancestor_owner != project_root:
            warn(
                "uv could not report the workspace root; using the explicit ancestor "
                f"workspace declaration instead: {exc}"
            )
            return ancestor_owner
        return project_root

    output = (completed.stdout or "").strip()
    uv_owner = (
        Path(output.splitlines()[-1]).resolve() if output else project_root
    )
    if ancestor_owner != project_root:
        if uv_owner != ancestor_owner:
            warn(
                "The current project is both an inner workspace root and a member of "
                f"an ancestor workspace. Aurauv gives environment ownership to {ancestor_owner}."
            )
        return ancestor_owner
    return uv_owner


def expand_workspace_members(owner: Path) -> tuple[Path, ...]:
    pyproject = load_toml(owner / "pyproject.toml")
    workspace = pyproject.get("tool", {}).get("uv", {}).get("workspace")
    if not isinstance(workspace, dict):
        return (owner.resolve(),)
    patterns = workspace.get("members", [])
    excludes = workspace.get("exclude", [])
    if not isinstance(patterns, list):
        return (owner.resolve(),)
    exclude_patterns = [item for item in excludes if isinstance(item, str)] if isinstance(
        excludes, list
    ) else []
    found: list[Path] = [owner.resolve()]
    for pattern in patterns:
        if not isinstance(pattern, str):
            continue
        for candidate in owner.glob(pattern):
            if not candidate.is_dir() or not (candidate / "pyproject.toml").is_file():
                continue
            relative = PurePosixPath(candidate.relative_to(owner).as_posix())
            if any(relative.match(item) for item in exclude_patterns):
                continue
            found.append(candidate.resolve())
    return tuple(dict.fromkeys(found))


def _git_output(project_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(project_root), *args],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def git_submodule_context(project_root: Path) -> tuple[bool, Path | None]:
    superproject = _git_output(project_root, "rev-parse", "--show-superproject-working-tree")
    if not superproject:
        return False, None
    return True, Path(superproject).resolve()


def git_submodule_paths(owner: Path) -> tuple[Path, ...]:
    gitmodules = owner / ".gitmodules"
    if not gitmodules.is_file():
        return ()
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(owner),
            "config",
            "--file",
            str(gitmodules),
            "--get-regexp",
            r"^submodule\..*\.path$",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return ()
    paths: list[Path] = []
    for line in (completed.stdout or "").splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            paths.append((owner / parts[1]).resolve())
    return tuple(paths)


def discover_topology(
    uv: Path,
    uv_args: Sequence[str],
    *,
    cwd: Path,
    runner: CommandRunner,
) -> Topology:
    invocation_root = requested_project_root(uv_args, cwd)
    owner = find_workspace_owner(uv, invocation_root, runner)
    invocation_pyproject = load_toml(invocation_root / "pyproject.toml")
    owns_inner_workspace = isinstance(
        invocation_pyproject.get("tool", {}).get("uv", {}).get("workspace"), dict
    )
    is_member = owner != invocation_root
    is_submodule, superproject = git_submodule_context(invocation_root)
    if is_member and owns_inner_workspace:
        kind = "workspace-member-root"
    elif is_member:
        kind = "workspace-member"
    elif is_submodule:
        kind = "unmanaged-submodule"
    elif owns_inner_workspace:
        kind = "workspace-root"
    else:
        kind = "standalone"
    return Topology(
        invocation_root=invocation_root,
        environment_owner=owner,
        kind=kind,
        is_workspace_member=is_member,
        is_workspace_root=owns_inner_workspace and not is_member,
        is_git_submodule=is_submodule,
        superproject_root=superproject,
        git_submodule_paths=git_submodule_paths(owner),
        uv_workspace_members=expand_workspace_members(owner),
    )
