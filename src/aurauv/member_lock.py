"""Standalone lock maintenance for workspace members that are independently cloned."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from .errors import AurauvError, CommandError
from .models import MemberSpec
from .process import CommandRunner
from .utils import info, load_toml, warn


def _validate_metadata_strategy(member: MemberSpec) -> None:
    pyproject = load_toml(member.path / "pyproject.toml")
    workspace = pyproject.get("tool", {}).get("uv", {}).get("workspace")
    if isinstance(workspace, dict):
        raise AurauvError(
            f"Member {member.name!r} is itself a workspace root. The metadata standalone-lock "
            "strategy cannot reproduce nested members; configure a future richer strategy."
        )
    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    if isinstance(sources, dict):
        for package, value in sources.items():
            entries = value if isinstance(value, list) else [value]
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if entry.get("workspace") is True or "path" in entry:
                    raise AurauvError(
                        f"Member {member.name!r} source {package!r} uses workspace/path metadata. "
                        "The lightweight metadata lock strategy would not be reproducible."
                    )


def _copy_context(member: MemberSpec, destination: Path) -> None:
    for relative_text in member.metadata_files:
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise AurauvError(
                f"Member {member.name!r} metadata file must stay inside the member: {relative}"
            )
        source = member.path / relative
        if not source.exists():
            if relative.name in {"uv.lock", "README.md"}:
                continue
            raise AurauvError(
                f"Member {member.name!r} metadata context does not exist: {source}"
            )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    pyproject = destination / "pyproject.toml"
    if not pyproject.is_file():
        raise AurauvError(
            f"Member {member.name!r} metadata-files must include pyproject.toml."
        )


def _replace_lock(source: Path, destination: Path) -> bool:
    generated = source.read_bytes()
    if destination.is_file() and destination.read_bytes() == generated:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=".uv.lock.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(generated)
            file.flush()
            os.fsync(file.fileno())
        if destination.exists():
            shutil.copymode(destination, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def reconcile_member_lock(
    uv: Path,
    member: MemberSpec,
    runner: CommandRunner,
    *,
    check_only: bool,
    env: dict[str, str],
    upgrade_packages: Sequence[str] = (),
    offline: bool = False,
) -> bool:
    if member.standalone_lock == "none":
        return False
    if member.standalone_lock != "metadata":
        raise AurauvError(f"Unsupported member lock strategy: {member.standalone_lock}")
    _validate_metadata_strategy(member)
    with tempfile.TemporaryDirectory(prefix=f"aurauv-{member.name}-lock-") as temporary:
        isolated = Path(temporary)
        _copy_context(member, isolated)
        check = [str(uv), "lock", "--check", "--project", str(isolated)]
        if offline:
            check.insert(1, "--offline")
        try:
            runner.run(
                check,
                cwd=isolated,
                capture_output=True,
                env=env,
                announce=False,
            )
        except CommandError as check_error:
            if check_only:
                raise AurauvError(
                    f"Standalone lock for member {member.name!r} is stale or unresolvable. "
                    f"Read-only mode did not modify it.\n{check_error}"
                ) from check_error
        else:
            info(f"Standalone lock is current for member {member.name!r}.")
            return False

        lock_command = [str(uv), "lock", "--project", str(isolated)]
        if offline:
            lock_command.insert(1, "--offline")
        for package in upgrade_packages:
            lock_command.extend(("--upgrade-package", package))
        runner.run(lock_command, cwd=isolated, env=env)
        generated = isolated / "uv.lock"
        if not generated.is_file():
            raise AurauvError(f"uv did not generate a standalone lock for {member.name!r}.")
        changed = _replace_lock(generated, member.path / "uv.lock")
        if changed:
            info(f"Updated {member.path / 'uv.lock'} atomically.")
        else:
            info(f"Standalone resolution did not change {member.name!r} uv.lock.")
        return changed
