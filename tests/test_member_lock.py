from __future__ import annotations

from pathlib import Path

from aurauv.member_lock import reconcile_member_lock
from aurauv.models import MemberSpec
from aurauv.process import CommandRunner
from helpers import make_wheel


def test_metadata_member_lock_is_generated_outside_workspace(tmp_path: Path, uv_path: Path) -> None:
    wheel = make_wheel(tmp_path, "member-dep", "member_dep")
    root = tmp_path / "root"
    member = root / "modules/member"
    member.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """[project]\nname='root'\nversion='0.1.0'\nrequires-python='>=3.11'\n"
        "dependencies=['member']\n[tool.uv.sources]\nmember={workspace=true}\n"
        "[tool.uv.workspace]\nmembers=['modules/member']\n""",
        encoding="utf-8",
    )
    (member / "README.md").write_text("member", encoding="utf-8")
    (member / "pyproject.toml").write_text(
        "[project]\nname='member'\nversion='0.1.0'\nrequires-python='>=3.11'\n"
        f"dependencies=['member-dep @ {wheel.as_uri()}']\n",
        encoding="utf-8",
    )
    spec = MemberSpec(
        name="member",
        path=member,
        distribution="member",
        standalone_lock="metadata",
        metadata_files=("pyproject.toml", "uv.lock", "README.md"),
    )
    changed = reconcile_member_lock(
        uv_path,
        spec,
        CommandRunner(),
        check_only=False,
        env={"UV_PYTHON_DOWNLOADS": "never"},
        offline=True,
    )
    assert changed is True
    assert (member / "uv.lock").is_file()
    changed_again = reconcile_member_lock(
        uv_path,
        spec,
        CommandRunner(),
        check_only=False,
        env={"UV_PYTHON_DOWNLOADS": "never"},
        offline=True,
    )
    assert changed_again is False
    assert not (member / ".venv").exists()
