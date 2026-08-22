from __future__ import annotations

from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_built_zipapp_runs_without_installing_package(tmp_path: Path) -> None:
    archive = tmp_path / "aurauv.pyz"
    build = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_zipapp.py"),
            "--output",
            str(archive),
        ],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert build.returncode == 0, build.stderr
    assert archive.is_file()
    completed = subprocess.run(
        [sys.executable, str(archive), "aura", "version"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "aurauv 0.1.0"


def test_project_launcher_keeps_project_local_and_supports_version() -> None:
    setup = REPO_ROOT / "templates" / "standalone" / "deployment" / "setup.py"
    completed = subprocess.run(
        [sys.executable, str(setup), "aura", "version"],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "aurauv 0.1.0"
