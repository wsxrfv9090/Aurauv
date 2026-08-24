from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "aurauv 0.1.2"
TEMPLATE_KINDS = ("standalone", "workspace-member", "workspace-root")
TRACKED_ARCHIVES = (
    REPO_ROOT / "bootstrap" / "aurauv.pyz",
    *(
        REPO_ROOT / "templates" / kind / "deployment" / "aurauv.pyz"
        for kind in TEMPLATE_KINDS
    ),
)


def _run_version(command: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == EXPECTED_VERSION


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
    _run_version([sys.executable, str(archive), "aura", "version"])


def test_tracked_zipapps_are_identical_and_report_release_version() -> None:
    payloads = [archive.read_bytes() for archive in TRACKED_ARCHIVES]
    assert len(set(payloads)) == 1
    for archive in TRACKED_ARCHIVES:
        _run_version([sys.executable, str(archive), "aura", "version"])


def test_project_launcher_keeps_project_local_and_supports_version() -> None:
    setup = REPO_ROOT / "templates" / "standalone" / "deployment" / "setup.py"
    _run_version([sys.executable, str(setup), "aura", "version"])


def test_python_and_shell_bootstrap_launchers_report_release_version(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    environment["AURAUV_BOOTSTRAP_PYTHON"] = sys.executable
    uv_stub = tmp_path / "uv"
    uv_stub.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    uv_stub.chmod(0o755)
    environment["PATH"] = f"{tmp_path}{os.pathsep}{environment.get('PATH', '')}"

    _run_version(
        [sys.executable, str(REPO_ROOT / "bootstrap" / "aurauv.py"), "aura", "version"],
        env=environment,
    )
    _run_version(
        ["bash", str(REPO_ROOT / "bootstrap" / "aurauv.sh"), "aura", "version"],
        env=environment,
    )
    _run_version(
        [
            "bash",
            str(REPO_ROOT / "templates" / "standalone" / "deployment" / "setup.sh"),
            "aura",
            "version",
        ],
        env=environment,
    )


def test_template_launcher_copies_and_windows_forwarding_contract_are_current() -> None:
    canonical = REPO_ROOT / "templates" / "deployment"
    for filename in ("setup.py", "setup.sh", "setup.bat"):
        expected = (canonical / filename).read_bytes()
        for kind in TEMPLATE_KINDS:
            actual = REPO_ROOT / "templates" / kind / "deployment" / filename
            assert actual.read_bytes() == expected

    setup_bat = (canonical / "setup.bat").read_text(encoding="utf-8")
    assert 'set "SETUP=%~dp0setup.py"' in setup_bat
    assert '"%SETUP%" %*' in setup_bat
    assert "exit /b %ERRORLEVEL%" in setup_bat
    assert setup_bat.count("sys.version_info < (3, 11)") == 3
    assert "sys.version_info ^< (3, 11)" not in setup_bat

    bootstrap_bat = (REPO_ROOT / "bootstrap" / "aurauv.bat").read_text(encoding="utf-8")
    assert 'set "ENTRY=%~dp0aurauv.py"' in bootstrap_bat
    assert '"%ENTRY%" %*' in bootstrap_bat
    assert "where uv >nul 2>nul" in bootstrap_bat
    assert bootstrap_bat.count("sys.version_info < (3, 11)") == 3
    assert "sys.version_info ^< (3, 11)" not in bootstrap_bat
