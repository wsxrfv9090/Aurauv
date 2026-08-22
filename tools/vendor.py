#!/usr/bin/env python3
"""Vendor Aurauv into a target uv project without editing its pyproject.toml."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import zipfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
SOURCE_PACKAGE = SOURCE_ROOT / "aurauv"
TEMPLATE_DIR = REPOSITORY_ROOT / "templates" / "deployment"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Vendor Aurauv into <target>/deployment without editing pyproject.toml."
    )
    value.add_argument("target", type=Path, help="Target project root")
    value.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing Aurauv runtime and setup launchers.",
    )
    value.add_argument(
        "--runtime",
        choices=("pyz", "source"),
        default="pyz",
        help="Vendor one aurauv.pyz (default) or an editable aurauv/ source directory.",
    )
    value.add_argument(
        "--with-colab-bootstrap",
        action="store_true",
        help="Also copy templates/colab/bootstrap.py to <target>/colab/bootstrap.py.",
    )
    return value


def build_pyz(destination: Path) -> None:
    """Build a deterministic, standard-library-only zipapp."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    main_source = (
        "from aurauv.cli import main\n"
        "raise SystemExit(main())\n"
    ).encode("utf-8")
    files: list[tuple[str, bytes]] = [("__main__.py", main_source)]
    for source in sorted(SOURCE_PACKAGE.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        relative = source.relative_to(SOURCE_ROOT).as_posix()
        files.append((relative, source.read_bytes()))

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as raw:
        raw.write(b"#!/usr/bin/env python3\n")
        with zipfile.ZipFile(
            raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, payload in files:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    temporary.chmod(temporary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    temporary.replace(destination)


def main() -> int:
    args = parser().parse_args()
    target = args.target.expanduser().resolve()
    if not (target / "pyproject.toml").is_file():
        print(f"[vendor] ERROR: target has no pyproject.toml: {target}", file=sys.stderr)
        return 1

    deployment = target / "deployment"
    source_runtime = deployment / "aurauv"
    pyz_runtime = deployment / "aurauv.pyz"
    launchers = [deployment / name for name in ("setup.py", "setup.sh", "setup.bat")]
    occupied = source_runtime.exists() or pyz_runtime.exists() or any(
        path.exists() for path in launchers
    )
    if occupied and not args.force:
        print(
            "[vendor] ERROR: Aurauv/setup files already exist; rerun with --force "
            "after reviewing git status.",
            file=sys.stderr,
        )
        return 1

    deployment.mkdir(parents=True, exist_ok=True)
    if source_runtime.exists():
        shutil.rmtree(source_runtime)
    pyz_runtime.unlink(missing_ok=True)

    if args.runtime == "pyz":
        build_pyz(pyz_runtime)
        runtime_destination = pyz_runtime
    else:
        shutil.copytree(
            SOURCE_PACKAGE,
            source_runtime,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        runtime_destination = source_runtime

    for name in ("setup.py", "setup.sh", "setup.bat"):
        destination = deployment / name
        destination.unlink(missing_ok=True)
        shutil.copy2(TEMPLATE_DIR / name, destination)
    shell = deployment / "setup.sh"
    shell.chmod(shell.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    if args.with_colab_bootstrap:
        destination = target / "colab" / "bootstrap.py"
        if destination.exists() and not args.force:
            print(f"[vendor] ERROR: {destination} exists; pass --force.", file=sys.stderr)
            return 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY_ROOT / "templates" / "colab" / "bootstrap.py", destination)

    print(f"[vendor] Aurauv runtime: {runtime_destination}")
    print(f"[vendor] Entry point: {deployment / 'setup.py'}")
    print("[vendor] Add `.aurauv/` to .gitignore and merge a [tool.aurauv] contract.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
