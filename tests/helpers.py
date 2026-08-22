from __future__ import annotations

import base64
import csv
from hashlib import sha256
from io import StringIO
from pathlib import Path
import zipfile


def make_wheel(directory: Path, distribution: str, module: str, version: str = "0.1.0") -> Path:
    normalized = distribution.replace("-", "_")
    wheel = directory / f"{normalized}-{version}-py3-none-any.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    files = {
        f"{module}/__init__.py": f"__version__ = {version!r}\nVALUE = {module!r}\n".encode(),
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            f"Version: {version}\n"
            "Requires-Python: >=3.11\n"
            "\n"
        ).encode(),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: aurauv-tests\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
            "\n"
        ).encode(),
    }
    rows: list[list[str]] = []
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(sha256(data).digest()).decode().rstrip("=")
        rows.append([name, f"sha256={digest}", str(len(data))])
    rows.append([f"{dist_info}/RECORD", "", ""])
    stream = StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
    files[f"{dist_info}/RECORD"] = stream.getvalue().encode()
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return wheel


def write_basic_project(path: Path, text: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text(text, encoding="utf-8")
