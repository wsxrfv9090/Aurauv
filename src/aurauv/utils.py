"""Standard-library helpers shared by Aurauv."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, Iterable, Sequence

from .errors import AurauvError


_VERSION = re.compile(r"\d+")
_REQUIREMENT_HEAD = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9_.-]+)\s*(?:\[(?P<extras>[^]]+)\])?"
)


def info(message: str) -> None:
    print(f"[aurauv] INFO: {message}")


def warn(message: str) -> None:
    print(f"[aurauv] WARN: {message}", file=sys.stderr)


def error(message: str) -> None:
    print(f"[aurauv] ERROR: {message}", file=sys.stderr)


def command_text(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as file:
            value = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise AurauvError(f"Could not read valid TOML from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AurauvError(f"TOML root in {path} is not a table.")
    return value


def parse_version(value: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in _VERSION.findall(value))
    if not parts:
        raise AurauvError(f"Could not parse a version from {value!r}.")
    return parts


def version_at_least(actual: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    width = max(len(actual), len(minimum))
    return actual + (0,) * (width - len(actual)) >= minimum + (0,) * (
        width - len(minimum)
    )


def normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value.strip()).lower()


def requirement_name_and_extras(requirement: str) -> tuple[str, frozenset[str]]:
    match = _REQUIREMENT_HEAD.match(requirement)
    if match is None:
        raise AurauvError(f"Invalid dependency requirement: {requirement!r}")
    name = normalized_name(match.group("name"))
    raw = match.group("extras")
    if raw is None:
        return name, frozenset()
    extras = frozenset(normalized_name(item) for item in raw.split(",") if item.strip())
    return name, extras


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return text_sha256(text)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"Ignoring unreadable state file {path}: {exc}")
        return None
    if not isinstance(payload, dict):
        warn(f"Ignoring non-object state file {path}.")
        return None
    return payload


def contains_option(args: Sequence[str], name: str) -> bool:
    return any(item == name or item.startswith(f"{name}=") for item in args)


def option_values(args: Sequence[str], name: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(args):
        item = args[index]
        if item.startswith(f"{name}="):
            values.append(item.split("=", 1)[1])
        elif item == name and index + 1 < len(args):
            values.append(args[index + 1])
            index += 1
        index += 1
    return values


def option_value(args: Sequence[str], name: str) -> str | None:
    values = option_values(args, name)
    return values[-1] if values else None


def insert_after_command(
    args: Sequence[str], command_index: int, injected: Iterable[str]
) -> list[str]:
    result = list(args)
    position = command_index + 1
    result[position:position] = list(injected)
    return result


def discover_nearest_pyproject(start: Path) -> Path:
    resolved = start.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise AurauvError(f"Could not find pyproject.toml from {start}.")


def resolve_relative(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def machine_fingerprint() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine().lower(),
        "python_implementation": platform.python_implementation(),
    }


def unique_preserving_order(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return tuple(result)
