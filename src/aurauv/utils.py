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
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .errors import AurauvError


_VERSION = re.compile(r"\d+")
_REQUIREMENT_HEAD = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9_.-]+)\s*(?:\[(?P<extras>[^]]+)\])?"
)
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s]+")
_SENSITIVE_OPTIONS = {
    "--api-key",
    "--client-secret",
    "--credential",
    "--credentials",
    "--password",
    "--secret",
    "--token",
}
_SENSITIVE_QUERY_PARTS = {
    "api-key",
    "apikey",
    "auth",
    "authorization",
    "credential",
    "credentials",
    "key",
    "password",
    "passwd",
    "secret",
    "sig",
    "signature",
    "token",
}
_REDACTED = "<redacted>"


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


def _sensitive_query_key(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    parts = set(normalized.split("-"))
    return normalized in _SENSITIVE_QUERY_PARTS or bool(parts & _SENSITIVE_QUERY_PARTS)


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = f"{_REDACTED}@{netloc.rsplit('@', 1)[1]}"
    query = parsed.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        if pairs:
            redacted_pairs = [
                (key, _REDACTED if _sensitive_query_key(key) else item)
                for key, item in pairs
            ]
            query = urlencode(redacted_pairs).replace("%3Credacted%3E", _REDACTED)
        elif _sensitive_query_key(query.partition("=")[0]):
            query = _REDACTED
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))


def redact_text(value: str) -> str:
    """Remove credentials embedded in URLs without changing non-URL text."""

    return _URL.sub(lambda match: _redact_url(match.group(0)), value)


def _sensitive_values(command: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    index = 0
    while index < len(command):
        token = str(command[index])
        name, equals, inline = token.partition("=")
        if name.lower() in _SENSITIVE_OPTIONS:
            if equals and inline:
                values.append(inline)
            elif index + 1 < len(command):
                value = str(command[index + 1])
                if value:
                    values.append(value)
                index += 1
        index += 1
    return tuple(sorted(set(values), key=len, reverse=True))


def redact_command(command: Sequence[str]) -> tuple[str, ...]:
    """Return a display-only argv with secret option values and URL credentials removed."""

    result: list[str] = []
    index = 0
    while index < len(command):
        token = str(command[index])
        name, equals, _inline = token.partition("=")
        if name.lower() in _SENSITIVE_OPTIONS:
            if equals:
                result.append(f"{name}={_REDACTED}")
            else:
                result.append(token)
                if index + 1 < len(command):
                    result.append(_REDACTED)
                    index += 1
        else:
            result.append(redact_text(token))
        index += 1
    return tuple(result)


def redact_diagnostic(value: str, command: Sequence[str]) -> str:
    """Redact command-derived secrets and URL credentials from subprocess diagnostics."""

    redacted = value
    for secret in _sensitive_values(command):
        redacted = redacted.replace(secret, _REDACTED)
    return redact_text(redacted)


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
