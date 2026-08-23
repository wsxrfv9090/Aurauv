"""In-memory metadata checkpoints for multi-stage project mutations."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .errors import AurauvError


@dataclass(frozen=True, slots=True)
class FileCheckpoint:
    path: Path
    content: bytes | None
    mode: int | None


class MetadataTransaction:
    """Restore an exact set of small metadata files without persistent backups."""

    def __init__(self, paths: Iterable[Path]) -> None:
        unique = tuple(dict.fromkeys(path.absolute() for path in paths))
        checkpoints: list[FileCheckpoint] = []
        for path in unique:
            if path.is_symlink():
                raise AurauvError(
                    f"Transactional metadata must not be a symlink: {path}"
                )
            if path.exists() and not path.is_file():
                raise AurauvError(
                    f"Transactional metadata must be a regular file: {path}"
                )
            checkpoints.append(
                FileCheckpoint(
                    path=path,
                    content=path.read_bytes() if path.is_file() else None,
                    mode=stat.S_IMODE(path.stat().st_mode) if path.is_file() else None,
                )
            )
        self._checkpoints = tuple(checkpoints)
        self._committed = False

    def existed(self, path: Path) -> bool:
        absolute = path.absolute()
        return any(
            checkpoint.path == absolute and checkpoint.content is not None
            for checkpoint in self._checkpoints
        )

    @staticmethod
    def _replace(path: Path, content: bytes, mode: int) -> None:
        if not path.parent.is_dir():
            raise AurauvError(
                f"Cannot restore transactional metadata because its parent is missing: {path}"
            )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".aurauv-restore",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def rollback(self) -> tuple[Path, ...]:
        if self._committed:
            raise AurauvError("Cannot roll back a committed metadata transaction.")
        restored: list[Path] = []
        failures: list[str] = []
        for checkpoint in self._checkpoints:
            path = checkpoint.path
            try:
                if checkpoint.content is None:
                    if path.exists() or path.is_symlink():
                        if not path.is_file() and not path.is_symlink():
                            raise AurauvError(
                                f"Refusing to remove non-file transaction output: {path}"
                            )
                        path.unlink()
                        restored.append(path)
                    continue
                current = path.read_bytes() if path.is_file() else None
                current_mode = (
                    stat.S_IMODE(path.stat().st_mode) if path.is_file() else None
                )
                if current == checkpoint.content and current_mode == checkpoint.mode:
                    continue
                assert checkpoint.mode is not None
                self._replace(path, checkpoint.content, checkpoint.mode)
                restored.append(path)
            except (OSError, AurauvError) as exc:
                failures.append(f"{path}: {exc}")
        if failures:
            raise AurauvError(
                "Could not fully restore mutation metadata:\n" + "\n".join(failures)
            )
        return tuple(restored)

    def commit(self) -> None:
        self._committed = True
