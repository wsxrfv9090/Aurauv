"""Subprocess execution with consistent diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from .errors import CommandError
from .utils import command_text, info


class CommandRunner:
    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        capture_output: bool = False,
        allowed_exit_codes: Sequence[int] = (0,),
        timeout: float | None = None,
        env: Mapping[str, str] | None = None,
        announce: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        normalized = tuple(str(part) for part in command)
        if announce:
            info(f"Command: {command_text(normalized)}")
        completed = subprocess.run(
            normalized,
            cwd=cwd,
            check=False,
            text=True,
            capture_output=capture_output,
            timeout=timeout,
            env=None if env is None else {**os.environ, **env},
        )
        if completed.returncode not in set(allowed_exit_codes):
            raise CommandError(
                command=normalized,
                returncode=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
            )
        return completed
