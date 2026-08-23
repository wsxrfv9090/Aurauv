"""Subprocess execution with consistent diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

from .errors import CommandError
from .utils import command_text, info, redact_command, redact_diagnostic


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
        display_command = redact_command(normalized)
        if announce:
            info(f"Command: {command_text(display_command)}")
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
                command=display_command,
                returncode=completed.returncode,
                stdout=redact_diagnostic(completed.stdout or "", normalized),
                stderr=redact_diagnostic(completed.stderr or "", normalized),
            )
        return completed
