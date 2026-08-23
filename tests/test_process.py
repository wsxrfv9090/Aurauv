from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aurauv.errors import CommandError
from aurauv.process import CommandRunner
from aurauv.utils import redact_diagnostic


def test_command_logging_redacts_secrets_without_changing_subprocess_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    actual: list[tuple[str, ...]] = []

    def run(command, **kwargs):
        actual.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    command = (
        "uv",
        "publish",
        "--token",
        "publish-secret",
        "--index-url",
        "https://user:index-secret@example.invalid/simple?token=query-secret",
    )
    CommandRunner().run(command, cwd=tmp_path)

    output = capsys.readouterr().out
    assert actual == [command]
    assert "publish-secret" not in output
    assert "index-secret" not in output
    assert "query-secret" not in output
    assert output.count("<redacted>") >= 3
    assert "example.invalid/simple" in output


def test_command_error_redacts_inline_secret_and_echoed_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command,
            2,
            "",
            (
                "failed token=inline-secret at "
                "https://name:url-secret@example.invalid/upload?signature=query-secret"
            ),
        )

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(CommandError) as captured:
        CommandRunner().run(
            ("uv", "publish", "--token=inline-secret"),
            cwd=tmp_path,
        )

    message = str(captured.value)
    assert "inline-secret" not in message
    assert "url-secret" not in message
    assert "query-secret" not in message
    assert "--token=<redacted>" in message
    assert "example.invalid/upload" in message


def test_empty_secret_value_does_not_corrupt_diagnostics() -> None:
    assert redact_diagnostic("native diagnostic", ("uv", "publish", "--token", "")) == (
        "native diagnostic"
    )
