"""User-facing errors raised by Aurauv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


class AurauvError(RuntimeError):
    """A failure that should be presented without a Python traceback."""


@dataclass(slots=True)
class CommandError(AurauvError):
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    def __str__(self) -> str:
        detail = (self.stderr or self.stdout).strip()
        suffix = f"\n{detail}" if detail else ""
        return (
            f"Command failed with exit code {self.returncode}: "
            f"{' '.join(self.command)}{suffix}"
        )


@dataclass(slots=True)
class ProviderPreflightError(AurauvError):
    provider: str
    route: str
    option: str
    reason: str
    fallback_safe: bool = True

    def __str__(self) -> str:
        return (
            f"Provider {self.provider!r} rejected route "
            f"{self.route}={self.option}: {self.reason}"
        )


@dataclass(slots=True)
class ProviderVerificationError(AurauvError):
    provider: str
    route: str
    option: str
    reason: str
    state: dict[str, Any]
    repairable: bool = False
    fallback_safe: bool = False

    def __str__(self) -> str:
        return (
            f"Provider {self.provider!r} verification failed for "
            f"{self.route}={self.option}: {self.reason}"
        )


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AurauvError(message)


def require_nonempty(items: Sequence[Any], message: str) -> None:
    if not items:
        raise AurauvError(message)
