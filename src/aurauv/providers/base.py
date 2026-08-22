"""Provider interface for machine-routed dependency families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import (
    AurauvConfig,
    DetectionResult,
    ProviderSpec,
    RouteSpec,
    VerificationResult,
)
from ..process import CommandRunner


class Provider:
    """Base class implemented by package-family integrations."""

    def __init__(
        self,
        spec: ProviderSpec,
        *,
        config: AurauvConfig,
        runner: CommandRunner,
    ) -> None:
        self.spec = spec
        self.config = config
        self.name = spec.name
        self.project_root = spec.project
        self.route = config.routes[spec.route]
        self.runner = runner

    def validate_contract(self) -> dict[str, Any]:
        raise NotImplementedError

    def detect(self) -> DetectionResult | None:
        return None

    def infer_installed(self, python: Path) -> DetectionResult | None:
        return None

    def machine_fingerprint(self) -> dict[str, Any]:
        """Return provider-specific hardware facts that invalidate persisted routes.

        Providers should keep this probe cheap and side-effect free.  A future Triton
        or GUI/headless provider can contribute its own platform/runtime signature
        without changing Aurauv's state layer.
        """

        return {}

    def preflight(self, option: str) -> dict[str, Any]:
        return {}

    def verify(self, python: Path, *, option: str, cwd: Path) -> VerificationResult:
        raise NotImplementedError

    def protected_packages(self) -> tuple[str, ...]:
        return ()

    def option_details(self, option: str) -> dict[str, Any]:
        return {}

    def current_target_install_args(self, option: str) -> tuple[str, ...]:
        """Extra ``uv pip install`` arguments for current-interpreter bootstraps."""

        return ()
