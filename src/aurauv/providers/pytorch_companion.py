"""Optional binary companions that must remain compatible with routed PyTorch."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import AurauvError, ProviderVerificationError
from ..models import VerificationResult
from ..utils import normalized_name
from .base import Provider

_BINARY_BACKEND = re.compile(
    r"(?:\+|[.-])(?P<backend>cu\d{3,4}|cpu|rocm\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PytorchCompanionContract:
    packages: tuple[str, ...]
    imports: dict[str, str]
    base_provider: str


def _cuda_backend(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parts = value.split(".", 2)
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return None
    return f"cu{int(parts[0])}{int(parts[1])}"


def _module_backend(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = _BINARY_BACKEND.search(value)
    return match.group("backend").lower() if match is not None else None


class PytorchCompanionProvider(Provider):
    """Verify configured torch-adjacent distributions only when uv installed them.

    A uv lock contains packages for multiple marker and extra branches.  Treating raw lock
    membership as activation would require Aurauv to duplicate uv's resolver.  Instead this
    provider runs after routed sync and activates exactly the distributions present in the
    managed interpreter.  A normal ``uv sync --check`` remains responsible for proving that
    the selected lock branch and environment agree.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._contract: PytorchCompanionContract | None = None
        self._active_packages: tuple[str, ...] | None = None

    @property
    def contract(self) -> PytorchCompanionContract:
        if self._contract is None:
            raise AurauvError(
                "PyTorch companion provider contract has not been validated."
            )
        return self._contract

    def _packages(self) -> tuple[str, ...]:
        value = self.spec.options.get("packages")
        if (
            not isinstance(value, list)
            or not value
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            raise AurauvError(
                f"Provider {self.name!r} packages must be a non-empty string array."
            )
        packages = tuple(dict.fromkeys(normalized_name(item) for item in value))
        if len(packages) != len(value):
            raise AurauvError(
                f"Provider {self.name!r} packages must be unique after normalization."
            )
        return packages

    def _imports(self, packages: tuple[str, ...]) -> dict[str, str]:
        default = {package: package.replace("-", "_") for package in packages}
        value = self.spec.options.get("imports", default)
        if not isinstance(value, dict):
            raise AurauvError(f"Provider {self.name!r} imports must be a string table.")
        imports: dict[str, str] = {}
        for distribution, module in value.items():
            if (
                not isinstance(distribution, str)
                or not isinstance(module, str)
                or not module
            ):
                raise AurauvError(
                    f"Provider {self.name!r} imports must map strings to strings."
                )
            imports[normalized_name(distribution)] = module
        if set(imports) != set(packages):
            raise AurauvError(
                f"Provider {self.name!r} imports must map exactly: {', '.join(packages)}."
            )
        return imports

    def _base_provider(self) -> str:
        value = self.spec.options.get("base-provider", self.route.detector)
        if not isinstance(value, str) or not value:
            raise AurauvError(
                f"Provider {self.name!r} requires base-provider or a route detector."
            )
        base = self.config.providers.get(value)
        if base is None:
            raise AurauvError(
                f"Provider {self.name!r} refers to unknown base-provider {value!r}."
            )
        if base.kind != "pytorch" or base.route != self.route.name:
            raise AurauvError(
                f"Provider {self.name!r} base-provider must be a pytorch provider on route "
                f"{self.route.name!r}."
            )
        providers = self.route.providers
        if value not in providers or self.name not in providers:
            raise AurauvError(
                f"Provider {self.name!r} and base-provider {value!r} must both be attached "
                f"to route {self.route.name!r}."
            )
        if providers.index(value) >= providers.index(self.name):
            raise AurauvError(
                f"Route {self.route.name!r} must list base-provider {value!r} before "
                f"companion {self.name!r}."
            )
        return value

    def validate_contract(self) -> dict[str, Any]:
        if self.project_root != self.config.owner_root:
            raise AurauvError(
                f"PyTorch companion provider {self.name!r} must use the environment owner "
                "project because activation is derived from the shared environment."
            )
        packages = self._packages()
        imports = self._imports(packages)
        base_provider = self._base_provider()
        base_spec = self.config.providers[base_provider]
        raw_base_packages = base_spec.options.get("packages", ["torch", "torchvision"])
        base_packages = (
            {
                normalized_name(item)
                for item in raw_base_packages
                if isinstance(item, str) and item
            }
            if isinstance(raw_base_packages, list)
            else set()
        )
        overlap = sorted(set(packages) & base_packages)
        if overlap:
            raise AurauvError(
                f"Provider {self.name!r} overlaps base-provider {base_provider!r}: "
                + ", ".join(overlap)
            )
        self._contract = PytorchCompanionContract(packages, imports, base_provider)
        return {
            "project": str(self.project_root),
            "base_provider": base_provider,
            "packages": list(packages),
            "imports": imports,
            "activation": "installed-after-sync",
        }

    def preflight(self, option: str) -> dict[str, Any]:
        return {
            "activation": "installed-after-sync",
            "packages": list(self.contract.packages),
            "selected_option": option,
        }

    def _probe(self, python: Path, *, cwd: Path) -> dict[str, Any]:
        payload = json.dumps(self.contract.imports)
        code = rf"""
from __future__ import annotations
import importlib
import importlib.metadata as metadata
import json

imports = json.loads({payload!r})
result = {{
    "active_packages": [],
    "inactive_packages": [],
    "distributions": {{}},
    "import_errors": {{}},
    "module_versions": {{}},
    "torch_cuda_version": None,
}}
for distribution, module_name in imports.items():
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        result["inactive_packages"].append(distribution)
        continue
    result["active_packages"].append(distribution)
    result["distributions"][distribution] = version
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        result["import_errors"][distribution] = repr(exc)
    else:
        result["module_versions"][distribution] = getattr(module, "__version__", None)
try:
    import torch
except Exception:
    pass
else:
    result["torch_cuda_version"] = getattr(torch.version, "cuda", None)
print(json.dumps(result, sort_keys=True))
"""
        completed = subprocess.run(
            [str(python), "-c", code],
            cwd=cwd,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise AurauvError(
                f"Could not inspect PyTorch companions with {python}: {detail}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AurauvError(
                f"PyTorch companion probe returned invalid JSON: {completed.stdout!r}"
            ) from exc
        if not isinstance(result, dict):
            raise AurauvError(
                "PyTorch companion probe returned a non-object JSON value."
            )
        return result

    def verify(self, python: Path, *, option: str, cwd: Path) -> VerificationResult:
        state = self._probe(python, cwd=cwd)
        active = state.get("active_packages", [])
        if not isinstance(active, list) or not all(
            isinstance(item, str) for item in active
        ):
            raise AurauvError(
                "PyTorch companion probe returned invalid active_packages."
            )
        self._active_packages = tuple(active)
        state["activation"] = "installed-after-sync"
        state["configured_packages"] = list(self.contract.packages)
        module_versions = state.get("module_versions", {})
        module_backends = (
            {
                package: _module_backend(version)
                for package, version in module_versions.items()
            }
            if isinstance(module_versions, dict)
            else {}
        )
        state["module_backends"] = module_backends

        import_errors = state.get("import_errors", {})
        if not isinstance(import_errors, dict):
            raise AurauvError("PyTorch companion probe returned invalid import_errors.")
        if import_errors:
            raise ProviderVerificationError(
                provider=self.name,
                route=self.route.name,
                option=option,
                reason="one or more installed PyTorch companion distributions are unimportable",
                state=state,
                repairable=True,
            )

        expected_backend = (
            _cuda_backend(state.get("torch_cuda_version"))
            if option == "cuda"
            else "cpu"
        )
        state["expected_backend"] = expected_backend
        if option == "cuda" and expected_backend is None:
            raise ProviderVerificationError(
                provider=self.name,
                route=self.route.name,
                option=option,
                reason="torch did not expose a CUDA runtime for companion verification",
                state=state,
            )
        mismatches = {
            package: backend
            for package, backend in module_backends.items()
            if backend is not None and backend != expected_backend
        }
        state["backend_mismatches"] = mismatches
        if mismatches:
            summary = ", ".join(
                f"{package}={backend}"
                for package, backend in sorted(mismatches.items())
            )
            raise ProviderVerificationError(
                provider=self.name,
                route=self.route.name,
                option=option,
                reason=(
                    f"companion wheel backend mismatch ({summary}); expected "
                    f"{expected_backend}"
                ),
                state=state,
                repairable=True,
            )
        return VerificationResult(self.name, self.route.name, option, state)

    def protected_packages(self) -> tuple[str, ...]:
        return (
            self.contract.packages
            if self._active_packages is None
            else self._active_packages
        )

    def option_details(self, option: str) -> dict[str, Any]:
        return {
            "base_provider": self.contract.base_provider,
            "activation": "installed-after-sync",
            "packages": list(self.contract.packages),
            "selected_option": option,
        }
