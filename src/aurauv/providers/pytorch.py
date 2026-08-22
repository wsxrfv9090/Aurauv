"""PyTorch provider for CPU, Apple MPS, and NVIDIA CUDA routing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any
from urllib.parse import urlparse

from ..errors import AurauvError, ProviderPreflightError, ProviderVerificationError
from ..models import DetectionResult, VerificationResult
from ..utils import load_toml, normalized_name, requirement_name_and_extras
from .base import Provider


_CUDA_BACKEND = re.compile(r"cu(\d{3,4})")
_APPLE_ARM = {"arm64", "aarch64"}
_REQUIRED_OPTIONS = {"cpu", "mps", "cuda"}


@dataclass(frozen=True, slots=True)
class PytorchContract:
    packages: tuple[str, ...]
    imports: dict[str, str]
    option_extras: dict[str, str]
    option_indexes: dict[str, str]
    indexes: dict[str, str]
    mps_requires_available: bool

    def index_url(self, option: str) -> str:
        return self.indexes[self.option_indexes[option]]

    def backend(self, option: str) -> str:
        path = urlparse(self.index_url(option)).path.rstrip("/")
        backend = path.rsplit("/", 1)[-1]
        if backend == "cpu" or _CUDA_BACKEND.fullmatch(backend):
            return backend
        raise AurauvError(
            f"Could not derive a PyTorch backend from index URL {self.index_url(option)!r}."
        )

    def expected_cuda_version(self, option: str) -> str | None:
        if option != "cuda":
            return None
        match = _CUDA_BACKEND.fullmatch(self.backend(option))
        if match is None:
            raise AurauvError("The CUDA option must use a /whl/cuXYZ PyTorch index.")
        digits = match.group(1)
        if len(digits) == 3:
            return f"{int(digits[:-1])}.{int(digits[-1])}"
        return f"{int(digits[:-2])}.{int(digits[-2:])}"


class PytorchProvider(Provider):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._contract: PytorchContract | None = None

    @property
    def contract(self) -> PytorchContract:
        if self._contract is None:
            raise AurauvError("PyTorch provider contract has not been validated.")
        return self._contract

    def _string_map(self, name: str, default: dict[str, str]) -> dict[str, str]:
        value = self.spec.options.get(name, default)
        if not isinstance(value, dict):
            raise AurauvError(f"Provider {self.name!r} option {name!r} must be a table.")
        result: dict[str, str] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not isinstance(item, str) or not key or not item:
                raise AurauvError(
                    f"Provider {self.name!r} option {name!r} must map strings to strings."
                )
            result[key] = item
        return result

    def _packages(self) -> tuple[str, ...]:
        value = self.spec.options.get("packages", ["torch", "torchvision"])
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item for item in value
        ):
            raise AurauvError(
                f"Provider {self.name!r} packages must be a non-empty string array."
            )
        normalized = tuple(dict.fromkeys(value))
        if "torch" not in {normalized_name(item) for item in normalized}:
            raise AurauvError("The PyTorch provider packages must include torch.")
        return normalized

    def _build_contract(self) -> PytorchContract:
        if not _REQUIRED_OPTIONS.issubset(self.route.options):
            raise AurauvError(
                f"PyTorch route {self.route.name!r} must define cpu, mps, and cuda options."
            )
        packages = self._packages()
        option_extras = self._string_map(
            "extras", {"cpu": "cpu", "mps": "mps", "cuda": "cuda"}
        )
        option_indexes = self._string_map(
            "indexes",
            {"cpu": "pytorch-cpu", "mps": "pytorch-cpu", "cuda": "pytorch-cuda"},
        )
        imports = self._string_map(
            "imports", {package: package.replace("-", "_") for package in packages}
        )
        if set(option_extras) != _REQUIRED_OPTIONS:
            raise AurauvError("PyTorch provider extras must map exactly cpu, mps, and cuda.")
        if set(option_indexes) != _REQUIRED_OPTIONS:
            raise AurauvError("PyTorch provider indexes must map exactly cpu, mps, and cuda.")
        if not set(packages).issubset(imports):
            missing = sorted(set(packages) - set(imports))
            raise AurauvError(f"PyTorch import mapping is missing: {', '.join(missing)}")
        mps_requires_available = self.spec.options.get("mps-requires-available", True)
        if not isinstance(mps_requires_available, bool):
            raise AurauvError("mps-requires-available must be boolean.")

        pyproject_path = self.project_root / "pyproject.toml"
        pyproject = load_toml(pyproject_path)
        project = pyproject.get("project", {})
        optional = project.get("optional-dependencies", {})
        if not isinstance(optional, dict):
            raise AurauvError(f"{pyproject_path} has no optional-dependencies table.")
        expected_names = {normalized_name(item) for item in packages}
        dependencies = project.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise AurauvError(f"{pyproject_path}: project.dependencies must be an array.")
        unconditional = {
            requirement_name_and_extras(item)[0]
            for item in dependencies
            if isinstance(item, str)
        } & expected_names
        if unconditional:
            raise AurauvError(
                "Machine-routed PyTorch packages must not be unconditional dependencies: "
                + ", ".join(sorted(unconditional))
            )
        for option, extra in option_extras.items():
            requirements = optional.get(extra)
            if not isinstance(requirements, list):
                raise AurauvError(
                    f"PyTorch option {option!r} refers to missing extra {extra!r} in "
                    f"{pyproject_path}."
                )
            names = {
                requirement_name_and_extras(item)[0]
                for item in requirements
                if isinstance(item, str)
            }
            missing = expected_names - names
            if missing:
                raise AurauvError(
                    f"Extra {extra!r} is missing PyTorch packages: {', '.join(sorted(missing))}."
                )

        uv = pyproject.get("tool", {}).get("uv", {})
        raw_indexes = uv.get("index", []) if isinstance(uv, dict) else []
        if not isinstance(raw_indexes, list):
            raise AurauvError(f"{pyproject_path}: tool.uv.index must be an array.")
        indexes: dict[str, str] = {}
        explicit: dict[str, bool] = {}
        for entry in raw_indexes:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            url = entry.get("url")
            if isinstance(name, str) and isinstance(url, str):
                indexes[name] = url
                explicit[name] = entry.get("explicit") is True
        for index_name in set(option_indexes.values()):
            if index_name not in indexes:
                raise AurauvError(f"PyTorch index {index_name!r} is not defined.")
            if not explicit.get(index_name):
                raise AurauvError(f"PyTorch index {index_name!r} must be explicit = true.")
            parsed = urlparse(indexes[index_name])
            if parsed.scheme != "https" or parsed.hostname != "download.pytorch.org":
                raise AurauvError(
                    f"PyTorch index {index_name!r} must use https://download.pytorch.org."
                )
        for option in ("cpu", "mps"):
            if urlparse(indexes[option_indexes[option]]).path.rstrip("/") != "/whl/cpu":
                raise AurauvError(
                    f"PyTorch {option} must route through the official /whl/cpu index."
                )
        cuda_path = urlparse(indexes[option_indexes["cuda"]]).path.rstrip("/")
        if _CUDA_BACKEND.fullmatch(cuda_path.rsplit("/", 1)[-1]) is None:
            raise AurauvError("PyTorch cuda index must end with /whl/cuXYZ.")

        sources = uv.get("sources", {}) if isinstance(uv, dict) else {}
        if not isinstance(sources, dict):
            raise AurauvError(f"{pyproject_path}: tool.uv.sources must be a table.")
        for package in packages:
            entries = sources.get(package)
            if not isinstance(entries, list):
                raise AurauvError(
                    f"{pyproject_path}: tool.uv.sources.{package} must be an array."
                )
            actual = {
                entry.get("extra"): entry.get("index")
                for entry in entries
                if isinstance(entry, dict)
                and isinstance(entry.get("extra"), str)
                and isinstance(entry.get("index"), str)
            }
            for option, extra in option_extras.items():
                if actual.get(extra) != option_indexes[option]:
                    raise AurauvError(
                        f"tool.uv.sources.{package} must map extra {extra!r} to "
                        f"index {option_indexes[option]!r}."
                    )

        contract = PytorchContract(
            packages=packages,
            imports=imports,
            option_extras=option_extras,
            option_indexes=option_indexes,
            indexes={name: indexes[name] for name in set(option_indexes.values())},
            mps_requires_available=mps_requires_available,
        )
        contract.expected_cuda_version("cuda")
        return contract

    def _validate_owner_forwarding(self) -> dict[str, Any]:
        if self.project_root == self.config.owner_root:
            return {"mode": "provider declared by environment owner"}
        provider_pyproject = load_toml(self.project_root / "pyproject.toml")
        distribution = provider_pyproject.get("project", {}).get("name")
        if not isinstance(distribution, str) or not distribution:
            raise AurauvError(f"Provider project {self.project_root} has no project.name.")
        distribution_name = normalized_name(distribution)
        owner_optional = self.config.pyproject.get("project", {}).get(
            "optional-dependencies", {}
        )
        if not isinstance(owner_optional, dict):
            raise AurauvError("Owner optional-dependencies must be a table.")
        details: dict[str, Any] = {}
        for option, provider_extra in self.contract.option_extras.items():
            root_extras = self.route.options[option].extras
            requirements: list[str] = []
            for root_extra in root_extras:
                values = owner_optional.get(root_extra, [])
                if isinstance(values, list):
                    requirements.extend(item for item in values if isinstance(item, str))
            found = False
            for requirement in requirements:
                name, extras = requirement_name_and_extras(requirement)
                if name == distribution_name and normalized_name(provider_extra) in extras:
                    found = True
                    break
            if not found:
                raise AurauvError(
                    f"Owner route {self.route.name}={option} must depend on "
                    f"{distribution}[{provider_extra}] through extras "
                    f"{', '.join(root_extras)}."
                )
            details[option] = {
                "owner_extras": list(root_extras),
                "provider_requirement": f"{distribution}[{provider_extra}]",
            }
        return {"mode": "owner extras forward to member provider", "options": details}

    def validate_contract(self) -> dict[str, Any]:
        self._contract = self._build_contract()
        return {
            "project": str(self.project_root),
            "packages": list(self.contract.packages),
            "cpu_index": self.contract.index_url("cpu"),
            "mps_index": self.contract.index_url("mps"),
            "cuda_index": self.contract.index_url("cuda"),
            "cuda_backend": self.contract.backend("cuda"),
            "expected_cuda": self.contract.expected_cuda_version("cuda"),
            "owner_routing": self._validate_owner_forwarding(),
        }

    @staticmethod
    def _nvidia_smi_probe() -> tuple[bool, dict[str, Any]]:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return False, {"nvidia_smi": None, "reason": "nvidia-smi is not on PATH"}
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                check=False,
                text=True,
                capture_output=True,
                timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, {"nvidia_smi": executable, "reason": str(exc)}
        output = (completed.stdout or "").strip()
        failure = (completed.stderr or "").strip()
        if completed.returncode != 0 or not output:
            return False, {
                "nvidia_smi": executable,
                "returncode": completed.returncode,
                "reason": failure or output or "nvidia-smi returned no GPU",
            }
        return True, {
            "nvidia_smi": executable,
            "gpus": [line.strip() for line in output.splitlines() if line.strip()],
        }

    def machine_fingerprint(self) -> dict[str, Any]:
        available, details = self._nvidia_smi_probe()
        return {
            "system": platform.system(),
            "machine": platform.machine().lower(),
            "nvidia_available": available,
            "nvidia": details,
        }

    def detect(self) -> DetectionResult | None:
        system = platform.system()
        machine = platform.machine().lower()
        if system == "Darwin" and machine in _APPLE_ARM:
            return DetectionResult(
                option="mps",
                reason="Apple Silicon macOS detected; MPS uses the CPU wheel index",
                details={"system": system, "machine": machine},
            )
        available, details = self._nvidia_smi_probe()
        if available and system != "Darwin":
            return DetectionResult(
                option="cuda",
                reason="nvidia-smi successfully queried an NVIDIA GPU",
                details=details,
            )
        return DetectionResult(
            option="cpu",
            reason="no usable Apple MPS or NVIDIA CUDA runtime was detected",
            details={"system": system, "machine": machine, **details},
        )

    def _probe(self, python: Path, *, cwd: Path) -> dict[str, Any]:
        payload = json.dumps(self.contract.imports)
        code = rf'''
from __future__ import annotations
import importlib
import importlib.metadata as metadata
import json

imports = json.loads({payload!r})
result = {{
    "distributions": {{}},
    "import_errors": {{}},
    "module_versions": {{}},
    "cuda_version": None,
    "cuda_available": False,
    "mps_built": False,
    "mps_available": False,
}}
for distribution, module_name in imports.items():
    try:
        result["distributions"][distribution] = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        result["distributions"][distribution] = None
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
    result["cuda_version"] = getattr(torch.version, "cuda", None)
    result["cuda_available"] = bool(torch.cuda.is_available())
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        result["mps_built"] = bool(mps.is_built())
        result["mps_available"] = bool(mps.is_available())
print(json.dumps(result, sort_keys=True))
'''
        completed = subprocess.run(
            [str(python), "-c", code],
            cwd=cwd,
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise AurauvError(f"Could not inspect PyTorch with {python}: {detail}")
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AurauvError(
                f"PyTorch probe returned invalid JSON: {completed.stdout!r}"
            ) from exc
        if not isinstance(result, dict):
            raise AurauvError("PyTorch probe returned a non-object JSON value.")
        return result

    def infer_installed(self, python: Path) -> DetectionResult | None:
        if not python.is_file():
            return None
        state = self._probe(python, cwd=self.config.owner_root)
        if state.get("import_errors"):
            return None
        distributions = state.get("distributions", {})
        if not isinstance(distributions, dict) or not all(distributions.values()):
            return None
        if state.get("cuda_version"):
            return DetectionResult("cuda", "existing PyTorch CUDA wheel", state)
        if (
            platform.system() == "Darwin"
            and platform.machine().lower() in _APPLE_ARM
            and state.get("mps_built")
        ):
            return DetectionResult("mps", "existing MPS-capable PyTorch wheel", state)
        return DetectionResult("cpu", "existing CPU PyTorch wheel", state)

    def preflight(self, option: str) -> dict[str, Any]:
        system = platform.system()
        machine = platform.machine().lower()
        if option == "cpu":
            return {"system": system, "machine": machine, "runtime": "cpu"}
        if option == "mps":
            if system != "Darwin" or machine not in _APPLE_ARM:
                raise ProviderPreflightError(
                    self.name,
                    self.route.name,
                    option,
                    "MPS requires Apple Silicon macOS; the wheel still comes from /whl/cpu.",
                )
            return {
                "system": system,
                "machine": machine,
                "runtime": "mps",
                "wheel_backend": self.contract.backend(option),
            }
        if option == "cuda":
            if system == "Darwin":
                raise ProviderPreflightError(
                    self.name, self.route.name, option, "CUDA wheels are unsupported on macOS."
                )
            available, details = self._nvidia_smi_probe()
            if not available:
                raise ProviderPreflightError(
                    self.name,
                    self.route.name,
                    option,
                    str(details.get("reason", "NVIDIA driver communication failed")),
                )
            return {
                **details,
                "runtime": "cuda",
                "wheel_backend": self.contract.backend(option),
                "expected_cuda": self.contract.expected_cuda_version(option),
                "nvcc_required_for_wheel": False,
            }
        raise AurauvError(f"PyTorch does not know option {option!r}.")

    def verify(self, python: Path, *, option: str, cwd: Path) -> VerificationResult:
        state = self._probe(python, cwd=cwd)
        distributions = state.get("distributions", {})
        import_errors = state.get("import_errors", {})
        if not isinstance(distributions, dict) or any(
            not distributions.get(package) for package in self.contract.packages
        ) or import_errors:
            raise ProviderVerificationError(
                provider=self.name,
                route=self.route.name,
                option=option,
                reason="one or more configured PyTorch distributions are missing or unimportable",
                state=state,
                repairable=True,
            )
        cuda_version = state.get("cuda_version")
        if option == "cuda":
            expected = self.contract.expected_cuda_version(option)
            if str(cuda_version or "") != expected:
                raise ProviderVerificationError(
                    self.name,
                    self.route.name,
                    option,
                    f"torch reports CUDA {cuda_version!r}; configured index requires {expected!r}",
                    state,
                    repairable=True,
                )
            if not state.get("cuda_available"):
                raise ProviderVerificationError(
                    self.name,
                    self.route.name,
                    option,
                    "CUDA wheel is installed, but torch.cuda.is_available() is False",
                    state,
                    fallback_safe=True,
                )
        else:
            if cuda_version is not None:
                raise ProviderVerificationError(
                    self.name,
                    self.route.name,
                    option,
                    f"a CUDA wheel is installed in non-CUDA option ({cuda_version})",
                    state,
                    repairable=True,
                )
            if option == "mps" and not state.get("mps_built"):
                raise ProviderVerificationError(
                    self.name,
                    self.route.name,
                    option,
                    "installed CPU-index wheel was not built with MPS support",
                    state,
                    repairable=True,
                    fallback_safe=True,
                )
            if (
                option == "mps"
                and self.contract.mps_requires_available
                and not state.get("mps_available")
            ):
                raise ProviderVerificationError(
                    self.name,
                    self.route.name,
                    option,
                    "torch.backends.mps.is_available() is False",
                    state,
                    fallback_safe=True,
                )
        return VerificationResult(self.name, self.route.name, option, state)

    def protected_packages(self) -> tuple[str, ...]:
        return self.contract.packages

    def option_details(self, option: str) -> dict[str, Any]:
        return {
            "provider_project": str(self.project_root),
            "index": self.contract.index_url(option),
            "wheel_backend": self.contract.backend(option),
            "expected_cuda": self.contract.expected_cuda_version(option),
            "packages": list(self.contract.packages),
        }

    def current_target_install_args(self, option: str) -> tuple[str, ...]:
        return ("--torch-backend", self.contract.backend(option))
