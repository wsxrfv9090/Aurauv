"""Route optional llama-cpp-python CPU/CUDA wheels and verify native devices.

uv owns installation and artifact hashes. This provider validates explicit
upstream indexes, probes NVIDIA hardware without importing PyTorch, and verifies
the selected native backend in a bounded subprocess. It never loads a model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Any
from urllib.parse import urlparse

from ..errors import AurauvError, ProviderPreflightError, ProviderVerificationError
from ..models import DetectionResult, VerificationResult
from ..utils import requirement_name_and_extras
from .base import Provider


_PACKAGE = "llama-cpp-python"
_CUDA_LIBRARIES = r"""
import importlib.metadata as metadata
import json
import sys

result = {"installed": False, "directories": []}
try:
    metadata.version("llama-cpp-python")
except metadata.PackageNotFoundError:
    pass
else:
    result["installed"] = True
    major = int(sys.argv[1])
    suffix = "" if major >= 13 else f"-cu{major}"
    runtime_soname = "11.0" if major == 11 else str(major)
    packages = {
        f"nvidia-cuda-runtime{suffix}": [f"libcudart.so.{runtime_soname}"],
        f"nvidia-cublas{suffix}": [f"libcublas.so.{major}", f"libcublasLt.so.{major}"],
    }
    try:
        for package, libraries in packages.items():
            dist = metadata.distribution(package)
            files = dist.files or []
            for library in libraries:
                matches = [dist.locate_file(file).resolve() for file in files
                           if file.name == library and dist.locate_file(file).is_file()]
                if len(matches) != 1:
                    raise RuntimeError(f"{package} must supply exactly one {library}")
                directory = str(matches[0].parent)
                if directory not in result["directories"]:
                    result["directories"].append(directory)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result))
"""


_PROBE = r"""
import ctypes as c
import importlib.metadata as metadata
import json

result = {"installed": False}
try:
    result["version"] = metadata.version("llama-cpp-python")
except metadata.PackageNotFoundError:
    pass
else:
    result["installed"] = True
    try:
        import llama_cpp
        from llama_cpp import llama_cpp as api
        api.llama_backend_init()
        result["gpu_offload"] = bool(api.llama_supports_gpu_offload())
        result["system_info"] = api.llama_print_system_info().decode(errors="replace")
        lib = api._lib
        def bind(name, restype, *argtypes):
            "Bind the ggml C API exported through libllama dependencies."
            function = getattr(lib, name)
            function.restype = restype
            function.argtypes = list(argtypes)
            return function
        count = bind("ggml_backend_dev_count", c.c_size_t)
        get = bind("ggml_backend_dev_get", c.c_void_p, c.c_size_t)
        reg = bind("ggml_backend_dev_backend_reg", c.c_void_p, c.c_void_p)
        reg_name = bind("ggml_backend_reg_name", c.c_char_p, c.c_void_p)
        description = bind("ggml_backend_dev_description", c.c_char_p, c.c_void_p)
        initialize = bind("ggml_backend_dev_init", c.c_void_p, c.c_void_p, c.c_char_p)
        allocate = bind("ggml_backend_alloc_buffer", c.c_void_p, c.c_void_p, c.c_size_t)
        clear = bind("ggml_backend_buffer_clear", None, c.c_void_p, c.c_uint8)
        synchronize = bind("ggml_backend_synchronize", None, c.c_void_p)
        free_buffer = bind("ggml_backend_buffer_free", None, c.c_void_p)
        free_backend = bind("ggml_backend_free", None, c.c_void_p)
        devices = []
        for index in range(count()):
            device = get(index)
            name = reg_name(reg(device)).decode()
            entry = {"backend": name, "description": description(device).decode()}
            if name in {"CPU", "CUDA"}:
                backend = initialize(device, None)
                if not backend:
                    raise RuntimeError(f"Cannot initialize {name}")
                buffer = None
                try:
                    buffer = allocate(backend, 4096)
                    if not buffer:
                        raise RuntimeError(f"Cannot allocate {name} buffer")
                    clear(buffer, 0)
                    synchronize(backend)
                    entry["allocation_verified"] = True
                finally:
                    if buffer:
                        free_buffer(buffer)
                    free_backend(backend)
            devices.append(entry)
        result["devices"] = devices
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, sort_keys=True))
"""


class LlamaCppProvider(Provider):
    """Own wheel contracts and native-runtime probes for one CPU/CUDA route.

    Construction is side-effect free. validate_contract precedes routing;
    preflight marks an explicitly activated installation as required. verify
    also checks retained installations after inexact sync or run --no-sync,
    while an unselected, absent optional dependency is recorded as inactive.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Store collaborators without touching the environment or hardware."""
        super().__init__(*args, **kwargs)
        self._required = False
        self._indexes: dict[str, str] = {}

    def validate_contract(self) -> dict[str, Any]:
        """Validate owner-local optional extras and explicit upstream wheel sources."""
        if self.project_root != self.config.owner_root:
            raise AurauvError("llama-cpp provider must be declared by the environment owner.")
        if set(self.route.options) != {"cpu", "cuda"}:
            raise AurauvError("llama-cpp route must define exactly cpu and cuda.")
        options = self.spec.options.get("indexes")
        if (not isinstance(options, dict) or set(options) != {"cpu", "cuda"}
                or not all(isinstance(value, str) for value in options.values())):
            raise AurauvError("llama-cpp indexes must map exactly cpu and cuda.")
        project = self.config.pyproject
        optional = project.get("project", {}).get("optional-dependencies", {})
        uv = project.get("tool", {}).get("uv", {})
        indexes = {item.get("name"): item for item in uv.get("index", []) if isinstance(item, dict)}
        sources = uv.get("sources", {}).get(_PACKAGE)
        if (not isinstance(sources, list) or len(sources) != 2
                or not all(isinstance(item, dict) for item in sources)):
            raise AurauvError("llama-cpp-python needs extra-specific tool.uv.sources entries.")
        controlled = {extra for option in self.route.options.values() for extra in option.extras}
        other_requirements = list(project.get("project", {}).get("dependencies", []))
        for extra, requirements in optional.items():
            if extra not in controlled:
                other_requirements.extend(requirements)
        if any(requirement_name_and_extras(item)[0] == _PACKAGE for item in other_requirements):
            raise AurauvError("llama-cpp-python must occur only in its CPU/CUDA route extras.")
        for option in ("cpu", "cuda"):
            extras = self.route.options[option].extras
            if len(extras) != 1:
                raise AurauvError("Each llama-cpp option must select exactly one extra.")
            extra = extras[0]
            names = {requirement_name_and_extras(item)[0] for item in optional.get(extra, [])}
            if _PACKAGE not in names:
                raise AurauvError(f"llama-cpp extra {extra!r} must include {_PACKAGE}.")
            index = indexes.get(options[option], {})
            url = index.get("url", "")
            parsed = urlparse(url)
            backend = parsed.path.rstrip("/").rsplit("/", 1)[-1]
            valid_backend = backend == "cpu" if option == "cpu" else bool(re.fullmatch(r"cu\d{3,4}", backend))
            if (not index.get("explicit") or parsed.scheme != "https"
                    or parsed.netloc != "abetlen.github.io"
                    or not parsed.path.startswith("/llama-cpp-python/whl/")
                    or not valid_backend):
                raise AurauvError(f"llama-cpp {option} requires an explicit upstream {option} wheel index.")
            matches = [item for item in sources if item.get("extra") == extra]
            if len(matches) != 1 or matches[0] != {"index": options[option], "extra": extra}:
                raise AurauvError(f"llama-cpp source for {extra!r} must select index {options[option]!r}.")
            self._indexes[option] = url
            if option == "cuda":
                suffix = "" if self._cuda_major() >= 13 else f"-cu{self._cuda_major()}"
                required = {f"nvidia-cuda-runtime{suffix}", f"nvidia-cublas{suffix}"}
                if not required.issubset(names):
                    raise AurauvError(
                        "llama-cpp CUDA extra must declare its NVIDIA runtime packages: "
                        + ", ".join(sorted(required))
                    )
        return {"package": _PACKAGE, "indexes": dict(self._indexes), "activation_extra": self.route.activation_extra}

    def _cuda_major(self) -> int:
        """Read the two-digit CUDA major from the validated upstream index."""
        backend = urlparse(self._indexes["cuda"]).path.rstrip("/").rsplit("/", 1)[-1]
        return int(backend[2:4])

    def runtime_environment(self, python: Path, *, option: str, cwd: Path) -> dict[str, str]:
        """Locate installed NVIDIA libraries in the target interpreter for Linux CUDA.

        Upstream wheels exclude cudart and cuBLAS. Discover their locked NVIDIA
        distributions without importing Torch, then prepend the library directories
        for child processes. Preserve the caller's search path and parent environment.
        Missing optional llama.cpp leaves the environment unchanged.
        """
        if option != "cuda":
            return {}
        try:
            completed = subprocess.run(
                [str(python), "-c", _CUDA_LIBRARIES, str(self._cuda_major())], cwd=cwd,
                text=True, capture_output=True, check=False, timeout=10,
            )
            if completed.returncode:
                raise ValueError(completed.stderr.strip())
            state = json.loads(completed.stdout)
            if not isinstance(state, dict):
                raise ValueError("runtime result is not an object")
            if state.get("error"):
                raise ValueError(state["error"])
            if not state.get("installed"):
                return {}
            directories = state.get("directories")
            if not isinstance(directories, list) or not directories or not all(
                isinstance(path, str) and Path(path).is_absolute() for path in directories
            ):
                raise ValueError("NVIDIA runtime library directories are missing")
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            raise AurauvError(
                f"Cannot locate llama.cpp CUDA runtime dependencies: {exc}. "
                "Synchronize the CUDA extra with its NVIDIA runtime packages."
            ) from exc
        inherited = os.environ.get("LD_LIBRARY_PATH")
        path = os.pathsep.join(dict.fromkeys(directories))
        if inherited:
            path += os.pathsep + inherited
        return {"LD_LIBRARY_PATH": path}

    @staticmethod
    def _nvidia() -> dict[str, Any]:
        """Read the NVIDIA driver query with an eight-second timeout."""
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return {"present": False, "available": False}
        try:
            completed = subprocess.run(
                [executable, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=8, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"present": True, "available": False, "error": str(exc)}
        return {
            "present": True,
            "available": completed.returncode == 0 and bool(completed.stdout.strip()),
            "gpus": completed.stdout.strip().splitlines(),
            "error": completed.stderr.strip(),
        }

    def machine_fingerprint(self) -> dict[str, Any]:
        """Return hardware facts that invalidate selections copied between hosts."""
        return {"system": platform.system(), "machine": platform.machine(), "nvidia": self._nvidia()}

    def detect(self) -> DetectionResult:
        """Select CUDA when NVIDIA is present; leave driver failures to preflight."""
        facts = self.machine_fingerprint()
        option = "cuda" if facts["nvidia"]["present"] else "cpu"
        return DetectionResult(option, "NVIDIA runtime presence selects the llama.cpp wheel", facts)

    def preflight(self, option: str) -> dict[str, Any]:
        """Check an activated route before uv installs; never authorize fallback."""
        self._required = True
        facts = self.machine_fingerprint()
        if facts["system"] != "Linux":
            raise ProviderPreflightError(self.name, self.route.name, option,
                "This llama-cpp provider currently supports Linux CPU/CUDA wheels.", fallback_safe=False)
        if option == "cuda" and (
            facts["machine"].lower() not in {"x86_64", "amd64"}
            or not facts["nvidia"]["available"]
        ):
            raise ProviderPreflightError(self.name, self.route.name, option,
                "CUDA wheels require x86-64 Linux and a working NVIDIA driver.", fallback_safe=False)
        return {**facts, "index": self._indexes[option], "nvcc_required": False}

    def _probe(self, python: Path, cwd: Path, *, option: str) -> dict[str, Any]:
        """Inspect native devices and allocate/clear 4 KiB per CPU/CUDA device."""
        env = {**os.environ, **self.runtime_environment(python, option=option, cwd=cwd)}
        try:
            completed = subprocess.run([str(python), "-c", _PROBE], cwd=cwd,
                text=True, capture_output=True, check=False, timeout=30, env=env)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise AurauvError(f"llama-cpp probe failed: {exc}") from exc
        if completed.returncode:
            raise AurauvError(f"llama-cpp native probe failed: {completed.stderr.strip()}")
        try:
            state = json.loads(completed.stdout)
            if not isinstance(state, dict):
                raise ValueError("probe result is not an object")
        except (ValueError, TypeError) as exc:
            raise AurauvError("llama-cpp probe returned invalid JSON.") from exc
        return state

    def verify(self, python: Path, *, option: str, cwd: Path) -> VerificationResult:
        """Verify the chosen backend, including native allocation, without model I/O."""
        state = self._probe(python, cwd, option=option)
        if not state.get("installed") and not self._required:
            return VerificationResult(self.name, self.route.name, option, {**state, "active": False})
        devices = state.get("devices", [])
        required = "CUDA" if option == "cuda" else "CPU"
        ready = any(item.get("backend") == required and item.get("allocation_verified") for item in devices)
        wrong_build = (not state.get("gpu_offload")) if option == "cuda" else bool(state.get("gpu_offload"))
        if state.get("error") or not state.get("installed") or wrong_build or not ready:
            raise ProviderVerificationError(self.name, self.route.name, option,
                f"Expected a working {required} llama.cpp build; {state.get('error', 'native backend mismatch')}",
                state, repairable=False, fallback_safe=False)
        return VerificationResult(self.name, self.route.name, option, {**state, "active": True})

    def protected_packages(self) -> tuple[str, ...]:
        """Name the distribution owned by this provider."""
        return (_PACKAGE,)

    def option_details(self, option: str) -> dict[str, Any]:
        """Describe the selected source without importing the runtime."""
        return {"package": _PACKAGE, "index": self._indexes[option]}

    def current_target_install_args(self, option: str) -> tuple[str, ...]:
        """Preserve the wheel source for lock-export/current-interpreter installs."""
        return ("--extra-index-url", self._indexes[option], "--only-binary", _PACKAGE)
