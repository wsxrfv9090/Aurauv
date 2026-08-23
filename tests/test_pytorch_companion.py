from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from aurauv.config import load_config
from aurauv.errors import ProviderVerificationError
from aurauv.models import EnvironmentIdentity, RouteSelection, Topology
from aurauv.process import CommandRunner
from aurauv.providers.registry import build_provider
from aurauv.state import read_route_state, write_state

PYTORCH_PROJECT = Path(__file__).resolve().parent / "fixtures" / "pytorch_project.toml"


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "companion-project"
    root.mkdir()
    original = PYTORCH_PROJECT.read_text(encoding="utf-8")
    original += r"""

[tool.aurauv]
schema-version = 1
project = "Companion project"
minimum-uv = "0.10.0"
python-request = "3.13"

[tool.aurauv.routes.accelerator]
default = "auto"
detector = "pytorch"
providers = ["pytorch", "pytorch-companions"]
fallbacks = { cuda = "cpu", mps = "cpu" }

[tool.aurauv.routes.accelerator.options.cpu]
extras = ["cpu"]

[tool.aurauv.routes.accelerator.options.mps]
extras = ["mps"]

[tool.aurauv.routes.accelerator.options.cuda]
extras = ["cuda"]

[tool.aurauv.providers.pytorch]
type = "pytorch"
project = "."
route = "accelerator"
packages = ["torch", "torchvision"]
extras = { cpu = "cpu", mps = "mps", cuda = "cuda" }
indexes = { cpu = "pytorch-cpu", mps = "pytorch-cpu", cuda = "pytorch-cuda" }

[tool.aurauv.providers.pytorch-companions]
type = "pytorch-companion"
project = "."
route = "accelerator"
base-provider = "pytorch"
packages = ["torchaudio", "torchcodec"]
imports = { torchaudio = "torchaudio", torchcodec = "torchcodec" }
"""
    (root / "pyproject.toml").write_text(original, encoding="utf-8")
    return root


def _provider(root: Path):
    config = load_config(root)
    provider = build_provider(
        config.providers["pytorch-companions"],
        config=config,
        runner=CommandRunner(),
    )
    return config, provider


def _fake_distribution(
    site: Path,
    distribution: str,
    module: str,
    *,
    distribution_version: str,
    module_version: str,
) -> None:
    package = site / module
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f"__version__ = {module_version!r}\n",
        encoding="utf-8",
    )
    normalized = distribution.replace("-", "_")
    metadata = site / f"{normalized}-{distribution_version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        f"Name: {distribution}\n"
        f"Version: {distribution_version}\n\n",
        encoding="utf-8",
    )


def _fake_torch(site: Path, cuda: str | None) -> None:
    package = site / "torch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f"class _Version:\n    cuda = {cuda!r}\nversion = _Version()\n",
        encoding="utf-8",
    )


def test_companion_contract_is_route_attached_and_not_a_detector(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    _config, provider = _provider(root)

    details = provider.validate_contract()

    assert details == {
        "project": str(root),
        "base_provider": "pytorch",
        "packages": ["torchaudio", "torchcodec"],
        "imports": {"torchaudio": "torchaudio", "torchcodec": "torchcodec"},
        "activation": "installed-after-sync",
    }
    assert provider.detect() is None
    assert provider.infer_installed(Path(sys.executable)) is None
    assert provider.preflight("cuda")["selected_option"] == "cuda"


def test_current_interpreter_verification_records_installed_companions_in_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    config, provider = _provider(root)
    provider.validate_contract()
    site = tmp_path / "site"
    site.mkdir()
    _fake_torch(site, "13.0")
    _fake_distribution(
        site,
        "torchaudio",
        "torchaudio",
        distribution_version="2.11.0",
        module_version="2.11.0+cu130",
    )
    _fake_distribution(
        site,
        "torchcodec",
        "torchcodec",
        distribution_version="0.16.0",
        module_version="0.16.0+cu130",
    )
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(site) if not existing else f"{site}{os.pathsep}{existing}",
    )

    verified = provider.verify(Path(sys.executable), option="cuda", cwd=root)

    assert verified.details["active_packages"] == ["torchaudio", "torchcodec"]
    assert verified.details["inactive_packages"] == []
    assert verified.details["module_backends"] == {
        "torchaudio": "cu130",
        "torchcodec": "cu130",
    }
    assert verified.details["expected_backend"] == "cu130"

    topology = Topology(
        invocation_root=root,
        environment_owner=root,
        kind="standalone",
        is_workspace_member=False,
        is_workspace_root=False,
        is_git_submodule=False,
        superproject_root=None,
        git_submodule_paths=(),
        uv_workspace_members=(root,),
    )
    environment = EnvironmentIdentity(
        target="current",
        path=Path(sys.prefix).resolve(),
        state_path=root / ".aurauv/state/current.json",
    )
    selection = RouteSelection(
        route="accelerator",
        requested="cuda",
        selected="cuda",
        origin="test",
    )
    machine_signature = {"base": {}, "providers": {}}
    state_path = write_state(
        config,
        topology,
        environment,
        {"accelerator": selection},
        {provider.name: verified.details},
        uv_version="uv 0.12.5",
        python_executable=Path(sys.executable),
        python_version="3.13",
        pyproject_sha256="test",
        lock_sha256="test",
        machine_signature=machine_signature,
    )
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    result = payload["provider_results"]["pytorch-companions"]
    assert result["distributions"] == {
        "torchaudio": "2.11.0",
        "torchcodec": "0.16.0",
    }
    assert result["module_backends"] == {
        "torchaudio": "cu130",
        "torchcodec": "cu130",
    }

    assert read_route_state(
        config,
        topology,
        environment,
        refresh=False,
        machine_signature=machine_signature,
    ) == {"accelerator": "cuda"}
    pyproject_path = root / "pyproject.toml"
    pyproject_path.write_text(
        pyproject_path.read_text(encoding="utf-8").replace(
            'imports = { torchaudio = "torchaudio", torchcodec = "torchcodec" }',
            'imports = { torchaudio = "torchaudio", torchcodec = "torchcodec._core" }',
        ),
        encoding="utf-8",
    )
    changed_config = load_config(root)
    assert (
        read_route_state(
            changed_config,
            topology,
            environment,
            refresh=False,
            machine_signature=machine_signature,
        )
        == {}
    )


def test_absent_companions_are_an_explicit_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    _config, provider = _provider(root)
    provider.validate_contract()
    monkeypatch.setattr(
        provider,
        "_probe",
        lambda *args, **kwargs: {
            "active_packages": [],
            "inactive_packages": ["torchaudio", "torchcodec"],
            "distributions": {},
            "import_errors": {},
            "module_versions": {},
            "torch_cuda_version": None,
        },
    )

    verified = provider.verify(Path(sys.executable), option="cpu", cwd=root)

    assert verified.details["active_packages"] == []
    assert verified.details["inactive_packages"] == ["torchaudio", "torchcodec"]
    assert provider.protected_packages() == ()


def test_companion_backend_mismatch_fails_without_authorizing_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _project(tmp_path)
    _config, provider = _provider(root)
    provider.validate_contract()
    monkeypatch.setattr(
        provider,
        "_probe",
        lambda *args, **kwargs: {
            "active_packages": ["torchaudio"],
            "inactive_packages": ["torchcodec"],
            "distributions": {"torchaudio": "2.11.0"},
            "import_errors": {},
            "module_versions": {"torchaudio": "2.11.0+cu130"},
            "torch_cuda_version": "13.2",
        },
    )

    with pytest.raises(ProviderVerificationError) as captured:
        provider.verify(Path(sys.executable), option="cuda", cwd=root)

    assert captured.value.repairable is True
    assert captured.value.fallback_safe is False
    assert captured.value.state["expected_backend"] == "cu132"
    assert captured.value.state["backend_mismatches"] == {"torchaudio": "cu130"}
    assert provider.protected_packages() == ("torchaudio",)
