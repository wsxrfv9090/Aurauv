from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from aurauv.config import load_config
from aurauv.errors import AurauvError, ProviderVerificationError
from aurauv.models import EnvironmentIdentity, RouteSelection, Topology
from aurauv.process import CommandRunner
from aurauv.providers.registry import build_provider
from aurauv.state import read_route_state, write_state


def _project(tmp_path: Path, *, conflicts: bool = True) -> Path:
    root = tmp_path / "exclusive-project"
    root.mkdir()
    conflict_table = (
        'conflicts = [[{ extra = "vision-gui" }, { extra = "vision-headless" }]]'
        if conflicts
        else ""
    )
    (root / "pyproject.toml").write_text(
        f'''[project]
name = "exclusive-project"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
vision-gui = ["vision-gui==1.0"]
vision-headless = ["vision-headless==1.0"]

[tool.uv]
package = false
{conflict_table}

[tool.aurauv]
schema-version = 1

[tool.aurauv.routes.vision-runtime]
default = "headless"
providers = ["vision"]
fallbacks = {{ gui = "headless" }}

[tool.aurauv.routes.vision-runtime.options.gui]
extras = ["vision-gui"]

[tool.aurauv.routes.vision-runtime.options.headless]
extras = ["vision-headless"]

[tool.aurauv.providers.vision]
type = "exclusive-distribution"
route = "vision-runtime"
family = ["vision-gui", "vision-headless", "vision-minimal"]
selections = {{ gui = "vision-gui", headless = "vision-headless" }}
module = "shared_cv"
required-attributes = {{ gui = ["optflow"], headless = ["optflow"] }}
''',
        encoding="utf-8",
    )
    return root


def _provider(root: Path):
    config = load_config(root)
    provider = build_provider(
        config.providers["vision"], config=config, runner=CommandRunner()
    )
    return config, provider


def _fake_environment(
    site: Path,
    *distributions: str,
    attribute: bool = True,
) -> None:
    module = site / "shared_cv"
    module.mkdir(parents=True, exist_ok=True)
    source = "__version__ = '1.0'\n"
    if attribute:
        source += "optflow = object()\n"
    (module / "__init__.py").write_text(source, encoding="utf-8")
    for distribution in distributions:
        normalized = distribution.replace("-", "_")
        metadata = site / f"{normalized}-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            f"Name: {distribution}\n"
            "Version: 1.0\n\n",
            encoding="utf-8",
        )


def _pythonpath(monkeypatch: pytest.MonkeyPatch, site: Path) -> None:
    existing = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(site) if not existing else f"{site}{os.pathsep}{existing}",
    )


def test_contract_maps_each_route_option_to_one_conflicting_distribution(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    _config, provider = _provider(root)

    details = provider.validate_contract()

    assert details["family"] == [
        "vision-gui",
        "vision-headless",
        "vision-minimal",
    ]
    assert details["selections"] == {
        "gui": "vision-gui",
        "headless": "vision-headless",
    }
    assert details["module"] == "shared_cv"
    assert provider.preflight("headless")["selected_distribution"] == "vision-headless"
    assert provider.protected_packages() == ("vision-headless",)


def test_contract_requires_route_extras_to_share_a_uv_conflict_group(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path, conflicts=False)

    with pytest.raises(AurauvError, match="tool.uv.conflicts"):
        load_config(root)


def test_installed_distribution_is_detected_only_when_exclusive_and_capable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _config, provider = _provider(root)
    provider.validate_contract()
    site = tmp_path / "site"
    _fake_environment(site, "vision-headless")
    _pythonpath(monkeypatch, site)

    detected = provider.infer_installed(Path(sys.executable))

    assert provider.detect() is None
    assert detected is not None
    assert detected.option == "headless"
    assert detected.details["active_distributions"] == ["vision-headless"]


def test_multiple_family_members_fail_closed_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _config, provider = _provider(root)
    provider.validate_contract()
    site = tmp_path / "site"
    _fake_environment(site, "vision-gui", "vision-headless")
    _pythonpath(monkeypatch, site)

    assert provider.infer_installed(Path(sys.executable)) is None
    with pytest.raises(ProviderVerificationError) as captured:
        provider.verify(Path(sys.executable), option="headless", cwd=root)

    assert captured.value.repairable is False
    assert captured.value.fallback_safe is False
    assert captured.value.state["active_distributions"] == [
        "vision-gui",
        "vision-headless",
    ]


def test_selected_distribution_import_and_capability_are_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _config, provider = _provider(root)
    provider.validate_contract()
    site = tmp_path / "site"
    _fake_environment(site, "vision-headless")
    _pythonpath(monkeypatch, site)

    verified = provider.verify(Path(sys.executable), option="headless", cwd=root)

    assert verified.details["distributions"] == {"vision-headless": "1.0"}
    assert verified.details["attributes"] == {"optflow": True}
    assert verified.details["missing_attributes"] == []


def test_missing_capability_is_repairable_but_never_authorizes_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path)
    _config, provider = _provider(root)
    provider.validate_contract()
    site = tmp_path / "site"
    _fake_environment(site, "vision-gui", attribute=False)
    _pythonpath(monkeypatch, site)

    with pytest.raises(ProviderVerificationError) as captured:
        provider.verify(Path(sys.executable), option="gui", cwd=root)

    assert captured.value.repairable is True
    assert captured.value.fallback_safe is False
    assert captured.value.state["missing_attributes"] == ["optflow"]


def test_provider_machine_fingerprint_invalidates_persisted_route(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path)
    config, provider = _provider(root)
    provider.validate_contract()
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
        target="project",
        path=root / ".venv",
        state_path=root / ".aurauv/state/test.json",
    )
    environment.path.mkdir()
    selection = RouteSelection(
        route="vision-runtime",
        requested="headless",
        selected="headless",
        origin="test",
    )
    machine = {"base": {}, "providers": {"vision": provider.machine_fingerprint()}}
    write_state(
        config,
        topology,
        environment,
        {"vision-runtime": selection},
        {"vision": {}},
        uv_version="uv test",
        python_executable=Path(sys.executable),
        python_version="3.13",
        pyproject_sha256="test",
        lock_sha256="test",
        machine_signature=machine,
    )

    assert read_route_state(
        config,
        topology,
        environment,
        refresh=False,
        machine_signature=machine,
    ) == {"vision-runtime": "headless"}
    changed = json.loads(json.dumps(machine))
    changed["providers"]["vision"]["machine"] = "changed"
    assert read_route_state(
        config,
        topology,
        environment,
        refresh=False,
        machine_signature=changed,
    ) == {}
