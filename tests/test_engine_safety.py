from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from aurauv.config import load_config
from aurauv.engine import AurauvEngine
from aurauv.errors import AurauvError
from aurauv.invocation import parse_invocation
from aurauv.models import (
    AurauvConfig,
    EnvironmentIdentity,
    PythonSelection,
    RouteSelection,
    Topology,
)
from aurauv.transaction import MetadataTransaction


def _engine(
    arguments: list[str],
    *,
    tmp_path: Path,
    uv_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AurauvEngine:
    monkeypatch.setenv("AURAUV_UV", str(uv_path))
    return AurauvEngine(parse_invocation(arguments), cwd=tmp_path)


@pytest.mark.parametrize("flag", ["--locked", "--frozen"])
def test_locked_and_frozen_disable_only_aurauv_side_effect_capabilities(
    flag: str,
    tmp_path: Path,
    uv_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(
        ["sync", flag], tmp_path=tmp_path, uv_path=uv_path, monkeypatch=monkeypatch
    )

    capabilities = engine._execution_capabilities()

    assert capabilities.may_update_uv is False
    assert capabilities.may_install_python is False
    assert capabilities.may_refresh_member_locks is False
    assert capabilities.may_repair_providers is False
    assert capabilities.may_resync_fallback is False
    assert capabilities.may_write_state is False


def test_normal_sync_keeps_existing_mutating_capabilities(
    tmp_path: Path,
    uv_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(
        ["sync"], tmp_path=tmp_path, uv_path=uv_path, monkeypatch=monkeypatch
    )

    capabilities = engine._execution_capabilities()

    assert capabilities.may_update_uv is True
    assert capabilities.may_install_python is True
    assert capabilities.may_refresh_member_locks is True
    assert capabilities.may_repair_providers is True
    assert capabilities.may_resync_fallback is True
    assert capabilities.may_write_state is True


@pytest.mark.parametrize("flag", ["--locked", "--frozen"])
def test_constrained_sync_verifies_without_repair_resync_or_state_write(
    flag: str,
    tmp_path: Path,
    uv_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(
        ["sync", flag], tmp_path=tmp_path, uv_path=uv_path, monkeypatch=monkeypatch
    )
    engine.capabilities = engine._execution_capabilities()
    observed: dict[str, object] = {}
    monkeypatch.setattr(engine, "_select", dict)
    monkeypatch.setattr(
        engine,
        "_reconcile_members",
        lambda **kwargs: observed.setdefault("members", kwargs),
    )
    monkeypatch.setattr(
        engine,
        "_routed_args",
        lambda original, command_index, selections, **kwargs: list(original),
    )
    monkeypatch.setattr(engine, "_run_uv", lambda *args, **kwargs: None)

    def verify(selections, *, allow_repair, allow_resync):
        observed["verify"] = (allow_repair, allow_resync)
        return selections, {"checked": {}}

    monkeypatch.setattr(engine, "_verify_with_runtime_fallback", verify)
    monkeypatch.setattr(
        engine,
        "_write_state",
        lambda *args, **kwargs: observed.setdefault("state_written", True),
    )

    assert engine._execute_sync() == 0
    assert observed["members"] == {"all_members": True, "check_only": True}
    assert observed["verify"] == (False, False)
    assert "state_written" not in observed


def _membership_project(tmp_path: Path, providers: str) -> Path:
    root = tmp_path / "membership"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        f"""[project]
name = "membership"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
local = []

[tool.aurauv]
schema-version = 1

[tool.aurauv.routes.runtime]
default = "local"
providers = {providers}

[tool.aurauv.routes.runtime.options.local]
extras = ["local"]

[tool.aurauv.providers.fake]
type = "future-test-provider"
project = "."
route = "runtime"
""",
        encoding="utf-8",
    )
    return root


def test_provider_must_be_listed_by_its_route(tmp_path: Path) -> None:
    root = _membership_project(tmp_path, "[]")

    with pytest.raises(AurauvError, match="is not listed"):
        load_config(root)


def test_route_rejects_duplicate_provider_membership(tmp_path: Path) -> None:
    root = _membership_project(tmp_path, '["fake", "fake"]')

    with pytest.raises(AurauvError, match="must not contain duplicates"):
        load_config(root)


def test_metadata_transaction_restores_existing_and_removes_created_files(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "pyproject.toml"
    created = tmp_path / "uv.lock"
    existing.write_text("before\n", encoding="utf-8")
    existing.chmod(0o640)
    transaction = MetadataTransaction((existing, created))

    existing.write_text("after\n", encoding="utf-8")
    existing.chmod(0o600)
    created.write_text("new lock\n", encoding="utf-8")

    restored = transaction.rollback()

    assert restored == (existing, created)
    assert existing.read_text(encoding="utf-8") == "before\n"
    assert existing.stat().st_mode & 0o777 == 0o640
    assert not created.exists()
    assert not list(tmp_path.glob("*.aurauv-restore"))


def _mutation_project(tmp_path: Path) -> tuple[Path, AurauvConfig]:
    root = tmp_path / "mutation"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        """[project]
name = "mutation"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
local = []

[tool.aurauv]
schema-version = 1

[tool.aurauv.routes.runtime]
default = "local"
providers = []

[tool.aurauv.routes.runtime.options.local]
extras = ["local"]
""",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("original lock\n", encoding="utf-8")
    return root, load_config(root)


def _prepared_mutation_engine(
    arguments: list[str],
    *,
    root: Path,
    config: AurauvConfig,
    uv_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> AurauvEngine:
    engine = _engine(
        arguments,
        tmp_path=root,
        uv_path=uv_path,
        monkeypatch=monkeypatch,
    )
    engine.config = config
    engine.topology = Topology(
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
    engine.python = PythonSelection(Path(sys.executable), "3.13", "test")
    engine.environment = EnvironmentIdentity(
        target="project",
        path=root / ".venv",
        state_path=root / ".aurauv/state/test.json",
    )
    return engine


def test_mutation_verification_failure_restores_metadata_and_environment(
    tmp_path: Path,
    uv_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _mutation_project(tmp_path)
    original_pyproject = (root / "pyproject.toml").read_bytes()
    original_lock = (root / "uv.lock").read_bytes()
    engine = _prepared_mutation_engine(
        ["add", "example"],
        root=root,
        config=config,
        uv_path=uv_path,
        monkeypatch=monkeypatch,
    )
    selection = RouteSelection(
        route="runtime",
        requested="local",
        selected="local",
        origin="test",
    )
    calls: list[list[str]] = []

    def run_uv(args, **kwargs):
        calls.append(list(args))
        if len(calls) == 1:
            (root / "pyproject.toml").write_text("mutated project\n", encoding="utf-8")
            (root / "uv.lock").write_text("mutated lock\n", encoding="utf-8")

    monkeypatch.setattr(engine, "_run_uv", run_uv)
    monkeypatch.setattr(engine, "_reconcile_members", lambda **kwargs: None)
    monkeypatch.setattr(engine, "_select", lambda: {"runtime": selection})
    monkeypatch.setattr(engine, "_fresh_sync_args", lambda selections: ["sync"])
    monkeypatch.setattr(
        engine,
        "_verify_with_runtime_fallback",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AurauvError("verification failed")
        ),
    )

    with pytest.raises(AurauvError, match="verification failed"):
        engine._execute_mutation("add")

    assert (root / "pyproject.toml").read_bytes() == original_pyproject
    assert (root / "uv.lock").read_bytes() == original_lock
    assert calls == [
        ["add", "--no-sync", "example"],
        ["sync"],
        ["sync", "--locked", "--project", str(root), "--extra", "local"],
    ]
    assert not engine.env_identity.state_path.exists()


def test_unrecoverable_environment_rollback_invalidates_restored_state(
    tmp_path: Path,
    uv_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _mutation_project(tmp_path)
    (root / "uv.lock").unlink()
    engine = _prepared_mutation_engine(
        ["add", "example"],
        root=root,
        config=config,
        uv_path=uv_path,
        monkeypatch=monkeypatch,
    )
    state_path = engine.env_identity.state_path
    state_path.parent.mkdir(parents=True)
    state_path.write_text("previous state\n", encoding="utf-8")
    transaction = MetadataTransaction(engine._mutation_transaction_paths())
    selection = RouteSelection(
        route="runtime",
        requested="local",
        selected="local",
        origin="test",
    )

    with pytest.raises(AurauvError, match="without a root uv.lock"):
        engine._rollback_mutation(
            transaction,
            selections={"runtime": selection},
            environment_may_have_changed=True,
            failure=AurauvError("verification failed"),
        )

    assert not state_path.exists()


def test_no_sync_nonzero_attempts_transaction_rollback_only_once(
    tmp_path: Path,
    uv_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, config = _mutation_project(tmp_path)
    engine = _prepared_mutation_engine(
        ["add", "--no-sync", "example"],
        root=root,
        config=config,
        uv_path=uv_path,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setattr(
        engine.runner,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 2, "", ""),
    )
    attempts = 0

    def rollback(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise AurauvError("rollback failed")

    monkeypatch.setattr(engine, "_rollback_mutation", rollback)

    with pytest.raises(AurauvError, match="rollback failed"):
        engine._execute_mutation("add")

    assert attempts == 1


@pytest.mark.parametrize('flag', ['--isolated', '--no-project', '--script', '--gui-script', '--help'])
def test_child_flags_do_not_bypass_project_routing(flag, tmp_path, uv_path, monkeypatch):
    engine = _engine(['run', 'python', 'app.py', flag], tmp_path=tmp_path, uv_path=uv_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(engine, 'passthrough', lambda: pytest.fail('child flag bypassed routing'))
    monkeypatch.setattr(engine, '_prepare_project', lambda **kwargs: None)
    monkeypatch.setattr(engine, '_execute_run', lambda: 42)
    assert engine.execute() == 42


@pytest.mark.parametrize('flag', ['--isolated', '--no-project', '--script', '-s', '--gui-script'])
def test_uv_ephemeral_flags_still_bypass_routing(flag, tmp_path, uv_path, monkeypatch):
    engine = _engine(['run', flag, 'app.py'], tmp_path=tmp_path, uv_path=uv_path, monkeypatch=monkeypatch)
    monkeypatch.setattr(engine, 'passthrough', lambda: 42)
    assert engine.execute() == 42


def test_current_uv_options_reach_presync_without_losing_following_group(tmp_path, uv_path, monkeypatch):
    engine = _engine(['run', '--no-editable-package', 'demo', '--group', 'test', '--system-certs', 'python', '--group', 'child'], tmp_path=tmp_path, uv_path=uv_path, monkeypatch=monkeypatch)
    _, engine.config = _mutation_project(tmp_path)
    engine.config.routes.clear()
    view = engine._run_option_view()
    assert view == ('run', '--no-editable-package', 'demo', '--group', 'test', '--system-certs')
    sync = engine._fresh_sync_args({}, source_args=view)
    assert '--system-certs' in sync
    assert sync[sync.index('--no-editable-package') + 1] == 'demo'
    assert sync[sync.index('--group') + 1] == 'test'
    assert 'child' not in sync


@pytest.mark.parametrize('option', ['--upgrade-group', '--prerelease-package'])
def test_new_uv_value_options_preserve_following_flags(option, tmp_path, uv_path, monkeypatch):
    engine = _engine(['run', option, 'demo', '--offline', 'python'], tmp_path=tmp_path, uv_path=uv_path, monkeypatch=monkeypatch)
    _, engine.config = _mutation_project(tmp_path)
    engine.config.routes.clear()
    view = engine._run_option_view()
    assert view == ('run', option, 'demo', '--offline')
    sync = engine._fresh_sync_args({}, source_args=view)
    assert sync[sync.index(option) + 1] == 'demo'
    assert '--offline' in sync


def test_module_flag_allows_following_uv_options(tmp_path, uv_path, monkeypatch):
    engine = _engine(['run', '-m', '--offline', 'demo', '--isolated'], tmp_path=tmp_path, uv_path=uv_path, monkeypatch=monkeypatch)
    assert engine._run_option_view() == ('run', '-m', '--offline')


def test_repair_sync_ignores_child_flags_by_default(tmp_path, uv_path, monkeypatch):
    engine = _engine(['run', 'python', 'app.py', '--offline', '--group', 'child'], tmp_path=tmp_path, uv_path=uv_path, monkeypatch=monkeypatch)
    _, engine.config = _mutation_project(tmp_path)
    engine.config.routes.clear()
    sync = engine._fresh_sync_args({})
    assert '--offline' not in sync
    assert '--group' not in sync


def test_explicit_run_separator_keeps_child_options_out_of_sync(tmp_path, uv_path, monkeypatch):
    engine = _engine(['run', '--offline', '--', 'python', '--isolated', '--group', 'child'], tmp_path=tmp_path, uv_path=uv_path, monkeypatch=monkeypatch)
    assert engine._run_option_view() == ('run', '--offline')
