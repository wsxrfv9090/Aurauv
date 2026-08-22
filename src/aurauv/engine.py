"""Aurauv orchestration around uv's native project commands."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Sequence

from .config import load_config
from .errors import (
    AurauvError,
    CommandError,
    ProviderVerificationError,
)
from .invocation import UV_COMMANDS
from .member_lock import reconcile_member_lock
from .models import (
    AuraOptions,
    AurauvConfig,
    EnvironmentIdentity,
    ParsedInvocation,
    PythonSelection,
    RouteSelection,
    Topology,
)
from .process import CommandRunner
from .providers import Provider, build_provider
from .routes import select_routes, selected_extras
from .runtime import (
    child_environment,
    confirm,
    ensure_python,
    ensure_uv_version,
    find_uv,
    uv_version,
)
from .state import (
    config_digest,
    environment_identity,
    environment_python,
    read_route_state,
    write_state,
)
from .topology import discover_topology
from .utils import (
    command_text,
    contains_option,
    file_sha256,
    info,
    insert_after_command,
    option_value,
    option_values,
    warn,
)


_MANAGED_COMMANDS = {"sync", "run", "add", "remove", "lock", "export", "tree"}
_ROUTED_COMMANDS = {"sync", "run", "export"}
_READ_ONLY_SYNC_FLAGS = {"--check", "--dry-run"}
_MEMBER_CHECK_ONLY_FLAGS = {"--check", "--dry-run", "--locked", "--frozen"}
_RUN_EPHEMERAL_FLAGS = {"--no-project", "--isolated", "--script", "--gui-script"}

# Options that belong to `uv run` before the child command.  We only need this
# parser to separate uv's option stream from arguments passed to the program.
# Unknown dash-prefixed options are retained until the first positional token,
# which keeps the wrapper forward-compatible with newer uv releases.
_RUN_VALUE_OPTIONS = {
    "--extra", "--no-extra", "--group", "--no-group", "--only-group",
    "--env-file", "--with", "--with-editable", "--with-requirements",
    "--package", "--python-platform", "--index", "--default-index",
    "--index-url", "--extra-index-url", "--find-links", "--index-strategy",
    "--keyring-provider", "--upgrade-package", "--resolution", "--prerelease",
    "--fork-strategy", "--exclude-newer", "--exclude-newer-package",
    "--no-sources-package", "--reinstall-package", "--link-mode",
    "--config-setting", "--config-settings-package",
    "--no-build-isolation-package", "--no-build-package",
    "--no-binary-package", "--cache-dir", "--refresh-package", "--python",
    "--color", "--allow-insecure-host", "--directory", "--project",
    "--config-file", "-w", "-P", "-C", "-f", "-i", "-p",
}
_RUN_FLAG_OPTIONS = {
    "--all-extras", "--no-dev", "--no-default-groups", "--all-groups",
    "--only-dev", "--no-editable", "--exact", "--no-env-file",
    "--isolated", "--active", "--no-sync", "--locked", "--frozen",
    "--all-packages", "--no-project", "--no-index", "--upgrade",
    "--no-sources", "--reinstall", "--compile-bytecode",
    "--no-build-isolation", "--no-build", "--no-binary", "--no-cache",
    "--refresh", "--managed-python", "--no-managed-python",
    "--no-python-downloads", "--quiet", "--verbose", "--native-tls",
    "--offline", "--no-progress", "--no-config", "-q", "-v", "-U", "-n",
}
_RUN_COMMAND_VALUE_OPTIONS = {"--module", "-m", "--script", "-s", "--gui-script"}

_SYNC_COPY_VALUE_OPTIONS = (
    "--extra", "--no-extra", "--group", "--no-group", "--only-group",
    "--package", "--python-platform", "--index", "--default-index",
    "--index-url", "--extra-index-url", "--find-links", "--index-strategy",
    "--keyring-provider", "--upgrade-package", "--resolution", "--prerelease",
    "--fork-strategy", "--exclude-newer", "--exclude-newer-package",
    "--no-sources-package", "--reinstall-package", "--link-mode",
    "--config-setting", "--config-settings-package",
    "--no-build-isolation-package", "--no-build-package",
    "--no-binary-package", "--cache-dir", "--refresh-package", "--python",
    "--color", "--allow-insecure-host", "--config-file",
)
_SYNC_COPY_FLAGS = (
    "--no-dev", "--no-default-groups", "--all-groups", "--only-dev",
    "--no-editable", "--active", "--locked", "--frozen", "--all-packages",
    "--no-index", "--upgrade", "--no-sources", "--reinstall",
    "--compile-bytecode", "--no-build-isolation", "--no-build",
    "--no-binary", "--no-cache", "--refresh", "--managed-python",
    "--no-managed-python", "--no-python-downloads", "--native-tls",
    "--offline", "--no-progress", "--no-config",
)


class AurauvEngine:
    def __init__(
        self,
        parsed: ParsedInvocation,
        *,
        cwd: Path | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self.parsed = parsed
        self.cwd = (cwd or Path.cwd()).resolve()
        self.runner = runner or CommandRunner()
        self.uv = find_uv()
        self.topology: Topology | None = None
        self.config: AurauvConfig | None = None
        self.providers: dict[str, Provider] = {}
        self.provider_contracts: dict[str, dict[str, Any]] = {}
        self.python: PythonSelection | None = None
        self.environment: EnvironmentIdentity | None = None
        self.uv_version_text = ""
        self.effective_aura = parsed.aura

    def _prepare_project(
        self, *, target: str | None = None, read_only: bool = False
    ) -> None:
        if self.topology is not None:
            return
        topology_args = (
            self._run_option_view()
            if self.parsed.command == "run"
            else self.parsed.uv_args
        )
        topology = discover_topology(
            self.uv,
            topology_args,
            cwd=self.cwd,
            runner=self.runner,
        )
        config = load_config(topology.environment_owner)
        if topology.kind == "unmanaged-submodule" and not config.allow_unmanaged_submodule:
            raise AurauvError(
                f"{topology.invocation_root} is a Git submodule but is not managed by an "
                "ancestor uv workspace. Aurauv will not create a second .venv inside it. "
                "Add it to the superproject workspace and [tool.aurauv.members], or run the "
                "submodule as a truly standalone clone outside the superproject."
            )
        if topology.is_workspace_member:
            configured = [
                member
                for member in config.members.values()
                if member.path == topology.invocation_root
            ]
            if topology.is_git_submodule and not configured:
                raise AurauvError(
                    "The current Git submodule is a uv workspace member, but the environment "
                    "owner does not declare it in [tool.aurauv.members]. Aurauv refuses to "
                    "guess its standalone-lock contract."
                )
        python_submodules = {
            path for path in topology.git_submodule_paths
            if (path / "pyproject.toml").is_file()
        }
        workspace_members = set(topology.uv_workspace_members)
        unmanaged_python = sorted(python_submodules - workspace_members)
        if unmanaged_python and not config.allow_unmanaged_submodule:
            raise AurauvError(
                "The environment owner contains Python Git submodules that are not uv "
                "workspace members: "
                + ", ".join(str(path) for path in unmanaged_python)
                + ". Aurauv cannot guarantee one shared .venv until they are added to "
                "tool.uv.workspace.members (or explicitly exempted)."
            )
        configured_member_paths = {member.path for member in config.members.values()}
        missing_contracts = sorted((python_submodules & workspace_members) - configured_member_paths)
        if missing_contracts and not config.allow_unmanaged_submodule:
            raise AurauvError(
                "Python Git submodules are present in the uv workspace but have no "
                "[tool.aurauv.members] ownership/standalone-lock contract: "
                + ", ".join(str(path) for path in missing_contracts)
            )
        self.topology = topology
        self.config = config
        self.effective_aura = (
            replace(
                self.parsed.aura,
                install_python=False,
                update_uv=False,
                no_input=True,
            )
            if read_only
            else self.parsed.aura
        )
        self.uv_version_text = ensure_uv_version(
            self.uv, config, self.effective_aura, self.runner
        )
        selected_target = target or self.effective_aura.target
        self.python = ensure_python(
            self.uv,
            config,
            self.effective_aura,
            self.runner,
            target=selected_target,
        )
        self.environment = environment_identity(
            config,
            target=selected_target,
            uv_args=topology_args,
            current_prefix=Path(sys.prefix),
        )
        self.providers = {
            name: build_provider(spec, config=config, runner=self.runner)
            for name, spec in config.providers.items()
        }
        self.provider_contracts = {
            name: provider.validate_contract()
            for name, provider in self.providers.items()
        }
        self._announce_topology()

    @property
    def cfg(self) -> AurauvConfig:
        if self.config is None:
            raise AurauvError("Aurauv project context has not been prepared.")
        return self.config

    @property
    def topo(self) -> Topology:
        if self.topology is None:
            raise AurauvError("Aurauv topology has not been prepared.")
        return self.topology

    @property
    def py(self) -> PythonSelection:
        if self.python is None:
            raise AurauvError("Aurauv Python has not been prepared.")
        return self.python

    @property
    def env_identity(self) -> EnvironmentIdentity:
        if self.environment is None:
            raise AurauvError("Aurauv environment has not been prepared.")
        return self.environment

    def _announce_topology(self) -> None:
        topology = self.topo
        if topology.kind == "standalone":
            info(f"Standalone project: {topology.invocation_root}")
        elif topology.kind == "workspace-root":
            info(
                f"Workspace root owns one environment: {topology.environment_owner / '.venv'}"
            )
            if topology.git_submodule_paths:
                info(
                    "Detected Git submodules: "
                    + ", ".join(
                        str(path.relative_to(topology.environment_owner))
                        for path in topology.git_submodule_paths
                    )
                )
        elif topology.kind in {"workspace-member", "workspace-member-root"}:
            label = "workspace member that is also an inner root" if topology.kind.endswith(
                "root"
            ) else "workspace member"
            info(
                f"Current project is a {label}: {topology.invocation_root}; "
                f"environment owner is {topology.environment_owner}."
            )
            if topology.is_git_submodule:
                info("Git submodule semantics are active; no member-local .venv will be used.")
        if self.cfg.members:
            descriptions = [
                f"{member.name}={member.distribution} "
                f"({member.path.relative_to(self.cfg.owner_root)}; "
                f"standalone-lock={member.standalone_lock})"
                for member in self.cfg.members.values()
            ]
            info("Managed workspace members: " + "; ".join(descriptions))

    def _child_env(self) -> dict[str, str]:
        return dict(child_environment(self.py))

    def _environment_python(self) -> Path:
        return environment_python(self.env_identity)

    def _machine_signature(self) -> dict[str, Any]:
        from .utils import machine_fingerprint

        return {
            "base": machine_fingerprint(),
            "providers": {
                name: provider.machine_fingerprint()
                for name, provider in self.providers.items()
            },
        }

    def _persisted(self) -> dict[str, str]:
        return read_route_state(
            self.cfg,
            self.topo,
            self.env_identity,
            refresh=self.effective_aura.refresh,
            machine_signature=self._machine_signature(),
        )

    def _select(self, uv_args: Sequence[str] | None = None) -> dict[str, RouteSelection]:
        return select_routes(
            self.cfg,
            self.providers,
            self.effective_aura,
            tuple(self.parsed.uv_args if uv_args is None else uv_args),
            self._persisted(),
            self._environment_python(),
        )

    def _validate_extra_flags(
        self, args: Sequence[str], selections: dict[str, RouteSelection]
    ) -> None:
        if contains_option(args, "--all-extras"):
            raise AurauvError(
                "--all-extras is unsafe for machine-routed mutually exclusive options. "
                "Select routes through automatic detection, --extra, or --aura-route."
            )
        excluded = set(option_values(args, "--no-extra"))
        for extra in selected_extras(self.cfg, selections):
            if extra in excluded:
                raise AurauvError(
                    f"--no-extra {extra} contradicts the selected Aurauv route."
                )

    def _routed_args(
        self,
        original: Sequence[str],
        command_index: int,
        selections: dict[str, RouteSelection],
        *,
        owner_project: bool = True,
        add_no_sync: bool = False,
        selection_args: Sequence[str] | None = None,
    ) -> list[str]:
        option_view = original if selection_args is None else selection_args
        self._validate_extra_flags(option_view, selections)
        injected: list[str] = []
        existing_extras = set(option_values(option_view, "--extra"))
        for extra in selected_extras(self.cfg, selections):
            if extra not in existing_extras:
                injected.extend(("--extra", extra))
        if owner_project and option_value(option_view, "--project") is None:
            injected.extend(("--project", str(self.cfg.owner_root)))
        if add_no_sync and not contains_option(original, "--no-sync"):
            injected.append("--no-sync")
        return insert_after_command(original, command_index, injected)

    def _run_option_view(self) -> tuple[str, ...]:
        """Return uv's `run` options without inspecting the child command arguments."""

        if self.parsed.command != "run" or self.parsed.command_index is None:
            return self.parsed.uv_args
        args = self.parsed.uv_args
        result = list(args[: self.parsed.command_index + 1])
        index = self.parsed.command_index + 1
        while index < len(args):
            token = args[index]
            if token == "--":
                break
            name = token.split("=", 1)[0]
            if name in _RUN_COMMAND_VALUE_OPTIONS:
                # Module/script selection consumes a value but is not a sync option.
                if "=" not in token:
                    index += 1
                break
            if name in _RUN_VALUE_OPTIONS:
                result.append(token)
                if "=" not in token and index + 1 < len(args):
                    index += 1
                    result.append(args[index])
                index += 1
                continue
            if name in _RUN_FLAG_OPTIONS or token.startswith("-"):
                result.append(token)
                index += 1
                continue
            break
        return tuple(result)

    def _fresh_sync_args(
        self,
        selections: dict[str, RouteSelection],
        *,
        reinstall_packages: Iterable[str] = (),
        no_sync: bool = False,
        inexact: bool = False,
        source_args: Sequence[str] | None = None,
    ) -> list[str]:
        source = tuple(self.parsed.uv_args if source_args is None else source_args)
        globals_before = (
            list(source[: self.parsed.command_index])
            if self.parsed.command_index is not None
            else []
        )
        args = [*globals_before, "sync"]
        if option_value(globals_before, "--project") is None:
            args.extend(("--project", str(self.cfg.owner_root)))
        existing_extras: set[str] = set()
        for name in _SYNC_COPY_VALUE_OPTIONS:
            for value in option_values(source, name):
                if name == "--extra":
                    existing_extras.add(value)
                args.extend((name, value))
        for flag in _SYNC_COPY_FLAGS:
            if contains_option(source, flag) and not contains_option(args, flag):
                args.append(flag)
        for extra in selected_extras(self.cfg, selections):
            if extra not in existing_extras:
                args.extend(("--extra", extra))
        if inexact and not contains_option(source, "--exact"):
            args.append("--inexact")
        if no_sync:
            args.append("--check")
        for package in reinstall_packages:
            args.extend(("--reinstall-package", package))
        return args

    def _run_uv(
        self,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return self.runner.run(
            [str(self.uv), *args],
            cwd=cwd or self.cwd,
            capture_output=capture_output,
            env=self._child_env(),
        )

    def _member_targets(self, *, all_members: bool = False) -> list[Any]:
        if all_members:
            return [
                member
                for member in self.cfg.members.values()
                if member.standalone_lock != "none"
            ]
        for member in self.cfg.members.values():
            if member.path == self.topo.invocation_root:
                return [member] if member.standalone_lock != "none" else []
        package = option_value(self.parsed.uv_args, "--package")
        if package:
            for member in self.cfg.members.values():
                from .utils import normalized_name

                if (
                    normalized_name(member.distribution) == normalized_name(package)
                    and member.standalone_lock != "none"
                ):
                    return [member]
        return []

    def _reconcile_members(
        self,
        *,
        all_members: bool,
        check_only: bool,
        upgrade_provider: str | None = None,
    ) -> None:
        packages: tuple[str, ...] = ()
        if upgrade_provider is not None:
            if upgrade_provider not in self.providers:
                raise AurauvError(f"Unknown provider {upgrade_provider!r}.")
            packages = self.providers[upgrade_provider].protected_packages()
        offline = contains_option(self.parsed.uv_args, "--offline")
        for member in self._member_targets(all_members=all_members):
            reconcile_member_lock(
                self.uv,
                member,
                self.runner,
                check_only=check_only,
                env=self._child_env(),
                upgrade_packages=packages,
                offline=offline,
            )

    def _verify_imports(self, python: Path) -> None:
        if not self.cfg.verify_imports:
            return
        code = (
            "import importlib; "
            f"mods={list(self.cfg.verify_imports)!r}; "
            "[importlib.import_module(name) for name in mods]; "
            "print('[aurauv] VERIFY: imports succeeded: ' + ', '.join(mods))"
        )
        self.runner.run([str(python), "-c", code], cwd=self.cfg.owner_root)

    def _verify_providers(
        self,
        selections: dict[str, RouteSelection],
        *,
        allow_repair: bool,
    ) -> dict[str, dict[str, Any]]:
        python = self._environment_python()
        if not python.is_file():
            raise AurauvError(
                f"The managed environment interpreter does not exist: {python}."
            )
        results: dict[str, dict[str, Any]] = {}
        repaired = False
        for provider_name, provider in self.providers.items():
            option = selections[provider.spec.route].selected
            try:
                verified = provider.verify(python, option=option, cwd=self.cfg.owner_root)
            except ProviderVerificationError as failure:
                if allow_repair and failure.repairable and not repaired:
                    packages = provider.protected_packages()
                    warn(
                        f"{failure}; retrying one routed uv sync with focused reinstall of "
                        f"{', '.join(packages)}."
                    )
                    self._run_uv(
                        self._fresh_sync_args(
                            selections, reinstall_packages=packages
                        ),
                        cwd=self.cfg.owner_root,
                    )
                    repaired = True
                    verified = provider.verify(
                        python, option=option, cwd=self.cfg.owner_root
                    )
                else:
                    raise
            results[provider_name] = verified.details
        self._verify_imports(python)
        return results

    def _verify_with_runtime_fallback(
        self,
        selections: dict[str, RouteSelection],
        *,
        allow_repair: bool,
        allow_resync: bool,
    ) -> tuple[dict[str, RouteSelection], dict[str, dict[str, Any]]]:
        """Verify providers and, only when authorized, resync a safe fallback.

        Preflight catches most invalid routes before uv changes the environment.  Some
        failures are visible only after import/runtime probing (for example a CUDA wheel
        with ``torch.cuda.is_available() == False``).  Those failures use the same explicit
        authorization model as preflight fallback; no silent downgrade is permitted.
        """

        try:
            return selections, self._verify_providers(
                selections, allow_repair=allow_repair
            )
        except ProviderVerificationError as failure:
            if not allow_resync or not failure.fallback_safe:
                raise
            route = self.cfg.routes[failure.route]
            explicit = self.effective_aura.fallback_overrides.get(failure.route)
            fallback = explicit or route.fallbacks.get(failure.option)
            if fallback is None:
                raise
            if fallback not in route.options:
                raise AurauvError(
                    f"Fallback {failure.route}={fallback} is not a configured option."
                )
            approved = explicit is not None or self.effective_aura.assume_yes
            if not approved:
                approved = confirm(
                    f"{failure}. Resynchronize with {failure.route}={fallback}?",
                    self.effective_aura,
                )
            if not approved:
                raise AurauvError(
                    f"{failure}. Runtime fallback to {failure.route}={fallback} "
                    "was not authorized."
                ) from failure

            preflight: dict[str, Any] = {}
            for provider_name in route.providers:
                preflight[provider_name] = self.providers[provider_name].preflight(
                    fallback
                )
            original = selections[failure.route]
            updated = dict(selections)
            updated[failure.route] = replace(
                original,
                selected=fallback,
                origin=f"authorized runtime fallback from {failure.option}",
                fallback_from=failure.option,
                fallback_reason=str(failure),
                details={**original.details, "runtime_fallback_preflight": preflight},
            )
            self._run_uv(self._fresh_sync_args(updated), cwd=self.cfg.owner_root)
            return updated, self._verify_providers(updated, allow_repair=allow_repair)

    def _write_state(
        self,
        selections: dict[str, RouteSelection],
        provider_results: dict[str, dict[str, Any]],
    ) -> None:
        path = write_state(
            self.cfg,
            self.topo,
            self.env_identity,
            selections,
            provider_results,
            uv_version=self.uv_version_text,
            python_executable=self._environment_python(),
            python_version=self.py.version,
            pyproject_sha256=file_sha256(self.cfg.owner_root / "pyproject.toml") or "",
            lock_sha256=file_sha256(self.cfg.owner_root / "uv.lock"),
            machine_signature=self._machine_signature(),
        )
        info(f"Persisted environment routes in {path}.")

    def passthrough(self) -> int:
        completed = self.runner.run(
            [str(self.uv), *self.parsed.uv_args],
            cwd=self.cwd,
            allowed_exit_codes=tuple(range(0, 256)),
        )
        return completed.returncode

    def execute(self) -> int:
        command = self.parsed.command
        if self.parsed.aura.no_route:
            return self.passthrough()
        if command not in _MANAGED_COMMANDS:
            return self.passthrough()
        if command == "run" and any(
            contains_option(self.parsed.uv_args, flag) for flag in _RUN_EPHEMERAL_FLAGS
        ):
            info("Ephemeral/no-project uv run detected; passing through without project routing.")
            return self.passthrough()
        read_only_prepare = (
            command == "sync"
            and any(
                contains_option(self.parsed.uv_args, flag)
                for flag in _READ_ONLY_SYNC_FLAGS
            )
        ) or (command == "lock" and contains_option(self.parsed.uv_args, "--check"))
        self._prepare_project(read_only=read_only_prepare)
        if command == "sync":
            return self._execute_sync()
        if command == "run":
            return self._execute_run()
        if command in {"add", "remove"}:
            return self._execute_mutation(command)
        if command == "lock":
            return self._execute_lock()
        if command == "export":
            return self._execute_export()
        if command == "tree":
            return self._execute_tree()
        raise AurauvError(f"Unhandled managed command {command!r}.")

    def _execute_sync(self) -> int:
        assert self.parsed.command_index is not None
        selections = self._select()
        check_only_members = any(
            contains_option(self.parsed.uv_args, flag) for flag in _MEMBER_CHECK_ONLY_FLAGS
        )
        self._reconcile_members(all_members=True, check_only=check_only_members)
        args = self._routed_args(
            self.parsed.uv_args,
            self.parsed.command_index,
            selections,
            owner_project=True,
        )
        self._run_uv(args)
        read_only = any(
            contains_option(args, flag) for flag in _READ_ONLY_SYNC_FLAGS
        )
        if contains_option(args, "--dry-run"):
            return 0
        selections, provider_results = self._verify_with_runtime_fallback(
            selections,
            allow_repair=not read_only,
            allow_resync=not read_only,
        )
        if not read_only:
            self._write_state(selections, provider_results)
        return 0

    def _execute_run(self) -> int:
        assert self.parsed.command_index is not None
        run_view = self._run_option_view()
        selections = self._select(run_view)
        no_sync = contains_option(run_view, "--no-sync")
        if not no_sync:
            # `uv run` should stay lightweight.  Member standalone locks are maintained
            # by sync/lock/add/remove or `aura lock-members`, not on every program run.
            self._run_uv(
                self._fresh_sync_args(
                    selections, inexact=True, source_args=run_view
                ),
                cwd=self.cfg.owner_root,
            )
        selections, provider_results = self._verify_with_runtime_fallback(
            selections,
            allow_repair=not no_sync,
            allow_resync=not no_sync,
        )
        if not no_sync:
            self._write_state(selections, provider_results)
        routed = self._routed_args(
            self.parsed.uv_args,
            self.parsed.command_index,
            selections,
            owner_project=True,
            selection_args=run_view,
        )
        if not no_sync:
            routed = insert_after_command(routed, self.parsed.command_index, ["--no-sync"])
        completed = self.runner.run(
            [str(self.uv), *routed],
            cwd=self.cwd,
            allowed_exit_codes=tuple(range(0, 256)),
            env=self._child_env(),
        )
        return completed.returncode

    def _execute_mutation(self, command: str) -> int:
        assert self.parsed.command_index is not None
        user_no_sync = contains_option(self.parsed.uv_args, "--no-sync")
        if user_no_sync:
            completed = self.runner.run(
                [str(self.uv), *self.parsed.uv_args],
                cwd=self.cwd,
                allowed_exit_codes=tuple(range(0, 256)),
                env=self._child_env(),
            )
            if completed.returncode == 0:
                self._reconcile_members(
                    all_members=False,
                    check_only=any(
                        contains_option(self.parsed.uv_args, flag)
                        for flag in {"--locked", "--frozen"}
                    ),
                )
            return completed.returncode

        mutation_args = insert_after_command(
            self.parsed.uv_args, self.parsed.command_index, ["--no-sync"]
        )
        self._run_uv(mutation_args)
        self._reconcile_members(
            all_members=False,
            check_only=any(
                contains_option(self.parsed.uv_args, flag)
                for flag in {"--locked", "--frozen"}
            ),
        )
        selections = self._select()
        self._run_uv(self._fresh_sync_args(selections), cwd=self.cfg.owner_root)
        selections, results = self._verify_with_runtime_fallback(
            selections, allow_repair=True, allow_resync=True
        )
        self._write_state(selections, results)
        return 0

    def _execute_lock(self) -> int:
        check_only = any(
            contains_option(self.parsed.uv_args, flag) for flag in _MEMBER_CHECK_ONLY_FLAGS
        )
        self._run_uv(self.parsed.uv_args)
        self._reconcile_members(all_members=True, check_only=check_only)
        return 0

    def _execute_export(self) -> int:
        assert self.parsed.command_index is not None
        selections = self._select()
        args = self._routed_args(
            self.parsed.uv_args,
            self.parsed.command_index,
            selections,
            owner_project=True,
        )
        self._run_uv(args)
        return 0

    def _execute_tree(self) -> int:
        assert self.parsed.command_index is not None
        selections = self._select()
        args = self._routed_args(
            self.parsed.uv_args,
            self.parsed.command_index,
            selections,
            owner_project=True,
        )
        self._run_uv(args)
        return 0

    def status(self) -> dict[str, Any]:
        self._prepare_project(read_only=True)
        selections = self._select()
        payload = {
            "aurauv": {
                "project": self.cfg.project_label,
                "uv": self.uv_version_text,
                "python": {
                    "executable": str(self.py.executable),
                    "version": self.py.version,
                    "origin": self.py.origin,
                },
                "environment": {
                    "target": self.env_identity.target,
                    "path": str(self.env_identity.path),
                    "state": str(self.env_identity.state_path),
                },
                "config_sha256": config_digest(self.cfg),
            },
            "topology": {
                "kind": self.topo.kind,
                "invocation_root": str(self.topo.invocation_root),
                "environment_owner": str(self.topo.environment_owner),
                "git_submodule": self.topo.is_git_submodule,
                "superproject_root": (
                    str(self.topo.superproject_root)
                    if self.topo.superproject_root is not None
                    else None
                ),
                "workspace_members": [str(path) for path in self.topo.uv_workspace_members],
                "git_submodules": [str(path) for path in self.topo.git_submodule_paths],
            },
            "routes": {
                name: {
                    "selected": selection.selected,
                    "requested": selection.requested,
                    "origin": selection.origin,
                    "fallback_from": selection.fallback_from,
                    "extras": list(
                        self.cfg.routes[name].options[selection.selected].extras
                    ),
                    "details": selection.details,
                }
                for name, selection in selections.items()
            },
            "providers": self.provider_contracts,
            "members": {
                name: {
                    "path": str(member.path),
                    "distribution": member.distribution,
                    "standalone_lock": member.standalone_lock,
                }
                for name, member in self.cfg.members.items()
            },
        }
        return payload

    def doctor(self) -> int:
        self._prepare_project(read_only=True)
        selections = self._select()
        self._reconcile_members(all_members=True, check_only=True)
        sync_args = self._fresh_sync_args(selections, no_sync=True)
        self._run_uv(sync_args, cwd=self.cfg.owner_root)
        results = self._verify_providers(selections, allow_repair=False)
        info("Doctor completed without modifying locks, environment, or Aurauv state.")
        if self.parsed.aura.json_output:
            print(json.dumps({"provider_results": results}, indent=2, sort_keys=True))
        return 0

    def lock_members(self, *, check_only: bool) -> int:
        self._prepare_project(read_only=check_only)
        self._reconcile_members(all_members=True, check_only=check_only)
        return 0

    def upgrade_provider(self, provider_name: str) -> int:
        """Refresh one routed package family in member and root lock contracts."""

        self._prepare_project()
        provider = self.providers.get(provider_name)
        if provider is None:
            available = ", ".join(sorted(self.providers)) or "(none)"
            raise AurauvError(
                f"Unknown provider {provider_name!r}; available providers: {available}."
            )
        packages = provider.protected_packages()
        if not packages:
            raise AurauvError(
                f"Provider {provider_name!r} declares no upgradable package family."
            )

        for member in self.cfg.members.values():
            if member.path == provider.project_root and member.standalone_lock != "none":
                reconcile_member_lock(
                    self.uv,
                    member,
                    self.runner,
                    check_only=False,
                    env=self._child_env(),
                    upgrade_packages=packages,
                    offline=contains_option(self.parsed.uv_args, "--offline"),
                )

        lock_args: list[str] = ["lock", "--project", str(self.cfg.owner_root)]
        if contains_option(self.parsed.uv_args, "--offline"):
            lock_args.insert(0, "--offline")
        for package in packages:
            lock_args.extend(("--upgrade-package", package))
        self._run_uv(lock_args, cwd=self.cfg.owner_root)

        selections = self._select()
        self._run_uv(self._fresh_sync_args(selections), cwd=self.cfg.owner_root)
        selections, results = self._verify_with_runtime_fallback(
            selections, allow_repair=True, allow_resync=True
        )
        self._write_state(selections, results)
        info(f"Upgraded provider {provider_name!r}: " + ", ".join(packages))
        return 0

    @staticmethod
    def _is_installable_project(path: Path) -> bool:
        from .utils import load_toml

        pyproject = load_toml(path / "pyproject.toml")
        uv = pyproject.get("tool", {}).get("uv", {})
        package = uv.get("package") if isinstance(uv, dict) else None
        if package is False:
            return False
        if package is True:
            return True
        return isinstance(pyproject.get("build-system"), dict)

    def bootstrap_current(self) -> int:
        self._prepare_project(target="current")
        selections = self._select()
        extras = selected_extras(self.cfg, selections)
        with tempfile.TemporaryDirectory(prefix="aurauv-bootstrap-") as temporary:
            requirements = Path(temporary) / "requirements.txt"
            export_args = [
                "export",
                "--locked",
                "--format",
                "requirements.txt",
                "--no-emit-project",
                "--no-emit-workspace",
                "--output-file",
                str(requirements),
                "--project",
                str(self.cfg.owner_root),
            ]
            for extra in extras:
                export_args.extend(("--extra", extra))
            if contains_option(self.parsed.uv_args, "--offline"):
                export_args.insert(0, "--offline")
            self._run_uv(export_args, cwd=self.cfg.owner_root)
            install_args = [
                "pip",
                "install",
                "--python",
                str(self.py.executable),
                "--requirements",
                str(requirements),
            ]
            for provider_name, provider in self.providers.items():
                option = selections[provider.spec.route].selected
                install_args.extend(provider.current_target_install_args(option))
            if contains_option(self.parsed.uv_args, "--offline"):
                install_args.insert(0, "--offline")
            self._run_uv(install_args, cwd=self.cfg.owner_root)

        editable_paths = [self.cfg.owner_root]
        editable_paths.extend(member.path for member in self.cfg.members.values())
        for path in editable_paths:
            pyproject = path / "pyproject.toml"
            if not pyproject.is_file() or not self._is_installable_project(path):
                continue
            self._run_uv(
                [
                    "pip",
                    "install",
                    "--python",
                    str(self.py.executable),
                    "--no-deps",
                    "--editable",
                    str(path),
                ],
                cwd=self.cfg.owner_root,
            )
        results = self._verify_providers(selections, allow_repair=False)
        self._write_state(selections, results)
        info("Current-interpreter bootstrap completed without exact-syncing the host kernel.")
        return 0
