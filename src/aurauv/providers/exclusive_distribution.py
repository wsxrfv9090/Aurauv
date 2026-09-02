"""Mutually exclusive distributions that expose one shared import module."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import AurauvError, ProviderVerificationError
from ..models import DetectionResult, VerificationResult
from ..utils import normalized_name, requirement_name_and_extras
from .base import Provider


@dataclass(frozen=True, slots=True)
class ExclusiveDistributionContract:
    family: tuple[str, ...]
    selections: dict[str, str]
    module: str
    required_attributes: dict[str, tuple[str, ...]]


class ExclusiveDistributionProvider(Provider):
    """Select and verify exactly one distribution from a shared-import family."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._contract: ExclusiveDistributionContract | None = None
        self._selected_package: str | None = None

    @property
    def contract(self) -> ExclusiveDistributionContract:
        if self._contract is None:
            raise AurauvError(
                "Exclusive distribution provider contract has not been validated."
            )
        return self._contract

    def _family(self) -> tuple[str, ...]:
        value = self.spec.options.get("family")
        if (
            not isinstance(value, list)
            or len(value) < 2
            or not all(isinstance(item, str) and item.strip() for item in value)
        ):
            raise AurauvError(
                f"Provider {self.name!r} family must contain at least two distribution names."
            )
        family = tuple(normalized_name(item) for item in value)
        if len(set(family)) != len(family):
            raise AurauvError(
                f"Provider {self.name!r} family must be unique after normalization."
            )
        return family

    def _selections(self, family: tuple[str, ...]) -> dict[str, str]:
        value = self.spec.options.get("selections")
        if not isinstance(value, dict):
            raise AurauvError(
                f"Provider {self.name!r} selections must map route options to distributions."
            )
        selections: dict[str, str] = {}
        for option, distribution in value.items():
            if (
                not isinstance(option, str)
                or not option
                or not isinstance(distribution, str)
                or not distribution.strip()
            ):
                raise AurauvError(
                    f"Provider {self.name!r} selections must map strings to strings."
                )
            selections[option] = normalized_name(distribution)
        if set(selections) != set(self.route.options):
            raise AurauvError(
                f"Provider {self.name!r} selections must map exactly the route options: "
                + ", ".join(self.route.options)
            )
        if len(set(selections.values())) != len(selections):
            raise AurauvError(
                f"Provider {self.name!r} route options must select distinct distributions."
            )
        unknown = sorted(set(selections.values()) - set(family))
        if unknown:
            raise AurauvError(
                f"Provider {self.name!r} selections are outside its family: "
                + ", ".join(unknown)
            )
        return selections

    def _module(self) -> str:
        value = self.spec.options.get("module")
        if not isinstance(value, str) or not value.strip():
            raise AurauvError(f"Provider {self.name!r} module must be a string.")
        return value.strip()

    def _required_attributes(self) -> dict[str, tuple[str, ...]]:
        value = self.spec.options.get("required-attributes", {})
        if not isinstance(value, dict):
            raise AurauvError(
                f"Provider {self.name!r} required-attributes must be a table."
            )
        unknown = sorted(set(value) - set(self.route.options))
        if unknown:
            raise AurauvError(
                f"Provider {self.name!r} required-attributes has unknown options: "
                + ", ".join(unknown)
            )
        result: dict[str, tuple[str, ...]] = {}
        for option in self.route.options:
            attributes = value.get(option, [])
            if not isinstance(attributes, list) or not all(
                isinstance(item, str) and item.strip() for item in attributes
            ):
                raise AurauvError(
                    f"Provider {self.name!r} required-attributes.{option} must be a string array."
                )
            cleaned = tuple(item.strip() for item in attributes)
            if len(set(cleaned)) != len(cleaned):
                raise AurauvError(
                    f"Provider {self.name!r} required-attributes.{option} must be unique."
                )
            result[option] = cleaned
        return result

    def _validate_owner_dependencies(
        self, family: tuple[str, ...], selections: dict[str, str]
    ) -> dict[str, Any]:
        if self.project_root != self.config.owner_root:
            raise AurauvError(
                f"Exclusive distribution provider {self.name!r} must use the environment "
                "owner project so the selected wheel is explicit in root route extras."
            )
        project = self.config.pyproject.get("project", {})
        if not isinstance(project, dict):
            raise AurauvError("Project metadata must be a table.")
        unconditional = project.get("dependencies", [])
        optional = project.get("optional-dependencies", {})
        if not isinstance(unconditional, list) or not isinstance(optional, dict):
            raise AurauvError(
                "Project dependencies and optional-dependencies must use array/table forms."
            )
        unconditional_family = sorted(
            {
                requirement_name_and_extras(item)[0]
                for item in unconditional
                if isinstance(item, str)
            }
            & set(family)
        )
        if unconditional_family:
            raise AurauvError(
                f"Provider {self.name!r} family must not be unconditional dependencies: "
                + ", ".join(unconditional_family)
            )

        details: dict[str, Any] = {}
        for option, selected in selections.items():
            requirements: list[str] = []
            root_extras = self.route.options[option].extras
            for extra in root_extras:
                values = optional.get(extra)
                if not isinstance(values, list):
                    raise AurauvError(
                        f"Root route extra {extra!r} must be a dependency array."
                    )
                requirements.extend(item for item in values if isinstance(item, str))
            routed_family = {
                requirement_name_and_extras(requirement)[0]
                for requirement in requirements
                if requirement_name_and_extras(requirement)[0] in family
            }
            if routed_family != {selected}:
                found = ", ".join(sorted(routed_family)) or "none"
                raise AurauvError(
                    f"Route {self.route.name}={option} must directly select only {selected!r} "
                    f"from provider {self.name!r}; found {found}."
                )
            details[option] = {
                "owner_extras": list(root_extras),
                "selected_distribution": selected,
            }

        return details

    def validate_contract(self) -> dict[str, Any]:
        family = self._family()
        selections = self._selections(family)
        module = self._module()
        required_attributes = self._required_attributes()
        owner_routing = self._validate_owner_dependencies(family, selections)
        self._contract = ExclusiveDistributionContract(
            family, selections, module, required_attributes
        )
        return {
            "project": str(self.project_root),
            "family": list(family),
            "selections": selections,
            "module": module,
            "required_attributes": {
                option: list(attributes)
                for option, attributes in required_attributes.items()
            },
            "owner_routing": owner_routing,
        }

    def machine_fingerprint(self) -> dict[str, Any]:
        return {"system": platform.system(), "machine": platform.machine().lower()}

    def _probe(
        self, python: Path, *, cwd: Path, attributes: tuple[str, ...]
    ) -> dict[str, Any]:
        payload = json.dumps(
            {
                "family": self.contract.family,
                "module": self.contract.module,
                "attributes": attributes,
            }
        )
        code = rf'''
from __future__ import annotations
import importlib
import importlib.metadata as metadata
import json

config = json.loads({payload!r})
result = {{
    "distributions": {{}},
    "active_distributions": [],
    "inactive_distributions": [],
    "import_error": None,
    "module_origin": None,
    "module_version": None,
    "attributes": {{}},
}}
for distribution in config["family"]:
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        result["inactive_distributions"].append(distribution)
    else:
        result["active_distributions"].append(distribution)
        result["distributions"][distribution] = version
if result["active_distributions"]:
    try:
        module = importlib.import_module(config["module"])
    except Exception as exc:
        result["import_error"] = repr(exc)
    else:
        result["module_origin"] = getattr(module, "__file__", None)
        result["module_version"] = getattr(module, "__version__", None)
        for path in config["attributes"]:
            current = module
            present = True
            for part in path.split("."):
                if not part or not hasattr(current, part):
                    present = False
                    break
                current = getattr(current, part)
            result["attributes"][path] = present
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
            raise AurauvError(
                f"Could not inspect exclusive distribution family with {python}: {detail}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AurauvError(
                "Exclusive distribution probe returned invalid JSON: "
                f"{completed.stdout!r}"
            ) from exc
        if not isinstance(result, dict):
            raise AurauvError(
                "Exclusive distribution probe returned a non-object JSON value."
            )
        return result

    def detect(self) -> DetectionResult | None:
        return None

    def infer_installed(self, python: Path) -> DetectionResult | None:
        if not python.is_file():
            return None
        state = self._probe(python, cwd=self.config.owner_root, attributes=())
        active = state.get("active_distributions")
        if not isinstance(active, list) or len(active) != 1:
            return None
        reverse = {
            distribution: option
            for option, distribution in self.contract.selections.items()
        }
        option = reverse.get(active[0])
        if option is None:
            return None
        attributes = self.contract.required_attributes[option]
        if attributes:
            state = self._probe(
                python, cwd=self.config.owner_root, attributes=attributes
            )
        missing = [
            attribute
            for attribute, present in state.get("attributes", {}).items()
            if not present
        ]
        if state.get("import_error") or missing:
            return None
        return DetectionResult(
            option,
            f"existing exclusive distribution {active[0]}",
            state,
        )

    def preflight(self, option: str) -> dict[str, Any]:
        try:
            selected = self.contract.selections[option]
        except KeyError as exc:
            raise AurauvError(
                f"Provider {self.name!r} does not know option {option!r}."
            ) from exc
        self._selected_package = selected
        return {
            "selected_distribution": selected,
            "module": self.contract.module,
            "required_attributes": list(self.contract.required_attributes[option]),
        }

    def verify(self, python: Path, *, option: str, cwd: Path) -> VerificationResult:
        selected = self.contract.selections[option]
        self._selected_package = selected
        state = self._probe(
            python,
            cwd=cwd,
            attributes=self.contract.required_attributes[option],
        )
        state["selected_distribution"] = selected
        state["configured_family"] = list(self.contract.family)
        active = state.get("active_distributions", [])
        if active != [selected]:
            raise ProviderVerificationError(
                provider=self.name,
                route=self.route.name,
                option=option,
                reason=(
                    f"expected exactly {selected!r}, but active family members are "
                    f"{active!r}"
                ),
                state=state,
            )
        if state.get("import_error"):
            raise ProviderVerificationError(
                provider=self.name,
                route=self.route.name,
                option=option,
                reason=f"shared module {self.contract.module!r} is unimportable",
                state=state,
                repairable=True,
            )
        attributes = state.get("attributes", {})
        missing = sorted(
            attribute
            for attribute in self.contract.required_attributes[option]
            if not isinstance(attributes, dict) or not attributes.get(attribute)
        )
        state["missing_attributes"] = missing
        if missing:
            raise ProviderVerificationError(
                provider=self.name,
                route=self.route.name,
                option=option,
                reason=(
                    f"shared module {self.contract.module!r} lacks required attributes: "
                    + ", ".join(missing)
                ),
                state=state,
                repairable=True,
            )
        return VerificationResult(self.name, self.route.name, option, state)

    def protected_packages(self) -> tuple[str, ...]:
        if self._selected_package is not None:
            return (self._selected_package,)
        return self.contract.family

    def option_details(self, option: str) -> dict[str, Any]:
        return {
            "selected_distribution": self.contract.selections[option],
            "family": list(self.contract.family),
            "module": self.contract.module,
            "required_attributes": list(self.contract.required_attributes[option]),
        }
