"""Route selection, persistence reuse, fallback authorization, and provider preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .errors import AurauvError, ProviderPreflightError
from .models import (
    AuraOptions,
    AurauvConfig,
    EnvironmentIdentity,
    RouteSelection,
)
from .providers.base import Provider
from .runtime import confirm
from .utils import option_values


def explicit_route_from_uv_extras(
    config: AurauvConfig, uv_args: Sequence[str]
) -> dict[str, str]:
    supplied = set(option_values(uv_args, "--extra"))
    result: dict[str, str] = {}
    for route in config.routes.values():
        candidates = [
            option.name
            for option in route.options.values()
            if option.extras and set(option.extras).issubset(supplied)
        ]
        if len(candidates) > 1:
            raise AurauvError(
                f"uv arguments select multiple options for route {route.name!r}: "
                f"{', '.join(candidates)}. Use one option only."
            )
        if candidates:
            result[route.name] = candidates[0]
    return result


def _preflight_route(
    route_name: str,
    option: str,
    providers: dict[str, Provider],
    config: AurauvConfig,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    route = config.routes[route_name]
    for provider_name in route.providers:
        details[provider_name] = providers[provider_name].preflight(option)
    return details


def select_routes(
    config: AurauvConfig,
    providers: dict[str, Provider],
    aura: AuraOptions,
    uv_args: Sequence[str],
    persisted: dict[str, str],
    environment_python: Path,
) -> dict[str, RouteSelection]:
    uv_overrides = explicit_route_from_uv_extras(config, uv_args)
    selections: dict[str, RouteSelection] = {}

    for route_name, route in config.routes.items():
        requested: str
        selected: str
        origin: str
        details: dict[str, Any] = {}
        if route_name in aura.route_overrides:
            requested = aura.route_overrides[route_name]
            selected = requested
            origin = "--aura-route"
        elif route_name in uv_overrides:
            requested = uv_overrides[route_name]
            selected = requested
            origin = "uv --extra"
        elif route_name in persisted:
            requested = persisted[route_name]
            selected = requested
            origin = "persisted environment state"
        else:
            inferred = None
            if environment_python.is_file() and route.detector is not None:
                inferred = providers[route.detector].infer_installed(environment_python)
            if inferred is not None and inferred.option in route.options:
                requested = inferred.option
                selected = inferred.option
                origin = "existing installed provider"
                details = inferred.details
            elif route.default != "auto":
                requested = route.default
                selected = route.default
                origin = "route default"
            else:
                if route.detector is None:
                    raise AurauvError(f"Route {route_name!r} has no automatic detector.")
                detected = providers[route.detector].detect()
                if detected is None:
                    raise AurauvError(
                        f"Provider {route.detector!r} could not detect route {route_name!r}."
                    )
                requested = detected.option
                selected = detected.option
                origin = f"automatic detection by {route.detector}"
                details = detected.details

        if selected not in route.options:
            raise AurauvError(
                f"Unknown option {selected!r} for route {route_name!r}; "
                f"available: {', '.join(route.options)}."
            )
        fallback_from: str | None = None
        fallback_reason: str | None = None
        try:
            preflight = _preflight_route(route_name, selected, providers, config)
        except ProviderPreflightError as failure:
            explicit_fallback = aura.fallback_overrides.get(route_name)
            fallback = explicit_fallback or route.fallbacks.get(selected)
            if fallback is None:
                raise
            if fallback not in route.options:
                raise AurauvError(
                    f"Fallback {route_name}={fallback} is not a configured route option."
                )
            approved = explicit_fallback is not None or aura.assume_yes
            if not approved:
                approved = confirm(
                    f"{failure}. Fall back to {route_name}={fallback}?", aura
                )
            if not approved:
                raise AurauvError(
                    f"{failure}. Fallback to {route_name}={fallback} was not authorized."
                ) from failure
            fallback_from = selected
            fallback_reason = str(failure)
            selected = fallback
            preflight = _preflight_route(route_name, selected, providers, config)
            origin = f"authorized fallback from {fallback_from}"

        selections[route_name] = RouteSelection(
            route=route_name,
            requested=requested,
            selected=selected,
            origin=origin,
            fallback_from=fallback_from,
            fallback_reason=fallback_reason,
            details={**details, "preflight": preflight},
        )
    return selections


def selected_extras(
    config: AurauvConfig, selections: dict[str, RouteSelection]
) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for route_name in config.routes:
        option = config.routes[route_name].options[selections[route_name].selected]
        for extra in option.extras:
            if extra not in seen:
                result.append(extra)
                seen.add(extra)
    return tuple(result)
