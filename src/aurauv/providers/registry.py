"""Built-in provider registry."""

from __future__ import annotations

from ..errors import AurauvError
from ..models import AurauvConfig, ProviderSpec
from ..process import CommandRunner
from .base import Provider
from .pytorch import PytorchProvider


_PROVIDER_TYPES = {
    "pytorch": PytorchProvider,
}


def build_provider(
    spec: ProviderSpec, *, config: AurauvConfig, runner: CommandRunner
) -> Provider:
    provider_type = _PROVIDER_TYPES.get(spec.kind)
    if provider_type is None:
        raise AurauvError(
            f"Unknown provider type {spec.kind!r} for {spec.name!r}. "
            f"Built-in types: {', '.join(sorted(_PROVIDER_TYPES))}."
        )
    return provider_type(spec, config=config, runner=runner)
