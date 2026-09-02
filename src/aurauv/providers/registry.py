"""Built-in provider registry."""

from __future__ import annotations

from ..errors import AurauvError
from ..models import AurauvConfig, ProviderSpec
from ..process import CommandRunner
from .base import Provider
from .exclusive_distribution import ExclusiveDistributionProvider
from .pytorch import PytorchProvider
from .pytorch_companion import PytorchCompanionProvider


_PROVIDER_TYPES = {
    "exclusive-distribution": ExclusiveDistributionProvider,
    "pytorch": PytorchProvider,
    "pytorch-companion": PytorchCompanionProvider,
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
