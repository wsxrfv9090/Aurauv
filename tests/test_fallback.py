from __future__ import annotations

from pathlib import Path

import pytest

from aurauv.config import load_config
from aurauv.engine import AurauvEngine
from aurauv.errors import AurauvError, ProviderPreflightError, ProviderVerificationError
from aurauv.invocation import parse_invocation
from aurauv.models import AuraOptions, DetectionResult, RouteSelection
from aurauv.routes import select_routes


class FailingCudaProvider:
    def detect(self) -> DetectionResult:
        return DetectionResult("cuda", "test detector")

    def infer_installed(self, python: Path):
        return None

    def preflight(self, option: str):
        if option == "cuda":
            raise ProviderPreflightError(
                "fake", "accelerator", "cuda", "no usable NVIDIA runtime"
            )
        return {"runtime": option}


class RuntimeProvider:
    def preflight(self, option: str):
        return {"runtime": option}


def _config(tmp_path: Path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '''[project]
name = "fallback-demo"
version = "0.1.0"
requires-python = ">=3.11"

[project.optional-dependencies]
cpu = []
cuda = []

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cuda" }]]

[tool.aurauv]
schema-version = 1
minimum-uv = "0.10.0"

[tool.aurauv.routes.accelerator]
default = "auto"
detector = "fake"
providers = ["fake"]
fallbacks = { cuda = "cpu" }

[tool.aurauv.routes.accelerator.options.cpu]
extras = ["cpu"]

[tool.aurauv.routes.accelerator.options.cuda]
extras = ["cuda"]

[tool.aurauv.providers.fake]
type = "future-test-provider"
project = "."
route = "accelerator"
''',
        encoding="utf-8",
    )
    return load_config(root)


def _aura(*, explicit_fallback: bool) -> AuraOptions:
    return AuraOptions(
        route_overrides={},
        fallback_overrides={"accelerator": "cpu"} if explicit_fallback else {},
        assume_yes=False,
        no_input=True,
        install_python=None,
        update_uv=None,
        refresh=False,
        target="project",
        json_output=False,
        no_route=False,
    )


def test_preflight_fallback_requires_authorization(tmp_path: Path) -> None:
    config = _config(tmp_path)
    provider = FailingCudaProvider()
    with pytest.raises(AurauvError, match="not authorized"):
        select_routes(
            config,
            {"fake": provider},
            _aura(explicit_fallback=False),
            (),
            {},
            tmp_path / "missing-python",
        )

    selected = select_routes(
        config,
        {"fake": provider},
        _aura(explicit_fallback=True),
        (),
        {},
        tmp_path / "missing-python",
    )
    assert selected["accelerator"].selected == "cpu"
    assert selected["accelerator"].fallback_from == "cuda"


def test_runtime_fallback_resyncs_only_when_explicitly_authorized(
    tmp_path: Path, uv_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    monkeypatch.setenv("AURAUV_UV", str(uv_path))
    parsed = parse_invocation(
        ["--aura-fallback", "accelerator=cpu", "sync"]
    )
    engine = AurauvEngine(parsed, cwd=config.owner_root)
    engine.config = config
    engine.providers = {"fake": RuntimeProvider()}  # type: ignore[assignment]
    selection = RouteSelection(
        route="accelerator",
        requested="cuda",
        selected="cuda",
        origin="test",
    )
    calls = {"verify": 0, "sync": 0}

    def verify(selections, *, allow_repair):
        calls["verify"] += 1
        if calls["verify"] == 1:
            raise ProviderVerificationError(
                "fake",
                "accelerator",
                "cuda",
                "runtime unavailable",
                {},
                fallback_safe=True,
            )
        assert selections["accelerator"].selected == "cpu"
        return {"fake": {"runtime": "cpu"}}

    monkeypatch.setattr(engine, "_verify_providers", verify)
    monkeypatch.setattr(engine, "_fresh_sync_args", lambda selections: ["sync"])
    monkeypatch.setattr(
        engine,
        "_run_uv",
        lambda *args, **kwargs: calls.__setitem__("sync", calls["sync"] + 1),
    )

    updated, results = engine._verify_with_runtime_fallback(
        {"accelerator": selection}, allow_repair=False, allow_resync=True
    )
    assert updated["accelerator"].selected == "cpu"
    assert updated["accelerator"].fallback_from == "cuda"
    assert results["fake"]["runtime"] == "cpu"
    assert calls == {"verify": 2, "sync": 1}
