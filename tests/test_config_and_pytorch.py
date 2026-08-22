from __future__ import annotations

from pathlib import Path

import pytest

from aurauv.config import load_config
from aurauv.process import CommandRunner
from aurauv.providers.registry import build_provider


PYTORCH_PROJECT = Path(__file__).resolve().parent / "fixtures" / "pytorch_project.toml"


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "sample-ml"
    root.mkdir()
    original = PYTORCH_PROJECT.read_text(encoding="utf-8")
    original += r'''

[tool.aurauv]
schema-version = 1
project = "Sample ML"
minimum-uv = "0.10.0"
python-request = "3.13"

[tool.aurauv.routes.accelerator]
default = "auto"
detector = "pytorch"
providers = ["pytorch"]
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
'''
    (root / "pyproject.toml").write_text(original, encoding="utf-8")
    return root


def test_pytorch_contract_has_one_cuda_source(tmp_path: Path) -> None:
    root = _project(tmp_path)
    config = load_config(root)
    provider = build_provider(config.providers["pytorch"], config=config, runner=CommandRunner())
    details = provider.validate_contract()
    assert details["cuda_backend"] == "cu132"
    assert details["expected_cuda"] == "13.2"
    source = Path(provider.__class__.__module__.replace(".", "/"))
    code = (Path(__file__).resolve().parents[1] / "src/aurauv/providers/pytorch.py").read_text()
    assert "cu132" not in code


def test_explicit_cpu_preflight_is_valid_on_any_machine(tmp_path: Path) -> None:
    root = _project(tmp_path)
    config = load_config(root)
    provider = build_provider(config.providers["pytorch"], config=config, runner=CommandRunner())
    provider.validate_contract()
    assert provider.preflight("cpu")["runtime"] == "cpu"
