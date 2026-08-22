from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "templates" / "deployment" / "setup.py"


def _load_setup():
    spec = importlib.util.spec_from_file_location("aurauv_template_setup", SETUP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_project_launcher_default_and_device_aliases() -> None:
    setup = _load_setup()

    assert setup._normalize_args([]) == ["sync"]
    assert setup._normalize_args(["cpu"]) == ["--aura-device", "cpu", "sync"]
    assert setup._normalize_args(["cuda", "--fallback", "cpu"]) == [
        "--aura-device",
        "cuda",
        "--aura-fallback",
        "cpu",
        "sync",
    ]
    assert setup._normalize_args(["cuda", "run", "pytest"]) == [
        "--aura-device",
        "cuda",
        "run",
        "pytest",
    ]
    assert setup._normalize_args(["cpu", "--offline", "--dry-run"]) == [
        "--aura-device",
        "cpu",
        "sync",
        "--offline",
        "--dry-run",
    ]


def test_project_launcher_preserves_normal_uv_invocations() -> None:
    setup = _load_setup()

    raw = ["--aura-device", "cuda", "sync", "--offline"]
    assert setup._normalize_args(raw) == raw
