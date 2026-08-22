from __future__ import annotations

from pathlib import Path
import shutil

import pytest


@pytest.fixture
def uv_path() -> Path:
    value = shutil.which("uv")
    if value is None:
        pytest.skip("uv is unavailable")
    return Path(value).resolve()
