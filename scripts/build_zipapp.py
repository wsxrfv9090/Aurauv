#!/usr/bin/env python3
"""Build Aurauv's deterministic, standard-library-only zipapp."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.vendor import build_pyz  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "bootstrap" / "aurauv.pyz",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    build_pyz(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
