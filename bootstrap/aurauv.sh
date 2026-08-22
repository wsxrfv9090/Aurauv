#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'MSG'
[aurauv] ERROR: uv was not found on PATH.
Aurauv stopped before changing pyproject.toml, uv.lock, .venv, Python, or project state.
MSG
  exit 1
fi

PYTHON=""
CANDIDATES=()
if [[ -n "${AURAUV_BOOTSTRAP_PYTHON:-}" ]]; then CANDIDATES+=("$AURAUV_BOOTSTRAP_PYTHON"); fi
CANDIDATES+=(python3 python)
for candidate in "${CANDIDATES[@]}"; do
  if command -v "$candidate" >/dev/null 2>&1 \
    && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
    PYTHON="$(command -v "$candidate")"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo "[aurauv] ERROR: Aurauv requires an existing Python >= 3.11 to start." >&2
  exit 1
fi
exec "$PYTHON" "$SCRIPT_DIR/aurauv.py" "$@"
