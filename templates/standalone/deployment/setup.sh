#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=""
CANDIDATES=()
if [[ -n "${AURAUV_BOOTSTRAP_PYTHON:-}" ]]; then
  CANDIDATES+=("$AURAUV_BOOTSTRAP_PYTHON")
fi
CANDIDATES+=(python3 python)

for candidate in "${CANDIDATES[@]}"; do
  if command -v "$candidate" >/dev/null 2>&1 && \
     "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' >/dev/null 2>&1; then
    PYTHON="$(command -v "$candidate")"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  cat >&2 <<'MSG'
[aurauv-launcher] Python >= 3.11 was not found on PATH.
Aurauv itself needs Python 3.11+ to start; the project's newer Python may then be installed by uv with authorization.
Set AURAUV_BOOTSTRAP_PYTHON=/path/to/python when needed.
MSG
  exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/setup.py" "$@"
