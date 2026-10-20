#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VENV=${QWENRELAY_VENV:-$ROOT/.venv}
PYTHON=${QWENRELAY_PYTHON:-$VENV/bin/python}
if ! command -v "$PYTHON" >/dev/null; then
  echo "Python not found: $PYTHON. Run scripts/install-qwenrelay.sh or export QWENRELAY_VENV / QWENRELAY_PYTHON." >&2
  exit 1
fi
exec "$PYTHON" "$ROOT/scripts/qwenrelay.py" serve "$@"
