#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${QWENRELAY_PYTHON:-$ROOT/.venv/bin/python}
exec "$PYTHON" "$ROOT/scripts/qwenrelay.py" serve "$@"
