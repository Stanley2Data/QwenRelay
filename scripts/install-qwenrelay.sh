#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# New venv only. No model download, model start, GPU probe or host service change.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3.12}
VENV=${QWENRELAY_VENV:-$ROOT/.venv}
BUILD=${QWENRELAY_BUILD:-$ROOT/.build}
if [ -e "$VENV" ]; then
  echo "Choose a new QWENRELAY_VENV; refusing to modify an existing environment." >&2
  exit 1
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
"$PYTHON" "$ROOT/scripts/qwenrelay.py" check-pack
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install -r "$ROOT/requirements-qwenrelay.txt"
mkdir -p -- "$BUILD"
SOURCE=${VLLM_SOURCE:-$BUILD/vllm-pristine}
if [ -z "${VLLM_SOURCE:-}" ]; then
  if [ -e "$SOURCE" ]; then echo "Set VLLM_SOURCE to a pinned clean checkout, or choose a new QWENRELAY_BUILD." >&2; exit 1; fi
  git init -q "$SOURCE"
  git -C "$SOURCE" fetch --depth=1 https://github.com/vllm-project/vllm.git db9527a46873454610df6dbedf79a36d6bf1a7f6
  git -C "$SOURCE" -c advice.detachedHead=false checkout --detach FETCH_HEAD
fi
"$VENV/bin/python" "$ROOT/scripts/qwenrelay.py" rebuild --vllm-source "$SOURCE" --target "$BUILD/final-F"
"$VENV/bin/python" "$ROOT/scripts/qwenrelay.py" install-source --source "$BUILD/final-F"
"$VENV/bin/python" "$ROOT/scripts/qwenrelay.py" check-runtime
echo "Installed. Fetch or validate the locked model assets before starting. No model was started."
