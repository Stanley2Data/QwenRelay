#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# New venv only. No model download, model start, GPU probe or host service change.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3.12}
VENV=${QWENRELAY_VENV:-$ROOT/.venv}
BUILD=${QWENRELAY_BUILD:-$ROOT/.build}
VENV=$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$VENV")
BUILD=$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$BUILD")
SOURCE=${VLLM_SOURCE:-$BUILD/vllm-pristine}
if [ -e "$VENV" ] || [ -L "$VENV" ]; then
  echo "Choose a new QWENRELAY_VENV; refusing to modify an existing environment." >&2
  exit 1
fi
if [ -e "$BUILD/final-F" ] || [ -L "$BUILD/final-F" ]; then
  echo "Choose a new QWENRELAY_BUILD; final-F already exists." >&2
  exit 1
fi
if [ -z "${VLLM_SOURCE:-}" ] && { [ -e "$SOURCE" ] || [ -L "$SOURCE" ]; }; then
  echo "Set VLLM_SOURCE to a pinned clean checkout, or choose a new QWENRELAY_BUILD." >&2
  exit 1
fi
if [ -n "${VLLM_SOURCE:-}" ] && [ ! -d "$SOURCE/vllm" ]; then
  echo "VLLM_SOURCE must contain a pristine vllm source directory." >&2
  exit 1
fi
"$PYTHON" -c 'import platform,sys
if sys.platform != "linux" or platform.machine() != "x86_64":
    sys.exit("QwenRelay installation requires native Linux x86_64")
if sys.version_info[:2] != (3,12):
    sys.exit("Python 3.12 required")'
for tool in git patch cc c++; do
  command -v "$tool" >/dev/null || { echo "Required build tool missing: $tool" >&2; exit 1; }
done
case "$(patch --version)" in
  *"GNU patch"*) ;;
  *) echo "GNU patch is required." >&2; exit 1 ;;
esac
"$PYTHON" "$ROOT/scripts/qwenrelay.py" check-pack
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install -r "$ROOT/requirements-qwenrelay.txt"
mkdir -p -- "$BUILD"
if [ -z "${VLLM_SOURCE:-}" ]; then
  git init -q "$SOURCE"
  git -C "$SOURCE" fetch --depth=1 https://github.com/vllm-project/vllm.git db9527a46873454610df6dbedf79a36d6bf1a7f6
  git -C "$SOURCE" -c advice.detachedHead=false checkout --detach FETCH_HEAD
fi
"$VENV/bin/python" "$ROOT/scripts/qwenrelay.py" rebuild --vllm-source "$SOURCE" --target "$BUILD/final-F"
"$VENV/bin/python" "$ROOT/scripts/qwenrelay.py" install-source --source "$BUILD/final-F"
"$VENV/bin/python" "$ROOT/scripts/qwenrelay.py" check-runtime
echo "Installed. Fetch or validate the locked model assets before starting. No model was started."
