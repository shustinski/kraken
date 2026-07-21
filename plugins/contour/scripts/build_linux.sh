#!/usr/bin/env bash
set -euo pipefail
PLUGIN_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "${PLUGIN_ROOT}/../.." && pwd)"
cd "${PLUGIN_ROOT}"

PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python"
fi

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
"${PYTHON}" -m pytest
"${PYTHON}" -m PyInstaller packaging/Contour.spec
tar -C dist -czf dist/contour-linux.tar.gz Contour
