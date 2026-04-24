#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
python -m pytest
pyinstaller packaging/Contour.spec
tar -C dist -czf dist/contour-linux.tar.gz Contour
