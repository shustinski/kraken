#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
python -m pytest
pyinstaller packaging/CSliser.spec
tar -C dist -czf dist/csliser-linux.tar.gz CSliser
