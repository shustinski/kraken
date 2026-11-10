"""List every test file carrying a module-level ``kraken_defect`` constant.

Used by the ``regression`` tier of ``scripts/quality-gates.ps1``: the gate
must run exactly the defect-regression files, and the list must stay in sync
with the source tree (a new ``kraken_defect = "D<n>"`` file is picked up
automatically). The scan reuses the AST helper from
:mod:`tools.quality.test_nodes` — no new dependencies, no pytest spawn.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.quality.test_nodes import module_constant_values  # noqa: E402


def _markers(path: Path) -> set[str]:
    """kraken_defect values of one file; empty when the file cannot be parsed.

    The tree mixes in vendor source files (some with a UTF-8 BOM, which
    ``ast.parse`` rejects); such files are not test files and cannot carry a
    marker, so a parse failure means "no marker here", not an error.
    """
    try:
        return module_constant_values(path, "kraken_defect")
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return set()


def defect_files(repo_root: Path) -> list[str]:
    """Repository-relative POSIX paths of test files with a kraken_defect marker."""
    files: set[Path] = set()
    for directory in ("tests", "plugins"):
        base = Path(repo_root) / directory
        if base.is_dir():
            files.update(base.rglob("*.py"))
    found = sorted(
        str(path.relative_to(repo_root)).replace("\\", "/")
        for path in files
        if "__pycache__" not in path.parts and _markers(path)
    )
    return found


def main() -> int:
    for line in defect_files(REPO_ROOT):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
