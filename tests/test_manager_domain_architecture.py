from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class CleanArchitectureTests(unittest.TestCase):
    def test_domain_has_no_inward_dependency_on_frameworks_or_outer_layers(self) -> None:
        forbidden = (
            "PyQt",
            "sqlalchemy",
            "fastapi",
            "pydantic",
            "httpx",
            "pathlib",
            "kraken_manager.application",
            "kraken_manager.infrastructure",
            "kraken_manager.presentation",
            "kraken_server",
            "kraken_agent",
        )
        violations: list[str] = []
        for path in (ROOT / "src" / "kraken_manager" / "domain").glob("*.py"):
            for module in imported_modules(path):
                if module.startswith(forbidden):
                    violations.append(f"{path.name}: {module}")
        self.assertEqual(violations, [])

    def test_application_depends_only_on_domain_and_stdlib(self) -> None:
        forbidden = (
            "PyQt",
            "sqlalchemy",
            "fastapi",
            "pydantic",
            "httpx",
            "pathlib",
            "kraken_manager.infrastructure",
            "kraken_manager.presentation",
            "kraken_server",
            "kraken_agent",
        )
        violations: list[str] = []
        for path in (ROOT / "src" / "kraken_manager" / "application").glob("*.py"):
            for module in imported_modules(path):
                if module.startswith(forbidden):
                    violations.append(f"{path.name}: {module}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
