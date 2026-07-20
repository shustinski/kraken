from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "kraken_manager"


def imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


class CleanArchitectureTests(unittest.TestCase):
    DOMAIN_FORBIDDEN = {
        "PyQt6",
        "fastapi",
        "pydantic",
        "sqlalchemy",
        "alembic",
        "psycopg",
        "httpx",
        "requests",
        "pathlib",
        "kraken_manager.application",
        "kraken_manager.infrastructure",
        "kraken_manager.presentation",
        "kraken_core.plugins",
        "kraken_server",
        "kraken_agent",
        "kraken_hub",
    }
    APPLICATION_FORBIDDEN = {
        "PyQt6",
        "fastapi",
        "pydantic",
        "sqlalchemy",
        "alembic",
        "psycopg",
        "httpx",
        "requests",
        "pathlib",
        "kraken_manager.infrastructure",
        "kraken_manager.presentation",
        "kraken_core.plugins",
        "kraken_server",
        "kraken_agent",
        "kraken_hub",
    }

    def assert_clean(self, package: str, forbidden: set[str]) -> None:
        violations: list[str] = []
        for path in sorted((SOURCE / package).rglob("*.py")):
            for imported in imports_in(path):
                if any(imported == prefix or imported.startswith(prefix + ".") for prefix in forbidden):
                    violations.append(f"{path.relative_to(ROOT)} imports {imported}")
        self.assertEqual([], violations, "\n".join(violations))

    def test_domain_has_no_outward_dependencies(self) -> None:
        self.assert_clean("domain", self.DOMAIN_FORBIDDEN)

    def test_application_depends_only_inward_and_on_stdlib(self) -> None:
        self.assert_clean("application", self.APPLICATION_FORBIDDEN)

    def test_presentation_does_not_depend_on_infrastructure(self) -> None:
        violations: list[str] = []
        for path in sorted((SOURCE / "presentation").rglob("*.py")):
            for imported in imports_in(path):
                if imported == "kraken_manager.infrastructure" or imported.startswith("kraken_manager.infrastructure."):
                    violations.append(f"{path.relative_to(ROOT)} imports {imported}")
        self.assertEqual([], violations, "\n".join(violations))


if __name__ == "__main__":
    unittest.main()

