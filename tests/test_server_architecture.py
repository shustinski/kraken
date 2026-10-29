"""Kraken Server is a composition root around an HTTP interface.

Domain and application rules stay in kraken_manager. The server package may
compose them, but its HTTP routes are not a single procedure split into
numbered sections, and those routes do not talk to the database driver.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "src" / "kraken_server"
ROUTES = SERVER / "interface" / "http" / "routes"

ROUTE_FORBIDDEN = {
    "sqlalchemy",
    "psycopg",
    "alembic",
    "kraken_server.persistent_services",
    "kraken_server.persistent_services_parts",
}


def imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


class ServerArchitectureTests(unittest.TestCase):
    def test_http_routes_are_resource_modules(self) -> None:
        numbered = [path for path in SERVER.rglob("*.py") if path.stem.startswith("section_")]
        self.assertEqual([], [str(path.relative_to(ROOT)) for path in numbered])
        self.assertTrue(ROUTES.is_dir())
        modules = [path.stem for path in ROUTES.glob("*.py") if path.stem != "__init__"]
        for name in (
            "session",
            "auth",
            "projects",
            "layers",
            "representations",
            "artifacts",
            "reviews",
            "jobs",
            "agent",
            "realtime",
        ):
            self.assertIn(name, modules)
        root_route_files = list(SERVER.glob("*_routes.py"))
        self.assertEqual([], [path.name for path in root_route_files])

    def test_route_modules_do_not_import_the_database(self) -> None:
        violations: list[str] = []
        for path in sorted(ROUTES.rglob("*.py")):
            for imported in imports_in(path):
                if any(imported == prefix or imported.startswith(prefix + ".") for prefix in ROUTE_FORBIDDEN):
                    violations.append(f"{path.relative_to(ROOT)} imports {imported}")
        self.assertEqual([], violations)

    def test_application_factory_only_composes_routes(self) -> None:
        factory = (SERVER / "interface" / "http" / "factory.py").read_text(encoding="utf-8")
        tree = ast.parse(factory)
        decorators = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list
        ]
        self.assertEqual([], [node.name for node in decorators])
