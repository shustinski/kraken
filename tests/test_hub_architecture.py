"""Hub layers point inward: domain, application, then adapters.

The package root is only the process entry. Qt lives in presentation, OS and
HTTP adapters live in infrastructure, and the local project service stays in
the composition root. Inner layers do not import the desktop shell.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT / "src" / "kraken_hub"


def imports_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
    return result


def violations(package: Path, forbidden: set[str]) -> list[str]:
    found: list[str] = []
    for path in sorted(package.rglob("*.py")):
        for imported in imports_in(path):
            if any(imported == prefix or imported.startswith(prefix + ".") for prefix in forbidden):
                found.append(f"{path.relative_to(ROOT)} imports {imported}")
    return found


class HubArchitectureTests(unittest.TestCase):
    def test_package_root_is_only_the_process_entry(self) -> None:
        root_modules = sorted(path.name for path in HUB.glob("*.py"))
        self.assertEqual(["__init__.py", "__main__.py"], root_modules)
        for layer in ("domain", "application", "infrastructure", "presentation", "composition"):
            self.assertTrue((HUB / layer / "__init__.py").is_file(), layer)

    def test_application_depends_only_inward(self) -> None:
        application = HUB / "application"
        self.assertTrue(application.is_dir(), "kraken_hub.application is missing")
        forbidden = {
            "PyQt6",
            "ctypes",
            "keyring",
            "urllib",
            "kraken_hub.infrastructure",
            "kraken_hub.presentation",
            "kraken_hub.composition",
        }
        self.assertEqual([], violations(application, forbidden), "\n".join(violations(application, forbidden)))

    def test_infrastructure_does_not_import_the_desktop_shell(self) -> None:
        forbidden = {
            "PyQt6",
            "kraken_hub.presentation",
            "kraken_hub.composition",
            "kraken_hub.presentation.qt.manager_app",
            "kraken_hub.composition.catalog",
        }
        found = violations(HUB / "infrastructure", forbidden)
        self.assertEqual([], found, "\n".join(found))

    def test_domain_has_no_outward_dependencies(self) -> None:
        domain = HUB / "domain"
        self.assertTrue(domain.is_dir(), "kraken_hub.domain is missing")
        forbidden = {
            "PyQt6",
            "ctypes",
            "keyring",
            "urllib",
            "pathlib",
            "kraken_hub.application",
            "kraken_hub.infrastructure",
            "kraken_hub.presentation",
            "kraken_hub.composition",
            "kraken_hub.presentation.qt.manager_app",
            "kraken_hub.infrastructure.remote.client",
            "kraken_hub.infrastructure.credentials.windows",
            "kraken_hub.composition.catalog",
        }
        self.assertEqual([], violations(domain, forbidden), "\n".join(violations(domain, forbidden)))

    def test_remote_client_lives_in_infrastructure_without_ui(self) -> None:
        client = HUB / "infrastructure" / "remote" / "client.py"
        parts = HUB / "infrastructure" / "remote" / "parts"
        self.assertTrue(client.is_file(), "remote client is still outside infrastructure")
        self.assertTrue(parts.is_dir(), "remote client parts are still outside infrastructure")
        self.assertFalse((HUB / "remote_client.py").exists(), "remote client facade still sits on the package root")
        self.assertFalse((HUB / "remote_client_parts").exists(), "remote client parts still sit on the package root")
        forbidden = {"PyQt6", "kraken_hub.presentation", "kraken_hub.presentation.qt.manager_app", "kraken_hub.composition.catalog"}
        self.assertEqual(
            [],
            violations(HUB / "infrastructure" / "remote", forbidden),
            "\n".join(violations(HUB / "infrastructure" / "remote", forbidden)),
        )

    def test_local_service_parts_live_under_the_composition_package(self) -> None:
        parts = HUB / "composition" / "parts"
        self.assertTrue((HUB / "composition").is_dir() and (HUB / "composition" / "__init__.py").is_file())
        self.assertTrue(parts.is_dir(), "local service parts are still beside the composition module")
        self.assertFalse((HUB / "composition_parts").exists())
        self.assertFalse((HUB / "composition.py").exists())


if __name__ == "__main__":
    unittest.main()
