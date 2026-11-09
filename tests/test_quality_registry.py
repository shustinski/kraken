"""Enforcement tests for the machine-readable quality registries.

The registries under ``docs/quality/registry/`` are the enforcement backbone
of the defect-to-regression policy. These tests make the build fail when:

* a registry file is missing, not valid JSON, or schema-inconsistent;
* a RESOLVED defect has no live regression test (node ids must resolve to
  actually-collected tests — resolved here by ``ast`` scanning so the check
  runs fast and offline, without spawning pytest);
* a ``kraken_defect = "D<n>"`` module marker in ``tests/`` references a
  defect that is not registered;
* a mandatory action is untracked (NOT_TESTED without ``exception_reason``)
  or lists test node ids that do not resolve;
* the generated sections of the markdown quality documents drift from the
  registries (enforced by running the generator in ``--check`` mode; a tampered
  section on a temp copy must be rejected).
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from tools.quality.registry_schema import validate_registry

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = REPO_ROOT / "docs" / "quality" / "registry"
DEFECTS_JSON = REGISTRY_DIR / "defects.json"
ACTIONS_JSON = REGISTRY_DIR / "actions.json"
TESTS_DIR = REPO_ROOT / "tests"
GENERATOR = REPO_ROOT / "tools" / "quality" / "generate_quality_docs.py"
QUALITY_DOCS = ("defects.md", "functional-inventory.md", "acceptance-matrix.md")

_AST_CACHE: dict[Path, ast.AST] = {}


def _parse(path: Path) -> ast.AST:
    if path not in _AST_CACHE:
        _AST_CACHE[path] = ast.parse(path.read_text(encoding="utf-8"))
    return _AST_CACHE[path]


def _is_test_class(node: ast.ClassDef) -> bool:
    """A class pytest collects: Test* with test_ methods or a TestCase subclass."""
    for base in node.bases:
        name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
        if name and "TestCase" in name:
            return True
    if node.name.startswith("Test"):
        return any(
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name.startswith("test_")
            for item in node.body
        )
    return False


def _file_has_tests(path: Path, tree: ast.AST) -> bool:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            return True
        if isinstance(node, ast.ClassDef) and _is_test_class(node):
            return True
    return False


def _module_level_imports(tree: ast.AST) -> set[str]:
    """Names bound at module level by imports (facade re-exports)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _resolve_node_id(node_id: str) -> list[str]:
    """Check one test node id against the real source tree.

    Accepted forms: ``path/to/test.py`` (file level: must exist and contain at
    least one test), ``path/to/test.py::test_name`` (module function, or a
    facade re-export via import), ``path/to/test.py::TestClass`` (collectable
    class with at least one test method), ``path/to/test.py::TestClass::test_name``.
    Returns a list of problems; empty means the node id resolves.
    """
    parts = node_id.split("::")
    path = REPO_ROOT / parts[0]
    if not path.is_file():
        return [f"{node_id}: file not found: {parts[0]}"]
    tree = _parse(path)
    if len(parts) == 1:
        if _file_has_tests(path, tree):
            return []
        return [f"{node_id}: no tests found in file"]
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    imports = _module_level_imports(tree)
    if len(parts) == 2:
        name = parts[1]
        if name in functions:
            return []
        if name in imports and name.startswith("test_"):
            return []  # facade re-export: pytest collects the imported name here
        if name in classes and _is_test_class(classes[name]):
            return []
        return [f"{node_id}: {name} not found (or has no tests) in {parts[0]}"]
    if len(parts) == 3:
        class_name, test_name = parts[1], parts[2]
        if class_name in classes:
            methods = {
                item.name
                for item in classes[class_name].body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if test_name in methods:
                return []
        return [f"{node_id}: {class_name}::{test_name} not found in {parts[0]}"]
    return [f"{node_id}: unsupported node id format"]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_defects_json_valid():
    problems = validate_registry(DEFECTS_JSON)
    assert not problems, "defects.json schema problems:\n" + "\n".join(problems)


def test_actions_json_valid():
    problems = validate_registry(ACTIONS_JSON)
    assert not problems, "actions.json schema problems:\n" + "\n".join(problems)


def test_resolved_defects_have_live_regression_tests():
    defects = _load(DEFECTS_JSON)["defects"]
    problems = []
    for defect in defects:
        if defect.get("status") != "RESOLVED":
            continue
        if not defect.get("regression_tests"):
            problems.append(f"{defect['id']}: RESOLVED defect has no regression_tests")
            continue
        for node_id in defect["regression_tests"]:
            problems.extend(f"{defect['id']}: {problem}" for problem in _resolve_node_id(node_id))
    assert not problems, "unresolved regression tests:\n" + "\n".join(problems)


def test_defect_markers_are_registered():
    registry_ids = {defect["id"] for defect in _load(DEFECTS_JSON)["defects"]}
    problems = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        tree = _parse(path)
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "kraken_defect" for target in node.targets):
                continue
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                problems.append(f"{path}: kraken_defect marker must be a string literal")
                continue
            defect_id = node.value.value
            if not re.fullmatch(r"D\d+", defect_id):
                problems.append(f"{path}: marker {defect_id!r} is not a D<n> defect id")
            elif defect_id not in registry_ids:
                problems.append(f"{path}: marker {defect_id} is not a registered defect")
    assert not problems, "unregistered defect markers:\n" + "\n".join(problems)


def test_mandatory_actions_tracked():
    actions = _load(ACTIONS_JSON)["actions"]
    problems = []
    for action in actions:
        if not action.get("mandatory"):
            continue
        if action.get("status") == "NOT_TESTED" and not action.get("exception_reason"):
            problems.append(f"{action['id']}: mandatory NOT_TESTED action without exception_reason")
        for node_id in action.get("tests", []):
            problems.extend(f"{action['id']}: {problem}" for problem in _resolve_node_id(node_id))
    assert not problems, "untracked mandatory actions:\n" + "\n".join(problems)


def _repo_python() -> Path:
    """The repo's own interpreter (the venv this repo runs with), falling back to the current one."""
    venv = REPO_ROOT / ".venv"
    for candidate in (venv / "Scripts" / "python.exe", venv / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def _run_generator(*args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [str(_repo_python()), str(GENERATOR), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=REPO_ROOT,
        timeout=60,
    )


def test_markdown_in_sync_with_json():
    result = _run_generator("--check")
    assert result.returncode == 0, "generator --check reported drift:\n" + result.stdout + result.stderr


def test_tampered_generated_section_fails_check(tmp_path: Path):
    """Tampering one rendered table on a temp copy of the docs must make --check exit 1."""
    quality_dir = tmp_path / "docs" / "quality"
    registry_dir = quality_dir / "registry"
    registry_dir.mkdir(parents=True)
    for name in QUALITY_DOCS:
        shutil.copy(REPO_ROOT / "docs" / "quality" / name, quality_dir / name)
    for name in ("defects.json", "actions.json"):
        shutil.copy(REGISTRY_DIR / name, registry_dir / name)
    doc_path = quality_dir / "defects.md"
    doc = doc_path.read_text(encoding="utf-8")
    begin = doc.index("quality-registry:generated-begin")
    end = doc.index("quality-registry:generated-end")
    section = doc[begin:end]
    tampered = section.replace("RESOLVED", "OPEN", 1)
    assert tampered != section, "expected a RESOLVED status cell in the generated defects table"
    doc_path.write_text(doc[:begin] + tampered + doc[end:], encoding="utf-8")
    result = _run_generator("--check", "--root", str(tmp_path))
    assert result.returncode == 1, "generator --check accepted a tampered generated section:\n" + result.stdout
    assert "defects.md" in result.stdout
