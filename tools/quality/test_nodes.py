"""AST-based test node resolution and marker scanning for the quality tooling.

Shared by the registry enforcement tests (``tests/test_quality_registry.py``)
and the coverage report (``tools/quality/coverage_report.py``).

Hand-rolled on purpose: no new dependencies, so the checks run fast and
offline against the source tree, without spawning pytest.
"""
from __future__ import annotations

import ast
from pathlib import Path

_AST_CACHE: dict[Path, ast.AST] = {}


def parse(path: Path) -> ast.AST:
    """Parse a Python file (cached per resolved path)."""
    resolved = Path(path).resolve()
    if resolved not in _AST_CACHE:
        _AST_CACHE[resolved] = ast.parse(resolved.read_text(encoding="utf-8"))
    return _AST_CACHE[resolved]


def is_test_class(node: ast.ClassDef) -> bool:
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


def list_test_nodes(path: Path) -> list[tuple[str | None, str]]:
    """Enumerate the collectable test nodes of a file as ``(class_name, func_name)``.

    Module-level ``test_*`` functions come with ``class_name=None``; test
    methods of collectable classes carry their class name. Parametrization
    suffixes are not represented: one node per function/method.
    """
    tree = parse(path)
    nodes: list[tuple[str | None, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            nodes.append((None, node.name))
        elif isinstance(node, ast.ClassDef) and is_test_class(node):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test_"):
                    nodes.append((node.name, item.name))
    return nodes


def file_has_tests(path: Path) -> bool:
    """Whether the file contains at least one collectable test.

    A collectable test class counts even when all of its test methods come
    from mixin bases (e.g. ``unittest.TestCase`` facades), which
    :func:`list_test_nodes` cannot see without cross-file resolution.
    """
    tree = parse(path)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            return True
        if isinstance(node, ast.ClassDef) and is_test_class(node):
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


def resolve_node_id(node_id: str, repo_root: Path) -> list[str]:
    """Check one test node id against the real source tree.

    Accepted forms: ``path/to/test.py`` (file level: must exist and contain at
    least one test), ``path/to/test.py::test_name`` (module function, or a
    facade re-export via import), ``path/to/test.py::TestClass`` (collectable
    class with at least one test method), ``path/to/test.py::TestClass::test_name``.
    Returns a list of problems; empty means the node id resolves.
    """
    parts = node_id.split("::")
    path = Path(repo_root) / parts[0]
    if not path.is_file():
        return [f"{node_id}: file not found: {parts[0]}"]
    try:
        tree = parse(path)
    except (OSError, SyntaxError, ValueError):
        return [f"{node_id}: cannot parse {parts[0]}"]
    if len(parts) == 1:
        if file_has_tests(path):
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
        if name in classes and is_test_class(classes[name]):
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


def _carries_mark(value: ast.expr, marker_name: str) -> bool:
    """Whether an expression contains the ``<something>.mark.<marker_name>`` attribute."""
    for sub in ast.walk(value):
        if isinstance(sub, ast.Attribute) and sub.attr == marker_name:
            if isinstance(sub.value, ast.Attribute) and sub.value.attr == "mark":
                return True
    return False


def marker_test_nodes(path: Path, marker_name: str) -> tuple[bool, set[tuple[str | None, str]]]:
    """Find which tests carry ``pytest.mark.<marker_name>``.

    Returns ``(module_level, nodes)``: ``module_level`` is True when the whole
    module is marked (a module-level ``pytestmark`` assignment carrying the
    mark); ``nodes`` holds the ``(class_name, func_name)`` pairs of individually
    marked test functions plus every test function of a marked test class.
    """
    tree = parse(path)
    module_level = False
    marked: set[tuple[str | None, str]] = set()
    marked_classes: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            if _carries_mark(node.value, marker_name):
                module_level = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(_carries_mark(dec, marker_name) for dec in node.decorator_list):
                marked.add((None, node.name))
        elif isinstance(node, ast.ClassDef):
            if any(_carries_mark(dec, marker_name) for dec in node.decorator_list) and is_test_class(node):
                marked_classes.add(node.name)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if any(_carries_mark(dec, marker_name) for dec in item.decorator_list):
                        marked.add((node.name, item.name))
    for cls, func in list_test_nodes(path):
        if cls in marked_classes:
            marked.add((cls, func))
    return module_level, marked


def module_constant_values(path: Path, name: str) -> set[str]:
    """String literal values assigned to a module-level constant (e.g. ``kraken_defect = "D7"``)."""
    tree = parse(path)
    values: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                values.add(node.value.value)
    return values
