"""Render the machine-owned sections of the quality markdown documents from the JSON registries.

The documents under ``docs/quality/`` carry rich hand-written history. To stop that
prose from drifting away from the machine-readable registries
(``docs/quality/registry/*.json`` — the single source of truth), each document embeds
exactly one generated section between markers::

    <!-- quality-registry:generated-begin (source: docs/quality/registry/<name>.json — do not edit inside) -->
    ...
    <!-- quality-registry:generated-end -->

Everything outside the markers is preserved verbatim by both modes:

* ``--check`` — renders the sections in memory and compares; exit 1 with a per-file
  diff summary when a section is missing or out of sync, exit 0 when in sync;
* ``--write`` — renders the sections into the documents, inserting the markers on the
  first run (right after the intro, before the first ``## `` heading).

Hand-rolled on purpose: no new dependencies, and a full run is a handful of small
file reads, well under the 10-second gate budget.
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.quality.registry_schema import validate_registry  # noqa: E402

END_MARKER = "<!-- quality-registry:generated-end -->"
GENERATED_HEADING = "## Автоматическая сводка из реестра (generated)"


@dataclass(frozen=True)
class Doc:
    """One rendered document: which registry it comes from and which table to build."""

    name: str
    doc_rel: str
    registry_rel: str
    render: Callable[[dict[str, Any]], str]

    @property
    def begin_marker(self) -> str:
        return f"<!-- quality-registry:generated-begin (source: {self.registry_rel} — do not edit inside) -->"


def _cell(value: Any) -> str:
    """One markdown table cell: pipes escaped, no newlines, empty/None becomes a dash."""
    if value is None:
        return "—"
    text = str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()
    return text or "—"


def _list_cell(values: Any) -> str:
    joined = ", ".join(str(item) for item in values) if values else ""
    return joined or "—"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [GENERATED_HEADING, "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def render_defects_md(registry: dict) -> str:
    """Table of ALL defects: ID | Компонент | Статус | Регрессии | Коммит | Доказательства."""
    rows = [
        [
            _cell(defect.get("id")),
            _cell(defect.get("component")),
            _cell(defect.get("status")),
            _list_cell(defect.get("regression_tests")),
            _cell(defect.get("fixed_in")),
            _list_cell(defect.get("evidence")),
        ]
        for defect in registry.get("defects", [])
    ]
    return _table(["ID", "Компонент", "Статус", "Регрессии", "Коммит", "Доказательства"], rows)


def render_actions_md(registry: dict) -> str:
    """Status table of ALL actions (for functional-inventory.md).

    Columns: ID | Поверхность | Статус | Обязательный (да/нет) | Тесты (count + first
    node id) | Исключение (exception_reason or —).
    """
    rows = []
    for action in registry.get("actions", []):
        tests = action.get("tests", [])
        tests_cell = f"{len(tests)} ({tests[0]})" if tests else "0"
        rows.append(
            [
                _cell(action.get("id")),
                _cell(action.get("surface")),
                _cell(action.get("status")),
                "да" if action.get("mandatory") else "нет",
                _cell(tests_cell),
                _cell(action.get("exception_reason")),
            ]
        )
    return _table(["ID", "Поверхность", "Статус", "Обязательный", "Тесты", "Исключение"], rows)


def render_matrix_md(registry: dict) -> str:
    """Table of ALL actions with current registry status + evidence paths (acceptance-matrix.md)."""
    rows = [
        [
            _cell(action.get("id")),
            _cell(action.get("surface")),
            _cell(action.get("status")),
            _list_cell(action.get("evidence")),
        ]
        for action in registry.get("actions", [])
    ]
    return _table(["ID", "Поверхность", "Статус", "Доказательства"], rows)


DOCS = (
    Doc("defects", "docs/quality/defects.md", "docs/quality/registry/defects.json", render_defects_md),
    Doc("actions", "docs/quality/functional-inventory.md", "docs/quality/registry/actions.json", render_actions_md),
    Doc("matrix", "docs/quality/acceptance-matrix.md", "docs/quality/registry/actions.json", render_matrix_md),
)


def _load_registry(root: Path, rel: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Load and validate one registry file. Returns (data, problems); data is None on error."""
    path = root / rel
    if not path.is_file():
        return None, [f"{rel}: file not found"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, [f"{rel}: invalid JSON: {exc}"]
    return data, list(validate_registry(path))


def _read_doc(path: Path) -> str:
    """Read a document preserving its exact line endings (no universal-newline translation)."""
    with path.open(encoding="utf-8", newline="") as handle:
        return handle.read()


def _load_all_registries(root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    registries: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    for rel in dict.fromkeys(doc.registry_rel for doc in DOCS):
        data, registry_problems = _load_registry(root, rel)
        problems.extend(registry_problems)
        if data is not None:
            registries[rel] = data
    return registries, problems


def _check_section(doc: Doc, text: str, rendered: str) -> list[str]:
    lines = text.splitlines()
    if doc.begin_marker not in lines or END_MARKER not in lines:
        return [f"{doc.doc_rel}: generated markers not found (run --write to insert them)"]
    begin = lines.index(doc.begin_marker)
    end = lines.index(END_MARKER)
    if end <= begin:
        return [f"{doc.doc_rel}: malformed generated markers (end marker not after begin marker)"]
    expected = [doc.begin_marker, *rendered.rstrip("\n").split("\n"), END_MARKER]
    actual = lines[begin : end + 1]
    if actual == expected:
        return []
    diff = difflib.unified_diff(
        actual,
        expected,
        fromfile=f"{doc.doc_rel} (current section)",
        tofile=f"rendered from {doc.registry_rel}",
        lineterm="",
    )
    return [f"{doc.doc_rel}: OUT OF SYNC with {doc.registry_rel}"] + list(diff)


def _upsert_section(doc: Doc, text: str, rendered: str) -> str:
    """Return the document text with the generated section replaced (or inserted on first run).

    When the markers exist, only the lines strictly between them are replaced. When they
    are absent (first run), the section is inserted right after the intro — before the
    first ``## `` heading — with blank-line separation on both sides.
    """
    lines = text.splitlines()
    content = rendered.rstrip("\n").split("\n")
    if doc.begin_marker in lines and END_MARKER in lines:
        begin = lines.index(doc.begin_marker)
        end = lines.index(END_MARKER)
        if end <= begin:
            raise ValueError(f"{doc.doc_rel}: malformed generated markers (end marker not after begin marker)")
        new_lines = lines[: begin + 1] + content + lines[end:]
    else:
        anchor = next((i for i, line in enumerate(lines) if line.startswith("## ")), None)
        if anchor is None:
            raise ValueError(f"{doc.doc_rel}: no '## ' heading to anchor the generated section")
        separator = [] if anchor and not lines[anchor - 1].strip() else [""]
        new_lines = lines[:anchor] + separator + [doc.begin_marker, *content, END_MARKER, ""] + lines[anchor:]
    newline = "\r\n" if "\r\n" in text else "\n"
    result = newline.join(new_lines)
    if text.endswith(("\n", "\r\n")):
        result += newline
    return result


def check_sync(root: Path) -> list[str]:
    """Compare every rendered section against the document. Empty list means in sync."""
    root = Path(root)
    registries, problems = _load_all_registries(root)
    for doc in DOCS:
        if doc.registry_rel not in registries:
            continue  # registry problem already reported
        doc_path = root / doc.doc_rel
        if not doc_path.is_file():
            problems.append(f"{doc.doc_rel}: file not found (run --write)")
            continue
        problems.extend(_check_section(doc, _read_doc(doc_path), doc.render(registries[doc.registry_rel])))
    return problems


def write_docs(root: Path) -> list[Path]:
    """Render every generated section into the documents. Returns the files actually changed."""
    root = Path(root)
    registries, problems = _load_all_registries(root)
    if problems:
        raise SystemExit("registries are invalid, nothing written:\n" + "\n".join(problems))
    changed: list[Path] = []
    for doc in DOCS:
        doc_path = root / doc.doc_rel
        if not doc_path.is_file():
            raise SystemExit(f"{doc.doc_rel}: file not found")
        text = _read_doc(doc_path)
        updated = _upsert_section(doc, text, doc.render(registries[doc.registry_rel]))
        if updated != text:
            doc_path.write_text(updated, encoding="utf-8", newline="")
            changed.append(doc_path)
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the machine-owned sections of the quality markdown documents from docs/quality/registry/*.json."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="exit 1 with a per-file diff summary when a generated section is missing or out of sync")
    mode.add_argument("--write", action="store_true", help="render the generated sections into the documents (inserts the markers on first run)")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repo root containing docs/quality (default: the repo this file belongs to); used by tests on temporary copies",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.check:
        problems = check_sync(root)
        if problems:
            print("\n".join(problems))
            print(f"quality docs OUT OF SYNC: {len(problems)} problem line(s); run 'python tools/quality/generate_quality_docs.py --write'")
            return 1
        print("quality docs in sync with the registries")
        return 0
    changed = write_docs(root)
    if changed:
        for path in changed:
            print(f"updated {path.relative_to(root).as_posix()}")
    else:
        print("quality docs already in sync; nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
