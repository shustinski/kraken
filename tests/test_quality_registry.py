"""Enforcement tests for the machine-readable quality registries.

The registries under ``docs/quality/registry/`` are the enforcement backbone
of the defect-to-regression policy. These tests make the build fail when:

* a registry file is missing, not valid JSON, or schema-inconsistent;
* a RESOLVED defect has no live regression test (node ids must resolve to
  actually-collected tests — resolved by the shared ``ast`` scanner in
  ``tools/quality/test_nodes.py`` so the check runs fast and offline,
  without spawning pytest);
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

from tools.quality.registry_schema import ACTION_TIERS, _require_enum, validate_registry
from tools.quality.test_nodes import parse, resolve_node_id

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = REPO_ROOT / "docs" / "quality" / "registry"
DEFECTS_JSON = REGISTRY_DIR / "defects.json"
ACTIONS_JSON = REGISTRY_DIR / "actions.json"
TESTS_DIR = REPO_ROOT / "tests"
GENERATOR = REPO_ROOT / "tools" / "quality" / "generate_quality_docs.py"
QUALITY_DOCS = ("defects.md", "functional-inventory.md", "acceptance-matrix.md")


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
            problems.extend(f"{defect['id']}: {problem}" for problem in resolve_node_id(node_id, REPO_ROOT))
    assert not problems, "unresolved regression tests:\n" + "\n".join(problems)


def test_defect_markers_are_registered():
    registry_ids = {defect["id"] for defect in _load(DEFECTS_JSON)["defects"]}
    problems = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        tree = parse(path)
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
            problems.extend(f"{action['id']}: {problem}" for problem in resolve_node_id(node_id, REPO_ROOT))
    assert not problems, "untracked mandatory actions:\n" + "\n".join(problems)


def _action_record(**overrides) -> dict:
    """A minimal valid synthetic action record; keyword overrides tweak one field."""
    record = {
        "id": "X-SYNTHETIC",
        "surface": "HUB",
        "entry_point": "синтетическая запись",
        "preconditions": [],
        "expected_behavior": "ничего",
        "side_effects": [],
        "error_conditions": [],
        "tier": "unit",
        "status": "PASS",
        "tests": [],
        "evidence": [],
        "mandatory": False,
        "beta": True,
    }
    record.update(overrides)
    return record


def _validate_records(tmp_path: Path, *records: dict) -> list[str]:
    path = tmp_path / "actions.json"
    path.write_text(json.dumps({"actions": list(records)}, ensure_ascii=False), encoding="utf-8")
    return validate_registry(path)


def test_action_requires_boolean_beta_flag(tmp_path: Path):
    record = _action_record()
    del record["beta"]
    problems = _validate_records(tmp_path, record)
    assert any("'beta' must be a boolean" in problem for problem in problems)

    for bad_beta in ("true", 1, None):
        problems = _validate_records(tmp_path, _action_record(beta=bad_beta))
        assert any("'beta' must be a boolean" in problem for problem in problems), f"beta={bad_beta!r} accepted"

    assert not _validate_records(tmp_path, _action_record(beta=False, beta_out_reason="вне объёма беты — тест"))


def test_beta_false_requires_beta_out_reason(tmp_path: Path):
    problems = _validate_records(tmp_path, _action_record(beta=False))
    assert any("'beta_out_reason' must be a non-empty string" in problem for problem in problems)

    problems = _validate_records(tmp_path, _action_record(beta=False, beta_out_reason=""))
    assert any("'beta_out_reason' must be a non-empty string" in problem for problem in problems)

    problems = _validate_records(tmp_path, _action_record(beta=False, beta_out_reason="вне объёма беты — сшивка"))
    assert not any("beta_out_reason" in problem for problem in problems)


def test_beta_true_must_not_carry_beta_out_reason(tmp_path: Path):
    problems = _validate_records(tmp_path, _action_record(beta=True, beta_out_reason="вне объёма беты — сшивка"))
    assert any("beta_out_reason" in problem for problem in problems)


def test_not_tested_rule_scopes_to_beta_not_mandatory(tmp_path: Path):
    problems = _validate_records(tmp_path, _action_record(beta=True, mandatory=False, status="NOT_TESTED"))
    assert any("NOT_TESTED action requires exception_reason" in problem for problem in problems)

    problems = _validate_records(
        tmp_path, _action_record(beta=False, mandatory=True, status="NOT_TESTED", beta_out_reason="вне объёма беты — тест")
    )
    assert not any("NOT_TESTED" in problem for problem in problems)

    problems = _validate_records(tmp_path, _action_record(beta=True, status="NOT_TESTED", exception_reason="почва"))
    assert not any("NOT_TESTED" in problem for problem in problems)


def test_live_is_a_valid_action_tier():
    assert "live" in ACTION_TIERS
    problems: list[str] = []
    _require_enum({"tier": "live"}, "tier", ACTION_TIERS, problems, "synthetic")
    assert problems == []


def test_new_beta_actions_are_registered():
    """The W3-palette scenarios bind tests to these ids; they must exist in the real registry."""
    new_ids = (
        "HUB-LOGIN-TABBED",
        "ADMIN-SERVER-BACKUP-RESTORE",
        "AGT-OP-VECTORIZE",
        "AGT-OP-DATASET-PREPARE",
        "AGT-OP-TRAIN",
        "AGT-OP-BINARY-SEGMENT",
    )
    actions = {action["id"]: action for action in _load(ACTIONS_JSON)["actions"]}
    for action_id in new_ids:
        assert action_id in actions, f"{action_id} missing from actions.json"
        record = actions[action_id]
        assert record["beta"] is True, f"{action_id} must be beta: true"
        assert "beta_out_reason" not in record, f"{action_id} must not carry beta_out_reason"
        assert record["expected_behavior"].strip(), f"{action_id} needs non-empty expected_behavior"


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
