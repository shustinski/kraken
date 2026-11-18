"""Tests for the coverage regression comparison of the quality gate.

Every fixture lives under ``tmp_path`` (a small JSON registry, a hand-written
base coverage report, a small test file and one JUnit XML); the real registry
and evidence are never touched. ``gate_compare.REPO_ROOT`` is monkeypatched to
the fake repo so node ids resolve against the fixture tree.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.quality import gate_compare  # noqa: E402

PASS_TEST = "tests/test_pass.py::test_x"
FAIL_TEST = "tests/test_fail.py::test_x"

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests">
  <testsuite name="pytest" tests="2" failures="1">
    <testcase classname="tests.test_pass" name="test_x" time="0.1"/>
    <testcase classname="tests.test_fail" name="test_x"><failure message="boom"/></testcase>
  </testsuite>
</testsuites>
"""


def _action(
    action_id: str,
    surface: str,
    tests: list[str],
    mandatory: bool = True,
    beta: bool = True,
) -> dict:
    return {
        "id": action_id,
        "surface": surface,
        "entry_point": f"entry point of {action_id}",
        "preconditions": [],
        "expected_behavior": f"expected behavior of {action_id}",
        "side_effects": [],
        "error_conditions": [],
        "tier": "unit",
        "status": "PASS",
        "tests": tests,
        "evidence": [],
        "mandatory": mandatory,
        "beta": beta,
    }


def _compare(
    tmp_path: Path,
    monkeypatch,
    actions: list[dict],
    ui_uncovered: list[str] | None = None,
    api_uncovered: list[str] | None = None,
) -> tuple[int, dict]:
    """Run gate_compare.main over a tmp fixture; return (exit code, document)."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "test_pass.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    (tests_dir / "test_fail.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    registry = tmp_path / "registry"
    registry.mkdir(exist_ok=True)
    (registry / "actions.json").write_text(json.dumps({"actions": actions}), encoding="utf-8")
    base = tmp_path / "coverage-report.json"
    base.write_text(
        json.dumps(
            {
                "metrics": {
                    "ui_actions": {"uncovered": ui_uncovered or []},
                    "api_operations": {"uncovered": api_uncovered or []},
                }
            }
        ),
        encoding="utf-8",
    )
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT, encoding="utf-8")
    monkeypatch.setattr(gate_compare, "REPO_ROOT", tmp_path)
    out = tmp_path / "compare.json"
    code = gate_compare.main(
        [
            "--registry",
            str(registry),
            "--base-report",
            str(base),
            "--new-junit",
            str(junit),
            "--out",
            str(out),
        ]
    )
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_passing_beta_action_tracked_without_regression(tmp_path: Path, monkeypatch):
    """Baseline: a beta action that passed in the base and passes now stays in base_pass."""
    code, doc = _compare(tmp_path, monkeypatch, [_action("A-PASS", "HUB", [PASS_TEST])])
    assert code == 0
    assert doc["base_pass"] == ["A-PASS"]
    assert doc["regressions"] == []
    assert doc["verdicts"] == {"A-PASS": "PASS"}


def test_beta_false_non_srv_pass_in_base_is_not_tracked(tmp_path: Path, monkeypatch):
    """A beta:false non-SRV action is out of the regression set even when mandatory."""
    # PASS in the base report (uncovered is empty), FAILing now: must not be tracked at all.
    code, doc = _compare(
        tmp_path,
        monkeypatch,
        [_action("A-OUT", "HUB", [FAIL_TEST], mandatory=True, beta=False)],
    )
    assert code == 0
    assert doc["base_pass"] == []
    assert doc["regressions"] == []
    assert doc["verdicts"] == {}


def test_beta_true_srv_action_is_tracked_and_regresses(tmp_path: Path, monkeypatch):
    """A beta:true SRV action is in the regression set and its PASS -> FAIL flip fails the gate."""
    code, doc = _compare(
        tmp_path,
        monkeypatch,
        [_action("S-IN", "SRV", [FAIL_TEST], beta=True)],
    )
    assert doc["base_pass"] == ["S-IN"]
    assert doc["regressions"] == ["S-IN"]
    assert doc["verdicts"] == {"S-IN": "FAIL"}
    assert code == 1


def test_mandatory_no_longer_gates_the_regression_set(tmp_path: Path, monkeypatch):
    """Only beta decides tracking: mandatory:true/beta:false is out, mandatory:false/beta:true is in."""
    code, doc = _compare(
        tmp_path,
        monkeypatch,
        [
            _action("A-OPT-BETA", "HUB", [PASS_TEST], mandatory=False, beta=True),
            _action("A-REQ-NOBETA", "HUB", [PASS_TEST], mandatory=True, beta=False),
            _action("S-OPT-BETA", "SRV", [PASS_TEST], mandatory=False, beta=True),
            _action("S-REQ-NOBETA", "SRV", [PASS_TEST], mandatory=True, beta=False),
        ],
    )
    assert code == 0
    assert doc["base_pass"] == ["A-OPT-BETA", "S-OPT-BETA"]
    assert doc["regressions"] == []
