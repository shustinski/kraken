"""Tests for the coverage report with explicit denominators.

Every fixture lives under ``tmp_path`` (small JSON registries plus small test
files and hand-written JUnit XML); the real registries and evidence are never
touched.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.quality.coverage_report import (  # noqa: E402
    METRIC_KEYS,
    build_report,
    build_source_metric,
    collect_junit,
    render_markdown,
)

STATUSES = ("PASS", "FAIL", "PARTIAL", "BLOCKED", "NOT_TESTED", "NOT_APPLICABLE")


def _action(
    action_id: str,
    surface: str,
    tier: str,
    tests: list[str] | None = None,
    error_conditions: list[str] | None = None,
    status: str = "PASS",
    mandatory: bool = True,
    beta: bool = True,
    exception_reason: str | None = None,
) -> dict:
    record = {
        "id": action_id,
        "surface": surface,
        "entry_point": f"entry point of {action_id}",
        "preconditions": [],
        "expected_behavior": f"expected behavior of {action_id}",
        "side_effects": [],
        "error_conditions": error_conditions or [],
        "tier": tier,
        "status": status,
        "tests": tests or [],
        "evidence": [],
        "mandatory": mandatory,
        "beta": beta,
    }
    if exception_reason is not None:
        record["exception_reason"] = exception_reason
    return record


def _defect(defect_id: str, status: str, regression_tests: list[str] | None = None) -> dict:
    return {
        "id": defect_id,
        "component": "fixture",
        "title": f"fixture defect {defect_id}",
        "status": status,
        "severity": "minor",
        "regression_tests": regression_tests or [],
        "evidence": [],
        "fixed_in": "",
        "repro": "fixture repro",
        **({"exception_reason": "fixture defect is not resolved"} if status != "RESOLVED" else {}),
    }


ACTIONS = {
    "actions": [
        _action("A-PASS", "HUB", tier="unit", tests=["tests/test_pass_action.py"]),
        _action(
            "A-SKIP",
            "HUB",
            tier="unit",
            tests=["tests/test_skip_action.py::test_only"],
            error_conditions=["skipped test carries no marker"],
            status="NOT_TESTED",
            exception_reason="skipped in the evidence",
        ),
        _action("A-FAIL", "HUB", tier="unit", tests=["tests/test_fail_action.py::test_only"], status="FAIL"),
        _action(
            "A-ERR",
            "MGR",
            tier="gui",
            tests=["tests/test_error_marker.py::test_bad_input"],
            error_conditions=["bad input rejected"],
            status="NOT_TESTED",
            exception_reason="marker test not in the evidence yet",
        ),
        _action(
            "A-NOTEST",
            "HUB",
            tier="smoke",
            error_conditions=["unmapped condition"],
            status="NOT_TESTED",
            exception_reason="no tests written yet",
        ),
        _action("A-NAA", "HUB", tier="unit", status="NOT_APPLICABLE"),
        _action("A-OPT", "HUB", tier="unit", mandatory=False, beta=False),
        _action("S1", "SRV", tier="integration", tests=["tests/test_pass_action.py"]),
    ]
}

DEFECTS = {
    "defects": [
        _defect("D1", "RESOLVED", ["tests/test_pass_action.py::test_one"]),
        _defect("D2", "RESOLVED", ["tests/test_skip_action.py::test_only"]),
        _defect("D3", "OPEN"),
        _defect("D4", "RESOLVED", ["tests/test_fail_action.py::test_only"]),
    ]
}

JUNIT_MAIN = """<?xml version="1.0" encoding="utf-8"?>
<testsuites name="pytest tests">
  <testsuite name="pytest" tests="7" failures="1" skipped="1">
    <testcase classname="tests.test_pass_action" name="test_one" time="0.1"/>
    <testcase classname="tests.test_pass_action" name="test_two" time="0.1"/>
    <testcase classname="tests.test_skip_action" name="test_only"><skipped message="skipped"/></testcase>
    <testcase classname="tests.test_fail_action" name="test_only"><failure message="boom"/></testcase>
    <testcase classname="tests.test_error_marker" name="test_bad_input" time="0.1"/>
    <testcase classname="tests.test_error_marker" name="test_plain" time="0.1"/>
    <testcase classname="tests.test_unknown_module" name="test_whatever" time="0.1"/>
  </testsuite>
</testsuites>
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A fake repo: small test files + fixture registries + one JUnit file."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_pass_action.py").write_text(
        "def test_one():\n    assert True\n\n\ndef test_two():\n    assert True\n", encoding="utf-8"
    )
    (tests_dir / "test_skip_action.py").write_text("def test_only():\n    assert True\n", encoding="utf-8")
    (tests_dir / "test_fail_action.py").write_text("def test_only():\n    assert True\n", encoding="utf-8")
    (tests_dir / "test_error_marker.py").write_text(
        "import pytest\n\n\n@pytest.mark.error_path\ndef test_bad_input():\n    assert True\n\n\n"
        "def test_plain():\n    assert True\n",
        encoding="utf-8",
    )
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "actions.json").write_text(json.dumps(ACTIONS, indent=2), encoding="utf-8")
    (registry / "defects.json").write_text(json.dumps(DEFECTS, indent=2), encoding="utf-8")
    (tmp_path / "junit-main.xml").write_text(JUNIT_MAIN, encoding="utf-8")
    return tmp_path


@pytest.fixture
def report(repo: Path) -> dict:
    return build_report(repo / "registry", [repo / "junit-main.xml"], repo_root=repo)


def _beta_report(tmp_path: Path, actions: list[dict]) -> dict:
    """Minimal registry (no JUnit evidence) for the beta-scoping of denominators."""
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "actions.json").write_text(json.dumps({"actions": actions}), encoding="utf-8")
    (registry / "defects.json").write_text(json.dumps({"defects": []}), encoding="utf-8")
    return build_report(registry, [], repo_root=tmp_path)


def test_beta_false_hub_action_leaves_error_paths_and_ui_actions(tmp_path: Path):
    """A beta:false HUB action is out of the error_paths and ui_actions denominators."""
    result = _beta_report(
        tmp_path,
        [
            _action("A-IN", "HUB", tier="unit", error_conditions=["cond one"]),
            _action("A-OUT", "HUB", tier="unit", error_conditions=["cond two"], beta=False),
        ],
    )
    assert result["ui_actions"]["total"] == 1
    assert result["ui_actions"]["uncovered"] == ["A-IN"]
    assert result["error_paths"]["total"] == 1
    assert result["error_paths"]["uncovered"] == ["A-IN::cond one"]


def test_beta_false_srv_action_leaves_api_operations(tmp_path: Path):
    result = _beta_report(tmp_path, [_action("S-OUT", "SRV", tier="integration", beta=False)])
    assert result["api_operations"]["total"] == 0
    assert result["api_operations"]["uncovered"] == []


def test_beta_true_srv_action_counted_in_api_operations(tmp_path: Path):
    result = _beta_report(tmp_path, [_action("S-IN", "SRV", tier="integration", beta=True)])
    assert result["api_operations"]["total"] == 1
    assert result["api_operations"]["uncovered"] == ["S-IN"]


def test_mandatory_no_longer_moves_any_denominator(tmp_path: Path):
    """mandatory is informational now: only beta decides membership of the denominators."""
    result = _beta_report(
        tmp_path,
        [
            _action("A-OPT-BETA", "HUB", tier="unit", mandatory=False, beta=True, error_conditions=["cond"]),
            _action("A-REQ-NOBETA", "HUB", tier="unit", mandatory=True, beta=False, error_conditions=["cond"]),
            _action("S-OPT-BETA", "SRV", tier="integration", mandatory=False, beta=True),
            _action("S-REQ-NOBETA", "SRV", tier="integration", mandatory=True, beta=False),
        ],
    )
    assert result["ui_actions"]["total"] == 1
    assert result["ui_actions"]["uncovered"] == ["A-OPT-BETA"]
    assert result["api_operations"]["total"] == 1
    assert result["api_operations"]["uncovered"] == ["S-OPT-BETA"]
    assert result["error_paths"]["total"] == 1
    assert result["error_paths"]["uncovered"] == ["A-OPT-BETA::cond"]


def test_denominators_are_explicit(report: dict):
    assert report["ui_actions"]["total"] == 5  # beta non-SRV, NOT_APPLICABLE and non-beta excluded
    assert report["api_operations"]["total"] == 1  # the single SRV action
    assert report["error_paths"]["total"] == 3  # one unit per error condition of beta actions
    assert report["state_transitions"]["total"] == 0  # no evidence files provided
    assert report["tiers"]["total"] == 8
    assert report["defect_regressions"]["total"] == 3  # RESOLVED only, OPEN D3 excluded
    assert report["source"]["total"] == 0  # no coverage data provided
    for metric in report.values():
        assert set(metric) >= {"covered", "total", "status_breakdown", "uncovered"}
        assert set(metric["status_breakdown"]) == set(STATUSES)


def test_all_pass_action_is_pass(report: dict):
    ui = report["ui_actions"]
    assert "A-PASS" not in ui["uncovered"]
    # A-ERR's single listed test passed in the JUnit, so the action passes too
    assert "A-ERR" not in ui["uncovered"]
    assert ui["covered"] == 2
    assert report["api_operations"]["covered"] == 1  # S1 shares the same passing file entry


def test_skipped_test_never_counts_as_pass(report: dict):
    ui = report["ui_actions"]
    assert "A-SKIP" in ui["uncovered"]
    assert ui["status_breakdown"]["NOT_TESTED"] == 2  # A-SKIP (skipped) + A-NOTEST (no tests)
    assert ui["covered"] == 2  # only A-PASS and A-ERR
    # a defect whose only regression test was skipped is not covered either
    assert "D2" in report["defect_regressions"]["uncovered"]


def test_failing_test_makes_action_fail(report: dict):
    ui = report["ui_actions"]
    assert "A-FAIL" in ui["uncovered"]
    assert ui["status_breakdown"]["FAIL"] == 1
    assert "D4" in report["defect_regressions"]["uncovered"]


def test_beta_not_tested_without_exception_reason_still_reported(tmp_path: Path):
    """The report does not enforce the registry schema; the registry test does."""
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "actions.json").write_text(
        json.dumps({"actions": [_action("A-ORPHAN", "HUB", tier="unit", status="NOT_TESTED")]}),
        encoding="utf-8",
    )
    (registry / "defects.json").write_text(json.dumps({"defects": []}), encoding="utf-8")
    result = build_report(registry, [], repo_root=tmp_path)
    ui = result["ui_actions"]
    assert ui["total"] == 1
    assert ui["uncovered"] == ["A-ORPHAN"]
    assert ui["status_breakdown"]["NOT_TESTED"] == 1


def test_not_applicable_and_optional_excluded_from_denominator(report: dict):
    ui = report["ui_actions"]
    assert ui["total"] == 5
    assert "A-NAA" not in ui["uncovered"]
    assert "A-OPT" not in ui["uncovered"]


def test_error_paths_without_markers_are_not_tested(tmp_path: Path):
    """Without pytest.mark.error_path anywhere, conditions are NOT_TESTED (visible, not a failure)."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_plain_file.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    registry = tmp_path / "registry"
    registry.mkdir()
    (registry / "actions.json").write_text(
        json.dumps(
            {
                "actions": [
                    _action(
                        "A-E",
                        "HUB",
                        tier="unit",
                        tests=["tests/test_plain_file.py::test_x"],
                        error_conditions=["cond one", "cond two"],
                        status="NOT_TESTED",
                        exception_reason="no error-path marker yet",
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    (registry / "defects.json").write_text(json.dumps({"defects": []}), encoding="utf-8")
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" tests="1">'
        '<testcase classname="tests.test_plain_file" name="test_x" time="0.1"/>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    result = build_report(registry, [junit], repo_root=tmp_path)
    error_paths = result["error_paths"]
    assert error_paths["total"] == 2
    assert error_paths["covered"] == 0
    assert error_paths["status_breakdown"]["NOT_TESTED"] == 2
    assert sorted(error_paths["uncovered"]) == ["A-E::cond one", "A-E::cond two"]
    # the passing test still passes the action itself — the missing marker only affects error_paths
    assert result["ui_actions"]["covered"] == 1


def test_state_transitions_absent_are_not_tested(report: dict, repo: Path):
    st = report["state_transitions"]
    assert st["total"] == 0
    assert st["covered"] == 0
    assert st["uncovered"] == []
    markdown = render_markdown(report, {"junit_files": ["junit-main.xml"], "base": "HEAD~1"})
    assert "NOT TESTED" in markdown
    assert "state_transitions" in markdown


def test_state_transitions_from_evidence(repo: Path):
    evidence = [
        {"state": "idle", "action": "start", "result": "PASS"},
        {"state": "idle", "action": "delete", "result": "FAIL"},
        {"state": "running", "action": "stop", "result": "ok"},
        {"state": "running", "action": "start", "result": "something-else"},
    ]
    path = repo / "transitions.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    result = build_report(repo / "registry", [], state_transition_files=[path], repo_root=repo)
    st = result["state_transitions"]
    assert st["total"] == 4
    assert st["covered"] == 2
    assert sorted(st["uncovered"]) == ["idle -> delete", "running -> start"]


def test_defect_regressions_covered_and_uncovered_split(report: dict):
    defects = report["defect_regressions"]
    assert defects["total"] == 3  # OPEN D3 is out of the denominator
    assert defects["covered"] == 1
    assert "D1" not in defects["uncovered"]
    assert sorted(defects["uncovered"]) == ["D2", "D4"]


def test_source_not_tested_without_coverage_data(report: dict):
    source = report["source"]
    assert source["total"] == 0
    assert source["covered"] == 0
    assert source["status_breakdown"]["NOT_TESTED"] == 1
    assert source["uncovered"] == []


def test_source_metric_with_coverage_data():
    coverage = {
        "files": {
            "src/a.py": {"summary": {"num_statements": 10, "covered_lines": 5}},
            "src/b.py": {
                "summary": {"num_statements": 4, "covered_lines": 4},
                "branch_total": 2,
                "branch_covered": 1,
            },
        }
    }
    metric = build_source_metric(["src/a.py", "src/b.py", "src/missing.py"], coverage)
    assert metric["total"] == 16  # 10 + 4 lines, 2 branches; the missing file contributes nothing
    assert metric["covered"] == 10  # 5 + 4 lines, 1 branch
    assert metric["status_breakdown"]["PARTIAL"] == 1
    assert metric["uncovered"] == ["changed-files"]
    full = build_source_metric(["src/b.py"], {"files": {"src/b.py": {"summary": {"num_statements": 4, "covered_lines": 4}}}})
    assert full["status_breakdown"]["PASS"] == 1
    assert full["uncovered"] == []
    none = build_source_metric(["src/a.py"], None)
    assert none["total"] == 0
    assert none["status_breakdown"]["NOT_TESTED"] == 1


def test_collect_junit_merges_worst_and_parses_unknown_ids(repo: Path):
    first = repo / "m1.xml"
    second = repo / "m2.xml"
    first.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
        '<testcase classname="tests.test_m" name="test_x" time="0.1"/>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    second.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">'
        '<testcase classname="tests.test_m" name="test_x"><skipped message="skipped"/></testcase>'
        '<testcase classname="tests.test_unknown" name="test_other" time="0.1"/>'
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    results = collect_junit([first, second], repo)
    assert results["tests/test_m.py::test_x"] == "SKIP"  # worst outcome wins
    assert results["tests/test_unknown.py::test_other"] == "PASS"


def test_unknown_junit_ids_are_ignored(report: dict):
    for metric in report.values():
        assert not any("test_whatever" in unit_id for unit_id in metric["uncovered"])


def test_cli_end_to_end_on_tmp_fixture(repo: Path, tmp_path: Path):
    (repo / ".git").mkdir()  # so the tool resolves the fixture repo, not the real one
    out_dir = tmp_path / "out" / "evidence"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "quality" / "coverage_report.py"),
            "--registry",
            str(repo / "registry"),
            "--junit",
            str(repo / "junit-main.xml"),
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    document = json.loads((out_dir / "coverage-report.json").read_text(encoding="utf-8"))
    assert set(document["metrics"]) == set(METRIC_KEYS)
    assert document["metrics"]["ui_actions"]["total"] == 5
    assert document["metrics"]["ui_actions"]["covered"] == 2  # JUnit evidence was actually consumed
    markdown = (out_dir / "coverage-report.md").read_text(encoding="utf-8")
    assert "ui_actions" in markdown
    assert "A-SKIP" in markdown
    assert "NOT TESTED" in markdown
