"""Coverage report with explicit denominators.

Builds ``coverage-report.{json,md}`` (written by the CLI to
``docs/quality/evidence/<date>/``) from the machine-readable registries plus
the JUnit evidence of actually executed tests.

Design rules (encoded in the report, not in prose):

* Every metric has an explicit denominator (``total``). A metric whose
  denominator source does not exist yet (e.g. state transitions before the
  stateful machines run) reports ``total: 0`` and is labeled NOT TESTED —
  never omitted.
* A test that is ``skipped`` in JUnit is NOT_TESTED for everything it covers;
  it is never counted as PASS.
* An action (or a defect regression) is PASS only if **all** of its listed
  tests are present in the provided JUnit files and passed. A listed test
  that is absent from the JUnit, or skipped, makes the unit NOT_TESTED; any
  failing listed test makes the unit FAIL (FAIL takes precedence over
  NOT_TESTED).
* Node ids expand against the source tree via AST: a file-level id
  (``tests/test_x.py``) covers every collectable test in the file; a
  class-level id (``tests/test_x.py::TestClass``) covers the class's test
  methods and requires the JUnit to carry class info; a function-level id
  covers that test (all its parametrizations).
* Unknown JUnit test ids (tests referenced by no registry unit) are ignored,
  not errors.
* The same test id occurring in several provided JUnit files is merged
  conservatively: the worst outcome wins (FAIL > SKIP > PASS).

Metric denominators:

* ``ui_actions``: mandatory actions with surface != SRV.
* ``api_operations``: actions with surface SRV (same PASS rule).
* ``error_paths``: one unit per ``error_conditions`` entry of each mandatory
  action (id ``<action_id>::<condition>``). A unit is covered when a test
  listed in the action's ``tests`` carries ``pytest.mark.error_path``
  (function/class decorator or module-level ``pytestmark``) and passed.
  Until such markers exist, the conditions are NOT_TESTED — the correct
  visible state, not a failure.
* ``state_transitions``: the ``(state, action)`` pairs recorded in the
  provided JSON evidence files (a JSON list of ``{"state", "action",
  "result"}`` objects; result PASS/OK → PASS, FAIL/FAILED/ERROR → FAIL,
  anything else → NOT_TESTED; a pair is PASS only when none of its recorded
  occurrences failed). No files → total 0, NOT TESTED.
* ``tiers``: the 8 tier names. A tier is covered when at least one test
  mapped to it passed. Mapping: first the action's ``tier`` field for every
  JUnit test that matches one of the action's listed tests; then a documented
  path heuristic for the rest (a ``stateful`` path segment → stateful;
  ``smoke`` → smoke; file name containing ``qt``/``gui`` → gui; file name
  containing ``end_to_end``/``e2e``/``live`` → e2e; a module-level
  ``kraken_defect`` constant or ``regression`` in the file name → regression;
  ``extended`` → extended; otherwise unit).
* ``defect_regressions``: RESOLVED defects; covered when every
  ``regression_tests`` node id resolves against the source tree and all of
  them are present in the JUnit and passed.
* ``source``: lines (+ branches when the coverage data reports them) in the
  files changed by ``git diff --name-only <base>``, measured from the
  provided pytest-cov JSON (``files.<path>.summary``). No coverage data, or
  no changed file present in it → total 0, NOT TESTED. Plain pytest-cov JSON
  carries no branch data, so branches are 0/0 unless the data provides
  per-file ``branch_total``/``branch_covered`` keys.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.quality.test_nodes import (  # noqa: E402
    list_test_nodes,
    marker_test_nodes,
    module_constant_values,
    resolve_node_id,
)

STATUSES = ("PASS", "FAIL", "PARTIAL", "BLOCKED", "NOT_TESTED", "NOT_APPLICABLE")
TIERS = ("smoke", "unit", "integration", "gui", "stateful", "e2e", "extended", "regression")
METRIC_KEYS = (
    "ui_actions",
    "api_operations",
    "error_paths",
    "state_transitions",
    "tiers",
    "defect_regressions",
    "source",
)

# JUnit outcome ordering for the conservative merge (worst wins).
_OUTCOME_ORDER = {"PASS": 0, "SKIP": 1, "FAIL": 2}
_PARAM_SUFFIX = re.compile(r"\[.*\]$")


def split_node_id(node_id: str) -> tuple[str, str | None, str | None]:
    """Split a node id into ``(file, class, func)``; missing parts are None."""
    parts = node_id.split("::")
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        return parts[0], None, parts[1]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(f"unsupported node id format: {node_id}")


def parse_junit(path: Path, repo_root: Path) -> dict[str, str]:
    """Parse one JUnit XML file into ``{test node id: "PASS" | "FAIL" | "SKIP"}``.

    Node ids are in pytest form ``path/to/test.py[::Class]::test_name``. The
    JUnit ``classname`` is a dotted path; the split point between module and
    class is found by checking which dotted prefix exists as a ``.py`` file
    under the repo root (both the legacy family — module only — and the xunit2
    family — module + class — are supported).
    """
    root = ET.parse(path).getroot()
    results: dict[str, str] = {}
    for case in root.iter("testcase"):
        dotted = case.get("classname", "")
        name = case.get("name", "")
        parts = dotted.split(".")
        node_parts: list[str] | None = None
        for count in range(len(parts), 0, -1):
            candidate = "/".join(parts[:count]) + ".py"
            if (Path(repo_root) / candidate).is_file():
                node_parts = [candidate, *parts[count:]]
                break
        if node_parts is None:
            node_parts = [dotted.replace(".", "/") + ".py"]
        node_id = "::".join([*node_parts, name])
        if case.find("failure") is not None or case.find("error") is not None:
            status = "FAIL"
        elif case.find("skipped") is not None:
            status = "SKIP"
        else:
            status = "PASS"
        results[node_id] = status
    return results


def collect_junit(paths: list[Path], repo_root: Path) -> dict[str, str]:
    """Merge several JUnit files; for a repeated test id the worst outcome wins."""
    merged: dict[str, str] = {}
    for path in paths:
        for node_id, status in parse_junit(Path(path), repo_root).items():
            previous = merged.get(node_id)
            if previous is None or _OUTCOME_ORDER[status] > _OUTCOME_ORDER[previous]:
                merged[node_id] = status
    return merged


def junit_matches_entry(junit_node_id: str, entry_file: str, entry_class: str | None, entry_func: str | None) -> bool:
    """Whether one JUnit test node id falls under one registry test entry."""
    jfile, jclass, jfunc = split_node_id(junit_node_id)
    if jfile != entry_file:
        return False
    if entry_func is not None:
        return _PARAM_SUFFIX.sub("", jfunc or "") == entry_func
    if entry_class is not None:
        # class-level entries require the JUnit to carry class info
        return jclass == entry_class
    return True


def evaluate_entry(entry: str, results: dict[str, str], repo_root: Path) -> str:
    """Status of one test entry (file, file::func, or file::Class) against the JUnit.

    PASS: every collectable test the entry covers is present in the results
    and passed. FAIL: at least one covered test is present and failed.
    NOT_TESTED: anything else (a covered test missing from the JUnit or
    skipped, or an entry whose file cannot be found at all).
    """
    try:
        entry_file, entry_class, entry_func = split_node_id(entry)
    except ValueError:
        return "NOT_TESTED"
    expected: list[tuple[str | None, str]] = []
    path = Path(repo_root) / entry_file
    if path.is_file():
        try:
            expected = list_test_nodes(path)
        except (OSError, SyntaxError, ValueError):
            expected = []
        if entry_func is not None:
            expected = [node for node in expected if node[1] == entry_func]
        elif entry_class is not None:
            expected = [node for node in expected if node[0] == entry_class]
    if expected:
        per_node = [
            [status for jid, status in results.items() if junit_matches_entry(jid, entry_file, cls, func)]
            for cls, func in expected
        ]
        if any("FAIL" in statuses for statuses in per_node):
            return "FAIL"
        if any(not statuses or "SKIP" in statuses for statuses in per_node):
            return "NOT_TESTED"
        return "PASS"
    # AST enumeration unavailable (file missing): judge by what the JUnit ran.
    statuses = [
        status
        for jid, status in results.items()
        if junit_matches_entry(jid, entry_file, entry_class, entry_func)
    ]
    if not statuses:
        return "NOT_TESTED"
    if "FAIL" in statuses:
        return "FAIL"
    if "SKIP" in statuses:
        return "NOT_TESTED"
    return "PASS"


def action_verdict(action: dict[str, Any], results: dict[str, str], repo_root: Path) -> str:
    """Measured status of one action: PASS only if all listed tests passed in the JUnit."""
    if action.get("status") == "NOT_APPLICABLE":
        return "NOT_APPLICABLE"
    tests = action.get("tests", [])
    if not tests:
        # a recorded blocker stays visible; anything else has no JUnit evidence
        return "BLOCKED" if action.get("status") == "BLOCKED" else "NOT_TESTED"
    verdicts = [evaluate_entry(entry, results, repo_root) for entry in tests]
    if "FAIL" in verdicts:
        return "FAIL"
    if all(verdict == "PASS" for verdict in verdicts):
        return "PASS"
    return "NOT_TESTED"


def _units_for_actions(
    actions: list[dict[str, Any]],
    results: dict[str, str],
    repo_root: Path,
    include: Callable[[dict[str, Any]], bool],
) -> dict[str, str]:
    return {
        action["id"]: action_verdict(action, results, repo_root)
        for action in actions
        if include(action)
    }


def _entry_carries_error_path(entry: str, repo_root: Path) -> bool:
    try:
        entry_file, entry_class, entry_func = split_node_id(entry)
    except ValueError:
        return False
    path = Path(repo_root) / entry_file
    if not path.is_file():
        return False
    try:
        module_level, marked = marker_test_nodes(path, "error_path")
    except (OSError, SyntaxError, ValueError):
        return False
    if module_level:
        return True
    if entry_func is not None:
        return (entry_class, entry_func) in marked
    if entry_class is not None:
        return any(cls == entry_class for cls, _ in marked)
    return bool(marked)


def error_condition_verdict(
    action: dict[str, Any], condition: str, results: dict[str, str], repo_root: Path
) -> str:
    """Status of one ``error_conditions`` entry of one action.

    Covered (PASS) when a test listed in the action's ``tests`` carries
    ``pytest.mark.error_path`` and passed; FAIL when such a test ran and
    failed and none passed; NOT_TESTED until marked tests exist and pass.
    """
    tests = action.get("tests", [])
    if not tests:
        return "NOT_TESTED"
    verdicts = [
        evaluate_entry(entry, results, repo_root) for entry in tests if _entry_carries_error_path(entry, repo_root)
    ]
    if not verdicts:
        return "NOT_TESTED"
    if "PASS" in verdicts:
        return "PASS"
    if "FAIL" in verdicts:
        return "FAIL"
    return "NOT_TESTED"


def _state_transition_units(files: list[Path]) -> dict[str, str]:
    occurrences: dict[tuple[str, str], list[str]] = {}
    for path in files:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"state transition evidence file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list of {{state, action, result}} records")
        for index, item in enumerate(data):
            if not isinstance(item, dict) or "state" not in item or "action" not in item:
                raise ValueError(f"{path}: record #{index} must be an object with 'state' and 'action'")
            result = str(item.get("result", "")).strip().upper()
            if result in ("PASS", "OK"):
                status = "PASS"
            elif result in ("FAIL", "FAILED", "ERROR"):
                status = "FAIL"
            else:
                status = "NOT_TESTED"
            occurrences.setdefault((str(item["state"]), str(item["action"])), []).append(status)
    units: dict[str, str] = {}
    for (state, action), statuses in occurrences.items():
        if "FAIL" in statuses:
            units[f"{state} -> {action}"] = "FAIL"
        elif "PASS" in statuses:
            units[f"{state} -> {action}"] = "PASS"
        else:
            units[f"{state} -> {action}"] = "NOT_TESTED"
    return units


def heuristic_tier(junit_node_id: str, repo_root: Path) -> str:
    """Documented path heuristic for JUnit tests no action maps to a tier (see module docstring)."""
    file_part = junit_node_id.split("::")[0]
    rel = Path(file_part)
    name = rel.name
    parts = [part.lower() for part in rel.parts]
    if "stateful" in parts:
        return "stateful"
    if "smoke" in parts or "smoke" in name:
        return "smoke"
    if "qt" in name or "gui" in name:
        return "gui"
    if "end_to_end" in name or "e2e" in name or "live" in name:
        return "e2e"
    try:
        if module_constant_values(Path(repo_root) / file_part, "kraken_defect"):
            return "regression"
    except (OSError, SyntaxError, ValueError):
        pass
    if "regression" in name:
        return "regression"
    if "extended" in name or "extended" in parts:
        return "extended"
    return "unit"


def _tier_units(actions: list[dict[str, Any]], results: dict[str, str], repo_root: Path) -> dict[str, str]:
    tiers_per_test: dict[str, set[str]] = {}
    for action in actions:
        tier = action.get("tier")
        if tier not in TIERS:
            continue
        for entry in action.get("tests", []):
            try:
                entry_file, entry_class, entry_func = split_node_id(entry)
            except ValueError:
                continue
            for jid in results:
                if junit_matches_entry(jid, entry_file, entry_class, entry_func):
                    tiers_per_test.setdefault(jid, set()).add(tier)
    for jid in results:
        if jid not in tiers_per_test:
            tiers_per_test[jid] = {heuristic_tier(jid, repo_root)}
    units = {tier: "NOT_TESTED" for tier in TIERS}
    for jid, status in results.items():
        for tier in tiers_per_test.get(jid, ()):
            if status == "PASS":
                units[tier] = "PASS"
            elif status == "FAIL" and units[tier] == "NOT_TESTED":
                units[tier] = "FAIL"
    return units


def _defect_units(defects: list[dict[str, Any]], results: dict[str, str], repo_root: Path) -> dict[str, str]:
    units: dict[str, str] = {}
    for defect in defects:
        if defect.get("status") != "RESOLVED":
            continue
        defect_id = defect["id"]
        tests = defect.get("regression_tests", [])
        if not tests or any(resolve_node_id(entry, repo_root) for entry in tests):
            # unresolvable node ids cannot be verified from evidence
            units[defect_id] = "NOT_TESTED"
            continue
        verdicts = [evaluate_entry(entry, results, repo_root) for entry in tests]
        if "FAIL" in verdicts:
            units[defect_id] = "FAIL"
        elif all(verdict == "PASS" for verdict in verdicts):
            units[defect_id] = "PASS"
        else:
            units[defect_id] = "NOT_TESTED"
    return units


def build_source_metric(changed_files: list[str], coverage_data: dict | None) -> dict[str, Any]:
    """The ``source`` metric: lines (+branches) of the changed files per the coverage data."""
    files = coverage_data.get("files", {}) if isinstance(coverage_data, dict) else {}
    lines_total = lines_covered = branches_total = branches_covered = 0
    for rel in changed_files:
        data = files.get(rel.replace("\\", "/"))
        if not isinstance(data, dict):
            continue
        summary = data.get("summary", {})
        lines_total += int(summary.get("num_statements", 0) or 0)
        lines_covered += int(summary.get("covered_lines", 0) or 0)
        branches_total += int(data.get("branch_total", 0) or 0)
        branches_covered += int(data.get("branch_covered", 0) or 0)
    total = lines_total + branches_total
    covered = lines_covered + branches_covered
    if coverage_data is None or total == 0:
        status = "NOT_TESTED"
    elif covered == total:
        status = "PASS"
    elif covered == 0:
        status = "FAIL"
    else:
        status = "PARTIAL"
    return {
        "covered": covered,
        "total": total,
        "status_breakdown": {state: (1 if state == status else 0) for state in STATUSES},
        "uncovered": [] if (status == "PASS" or total == 0) else ["changed-files"],
        "changed_files": list(changed_files),
    }


def _git_changed_files(repo_root: Path, base: str) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "diff", "--name-only", base],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _default_repo_root(registry_dir: Path) -> Path:
    directory = Path(registry_dir).resolve()
    for parent in (directory, *directory.parents):
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def _load_registry(registry_dir: Path, name: str) -> dict[str, Any]:
    path = Path(registry_dir) / name
    if not path.is_file():
        raise FileNotFoundError(f"registry file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be an object")
    return data


def _metric(units: dict[str, str]) -> dict[str, Any]:
    """Assemble one metric from a ``{unit id: status}`` mapping.

    NOT_APPLICABLE units are excluded from the denominator entirely.
    ``uncovered`` lists every denominator unit that is not PASS.
    """
    breakdown = {status: 0 for status in STATUSES}
    covered = 0
    uncovered: list[str] = []
    total = 0
    for unit_id, status in units.items():
        if status == "NOT_APPLICABLE":
            continue
        total += 1
        breakdown[status] += 1
        if status == "PASS":
            covered += 1
        else:
            uncovered.append(unit_id)
    return {
        "covered": covered,
        "total": total,
        "status_breakdown": breakdown,
        "uncovered": sorted(uncovered),
    }


def build_report(
    registry_dir: Path,
    junit_paths: list[Path],
    coverage_data: dict | None = None,
    state_transition_files: list[Path] | None = None,
    base: str = "HEAD~1",
    repo_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the coverage report; see the module docstring for denominators and rules."""
    registry_dir = Path(registry_dir)
    repo = Path(repo_root) if repo_root is not None else _default_repo_root(registry_dir)
    actions = _load_registry(registry_dir, "actions.json").get("actions", [])
    defects = _load_registry(registry_dir, "defects.json").get("defects", [])
    results = collect_junit([Path(path) for path in junit_paths], repo)
    error_units = {
        f"{action['id']}::{condition}": error_condition_verdict(action, condition, results, repo)
        for action in actions
        if action.get("mandatory")
        for condition in action.get("error_conditions", [])
    }
    return {
        "ui_actions": _metric(
            _units_for_actions(actions, results, repo, lambda a: bool(a.get("mandatory")) and a.get("surface") != "SRV")
        ),
        "api_operations": _metric(
            _units_for_actions(actions, results, repo, lambda a: a.get("surface") == "SRV")
        ),
        "error_paths": _metric(error_units),
        "state_transitions": _metric(_state_transition_units([Path(p) for p in (state_transition_files or [])])),
        "tiers": _metric(_tier_units(actions, results, repo)),
        "defect_regressions": _metric(_defect_units(defects, results, repo)),
        "source": build_source_metric(_git_changed_files(repo, base), coverage_data),
    }


def render_markdown(report: dict[str, dict[str, Any]], meta: dict[str, Any]) -> str:
    """Human-readable markdown: the denominator table plus the uncovered units per metric."""
    lines: list[str] = []
    add = lines.append
    add("# Quality coverage report")
    add("")
    add(f"- Generated: {meta.get('generated_at', '')}")
    add(f"- Registry: {meta.get('registry', '')}")
    add(f"- Source diff base: `{meta.get('base', '')}`")
    junit_files = meta.get("junit_files", [])
    if junit_files:
        add(f"- JUnit files: {len(junit_files)} ({', '.join(junit_files)})")
    else:
        add("- JUnit files: 0 (none provided — every JUnit-verifiable unit is NOT_TESTED)")
    add("")
    add("| Metric | Covered | Total | PASS | FAIL | PARTIAL | BLOCKED | NOT_TESTED | NOT_APPLICABLE |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in METRIC_KEYS:
        metric = report[key]
        breakdown = metric["status_breakdown"]
        add(
            f"| {key} | {metric['covered']} | {metric['total']} | {breakdown['PASS']} | {breakdown['FAIL']} | "
            f"{breakdown['PARTIAL']} | {breakdown['BLOCKED']} | {breakdown['NOT_TESTED']} | "
            f"{breakdown['NOT_APPLICABLE']} |"
        )
    add("")
    add("Rules: `Total` is the explicit denominator; a skipped test is NOT_TESTED, never PASS;")
    add("a unit is PASS only when all of its listed tests are present in the JUnit evidence and passed.")
    not_tested = [key for key in METRIC_KEYS if report[key]["total"] == 0]
    if not_tested:
        add("")
        add("## NOT TESTED (no denominator source)")
        add("")
        for key in not_tested:
            add(f"- `{key}`: NOT TESTED — no denominator source available.")
    add("")
    add("## Uncovered units")
    add("")
    for key in METRIC_KEYS:
        metric = report[key]
        if metric["total"] == 0:
            continue
        if not metric["uncovered"]:
            add(f"### {key}: fully covered ({metric['covered']}/{metric['total']})")
            add("")
            continue
        add(f"### {key} ({len(metric['uncovered'])} uncovered of {metric['total']})")
        add("")
        for unit_id in metric["uncovered"]:
            add(f"- `{unit_id}`")
        add("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the quality coverage report with explicit denominators.")
    parser.add_argument("--registry", required=True, type=Path, help="registry directory (docs/quality/registry)")
    parser.add_argument("--junit", nargs="*", default=[], type=Path, help="JUnit XML files (pytest --junitxml)")
    parser.add_argument("--coverage", type=Path, default=None, help="pytest-cov JSON file for the source metric")
    parser.add_argument(
        "--state-transitions",
        dest="state_transitions",
        nargs="*",
        default=[],
        type=Path,
        help="state-transition evidence JSON files",
    )
    parser.add_argument("--base", default="HEAD~1", help="git base for the source metric (default: HEAD~1)")
    parser.add_argument("--out", required=True, type=Path, help="output directory for coverage-report.{json,md}")
    args = parser.parse_args(argv)

    coverage_data = None
    if args.coverage is not None:
        coverage_data = json.loads(args.coverage.read_text(encoding="utf-8"))
    report = build_report(
        args.registry,
        args.junit,
        coverage_data=coverage_data,
        state_transition_files=args.state_transitions or None,
        base=args.base,
    )
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "registry": str(args.registry),
        "junit_files": [str(path) for path in args.junit],
        "base": args.base,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    document = {**meta, "metrics": report}
    (args.out / "coverage-report.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out / "coverage-report.md").write_text(render_markdown(report, meta), encoding="utf-8")
    covered = sum(metric["covered"] for metric in report.values())
    total = sum(metric["total"] for metric in report.values())
    print(f"coverage report: {covered}/{total} covered across {len(report)} metrics -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
