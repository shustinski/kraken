"""Audit fresh full-gate JUnit evidence for every beta PASS registry action."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from tools.quality.coverage_report import action_verdict, collect_junit
from tools.quality.test_nodes import resolve_node_id


def audit_runs(registry: Path, runs: list[Path], freeze: datetime, repo_root: Path) -> dict:
    actions = json.loads(registry.read_text(encoding="utf-8"))["actions"]
    beta_pass = [action for action in actions if action.get("beta") and action["status"] == "PASS"]
    problems: list[str] = []
    outcomes: list[dict[str, str]] = []
    run_records = []
    for action in beta_pass:
        if not action.get("tests"):
            problems.append(f"{action['id']}: no test binding")
        for node_id in action.get("tests", []):
            problems.extend(f"{action['id']}: {issue}" for issue in resolve_node_id(node_id, repo_root))
    for run in runs:
        junits = sorted(run.rglob("junit*.xml"))
        if not junits:
            problems.append(f"{run}: no JUnit files")
            continue
        for junit in junits:
            if datetime.fromtimestamp(junit.stat().st_mtime, timezone.utc) < freeze:
                problems.append(f"{junit}: older than freeze")
        results = collect_junit(junits, repo_root)
        outcomes.append(results)
        for action in beta_pass:
            verdict = action_verdict(action, results, repo_root)
            if verdict != "PASS":
                problems.append(f"{run}: {action['id']} measured {verdict}")
        comparison = run / "coverage-compare.json"
        report = run / "coverage-report.json"
        if not comparison.is_file() or not report.is_file():
            problems.append(f"{run}: missing coverage report or comparison")
        else:
            compared = json.loads(comparison.read_text(encoding="utf-8-sig"))
            if compared.get("regressions"):
                problems.append(f"{run}: PASS-to-FAIL regressions: {compared['regressions']}")
        run_records.append({"run": str(run), "junit_files": len(junits), "test_nodes": len(results)})
    if len(outcomes) == 2 and outcomes[0] != outcomes[1]:
        differences = sorted(set(outcomes[0]) ^ set(outcomes[1]) |
                             {node for node in outcomes[0].keys() & outcomes[1].keys()
                              if outcomes[0][node] != outcomes[1][node]})
        problems.append(f"full runs differ at {len(differences)} test nodes: {differences[:20]}")
    return {"status": "PASS" if not problems else "FAIL", "beta_pass_actions": len(beta_pass),
            "runs": run_records, "problems": problems}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("docs/quality/registry/actions.json"))
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--freeze", default="2026-11-18T00:00:00+00:00")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    freeze = datetime.fromisoformat(args.freeze)
    if freeze.tzinfo is None:
        parser.error("--freeze must include a timezone")
    if len(args.run) != 2:
        parser.error("exactly two --run directories are required")
    report = audit_runs(args.registry, args.run, freeze, Path.cwd())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
