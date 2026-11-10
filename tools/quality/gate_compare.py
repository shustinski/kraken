"""Coverage regression comparison for the local quality gate.

Compares the verdicts of the mandatory UI actions (``ui_actions`` denominator:
mandatory actions with surface != SRV) and the SRV API operations
(``api_operations`` denominator) between the previous committed coverage
report and the JUnit evidence of the current gate run.

* "PASS before" is derived from the base report: denominator units minus the
  report's ``uncovered`` list (which lists every non-PASS unit, so the
  difference is exactly the PASS set — skipped tests never count as PASS).
* The current verdicts are computed with the same code the coverage report
  uses (``action_verdict`` over the merged JUnit), so a PASS -> FAIL flip is
  detected with the report's own semantics.
* Newly NOT_TESTED/BLOCKED actions are informational only (``newly_not_tested``)
  and do not fail the gate.

Exit codes: 0 = no regression, 1 = regression detected, 2 = operational error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.quality.coverage_report import action_verdict, collect_junit  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect PASS -> FAIL coverage regressions for the quality gate.")
    parser.add_argument("--registry", required=True, type=Path, help="registry directory (docs/quality/registry)")
    parser.add_argument("--base-report", required=True, type=Path, help="previous committed coverage-report.json")
    parser.add_argument("--new-junit", nargs="*", default=[], type=Path, help="JUnit XML files of the current gate run")
    parser.add_argument("--out", type=Path, default=None, help="write the comparison document as JSON here")
    args = parser.parse_args(argv)

    try:
        actions = json.loads((args.registry / "actions.json").read_text(encoding="utf-8-sig")).get("actions", [])
        # utf-8-sig: the PowerShell gate may materialize the base report with a BOM
        base = json.loads(args.base_report.read_text(encoding="utf-8-sig"))
        metrics = base.get("metrics", {})
        if "ui_actions" not in metrics or "api_operations" not in metrics:
            raise ValueError("base report is missing the ui_actions/api_operations metrics")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"gate_compare: cannot read registry/base report: {exc}", file=sys.stderr)
        return 2

    ui_ids = {a["id"] for a in actions if a.get("mandatory") and a.get("surface") != "SRV"}
    api_ids = {a["id"] for a in actions if a.get("surface") == "SRV"}
    base_pass = (ui_ids - set(metrics["ui_actions"]["uncovered"])) | (api_ids - set(metrics["api_operations"]["uncovered"]))

    results = collect_junit([Path(p) for p in args.new_junit], REPO_ROOT)
    verdicts = {a["id"]: action_verdict(a, results, REPO_ROOT) for a in actions}
    regressions = sorted(aid for aid in base_pass if verdicts.get(aid) == "FAIL")
    newly_not_tested = sorted(aid for aid in base_pass if verdicts.get(aid) not in ("PASS", "FAIL"))

    document = {
        "base_report": str(args.base_report),
        "new_junit": [str(p) for p in args.new_junit],
        "base_pass": sorted(base_pass),
        "regressions": regressions,
        "newly_not_tested": newly_not_tested,
        "verdicts": {aid: verdicts[aid] for aid in sorted(base_pass)},
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "gate_compare: {0} regression(s); {1} newly not-tested; base PASS set: {2}".format(
            len(regressions), len(newly_not_tested), ", ".join(sorted(base_pass)) or "(none)"
        )
    )
    if regressions:
        print("regressed actions (PASS before, FAIL now): " + ", ".join(regressions), file=sys.stderr)
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
