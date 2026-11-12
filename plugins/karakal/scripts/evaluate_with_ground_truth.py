"""Score an exported labelling sample against its corrected masks (ground truth).

    python scripts/evaluate_with_ground_truth.py SAMPLE_DIR TRUTH_DIR [--points]

SAMPLE_DIR is an export of scripts/build_sample.py (its manifest.json); TRUTH_DIR holds the
corrected masks named like the frames. Writes ground_truth_report.md and .json into SAMPLE_DIR:
every model's IoU and real events against the truth, the precision of every detector and kind of
finding, the share of real errors the checks found, and how many frames really have an error by
how they were picked (the random part is the honest estimate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

from karakal.core.ground_truth import evaluate_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("truth_dir", type=Path)
    parser.add_argument("--points", action="store_true", default=None, help="the layer is points (read from the findings otherwise)")
    args = parser.parse_args(argv)
    report = evaluate_manifest(args.sample_dir, args.truth_dir, point=args.points)
    (args.sample_dir / "ground_truth_report.md").write_text(report.to_markdown(), encoding="utf-8")
    (args.sample_dir / "ground_truth_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
