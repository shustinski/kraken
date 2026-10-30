"""Synthetic chip photo sets with an answer key, for trying the heuristic check by eye and by score.

    python scripts/heuristic_synth_set.py OUT_DIR [--per-kind N] [--styles a,b] [--check]

Every style becomes its own folder of frames F_0000.png, F_0001.png, ... (open one folder in
Karakal's validation: one folder of photos picks the heuristic check). Beside the frames:

- answers.csv / answers.json: per frame the planted defect, its box in frame pixels (x, y, w, h;
  empty for a defect of the whole frame), the reason codes that count as finding it and the
  expected verdict;
- with --check, scorecard.md: what the heuristic check found against the answers, per style
  and defect kind, and every finding on a clean frame (a false finding).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT.parents[1] / "src"))

import numpy as np  # noqa: E402

from karakal.core.domain import BuildOptions, BuildResult, FrameRecord, ModelSpec  # noqa: E402
from karakal.core.heuristic_check import (  # noqa: E402
    CONTAMINATION_REASONS,
    GEOMETRY_REASONS,
    TOPOLOGY_REASONS,
    compute_heuristic_check,
)
from karakal.core.image_io import _grayscale_array_to_qimage  # noqa: E402
from karakal.core.synthetic_chip import (  # noqa: E402
    DEFECT_EXPECTATIONS,
    LayerStyle,
    PhotoStyle,
    SyntheticChipFrame,
    box_overlaps,
    make_frame,
)

STYLES: dict[str, tuple[LayerStyle, PhotoStyle]] = {
    "realistic": (LayerStyle(), PhotoStyle(light_gradient=0.3, vignette=0.15, grain=5.0, edge_roughness=0.6, noise=5.0)),
    "manhattan": (LayerStyle(), PhotoStyle()),
    "diagonals": (LayerStyle(diagonals=True), PhotoStyle()),
    "mixed_widths": (LayerStyle(mixed_widths=True), PhotoStyle()),
    "uneven_light": (LayerStyle(), PhotoStyle(light_gradient=0.5, vignette=0.25)),
    "vias": (LayerStyle(vias=True), PhotoStyle()),
}
# Kinds a style can carry: a via layer has no tracks to break.
TRACK_KINDS = ("open", "neck", "bridge")
FINDING_CODES = set(TOPOLOGY_REASONS + CONTAMINATION_REASONS + GEOMETRY_REASONS)


def _plan(style: LayerStyle, per_kind: int, clean: int) -> list[tuple[str, int]]:
    kinds = [kind for kind in DEFECT_EXPECTATIONS if kind != "clean" and not (style.vias and kind in TRACK_KINDS)]
    plan = [("clean", seed) for seed in range(clean)]
    plan += [(kind, 100 + seed) for kind in kinds for seed in range(per_kind)]
    # Mixed, so the matrix shows defects among clean frames, the same order every time.
    order = np.random.default_rng(7).permutation(len(plan))
    return [plan[index] for index in order]


def write_style(out_dir: Path, name: str, per_kind: int, clean: int) -> tuple[Path, dict[str, SyntheticChipFrame]]:
    layer, photo = STYLES[name]
    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    frames: dict[str, SyntheticChipFrame] = {}
    rows = []
    for index, (kind, seed) in enumerate(_plan(layer, per_kind, clean)):
        frame = make_frame(kind, seed, layer, photo)
        key = f"F_{index:04d}"
        frames[key] = frame
        image = _grayscale_array_to_qimage(np.ascontiguousarray(frame.image, dtype=np.uint8))
        if not image.save(str(folder / f"{key}.png")):
            raise OSError(f"cannot write {folder / key}.png")
        rows.append(
            {
                "frame": key,
                "kind": kind,
                "seed": seed,
                "expected_verdict": frame.expected_verdict or "",
                "expected_codes": list(frame.defects[0].expected_codes) if frame.defects else [],
                "boxes": [list(defect.box) for defect in frame.defects if defect.box is not None],
            }
        )
    (folder / "answers.json").write_text(
        json.dumps({"style": name, "layer": asdict(layer), "photo": asdict(photo), "frames": rows}, indent=2),
        encoding="utf-8",
    )
    with (folder / "answers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["frame", "kind", "expected_verdict", "expected_codes", "boxes x,y,w,h"])
        for row in rows:
            writer.writerow(
                [
                    row["frame"],
                    row["kind"],
                    row["expected_verdict"],
                    " ".join(row["expected_codes"]),
                    " | ".join(",".join(str(value) for value in box) for box in row["boxes"]),
                ]
            )
    return folder, frames


def check_style(folder: Path, frames: dict[str, SyntheticChipFrame]) -> tuple[list[str], list[str]]:
    """Scorecard rows of one style and its false findings on clean frames."""

    records = tuple(
        FrameRecord(key, key, model_mask_paths={"photos": str(folder / f"{key}.png")}) for key in sorted(frames)
    )
    result = compute_heuristic_check(
        BuildResult(records=records, model_specs=(ModelSpec("photos", "Photos", folder),), options=BuildOptions())
    )
    tally: dict[str, list[int]] = {}
    false_findings: list[str] = []
    for record in result.records:
        frame = frames[record.key]
        check = getattr(record.summary, "heuristic_check", None)
        reasons = tuple(getattr(check, "reasons", ()) or ())
        verdict = str(getattr(check, "verdict", "error"))
        counts = tally.setdefault(frame.kind, [0, 0, 0])
        counts[0] += 1
        if frame.kind == "clean":
            found = not any(reason.code in FINDING_CODES for reason in reasons)
            for reason in reasons:
                if reason.code in FINDING_CODES:
                    false_findings.append(f"{record.key}: {reason.code} {reason.bbox}")
        elif frame.kind == "blur":
            found = verdict == "unreadable"
        else:
            found = any(
                reason.code in defect.expected_codes
                and (
                    defect.box is None
                    or (reason.bbox is not None and box_overlaps(reason.bbox, defect.box, frame.image.shape))
                )
                for defect in frame.defects
                for reason in reasons
            )
        counts[1] += int(found)
        counts[2] += int(frame.expected_verdict is None or verdict == frame.expected_verdict)
    rows = [
        f"| {kind} | {total} | {found}/{total} | {verdict_ok}/{total} |"
        for kind, (total, found, verdict_ok) in sorted(tally.items())
    ]
    return rows, false_findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--per-kind", type=int, default=3, help="frames of every defect kind per style")
    parser.add_argument("--clean", type=int, default=12, help="clean frames per style (the run's norm)")
    parser.add_argument("--styles", default=",".join(STYLES), help=f"comma-separated: {', '.join(STYLES)}")
    parser.add_argument("--check", action="store_true", help="run the heuristic check and write scorecard.md")
    args = parser.parse_args()
    names = [name.strip() for name in args.styles.split(",") if name.strip()]
    unknown = [name for name in names if name not in STYLES]
    if unknown:
        parser.error(f"unknown styles: {', '.join(unknown)}")
    lines = ["# Heuristic check on synthetic chip photos", ""]
    for name in names:
        folder, frames = write_style(args.out_dir, name, args.per_kind, args.clean)
        print(f"{name}: {len(frames)} frames -> {folder}")
        if not args.check:
            continue
        rows, false_findings = check_style(folder, frames)
        lines += [f"## {name}", "", "| kind | frames | found | verdict |", "|---|---|---|---|", *rows, ""]
        lines += [f"False findings on clean frames: {len(false_findings)}", ""]
        lines += [f"- {item}" for item in false_findings[:20]]
        lines.append("")
        print("\n".join(rows))
        print(f"false findings on clean frames: {len(false_findings)}")
    if args.check:
        (args.out_dir / "scorecard.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
