"""Synthetic validation sets with an answer key: chip photos, network masks with planted errors, confidence.

    python scripts/validation_synth_set.py OUT_DIR [--per-case N] [--clean N] [--styles a,b]
        [--models 1|2] [--behaviour calibrated] [--encoding probability] [--shared] [--reaction ignores]
        [--check]

Every style becomes its own folder:

- photos/F_0000.png ...        the chip photos (Karakal's source photos);
- model_a/F_0000.png           the mask of model a (0/255) and model_a/F_0000_confidence.png, its
                               confidence; model_b/ the same for the second model;
- truth/F_0000.png             the real metal: what ground truth will be;
- answers.json / answers.csv   per frame the case, every planted item with its box (x, y, w, h in
                               frame pixels, empty for the whole frame), its diagnosis and codes.

Open model_a and model_b as layers in Karakal's validation, photos as the source.

With --check, scorecard.md measures today's validation methods on the set (the baseline the new
checks are compared with): per case how far its frames rise in the ranking (AUC against clean
frames), how the frame score follows the real error of model a (Spearman against 1 - IoU with the
truth), and for the single-result risk and the heuristic check of the photos whether their
findings land on the planted problems.
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from karakal.core.analysis_modes import model_output_confidence_metric_key  # noqa: E402
from karakal.core.analytics import compute_build_result_analytics  # noqa: E402
from karakal.core.domain import BuildOptions, BuildResult, FrameRecord, GeometryMode, ModelSpec  # noqa: E402
from karakal.core.findings import frame_findings  # noqa: E402
from karakal.core.heuristic_check import compute_heuristic_check  # noqa: E402
from karakal.core.synthetic_chip import LayerStyle, PhotoStyle  # noqa: E402
from karakal.core.synthetic_scorecard import (  # noqa: E402
    Scorecard,
    found_from_findings,
    rank_auc,
    spearman,
    true_error,
)
from karakal.core.synthetic_validation import (  # noqa: E402
    BEHAVIOURS,
    ENCODINGS,
    REACTIONS,
    SyntheticValidationFrame,
    cases_for,
    make_case,
)

STYLES: dict[str, tuple[LayerStyle, PhotoStyle]] = {
    "realistic": (LayerStyle(), PhotoStyle(light_gradient=0.3, vignette=0.15, grain=5.0, edge_roughness=0.6, noise=5.0)),
    "manhattan": (LayerStyle(), PhotoStyle()),
    "diagonals": (LayerStyle(diagonals=True), PhotoStyle()),
    "mixed_widths": (LayerStyle(mixed_widths=True), PhotoStyle()),
    "dark_metal": (LayerStyle(), PhotoStyle(metal_level=60.0, substrate_level=190.0)),
    "vias": (LayerStyle(vias=True), PhotoStyle()),
}
CONFIDENCE_SUFFIX = "_confidence"


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", np.ascontiguousarray(image, dtype=np.uint8))
    if not ok:
        raise OSError(f"cannot encode {path}")
    encoded.tofile(str(path))


def plan_cases(layer: LayerStyle, per_case: int, clean: int) -> list[tuple[str, int]]:
    cases = [case for case in cases_for(layer) if case != "clean"]
    plan = [("clean", seed) for seed in range(clean)]
    plan += [(case, 100 + seed) for case in cases for seed in range(per_case)]
    # Mixed, so the matrix shows problems among clean frames, the same order every time.
    order = np.random.default_rng(7).permutation(len(plan))
    return [plan[index] for index in order]


def write_style(
    out_dir: Path,
    name: str,
    *,
    per_case: int,
    clean: int,
    model_count: int,
    behaviour: str,
    encoding: str,
    shared: bool,
    reaction: str,
) -> tuple[Path, dict[str, SyntheticValidationFrame]]:
    layer, photo = STYLES[name]
    folder = out_dir / name
    frames: dict[str, SyntheticValidationFrame] = {}
    rows = []
    for index, (case, seed) in enumerate(plan_cases(layer, per_case, clean)):
        frame = make_case(
            case,
            seed,
            layer,
            photo,
            model_count=model_count,
            behaviour=behaviour,
            encoding=encoding,
            shared=shared,
            reaction=reaction,
        )
        key = f"F_{index:04d}"
        frames[key] = frame
        _write_png(folder / "photos" / f"{key}.png", frame.photo)
        _write_png(folder / "truth" / f"{key}.png", (frame.truth * 255).astype(np.uint8))
        for model_id, mask in frame.masks.items():
            _write_png(folder / f"model_{model_id}" / f"{key}.png", mask)
            _write_png(folder / f"model_{model_id}" / f"{key}{CONFIDENCE_SUFFIX}.png", frame.confidences[model_id])
        rows.append(
            {
                "frame": key,
                "case": case,
                "seed": seed,
                "items": [
                    {
                        "kind": item.kind,
                        "box": list(item.box) if item.box is not None else None,
                        "diagnosis": item.diagnosis,
                        "codes": list(item.codes),
                        "models": list(item.models),
                        "params": dict(item.params),
                    }
                    for item in frame.items
                ],
            }
        )
    settings = {
        "style": name,
        "geometry": "point" if layer.vias else "polygon",
        "layer": asdict(layer),
        "photo": asdict(photo),
        "models": model_count,
        "behaviour": behaviour,
        "encoding": encoding,
        "shared_errors": shared,
        "reaction": reaction,
        "confidence_suffix": CONFIDENCE_SUFFIX,
    }
    (folder / "answers.json").write_text(json.dumps({**settings, "frames": rows}, indent=2), encoding="utf-8")
    with (folder / "answers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["frame", "case", "kind", "diagnosis", "models", "codes", "box x,y,w,h"])
        for row in rows:
            for item in row["items"] or [{"kind": "", "diagnosis": "", "models": [], "codes": [], "box": None}]:
                writer.writerow(
                    [
                        row["frame"],
                        row["case"],
                        item["kind"],
                        item["diagnosis"],
                        " ".join(item["models"]),
                        " ".join(item["codes"]),
                        ",".join(str(value) for value in item["box"]) if item["box"] else "",
                    ]
                )
    return folder, frames


# --- Check -----------------------------------------------------------------------------------------


def _build_result(
    folder: Path, frames: dict[str, SyntheticValidationFrame], model_ids: tuple[str, ...], workers: int
) -> BuildResult:
    geometry = GeometryMode.POINT if next(iter(frames.values())).geometry == "point" else GeometryMode.MASK
    records = tuple(
        FrameRecord(
            key,
            key,
            original_path=str(folder / "photos" / f"{key}.png"),
            model_mask_paths={model_id: str(folder / f"model_{model_id}" / f"{key}.png") for model_id in model_ids},
            model_prob_paths={
                model_id: str(folder / f"model_{model_id}" / f"{key}{CONFIDENCE_SUFFIX}.png") for model_id in model_ids
            },
        )
        for key in sorted(frames)
    )
    specs = tuple(
        ModelSpec(
            model_id,
            f"Model {model_id}",
            folder / f"model_{model_id}",
            prob_folder=folder / f"model_{model_id}",
            original_folder=folder / "photos",
        )
        for model_id in model_ids
    )
    options = BuildOptions(geometry_mode=geometry, cache_enabled=False, max_workers=max(1, workers))
    return BuildResult(records=records, model_specs=specs, options=options)


def _metric_scores(result: BuildResult, metric_key: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for record in result.records:
        summary = record.summary
        value = None if summary is None else summary.metric_values.get(metric_key)
        if value is not None and np.isfinite(float(value)):
            scores[record.key] = float(value)
    return scores


def _ranking_rows(frames: dict[str, SyntheticValidationFrame], scores: dict[str, float]) -> list[str]:
    keys = [key for key in sorted(frames) if key in scores]
    if not keys:
        return ["| (no scores) | | | |"]
    values = np.asarray([scores[key] for key in keys], dtype=np.float64)
    # Percentile of badness: 1.0 is the worst frame of the run.
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    percentile = {key: float(order[index]) / max(1, len(keys) - 1) for index, key in enumerate(keys)}
    clean = [scores[key] for key in keys if frames[key].case == "clean"]
    by_case: dict[str, list[str]] = {}
    for key in keys:
        by_case.setdefault(frames[key].case, []).append(key)
    rows = []
    for case in sorted(by_case):
        case_keys = by_case[case]
        auc = None if case == "clean" else rank_auc([scores[key] for key in case_keys], clean)
        mean_percentile = float(np.mean([percentile[key] for key in case_keys]))
        rows.append(
            f"| {case} | {len(case_keys)} | {mean_percentile:.2f} | {'' if auc is None else f'{auc:.2f}'} |"
        )
    return rows


def _correlation_line(frames: dict[str, SyntheticValidationFrame], scores: dict[str, float], model_id: str) -> str:
    keys = [key for key in sorted(frames) if key in scores and model_id in frames[key].masks]
    errors = [true_error(frames[key], model_id) for key in keys]
    iou_rho = spearman([scores[key] for key in keys], [error["iou_error"] for error in errors])
    topo_rho = spearman([scores[key] for key in keys], [error["component_delta"] for error in errors])

    def text(value: float | None) -> str:
        return "n/a" if value is None else f"{value:+.2f}"

    return f"Spearman with the real error of model {model_id}: 1-IoU {text(iou_rho)}, object count change {text(topo_rho)}"


def _method_section(title: str, frames: dict[str, SyntheticValidationFrame], scores: dict[str, float]) -> list[str]:
    return [
        f"### {title}",
        "",
        "| case | frames | mean badness percentile | AUC vs clean |",
        "|---|---|---|---|",
        *_ranking_rows(frames, scores),
        "",
        _correlation_line(frames, scores, "a"),
        "",
    ]


def check_style(folder: Path, frames: dict[str, SyntheticValidationFrame], model_count: int, workers: int) -> list[str]:
    """Scorecard lines of today's validation methods on one style."""

    lines: list[str] = []
    model_ids = tuple(sorted(next(iter(frames.values())).masks))
    if len(model_ids) >= 2:
        result = compute_build_result_analytics(
            _build_result(folder, frames, model_ids, workers), metric_key="overall_frame_score"
        )
        lines += _method_section("Model comparison: overall_frame_score (disagreement)", frames, _metric_scores(result, "overall_frame_score"))
        lines += _method_section("Model comparison: topology_risk (object events)", frames, _metric_scores(result, "topology_risk"))
        # Topology events name what they find in our words: the code must match too.
        lines += [_findings_card(result, frames, "a", any_code=False, detector="topology").to_markdown("Model comparison: topology events of model a"), ""]

    single = _build_result(folder, frames, ("a",), workers)
    confidence_key = model_output_confidence_metric_key("a")
    result = compute_build_result_analytics(single, metric_key=confidence_key)
    if result.selected_metric_key == confidence_key:
        lines += _method_section(f"Confidence check: {confidence_key}", frames, _metric_scores(result, confidence_key))
    internal_key = "model_confidence::a"
    result = compute_build_result_analytics(single, metric_key=internal_key)
    if result.selected_metric_key == internal_key:
        lines += _method_section(f"Internal confidence of the mask: {internal_key}", frames, _metric_scores(result, internal_key))

    result = compute_build_result_analytics(single, metric_key="photo_mismatch_risk")
    if result.selected_metric_key == "photo_mismatch_risk":
        lines += _method_section("Photo check: photo_mismatch_risk (model a against the photo)", frames, _metric_scores(result, "photo_mismatch_risk"))
        lines += _method_section("Photo check: photo_quality_risk", frames, _metric_scores(result, "photo_quality_risk"))
        lines += [_findings_card(result, frames, "a", any_code=False, detector="photo_check").to_markdown("Photo check: findings of model a and of the photo"), ""]

    result = compute_build_result_analytics(single, metric_key="single_result_risk_score")
    lines += _method_section("Single-result risk: single_result_risk_score", frames, _metric_scores(result, "single_result_risk_score"))
    # Where the risk points: any finding on a planted problem counts (its codes differ from ours).
    lines += [_findings_card(result, frames, "a").to_markdown("Single-result risk: where its findings point"), ""]

    # The etch check reads the photos alone: it is asked about the photo's problems only.
    photos = BuildResult(
        records=tuple(
            FrameRecord(key, key, model_mask_paths={"photos": str(folder / "photos" / f"{key}.png")}) for key in sorted(frames)
        ),
        model_specs=(ModelSpec("photos", "Photos", folder / "photos"),),
        options=BuildOptions(cache_enabled=False, max_workers=max(1, workers)),
    )
    result = compute_heuristic_check(photos)
    lines += _method_section("Heuristic check of the photos: heuristic_check_score", frames, _metric_scores(result, "heuristic_check_score"))
    lines += [_findings_card(result, frames, "").to_markdown("Heuristic check: where its findings point"), ""]
    return lines


def _findings_card(
    result: BuildResult,
    frames: dict[str, SyntheticValidationFrame],
    model_id: str,
    *,
    any_code: bool = True,
    detector: str | None = None,
) -> Scorecard:
    """Findings of every frame against the planted items; ``model_id`` "" asks about the photo only."""

    card = Scorecard(any_code=any_code)
    for record in result.records:
        frame = frames[record.key]
        findings = [item for item in frame_findings(record.summary) if detector is None or item.detector == detector]
        found = found_from_findings(findings, frame.shape, model_id=model_id or None)
        card.add_frame(record.key, frame, found, model_id=model_id)
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--per-case", type=int, default=3, help="frames of every case per style")
    parser.add_argument("--clean", type=int, default=12, help="clean frames per style (the run's norm)")
    parser.add_argument("--styles", default=",".join(STYLES), help=f"comma-separated: {', '.join(STYLES)}")
    parser.add_argument("--models", type=int, default=2, choices=(1, 2, 3, 4), help="models per frame")
    parser.add_argument("--behaviour", default="calibrated", choices=BEHAVIOURS, help="confidence behaviour of the models")
    parser.add_argument("--encoding", default="probability", choices=ENCODINGS, help="how confidence maps are written")
    parser.add_argument("--shared", action="store_true", help="every model makes the same errors (correlated models)")
    parser.add_argument("--reaction", default="ignores", choices=REACTIONS, help="whether the models follow photo defects")
    parser.add_argument("--workers", type=int, default=1, help="analytics workers for --check")
    parser.add_argument("--check", action="store_true", help="measure today's validation methods and write scorecard.md")
    args = parser.parse_args()
    names = [name.strip() for name in args.styles.split(",") if name.strip()]
    unknown = [name for name in names if name not in STYLES]
    if unknown:
        parser.error(f"unknown styles: {', '.join(unknown)}")
    lines = [
        "# Today's validation methods on synthetic sets",
        "",
        f"models: {args.models}, confidence: {args.behaviour} ({args.encoding}), shared errors: {args.shared}, "
        f"models on photo defects: {args.reaction}",
        "",
        "AUC: the chance a frame of the case ranks worse than a clean frame (0.5 = blind, 1.0 = always).",
        "For harmless cases (boundary_jitter, point_jitter) a high AUC is bad: the method flags what is not wrong.",
        "",
    ]
    for name in names:
        folder, frames = write_style(
            args.out_dir,
            name,
            per_case=args.per_case,
            clean=args.clean,
            model_count=args.models,
            behaviour=args.behaviour,
            encoding=args.encoding,
            shared=args.shared,
            reaction=args.reaction,
        )
        print(f"{name}: {len(frames)} frames -> {folder}")
        if not args.check:
            continue
        section = [f"## {name}", "", *check_style(folder, frames, args.models, args.workers)]
        lines += section
        print("\n".join(section))
    if args.check:
        (args.out_dir / "scorecard.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
