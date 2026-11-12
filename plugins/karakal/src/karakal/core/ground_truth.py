"""Ground truth: the operator's corrected masks against the models and against the checks.

With ground truth the validation stops guessing:

- every model's mask is compared with the truth by the same engine as two models are compared
  (core/topology_events.py, reference "ground_truth"): IoU, Dice and the real events (opens,
  bridges, missing and extra objects, necks and bulges, shifted edges), all ``model_error``;
- every finding a check made without the truth is confirmed or not: it is confirmed when it lies
  on a real event of the same model (or is about the photo and lies on one of any model). That
  gives each detector and each kind of finding its precision, and each kind of real event the share
  the checks found (recall);
- the random part of a sample (core/sampling.py) estimates, without the checks' bias, how many
  frames really have an error.

Ground truth comes as a folder of masks named like the frames (``<frame>.png``, any image type).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .findings import DIAG_CONFIDENT_ERROR, DIAG_MODEL_ERROR, DIAG_UNKNOWN, Finding, findings_from_payload

# Findings that say a model is wrong; the others (a poor or shifted photo, doubt) are not judged by
# the truth of the masks, only counted.
ERROR_CLAIMS = frozenset({DIAG_MODEL_ERROR, DIAG_CONFIDENT_ERROR, DIAG_UNKNOWN})
from .image_formats import SUPPORTED_IMAGE_EXTENSION_SET
from .topology_events import TopologySettings, compare_to_reference

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard dependency of the app
    cv2 = None

GROUND_TRUTH_ID = "ground_truth"
DETECTOR_GROUND_TRUTH = "ground_truth"


@dataclass(slots=True)
class FrameTruth:
    frame: str
    model_id: str
    iou: float
    dice: float
    events: tuple[Finding, ...]


@dataclass(slots=True)
class KindCount:
    total: int = 0
    hit: int = 0

    @property
    def share(self) -> float | None:
        return None if self.total == 0 else self.hit / self.total


@dataclass(slots=True)
class GroundTruthReport:
    frames: list[FrameTruth] = field(default_factory=list)
    # (detector, kind) -> how many of its findings a real event confirms (error claims only).
    precision: dict[tuple[str, str], KindCount] = field(default_factory=dict)
    # (detector, kind, diagnosis) -> findings that claim no model error, and how many lie on one anyway.
    other: dict[tuple[str, str, str], KindCount] = field(default_factory=dict)
    # real event kind -> how many of them some check found.
    recall: dict[str, KindCount] = field(default_factory=dict)
    # strategy -> frames, frames with a real event.
    error_rate: dict[str, KindCount] = field(default_factory=dict)
    missing_truth: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "frames": [
                {"frame": item.frame, "model": item.model_id, "iou": item.iou, "dice": item.dice,
                 "events": [event.to_dict() for event in item.events]}
                for item in self.frames
            ],
            "precision": [
                {"detector": detector, "kind": kind, "findings": count.total, "confirmed": count.hit, "precision": count.share}
                for (detector, kind), count in sorted(self.precision.items())
            ],
            "other_findings": [
                {"detector": detector, "kind": kind, "diagnosis": diagnosis, "findings": count.total, "on_real_error": count.hit}
                for (detector, kind, diagnosis), count in sorted(self.other.items())
            ],
            "recall": [
                {"kind": kind, "events": count.total, "found": count.hit, "recall": count.share}
                for kind, count in sorted(self.recall.items())
            ],
            "error_rate": [
                {"strategy": strategy, "frames": count.total, "with_error": count.hit, "share": count.share}
                for strategy, count in sorted(self.error_rate.items())
            ],
            "missing_truth": list(self.missing_truth),
        }

    def to_markdown(self) -> str:
        lines = ["# Validation against ground truth", ""]
        if self.frames:
            ious = [item.iou for item in self.frames]
            lines += [f"Masks checked: {len(self.frames)}; mean IoU {np.mean(ious):.3f}; "
                      f"with a real event: {sum(1 for item in self.frames if item.events)}", ""]
        lines += ["## How often a finding was right (precision)", "", "| detector | kind | findings | confirmed | precision |", "|---|---|---|---|---|"]
        for (detector, kind), count in sorted(self.precision.items()):
            share = "" if count.share is None else f"{count.share:.2f}"
            lines.append(f"| {detector} | {kind} | {count.total} | {count.hit} | {share} |")
        if self.other:
            lines += ["", "## Findings that claim no model error", "", "| detector | kind | diagnosis | findings | on a real error |", "|---|---|---|---|---|"]
            for (detector, kind, diagnosis), count in sorted(self.other.items()):
                lines.append(f"| {detector} | {kind} | {diagnosis} | {count.total} | {count.hit} |")
        lines += ["", "## How many real errors the checks found (recall)", "", "| real event | count | found | recall |", "|---|---|---|---|"]
        for kind, count in sorted(self.recall.items()):
            share = "" if count.share is None else f"{count.share:.2f}"
            lines.append(f"| {kind} | {count.total} | {count.hit} | {share} |")
        lines += ["", "## Frames with a real error, by how they were picked", "", "| picked by | frames | with error | share |", "|---|---|---|---|"]
        for strategy, count in sorted(self.error_rate.items()):
            share = "" if count.share is None else f"{count.share:.2f}"
            lines.append(f"| {strategy} | {count.total} | {count.hit} | {share} |")
        lines += ["", "The random part is the honest estimate of how often the model is wrong; the others are biased by the checks that picked them."]
        if self.missing_truth:
            lines += ["", f"No ground truth for {len(self.missing_truth)} frames: {', '.join(self.missing_truth[:20])}"]
        return "\n".join(lines) + "\n"


def _read_mask(path: Path | str, threshold: float = 0.5) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    return gray >= int(round(threshold * 255))


def find_ground_truth(folder: Path | str, frames: Iterable[str]) -> dict[str, Path]:
    """The truth file of every frame that has one: ``<frame>.<image type>`` in ``folder``."""

    root = Path(folder)
    by_stem: dict[str, Path] = {}
    if root.is_dir():
        for path in root.iterdir():
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSION_SET:
                by_stem.setdefault(path.stem, path)
    found = {}
    for frame in frames:
        stem = Path(str(frame)).stem
        if str(frame) in by_stem:
            found[str(frame)] = by_stem[str(frame)]
        elif stem in by_stem:
            found[str(frame)] = by_stem[stem]
    return found


def _iou_dice(mask: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    both = int(np.count_nonzero(mask & truth))
    union = int(np.count_nonzero(mask | truth))
    total = int(np.count_nonzero(mask)) + int(np.count_nonzero(truth))
    return (1.0 if union == 0 else both / union), (1.0 if total == 0 else 2.0 * both / total)


def _boxes_touch(first: tuple[float, float, float, float] | None, second: tuple[float, float, float, float] | None, margin: float) -> bool:
    if first is None or second is None:
        return True
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    return ax - margin <= bx + bw and bx - margin <= ax + aw and ay - margin <= by + bh and by - margin <= ay + ah


def evaluate(
    frames: Mapping[str, Mapping[str, object]],
    truth_paths: Mapping[str, Path | str],
    *,
    point: bool = False,
    settings: TopologySettings | None = None,
    margin: float = 0.01,
) -> GroundTruthReport:
    """Score masks and findings against ground truth.

    ``frames``: frame key -> {"masks": {model: path}, "findings": [Finding or dict], "strategy": str}.
    ``margin``: how far (share of the frame) a finding may lie from a real event and still be it.
    """

    report = GroundTruthReport()
    for key, info in frames.items():
        truth_path = truth_paths.get(key)
        if truth_path is None:
            report.missing_truth.append(str(key))
            continue
        truth = _read_mask(truth_path)
        if truth is None:
            report.missing_truth.append(str(key))
            continue
        raw_findings = info.get("findings") or ()
        findings = [item for item in raw_findings if isinstance(item, Finding)] + list(
            findings_from_payload([item for item in raw_findings if not isinstance(item, Finding)])
        )
        real_events: dict[str, tuple[Finding, ...]] = {}
        for model_id, mask_path in dict(info.get("masks") or {}).items():
            if not mask_path:
                continue
            mask = _read_mask(mask_path)
            if mask is None:
                continue
            if mask.shape != truth.shape:
                mask = cv2.resize(mask.astype(np.uint8), (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            iou, dice = _iou_dice(mask, truth)
            events = compare_to_reference(
                mask, truth, model_id=str(model_id), reference_id=GROUND_TRUTH_ID, point=point, diagnosis=DIAG_MODEL_ERROR, settings=settings
            )
            real_events[str(model_id)] = events
            report.frames.append(FrameTruth(str(key), str(model_id), iou, dice, events))
        all_events = [event for events in real_events.values() for event in events]
        strategy = str(info.get("strategy") or "all")
        rate = report.error_rate.setdefault(strategy, KindCount())
        rate.total += 1
        rate.hit += int(bool(all_events))
        for finding in findings:
            pool = real_events.get(finding.model_id, ()) if finding.model_id else all_events
            confirmed = any(_boxes_touch(finding.box, event.box, margin) for event in pool)
            if finding.diagnosis in ERROR_CLAIMS:
                count = report.precision.setdefault((finding.detector, finding.kind), KindCount())
            else:
                count = report.other.setdefault((finding.detector, finding.kind, finding.diagnosis), KindCount())
            count.total += 1
            count.hit += int(confirmed)
        for model_id, events in real_events.items():
            mine = [item for item in findings if item.model_id in ("", model_id) and item.diagnosis in ERROR_CLAIMS]
            for event in events:
                found = any(_boxes_touch(item.box, event.box, margin) for item in mine if item.box is not None or event.box is None)
                count = report.recall.setdefault(event.kind, KindCount())
                count.total += 1
                count.hit += int(found)
    return report


def evaluate_manifest(manifest_path: Path | str, truth_folder: Path | str, *, point: bool | None = None) -> GroundTruthReport:
    """Score an exported sample (core/sample_export.py) once its truth masks are back."""

    path = Path(manifest_path)
    folder = path if path.is_dir() else path.parent
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    frames: dict[str, dict[str, object]] = {}
    for row in manifest.get("frames", []):
        masks = {model: str(folder / relative) for model, relative in (row.get("files", {}).get("masks") or {}).items() if relative}
        frames[str(row["frame"])] = {"masks": masks, "findings": row.get("findings") or [], "strategy": row.get("strategy")}
    truth = find_ground_truth(truth_folder, frames.keys())
    if point is None:
        point = any(item.get("kind") in {"missing_point", "extra_point", "merged_points", "shifted_point"}
                    for row in manifest.get("frames", []) for item in row.get("findings") or [])
    return evaluate(frames, truth, point=bool(point))


__all__ = [
    "DETECTOR_GROUND_TRUTH",
    "GROUND_TRUTH_ID",
    "FrameTruth",
    "GroundTruthReport",
    "KindCount",
    "evaluate",
    "evaluate_manifest",
    "find_ground_truth",
]
