"""Score validation checks against synthetic frames with known answers. Not used by the analysis itself.

Two levels:

- findings: what a check marked on a frame (code, box, diagnosis) against the planted items of
  ``synthetic_validation``: recall per kind, false findings, how often the diagnosis is right;
- frames: how well a frame score separates frames with a problem from clean ones (rank AUC) and
  how it follows the real error of the model (Spearman against 1 - IoU with the truth).

The same scoring later runs against real ground truth: only the source of the answers changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .findings import DIAG_UNKNOWN, Finding
from .synthetic_validation import DIAGNOSES, PlantedItem, SyntheticValidationFrame

BoxF = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class FoundItem:
    """One finding of a check. ``box``: x, y, w, h in frame pixels; ``None`` for the whole frame."""

    code: str
    box: BoxF | None = None
    diagnosis: str | None = None
    model_id: str | None = None


def found_from_findings(
    findings: Iterable[Finding], shape: tuple[int, int], *, model_id: str | None = None
) -> list[FoundItem]:
    """Findings of a check as scorecard items: frame pixels, ``unknown`` diagnosis left unscored.

    ``model_id``: keep only findings about that model's output and about the photo.
    """

    items = []
    for finding in findings:
        if model_id is not None and finding.model_id and finding.model_id != model_id:
            continue
        diagnosis = None if finding.diagnosis == DIAG_UNKNOWN else finding.diagnosis
        items.append(FoundItem(finding.kind, finding.pixel_box(shape), diagnosis, finding.model_id or None))
    return items


def from_relative_box(bbox: Sequence[float] | None, shape: tuple[int, int]) -> BoxF | None:
    """A frame-relative box (as risk reasons carry them) in frame pixels."""

    if bbox is None:
        return None
    height, width = shape
    x, y, w, h = (float(value) for value in bbox)
    return (x * width, y * height, w * width, h * height)


def boxes_touch(first: BoxF, second: Sequence[float], *, margin: float = 0.0) -> bool:
    fx, fy, fw, fh = first
    sx, sy, sw, sh = (float(value) for value in second)
    return fx - margin <= sx + sw and sx - margin <= fx + fw and fy - margin <= sy + sh and sy - margin <= fy + fh


def _matches(found: FoundItem, item: PlantedItem, aliases: Mapping[str, str], any_code: bool, margin: float) -> bool:
    if not any_code and aliases.get(found.code, found.code) not in item.codes:
        return False
    if item.box is None:
        # A problem of the whole frame: the right code anywhere names it; without codes only a
        # finding about the whole frame does, or any local finding would count.
        return not any_code or found.box is None
    if found.box is None:
        return False
    return boxes_touch(found.box, item.box, margin=margin)


@dataclass(slots=True)
class KindTally:
    planted: int = 0
    found: int = 0
    frames: int = 0


@dataclass(slots=True)
class Scorecard:
    """Findings of one check over many frames.

    ``aliases`` maps the check's own codes to the planted vocabulary; ``any_code`` counts any finding
    in the right place (for checks that do not name problems the way the answers do).
    """

    aliases: Mapping[str, str] = field(default_factory=dict)
    any_code: bool = False
    margin: float = 2.0
    kinds: dict[str, KindTally] = field(default_factory=dict)
    false_findings: list[tuple[str, FoundItem]] = field(default_factory=list)
    clean_frames: int = 0
    clean_frames_flagged: int = 0
    diagnosis_pairs: dict[tuple[str, str], int] = field(default_factory=dict)
    frames: int = 0

    def add_frame(
        self,
        key: str,
        frame: SyntheticValidationFrame,
        found: Iterable[FoundItem],
        *,
        model_id: str | None = None,
    ) -> None:
        findings = tuple(found)
        items = frame.items_for(model_id)
        positives = [item for item in items if item.is_positive]
        self.frames += 1
        seen_kinds: set[str] = set()
        matched_findings: set[int] = set()
        for item in positives:
            tally = self.kinds.setdefault(item.kind, KindTally())
            tally.planted += 1
            if item.kind not in seen_kinds:
                tally.frames += 1
                seen_kinds.add(item.kind)
            hits = [
                index
                for index, finding in enumerate(findings)
                if _matches(finding, item, self.aliases, self.any_code, self.margin)
            ]
            if hits:
                tally.found += 1
                matched_findings.update(hits)
                for index in hits:
                    diagnosis = findings[index].diagnosis
                    if diagnosis is not None:
                        pair = (item.diagnosis, str(diagnosis))
                        self.diagnosis_pairs[pair] = self.diagnosis_pairs.get(pair, 0) + 1
        for index, finding in enumerate(findings):
            if index not in matched_findings:
                self.false_findings.append((key, finding))
        if not positives:
            self.clean_frames += 1
            if findings:
                self.clean_frames_flagged += 1

    def recall(self, kind: str) -> float | None:
        tally = self.kinds.get(kind)
        if tally is None or tally.planted == 0:
            return None
        return tally.found / tally.planted

    @property
    def false_per_frame(self) -> float:
        return len(self.false_findings) / max(1, self.frames)

    @property
    def diagnosis_accuracy(self) -> float | None:
        total = sum(self.diagnosis_pairs.values())
        if total == 0:
            return None
        right = sum(count for (expected, given), count in self.diagnosis_pairs.items() if expected == given)
        return right / total

    def to_markdown(self, title: str = "") -> str:
        lines = [f"## {title}", ""] if title else []
        lines += ["| kind | planted | found | recall |", "|---|---|---|---|"]
        for kind, tally in sorted(self.kinds.items()):
            recall = tally.found / tally.planted if tally.planted else 0.0
            lines.append(f"| {kind} | {tally.planted} | {tally.found} | {recall:.2f} |")
        lines += [
            "",
            f"False findings: {len(self.false_findings)} ({self.false_per_frame:.2f} per frame); "
            f"frames without planted problems that got findings: {self.clean_frames_flagged}/{self.clean_frames}",
        ]
        if self.diagnosis_pairs:
            used = [diagnosis for diagnosis in DIAGNOSES if any(diagnosis in pair for pair in self.diagnosis_pairs)]
            lines += ["", "| expected \\ given | " + " | ".join(used) + " |", "|---" * (len(used) + 1) + "|"]
            for expected in used:
                row = [str(self.diagnosis_pairs.get((expected, given), 0)) for given in used]
                lines.append(f"| {expected} | " + " | ".join(row) + " |")
            accuracy = self.diagnosis_accuracy
            lines += ["", f"Diagnosis accuracy: {accuracy:.2f}" if accuracy is not None else ""]
        return "\n".join(lines)


# --- Frame scores ----------------------------------------------------------------------------------


def rank_auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """The chance a frame with the problem scores worse (higher) than a clean one; ties count half."""

    pos = np.asarray([value for value in positive if np.isfinite(value)], dtype=np.float64)
    neg = np.asarray([value for value in negative if np.isfinite(value)], dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start
        while stop + 1 < values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start : stop + 1]] = 0.5 * (start + stop)
        start = stop + 1
    return ranks


def spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    if float(np.std(ra)) == 0.0 or float(np.std(rb)) == 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def true_error(frame: SyntheticValidationFrame, model_id: str) -> dict[str, float]:
    """How wrong a model really is on a frame: 1 - IoU with the truth and the object count change."""

    import cv2

    mask = np.asarray(frame.masks[model_id]) > 0
    truth = np.asarray(frame.truth) > 0
    union = int(np.count_nonzero(mask | truth))
    iou = 1.0 if union == 0 else float(np.count_nonzero(mask & truth)) / union
    count_mask = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)[0] - 1
    count_truth = cv2.connectedComponents(truth.astype(np.uint8), connectivity=8)[0] - 1
    return {"iou_error": 1.0 - iou, "component_delta": float(abs(count_mask - count_truth))}


__all__ = [
    "FoundItem",
    "KindTally",
    "Scorecard",
    "boxes_touch",
    "found_from_findings",
    "from_relative_box",
    "rank_auc",
    "spearman",
    "true_error",
]
