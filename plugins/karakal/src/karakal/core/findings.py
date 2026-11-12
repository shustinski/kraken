"""One format for what a validation check finds on a frame.

A finding says what is wrong (``kind``), where (``box``, frame-relative x, y, w, h in 0..1, or
``None`` for the whole frame), how badly (``severity`` 0..1), whose fault it probably is
(``diagnosis``), about which model output (``model_id``, empty for the photo itself) and which
check said so (``detector`` and its version). Every check of the validation speaks it, so a frame
lists all its problems in one place, a sample can be picked by kind, an export can carry them, and
later every kind can be measured against ground truth.

Boxes are frame-relative, as the risk reasons and attention issues already are: they survive the
analysis working on a scaled-down frame.

Checks written before this format keep their own results; the adapters below turn them into
findings, also for results saved before findings existed (``frame_findings``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

FINDINGS_SCHEMA_VERSION = 1

Box = tuple[float, float, float, float]

# Whose fault a finding probably is. Without ground truth a check often cannot tell: "unknown".
DIAG_UNKNOWN = "unknown"
DIAG_OK = "ok"
DIAG_MODEL_ERROR = "model_error"
DIAG_CONFIDENT_ERROR = "confident_error"
DIAG_BAD_DATA = "bad_data"
DIAG_NOT_ALIGNED = "not_aligned"
DIAG_UNCERTAIN = "uncertain"
DIAGNOSES = (
    DIAG_UNKNOWN,
    DIAG_OK,
    DIAG_MODEL_ERROR,
    DIAG_CONFIDENT_ERROR,
    DIAG_BAD_DATA,
    DIAG_NOT_ALIGNED,
    DIAG_UNCERTAIN,
)

DETECTOR_SINGLE_RESULT_RISK = "single_result_risk"
DETECTOR_HEURISTIC_CHECK = "heuristic_check"
DETECTOR_ATTENTION = "attention"

# Severity words as the risk reasons use them.
_SEVERITY_LEVELS = ((0.50, "low"), (0.75, "medium"), (0.90, "high"))


def severity_label(value: float) -> str:
    for upper, label in _SEVERITY_LEVELS:
        if value < upper:
            return label
    return "critical"


def _clip01(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def _box(value: object) -> Box | None:
    if (
        isinstance(value, tuple)
        and len(value) == 4
        and all(type(item) is float and math.isfinite(item) for item in value)
        and value[2] >= 0.0
        and value[3] >= 0.0
    ):
        # The very tuple of the reason: a pickle of the summary stores it once.
        return value  # type: ignore[return-value]
    if not isinstance(value, (tuple, list)) or len(value) < 4:
        return None
    try:
        x, y, w, h = (float(item) for item in value[:4])
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, w, h)) or w < 0.0 or h < 0.0:
        return None
    return (x, y, w, h)


def _evidence_value(value: object) -> float | str:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else str(value)
    return str(value)


@dataclass(frozen=True, slots=True)
class Finding:
    kind: str
    detector: str
    severity: float
    box: Box | None = None
    diagnosis: str = DIAG_UNKNOWN
    model_id: str = ""
    detector_version: str = ""
    outline: tuple[tuple[float, float], ...] = ()
    evidence: tuple[tuple[str, float | str], ...] = ()
    detail: str = ""

    @property
    def severity_label(self) -> str:
        return severity_label(self.severity)

    @property
    def whole_frame(self) -> bool:
        return self.box is None

    def evidence_value(self, name: str, default: float | str | None = None) -> float | str | None:
        return dict(self.evidence).get(name, default)

    def pixel_box(self, shape: tuple[int, int]) -> tuple[float, float, float, float] | None:
        """The box in pixels of a frame of ``shape`` (height, width)."""

        if self.box is None:
            return None
        height, width = shape
        x, y, w, h = self.box
        return (x * width, y * height, w * width, h * height)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "detector": self.detector,
            "severity": self.severity,
            "box": None if self.box is None else list(self.box),
            "diagnosis": self.diagnosis,
        }
        if self.model_id:
            payload["model_id"] = self.model_id
        if self.detector_version:
            payload["detector_version"] = self.detector_version
        if self.outline:
            payload["outline"] = [list(point) for point in self.outline]
        if self.evidence:
            payload["evidence"] = dict(self.evidence)
        if self.detail:
            payload["detail"] = self.detail
        return payload

    @classmethod
    def from_dict(cls, payload: object) -> Finding | None:
        """A finding from ``to_dict`` output; ``None`` for anything unreadable."""

        if not isinstance(payload, Mapping):
            return None
        kind = str(payload.get("kind") or "")
        detector = str(payload.get("detector") or "")
        if not kind or not detector:
            return None
        diagnosis = str(payload.get("diagnosis") or DIAG_UNKNOWN)
        if diagnosis not in DIAGNOSES:
            diagnosis = DIAG_UNKNOWN
        outline: list[tuple[float, float]] = []
        for point in payload.get("outline") or ():
            if isinstance(point, (tuple, list)) and len(point) >= 2:
                try:
                    outline.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue
        evidence_raw = payload.get("evidence") or {}
        evidence = (
            tuple(sorted((str(key), _evidence_value(value)) for key, value in evidence_raw.items()))
            if isinstance(evidence_raw, Mapping)
            else ()
        )
        return cls(
            kind=kind,
            detector=detector,
            severity=_clip01(payload.get("severity")),
            box=_box(payload.get("box")),
            diagnosis=diagnosis,
            model_id=str(payload.get("model_id") or ""),
            detector_version=str(payload.get("detector_version") or ""),
            outline=tuple(outline),
            evidence=evidence,
            detail=str(payload.get("detail") or ""),
        )


def make_finding(
    kind: str,
    detector: str,
    severity: float,
    box: object = None,
    *,
    diagnosis: str = DIAG_UNKNOWN,
    model_id: str = "",
    detector_version: str = "",
    evidence: Mapping[str, object] | None = None,
    detail: str = "",
) -> Finding:
    """A finding with its values cleaned: severity clipped to 0..1, a box checked, evidence sorted."""

    return Finding(
        kind=str(kind),
        detector=str(detector),
        severity=_clip01(severity),
        box=_box(box),
        diagnosis=diagnosis if diagnosis in DIAGNOSES else DIAG_UNKNOWN,
        model_id=str(model_id or ""),
        detector_version=str(detector_version or ""),
        evidence=tuple(sorted((str(key), _evidence_value(value)) for key, value in (evidence or {}).items())),
        detail=str(detail or ""),
    )


def findings_to_payload(findings: Iterable[Finding]) -> dict[str, Any]:
    return {"schema_version": FINDINGS_SCHEMA_VERSION, "findings": [finding.to_dict() for finding in findings]}


def findings_from_payload(payload: object) -> tuple[Finding, ...]:
    """Findings from ``findings_to_payload`` output (or a bare list of finding dicts)."""

    items = payload.get("findings") if isinstance(payload, Mapping) else payload
    if not isinstance(items, (list, tuple)):
        return ()
    result = []
    for item in items:
        finding = Finding.from_dict(item)
        if finding is not None:
            result.append(finding)
    return tuple(result)


def sort_findings(findings: Iterable[Finding]) -> tuple[Finding, ...]:
    """Worst first; equal ones in a stable order."""

    return tuple(sorted(findings, key=lambda item: (-item.severity, item.detector, item.kind, item.box or (-1.0,))))


# --- Adapters from the checks written before findings ------------------------------------------------

# The single-result risk judges one mask: its shape problems point at the model; disagreement with
# the photo cannot tell the model from the photo yet (that is what checking against the photo will
# decide); a batch outlier may be either.
_RISK_DIAGNOSES = {
    "empty_mask": DIAG_MODEL_ERROR,
    "full_mask": DIAG_MODEL_ERROR,
    "tiny_components": DIAG_MODEL_ERROR,
    "fragmented_mask": DIAG_MODEL_ERROR,
    "rough_boundary": DIAG_MODEL_ERROR,
    "excess_holes": DIAG_MODEL_ERROR,
    "skeleton_complexity": DIAG_MODEL_ERROR,
    "edge_clipping": DIAG_MODEL_ERROR,
    "low_confidence": DIAG_UNCERTAIN,
}
# Notes of a check that are not problems of the frame.
_NOT_FINDINGS = frozenset({"insufficient_baseline"})

# Attention issues mark unsure places of a model output.
_ATTENTION_DIAGNOSES = {"original_mismatch": DIAG_UNKNOWN}


def findings_from_risk_reasons(
    reasons: Iterable[object],
    *,
    detector: str,
    detector_version: str = "",
    model_id: str = "",
    diagnoses: Mapping[str, str] | None = None,
    default_diagnosis: str = DIAG_UNKNOWN,
) -> tuple[Finding, ...]:
    """Risk reasons (code, severity word, contribution 0..1, frame-relative bbox, detail) as findings.

    The severity word is not kept: it follows from the contribution (``severity_label``).
    """

    lookup = diagnoses or {}
    result = []
    for reason in reasons:
        code = str(getattr(reason, "code", "") or "")
        if not code or code in _NOT_FINDINGS:
            continue
        base = code.split(".", 1)[0]
        result.append(
            make_finding(
                code,
                detector,
                float(getattr(reason, "contribution", 0.0) or 0.0),
                getattr(reason, "bbox", None),
                diagnosis=lookup.get(code, lookup.get(base, default_diagnosis)),
                model_id=model_id,
                detector_version=detector_version,
                detail=str(getattr(reason, "detail", "") or ""),
            )
        )
    return tuple(result)


def findings_from_single_result_risk(summary: object, *, model_id: str = "") -> tuple[Finding, ...]:
    if summary is None:
        return ()
    from .single_result_risk import SINGLE_RESULT_RISK_VERSION

    return sort_findings(
        findings_from_risk_reasons(
            tuple(getattr(summary, "reasons", ()) or ()),
            detector=DETECTOR_SINGLE_RESULT_RISK,
            detector_version=SINGLE_RESULT_RISK_VERSION,
            model_id=model_id,
            diagnoses=_RISK_DIAGNOSES,
        )
    )


def findings_from_heuristic_check(summary: object) -> tuple[Finding, ...]:
    """The etch check reads the chip photo itself: everything it finds is about the data.

    Every region with a box (not only the few listed reasons), every reason of the whole frame
    (width drift, batch outliers) and the capture issues of an unreadable frame.
    """

    if summary is None:
        return ()
    from .heuristic_check import HEURISTIC_CHECK_VERSION

    regions = tuple(getattr(summary, "regions", ()) or ())
    reasons = tuple(getattr(summary, "reasons", ()) or ())
    boxed = regions if regions else tuple(reason for reason in reasons if getattr(reason, "bbox", None) is not None)
    whole_frame = tuple(reason for reason in reasons if getattr(reason, "bbox", None) is None)
    return sort_findings(
        findings_from_risk_reasons(
            boxed + whole_frame,
            detector=DETECTOR_HEURISTIC_CHECK,
            detector_version=HEURISTIC_CHECK_VERSION,
            default_diagnosis=DIAG_BAD_DATA,
        )
    )


def findings_from_attention_issues(issues_by_model: Mapping[str, Iterable[object]] | None) -> tuple[Finding, ...]:
    """Attention issues (objects or their dicts) of every model as findings."""

    from .attention_issues import AttentionIssue

    result = []
    for model_id, issues in (issues_by_model or {}).items():
        for raw in issues or ():
            issue = raw if isinstance(raw, AttentionIssue) else AttentionIssue.from_dict(raw)
            if issue is None:
                continue
            result.append(
                make_finding(
                    issue.issue_type,
                    DETECTOR_ATTENTION,
                    issue.score,
                    issue.bbox,
                    diagnosis=_ATTENTION_DIAGNOSES.get(issue.issue_type, DIAG_UNCERTAIN),
                    model_id=str(issue.model_id or model_id or ""),
                    evidence={"issue_id": issue.issue_id} if issue.issue_id else None,
                    detail=issue.detail,
                )
            )
    return sort_findings(result)


def frame_findings(summary: object) -> tuple[Finding, ...]:
    """Everything found on a frame.

    Summaries made since findings exist carry them; older ones (saved projects, results pickled
    before) get them from the results of the checks they do carry.
    """

    if summary is None:
        return ()
    stored = getattr(summary, "findings", None)
    if stored:
        return tuple(stored)
    derived: list[Finding] = []
    derived.extend(findings_from_single_result_risk(getattr(summary, "single_result_risk", None)))
    derived.extend(findings_from_heuristic_check(getattr(summary, "heuristic_check", None)))
    derived.extend(findings_from_attention_issues(getattr(summary, "attention_issues_by_model", None)))
    return sort_findings(derived)


def summarize_findings(findings: Iterable[Finding]) -> dict[str, int]:
    """How many findings of every kind."""

    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "DETECTOR_ATTENTION",
    "DETECTOR_HEURISTIC_CHECK",
    "DETECTOR_SINGLE_RESULT_RISK",
    "DIAGNOSES",
    "DIAG_BAD_DATA",
    "DIAG_CONFIDENT_ERROR",
    "DIAG_MODEL_ERROR",
    "DIAG_NOT_ALIGNED",
    "DIAG_OK",
    "DIAG_UNCERTAIN",
    "DIAG_UNKNOWN",
    "FINDINGS_SCHEMA_VERSION",
    "Finding",
    "findings_from_attention_issues",
    "findings_from_heuristic_check",
    "findings_from_payload",
    "findings_from_risk_reasons",
    "findings_from_single_result_risk",
    "findings_to_payload",
    "frame_findings",
    "make_finding",
    "severity_label",
    "sort_findings",
    "summarize_findings",
]
