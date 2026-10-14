"""_core: debug candidates via debug."""
from __future__ import annotations
from typing import Any
from contour.application.processing import (
    ContourDebugCandidate,
)

def _debug_candidates_from_via_hits(hits: list[Any]) -> list[ContourDebugCandidate]:
    debug: list[ContourDebugCandidate] = []
    for index, hit in enumerate(hits):
        x_coord, y_coord, width, height = hit.to_axis_aligned_box()
        extra = dict(getattr(hit, "extra", {}) or {})
        debug.append(
            ContourDebugCandidate(
                contour_index=index,
                bbox=(x_coord, y_coord, width, height),
                area=float(width * height),
                perimeter=float(2 * (width + height)),
                roundness=float(getattr(hit, "annulus_coverage", 0.0) * 100.0),
                accepted=True,
                reason=f"accepted:{getattr(hit, 'strategy', 'heuristic')}",
                source=str(getattr(hit, "strategy", "heuristic")),
                score=float(getattr(hit, "score", 0.0)),
                template_index=extra.get("template_index"),
                metrics=dict(extra.get("features", {}) or {}),
            )
        )
    return debug

def _debug_candidates_from_via_debug(debug_payload: Any) -> list[ContourDebugCandidate]:
    payload = dict(debug_payload or {}) if isinstance(debug_payload, dict) else {}
    out: list[ContourDebugCandidate] = []
    index = 10_000
    payloads = [payload]
    payloads.extend(
        nested
        for key in ("template", "heuristic")
        if isinstance((nested := payload.get(key)), dict)
    )
    for candidate_payload in payloads:
        for group_name, accepted in (("below_threshold", False), ("rejected", False)):
            items = candidate_payload.get(group_name, [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                bbox_raw = item.get("bbox", [0, 0, 0, 0])
                if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) < 4:
                    bbox = (0, 0, 0, 0)
                else:
                    bbox = tuple(round(float(v)) for v in bbox_raw[:4])
                reason = str(item.get("reject_reason") or group_name)
                out.append(
                    ContourDebugCandidate(
                        contour_index=index,
                        bbox=bbox,
                        area=float(max(0, bbox[2]) * max(0, bbox[3])),
                        perimeter=float(2 * (max(0, bbox[2]) + max(0, bbox[3]))),
                        roundness=float(item.get("compactness", 0.0) or 0.0) * 100.0,
                        accepted=accepted,
                        reason=f"{group_name}:{reason}",
                        source=str(
                            item.get("polarity_hypothesis")
                            or candidate_payload.get("strategy")
                            or "via"
                        ),
                        score=float(item.get("score", 0.0) or 0.0),
                        template_index=item.get("template_index"),
                        metrics=dict(item.get("features", {}) or {}),
                    )
                )
                index += 1
    return out
