"""heuristic_detector: debug viz."""
from __future__ import annotations
import cv2
import numpy as np
from contour.vision.via_detection.result import ViaDetection

def _debug_viz(
    gray: np.ndarray,
    corrected: np.ndarray,
    mbright: np.ndarray,
    mdark: np.ndarray,
    acc: list[ViaDetection],
    below: list[ViaDetection],
    hard: list[ViaDetection],
    d: int,
) -> dict[str, np.ndarray]:
    h, w = gray.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    g = (0, 200, 0)
    yl = (0, 220, 255)
    rd = (0, 0, 255)
    for v in acc:
        cv2.drawMarker(
            out,
            (round(v.x), round(v.y)),
            g,
            markerType=cv2.MARKER_CROSS,
            markerSize=int(max(5, v.diameter_estimate + 1)),
            thickness=1,
        )
    for v in below:
        cv2.drawMarker(
            out,
            (round(v.x), round(v.y)),
            yl,
            markerType=cv2.MARKER_SQUARE,
            markerSize=int(max(5, d)),
            thickness=1,
        )
    for v in hard:
        if v.reject_reason and "low_contrast" not in str(v.reject_reason):
            cv2.drawMarker(
                out,
                (round(v.x), round(v.y)),
                rd,
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=int(max(5, d)),
                thickness=1,
            )
    return {
        "source_gray": base,
        "background_corrected": cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR),
        "local_max_bright": cv2.cvtColor(mbright, cv2.COLOR_GRAY2BGR) if mbright is not None else base,
        "local_max_dark": cv2.cvtColor(mdark, cv2.COLOR_GRAY2BGR) if mdark is not None else base,
        "overlay": cv2.addWeighted(base, 0.55, out, 0.7, 0) if out.size else base,
    }
