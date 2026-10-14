"""bright_tophat_dog: hard."""
from __future__ import annotations

def _hard(
    center: tuple[float, float],
    bbox: tuple[int, int, int, int],
    area: float,
    circularity: float,
    aspect: float,
    reason: str,
) -> BrightViaDetection:
    from contour.vision.via.bright_tophat_dog_parts.bright_via_detection import BrightViaDetection
    return BrightViaDetection(
        center=(float(center[0]), float(center[1])),
        bbox=bbox,
        area=float(area),
        circularity=float(circularity),
        aspect=float(aspect),
        brightness_score=0.0,
        local_peak_score=0.0,
        tophat_response=0.0,
        dog_response=0.0,
        metal_fraction=0.0,
        final_score=0.0,
        status="hard_reject",
        hard_reason=reason,
    )
