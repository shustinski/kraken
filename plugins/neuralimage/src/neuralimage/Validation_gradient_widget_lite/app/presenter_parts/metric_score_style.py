"""presenter: metric score style."""
from __future__ import annotations
from ...core.analysis_modes import (
    metric_level_key,
    metric_visual_ratio,
)
from ...core.backend_constants import (
    BCE_SCORE_CAP,
)


class MetricScoreStyleMixin:
    def _metric_score_style(self, value: float | None, metric_key: str) -> str:
        ratio = metric_visual_ratio(
            metric_key,
            value,
            point_match_radius=float(self.point_match_radius_spin.value()),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        level_key = metric_level_key(
            metric_key,
            value,
            point_match_radius=float(self.point_match_radius_spin.value()),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        higher_is_better = self._metric_higher_is_better(metric_key)
        family = str(metric_key or "").split("::", 1)[0]
        if ratio is None or level_key is None:
            background = "#2f3844"
            foreground = "#edf3fb"
        elif family == "model_confidence":
            if level_key == "score.level.low":
                background = "#1f5f3b"
                foreground = "#e9fff1"
            elif level_key == "score.level.moderate":
                background = "#6f7a18"
                foreground = "#f7ffd8"
            elif level_key == "score.level.elevated":
                background = "#a75d12"
                foreground = "#fff0dc"
            else:
                background = "#8c2f39"
                foreground = "#ffe9ec"
        elif higher_is_better:
            if ratio < 0.33:
                background = "#8c2f39"
                foreground = "#ffe9ec"
            elif ratio < 0.66:
                background = "#8a6a12"
                foreground = "#fff7da"
            else:
                background = "#1f5f3b"
                foreground = "#e9fff1"
        else:
            if ratio < 0.33:
                background = "#1f5f3b"
                foreground = "#e9fff1"
            elif ratio < 0.66:
                background = "#8a6a12"
                foreground = "#fff7da"
            else:
                background = "#8c2f39"
                foreground = "#ffe9ec"
        return f"padding: 6px 10px; border-radius: 8px; background-color: {background}; color: {foreground}; font-weight: 700;"

    def _metric_score_text(self, value: float | None, metric_key: str) -> str:
        if value is None:
            return "-"
        level_key = metric_level_key(
            metric_key,
            value,
            point_match_radius=float(self.point_match_radius_spin.value()),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        if level_key is None:
            return "-"
        level = self._t(level_key)
        if "::" in str(metric_key):
            return f"{level} {float(value) * 100.0:.1f}%"
        if str(metric_key) in {"overall_polygon_score", "iou_score", "dice_score", "polygon_bce_score", "overall_point_score", "precision_score", "recall_score", "f1_score", "localization_score"}:
            return f"{level} {float(value):.1f}"
        return f"{level} {float(value):.4f}"
