"""details_dialog: comparison score style."""
from __future__ import annotations
from ...core.analysis_modes import metric_level_key, metric_visual_ratio
from ...core.backend_constants import BCE_SCORE_CAP
from ...core.repository import (
    metric_higher_is_better,
)


class ComparisonScoreStyleMixin:
    def _comparison_score_metric_key(self) -> str:
        return str(self._preferred_metric_key or self._build_result.selected_metric_key or "overall_frame_score")

    def _comparison_score_style(self, value: float | None, metric_key: str | None = None) -> str:
        active_metric = str(metric_key or self._comparison_score_metric_key())
        ratio = metric_visual_ratio(
            active_metric,
            value,
            point_match_radius=float(getattr(self._build_result.options, "point_match_radius", 3.0) or 3.0),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        level_key = metric_level_key(
            active_metric,
            value,
            point_match_radius=float(getattr(self._build_result.options, "point_match_radius", 3.0) or 3.0),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        higher_is_better = metric_higher_is_better(active_metric)
        family = str(active_metric or "").split("::", 1)[0]
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
        return f"padding: 8px 12px; border-radius: 10px; background-color: {background}; color: {foreground}; font-weight: 700;"

    def _comparison_score_text(self, value: float | None, metric_key: str | None = None) -> str:
        if value is None:
            return "-"
        active_metric = str(metric_key or self._comparison_score_metric_key())
        level_key = metric_level_key(
            active_metric,
            value,
            point_match_radius=float(getattr(self._build_result.options, "point_match_radius", 3.0) or 3.0),
            bce_score_cap=float(BCE_SCORE_CAP),
        )
        if level_key is None:
            return "-"
        level = self._t(level_key)
        if "::" in active_metric:
            return f"{level} {float(value) * 100.0:.1f}%"
        return f"{level} {float(value):.4f}"
