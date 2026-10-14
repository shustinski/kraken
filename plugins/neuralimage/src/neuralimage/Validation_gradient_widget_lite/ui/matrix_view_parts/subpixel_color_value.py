"""matrix_view: subpixel color value."""
from __future__ import annotations
from PyQt6.QtGui import QColor
from ...core.analysis_modes import metric_level_key, metric_visual_ratio
from ...core.domain import FrameRecord
from ...core.repository import metric_higher_is_better
from neuralimage.Validation_gradient_widget_lite.ui.ui_constants import (
    MATRIX_BACKGROUND_ALT,
)


class SubpixelColorValueMixin:
    def _subpixel_color_for_value(self, score: float) -> QColor:
        metric_key = str(self._metric_key or "")
        ratio = metric_visual_ratio(
            metric_key,
            float(score),
            point_match_radius=float(self._point_match_radius),
            bce_score_cap=float(self._bce_score_cap),
        )
        if ratio is None:
            return QColor(MATRIX_BACKGROUND_ALT)
        level_key = metric_level_key(
            metric_key,
            float(score),
            point_match_radius=float(self._point_match_radius),
            bce_score_cap=float(self._bce_score_cap),
        )
        family = metric_key.split("::", 1)[0]
        higher_is_better = metric_higher_is_better(metric_key)
        if family == "model_confidence":
            if level_key == "score.level.low":
                return QColor(31, 95, 59, 235)
            if level_key == "score.level.moderate":
                return QColor(111, 122, 24, 235)
            if level_key == "score.level.elevated":
                return QColor(167, 93, 18, 235)
            return QColor(140, 47, 57, 235)
        if higher_is_better:
            if ratio < 0.33:
                return QColor(140, 47, 57, 235)
            if ratio < 0.66:
                return QColor(138, 106, 18, 235)
            return QColor(31, 95, 59, 235)
        if ratio < 0.33:
            return QColor(31, 95, 59, 235)
        if ratio < 0.66:
            return QColor(138, 106, 18, 235)
        return QColor(140, 47, 57, 235)

    def _background_color_for_record(self, record: FrameRecord) -> QColor:
        score = self._display_score(record)
        if score is None:
            return QColor(MATRIX_BACKGROUND_ALT)
        return self._background_color(score)

    def _tooltip_for_record(self, record: FrameRecord) -> str:
        if not bool(getattr(record, "score_ready", False)):
            suffix = f"\n{self._t('matrix.reference_frame')}" if self._reference_key == record.key else ""
            return f"{record.display_name}\n{self._t('matrix.mismatch_not_computed')}{suffix}"
        lines = [record.display_name]
        if record.absolute_score is not None:
            lines.append(f"{self._t('matrix.absolute_mismatch')}: {self._format_metric_value(record.absolute_score)}")
        if record.relative_score is not None:
            lines.append(f"{self._t('matrix.relative_mismatch')}: {record.relative_score * 100.0:.3f}%")
        if record.score_percentile is not None:
            lines.append(f"Score percentile: P{float(record.score_percentile):.1f}")
        if self._reference_key == record.key:
            lines.append(self._t('matrix.reference_frame'))
        return "\n".join(lines)

    def _hover_text(self, record: FrameRecord) -> str:
        if not bool(getattr(record, "score_ready", False)):
            return f"{record.display_name} | {self._t('matrix.mismatch_not_computed').lower()}"
        parts = [record.display_name]
        if self._reference_key == record.key:
            parts.append(self._t('matrix.reference_short'))
        if record.absolute_score is not None:
            parts.append(f"{self._t('matrix.absolute_short')} {self._format_metric_value(record.absolute_score)}")
        if record.relative_score is not None:
            parts.append(f"{self._t('matrix.relative_short')} {record.relative_score * 100.0:.3f}%")
        if record.score_percentile is not None:
            parts.append(f"P{float(record.score_percentile):.1f}")
        return " | ".join(parts)
