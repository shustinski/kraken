"""main_window: build preview panel."""
from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from neuralimage.Validation_gradient_widget_lite.app.state import ExtendPreviewPanel

from neuralimage.Validation_gradient_widget_lite.app.main_window_parts.expandable_score_card import _ExpandableScoreCard


class BuildPreviewPanelMixin:
    def _build_preview_panel(self, parent: QWidget, metric_keys: tuple[str, ...], build_result: BuildResult) -> ExtendPreviewPanel:
        group = QGroupBox(self._t("matrix.preview.group"), parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        frame_row = QWidget(group)
        frame_layout = QHBoxLayout(frame_row)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(8)
        frame_title = QLabel(self._t("matrix.preview.frame"), frame_row)
        frame_value = QLabel("-", frame_row)
        frame_value.setWordWrap(True)
        frame_value.setMinimumWidth(0)
        frame_value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        frame_layout.addWidget(frame_title)
        frame_layout.addWidget(frame_value, stretch=1)
        layout.addWidget(frame_row)

        subpixel_group = QGroupBox(self._t("details.subpixel_selection"), group)
        subpixel_layout = QVBoxLayout(subpixel_group)
        subpixel_layout.setContentsMargins(0, 0, 0, 0)
        subpixel_layout.setSpacing(6)
        subpixel_value = QLabel("-", subpixel_group)
        subpixel_value.setWordWrap(True)
        subpixel_value.hide()
        subpixel_score_card = _ExpandableScoreCard(self._t("details.selected_comparison_score"), subpixel_group)
        subpixel_score_card.hide()
        subpixel_layout.addWidget(subpixel_value)
        subpixel_layout.addWidget(subpixel_score_card)
        subpixel_group.hide()
        layout.addWidget(subpixel_group)

        score_cards: dict[str, QWidget] = {}
        seen_keys: set[str] = set()
        overall_card_count = 0
        component_card_count = 0
        overall_group = QGroupBox(self._t("matrix.preview.score_overall"), group)
        overall_layout = QVBoxLayout(overall_group)
        overall_layout.setContentsMargins(0, 0, 0, 0)
        overall_layout.setSpacing(8)
        overall_host = QWidget(overall_group)
        overall_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        overall_host_layout = QVBoxLayout(overall_host)
        overall_host_layout.setContentsMargins(0, 0, 0, 0)
        overall_host_layout.setSpacing(8)
        overall_layout.addWidget(overall_host)

        component_group = QGroupBox(self._t("matrix.preview.score_components"), group)
        component_layout = QVBoxLayout(component_group)
        component_layout.setContentsMargins(0, 0, 0, 0)
        component_layout.setSpacing(8)
        component_host = QWidget(component_group)
        component_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        component_host_layout = QVBoxLayout(component_host)
        component_host_layout.setContentsMargins(0, 0, 0, 0)
        component_host_layout.setSpacing(8)
        component_layout.addWidget(component_host)

        for metric_key in metric_keys:
            if metric_key in seen_keys:
                continue
            seen_keys.add(metric_key)
            target_parent = overall_host if self._is_overall_score_metric(metric_key) else component_host
            card = _ExpandableScoreCard(self._metric_text_for_key(metric_key, build_result), target_parent)
            card.hide()
            card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            target_layout = overall_host_layout if self._is_overall_score_metric(metric_key) else component_host_layout
            target_layout.addWidget(card)
            score_cards[str(metric_key)] = card
            if self._is_overall_score_metric(metric_key):
                overall_card_count += 1
            else:
                component_card_count += 1
        overall_host_layout.addStretch(1)
        component_host_layout.addStretch(1)
        overall_group.setVisible(overall_card_count > 0)
        component_group.setVisible(component_card_count > 0)
        layout.addWidget(overall_group)
        layout.addWidget(component_group)
        layout.addStretch(1)
        return ExtendPreviewPanel(
            group=group,
            frame_title=frame_title,
            frame_value=frame_value,
            subpixel_group=subpixel_group,
            subpixel_value=subpixel_value,
            subpixel_score_card=subpixel_score_card,
            overall_group=overall_group,
            component_group=component_group,
            score_cards=score_cards,
            histogram_cards={},
        )

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.core.domain import BuildResult
