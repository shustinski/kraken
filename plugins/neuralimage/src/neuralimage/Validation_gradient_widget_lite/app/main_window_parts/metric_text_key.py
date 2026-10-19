"""main_window: metric text key."""
from __future__ import annotations
from typing import TYPE_CHECKING
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QActionGroup
from kraken_core.theme import apply_app_theme, normalize_theme
from ...core.analysis_modes import ANALYSIS_MODE_OPTIONS, default_confidence_model_id
from ...ui.ui_constants import (
    CONFIDENCE_UNCERTAINTY_PROFILE_OPTIONS,
    DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE,
    DEFAULT_ANALYSIS_MODE,
    DEFAULT_GEOMETRY_MODE,
    DEFAULT_POLYGON_COMPARE_PROFILE,
    DEFAULT_POINT_EXTRACTION_MODE,
    DEFAULT_POLYGON_CONFIDENCE_SUMMARY,
    DEFAULT_MATRIX_LAYOUT_MODE,
    DEFAULT_MATRIX_SCORE_VIEW_MODE,
    DEFAULT_MATRIX_METRIC_KEY,
    EXTEND_WIDGET_STYLESHEET,
    GEOMETRY_MODE_OPTIONS,
    MATRIX_METRIC_OPTIONS,
    MATRIX_SCORE_VIEW_OPTIONS,
    POLYGON_CONFIDENCE_SUMMARY_OPTIONS,
    POLYGON_COMPARE_PROFILE_OPTIONS,
)


class MetricTextKeyMixin:
    def _setup_menu_bar(self) -> None:
        self._menu_bar.clear()
        theme_menu = self._menu_bar.addMenu("Тема")
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        self._theme_action_group = theme_group
        self._theme_dark_action = QAction("Темная", self)
        self._theme_dark_action.setCheckable(True)
        self._theme_dark_action.setData("dark")
        self._theme_light_action = QAction("Светлая", self)
        self._theme_light_action.setCheckable(True)
        self._theme_light_action.setData("light")
        for action in (self._theme_dark_action, self._theme_light_action):
            theme_group.addAction(action)
            theme_menu.addAction(action)
        self._theme_dark_action.triggered.connect(lambda: self._apply_theme("dark"))
        self._theme_light_action.triggered.connect(lambda: self._apply_theme("light"))
        self._sync_theme_menu_checks()
        self._menu_bar.setCornerWidget(self.language_toggle_button, Qt.Corner.TopRightCorner)

    def _apply_theme(self, theme: str) -> None:
        self._theme = normalize_theme(theme)
        apply_app_theme(self._theme)
        self.setStyleSheet(EXTEND_WIDGET_STYLESHEET if self._theme == "dark" else "")
        self._sync_theme_menu_checks()

    def _sync_theme_menu_checks(self) -> None:
        if hasattr(self, "_theme_dark_action"):
            self._theme_dark_action.setChecked(self._theme == "dark")
        if hasattr(self, "_theme_light_action"):
            self._theme_light_action.setChecked(self._theme == "light")

    def _populate_layout_mode_combo(self, selected_mode: str | None) -> None:
        current = str(selected_mode or DEFAULT_MATRIX_LAYOUT_MODE)
        self.layout_mode_combo.blockSignals(True)
        self.layout_mode_combo.clear()
        self.layout_mode_combo.addItem(self._t("matrix.layout.indexed"), "indexed_grid")
        self.layout_mode_combo.addItem(self._t("matrix.layout.manual"), "manual_grid")
        index = self.layout_mode_combo.findData(current)
        self.layout_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.layout_mode_combo.blockSignals(False)

    def _populate_matrix_score_view_combo(self, selected_mode: str | None) -> None:
        current = str(selected_mode or DEFAULT_MATRIX_SCORE_VIEW_MODE)
        self.matrix_score_view_combo.blockSignals(True)
        self.matrix_score_view_combo.clear()
        for label_key, key in MATRIX_SCORE_VIEW_OPTIONS:
            self.matrix_score_view_combo.addItem(self._t(label_key), key)
        index = self.matrix_score_view_combo.findData(current)
        self.matrix_score_view_combo.setCurrentIndex(index if index >= 0 else 0)
        self.matrix_score_view_combo.blockSignals(False)

    def _populate_analysis_mode_combo(self, selected_mode: str | None) -> None:
        current = str(selected_mode or DEFAULT_ANALYSIS_MODE)
        self.analysis_mode_combo.blockSignals(True)
        self.analysis_mode_combo.clear()
        for label_key, key in ANALYSIS_MODE_OPTIONS:
            self.analysis_mode_combo.addItem(self._t(label_key), key)
        index = self.analysis_mode_combo.findData(current)
        self.analysis_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.analysis_mode_combo.blockSignals(False)

    def _populate_geometry_mode_combo(self, selected_mode: str | None) -> None:
        current = str(selected_mode or DEFAULT_GEOMETRY_MODE)
        self.geometry_mode_combo.blockSignals(True)
        self.geometry_mode_combo.clear()
        labels = {
            "mask": self._t("geometry.mask"),
            "point": self._t("geometry.point"),
        }
        for label, key in GEOMETRY_MODE_OPTIONS:
            if str(key) == "auto":
                continue
            self.geometry_mode_combo.addItem(labels.get(str(key), str(label)), key)
        index = self.geometry_mode_combo.findData(current)
        self.geometry_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.geometry_mode_combo.blockSignals(False)

    def _populate_polygon_confidence_summary_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_POLYGON_CONFIDENCE_SUMMARY)
        self.polygon_confidence_summary_combo.blockSignals(True)
        self.polygon_confidence_summary_combo.clear()
        for label_key, key in POLYGON_CONFIDENCE_SUMMARY_OPTIONS:
            self.polygon_confidence_summary_combo.addItem(self._t(label_key), key)
        index = self.polygon_confidence_summary_combo.findData(current)
        self.polygon_confidence_summary_combo.setCurrentIndex(index if index >= 0 else 0)
        self.polygon_confidence_summary_combo.blockSignals(False)

    def _populate_polygon_compare_profile_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_POLYGON_COMPARE_PROFILE)
        self.polygon_compare_profile_combo.blockSignals(True)
        self.polygon_compare_profile_combo.clear()
        for label_key, key in POLYGON_COMPARE_PROFILE_OPTIONS:
            self.polygon_compare_profile_combo.addItem(self._t(label_key), key)
        index = self.polygon_compare_profile_combo.findData(current)
        if index < 0:
            index = self.polygon_compare_profile_combo.findData(DEFAULT_POLYGON_COMPARE_PROFILE)
        self.polygon_compare_profile_combo.setCurrentIndex(index if index >= 0 else 0)
        self.polygon_compare_profile_combo.blockSignals(False)

    def _populate_confidence_uncertainty_profile_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_CONFIDENCE_UNCERTAINTY_PROFILE)
        self.confidence_uncertainty_profile_combo.blockSignals(True)
        self.confidence_uncertainty_profile_combo.clear()
        for label_key, key in CONFIDENCE_UNCERTAINTY_PROFILE_OPTIONS:
            self.confidence_uncertainty_profile_combo.addItem(self._t(label_key), key)
        index = self.confidence_uncertainty_profile_combo.findData(current)
        self.confidence_uncertainty_profile_combo.setCurrentIndex(index if index >= 0 else 0)
        self.confidence_uncertainty_profile_combo.blockSignals(False)

    def _populate_metric_scope_combo(self, build_result: BuildResult | None, selected_scope: str | None) -> None:
        current = str(selected_scope or "")
        self.metric_scope_combo.blockSignals(True)
        self.metric_scope_combo.clear()
        if build_result is not None:
            for spec in build_result.model_specs:
                self.metric_scope_combo.addItem(str(spec.display_name), str(spec.model_id))
        if build_result is None and self.metric_scope_combo.count() <= 0 and current:
            self.metric_scope_combo.addItem(current, current)
        index = self.metric_scope_combo.findData(current)
        if index < 0 and build_result is not None:
            fallback = default_confidence_model_id(build_result)
            index = self.metric_scope_combo.findData(fallback)
        self.metric_scope_combo.setCurrentIndex(index if index >= 0 else 0)
        self.metric_scope_combo.setToolTip(self._t('analysis.confidence_model'))
        self.metric_scope_combo.blockSignals(False)

    def _metric_text_for_key(self, metric_key: str, build_result: BuildResult | None = None) -> str:
        metric_key_text = str(metric_key)
        for label_key, key, _group in MATRIX_METRIC_OPTIONS:
            if str(key) == metric_key_text:
                return self._t(label_key)
        translated = self._t(f"metric.{metric_key_text}")
        if translated != f"metric.{metric_key_text}":
            return translated
        if '::' in metric_key_text:
            family, model_id = metric_key_text.split('::', 1)
            model_name = model_id
            if build_result is not None:
                for spec in build_result.model_specs:
                    if spec.model_id == model_id:
                        model_name = spec.display_name
                        break
            if family == 'model_confidence':
                return f"{self._t('metric.model_confidence')} [{model_name}]"
            if family == 'model_uncertain_fraction':
                return f"{self._t('metric.model_uncertain_fraction')} [{model_name}]"
            if family == 'model_point_contrast':
                return f"{self._t('metric.model_point_contrast')} [{model_name}]"
        return metric_key_text

    @staticmethod
    def _is_overall_score_metric(metric_key: str) -> bool:
        return str(metric_key).startswith("overall_")

    def _populate_metric_combo(self, selected_metric_key: str | None) -> None:
        current = str(selected_metric_key or DEFAULT_MATRIX_METRIC_KEY)
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for label_key, key, _group in MATRIX_METRIC_OPTIONS:
            self.metric_combo.addItem(self._t(label_key), key)
        index = self.metric_combo.findData(current)
        self.metric_combo.setCurrentIndex(index if index >= 0 else 0)
        self.metric_combo.blockSignals(False)

    def _populate_frame_type_filter_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or 'all')
        self.frame_type_filter_combo.blockSignals(True)
        self.frame_type_filter_combo.clear()
        self.frame_type_filter_combo.addItem(self._t('frame_type.all'), 'all')
        self.frame_type_filter_combo.addItem(self._t('frame_type.polygon'), 'polygon')
        self.frame_type_filter_combo.addItem(self._t('frame_type.point'), 'point')
        index = self.frame_type_filter_combo.findData(current)
        self.frame_type_filter_combo.setCurrentIndex(index if index >= 0 else 0)
        self.frame_type_filter_combo.blockSignals(False)

    def _populate_point_extraction_mode_combo(self, selected_value: str | None) -> None:
        current = str(selected_value or DEFAULT_POINT_EXTRACTION_MODE)
        self.point_extraction_mode_combo.blockSignals(True)
        self.point_extraction_mode_combo.clear()
        self.point_extraction_mode_combo.addItem(self._t("point_extraction.component_centroids"), "component_centroids")
        self.point_extraction_mode_combo.addItem(self._t("point_extraction.local_maxima_legacy"), "local_maxima_legacy")
        index = self.point_extraction_mode_combo.findData(current)
        self.point_extraction_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        self.point_extraction_mode_combo.blockSignals(False)

if TYPE_CHECKING:
    from neuralimage.Validation_gradient_widget_lite.core.domain import BuildResult
