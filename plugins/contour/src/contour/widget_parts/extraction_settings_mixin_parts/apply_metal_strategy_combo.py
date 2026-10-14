"""extraction_settings_mixin: apply metal strategy combo."""
from __future__ import annotations
from contour.widget_parts._imports import (
    AutoTuneResult,
    ContourExtractionSettings,
    PreprocessingPipeline,
    QCheckBox,
    QComboBox,
    QSignalBlocker,
)


class ApplyMetalStrategyComboMixin:
    def _apply_auto_tune_result(self, result: AutoTuneResult) -> None:
        current_settings = self._current_contour_settings()
        tuned_settings = ContourExtractionSettings.from_dict(result.contour_settings.to_dict())
        for field in (
            "via_white_range_enabled",
            "via_white_range_min",
            "via_white_range_max",
            "via_black_range_enabled",
            "via_black_range_min",
            "via_black_range_max",
            "bright_via_min_final_score",
        ):
            setattr(tuned_settings, field, getattr(current_settings, field))
        self._pipeline = PreprocessingPipeline.from_dict(result.pipeline_config)
        self._populate_pipeline_list()
        self._set_extraction_settings(tuned_settings)
        self.process_current_image()

    def _on_via_output_diameter_changed(self, value: int) -> None:
        self._store_active_extraction_profile_settings()
        if hasattr(self, "_redraw_recognized_contacts_with_diameter"):
            self._redraw_recognized_contacts_with_diameter(int(value))

    def _metal_strategy_from_combo(self) -> str:
        from ...vision.metal_recovery.segmentation import normalize_metal_segmentation_strategy

        if not hasattr(self, "metal_segmentation_strategy_combo"):
            return "legacy_otsu"
        return normalize_metal_segmentation_strategy(
            self.metal_segmentation_strategy_combo.currentData() or "legacy_otsu"
        )

    def _apply_metal_strategy_to_combo(self, settings: object) -> None:
        from ...vision.metal_recovery.segmentation import resolve_metal_segmentation_strategy

        if not hasattr(self, "metal_segmentation_strategy_combo"):
            return
        strategy = resolve_metal_segmentation_strategy(
            getattr(settings, "metal_segmentation_strategy", "legacy_otsu"),
            use_wide_conductor_gradient=bool(getattr(settings, "metal_use_wide_conductor_gradient", False)),
        )
        index = self.metal_segmentation_strategy_combo.findData(strategy)
        if index < 0:
            index = self.metal_segmentation_strategy_combo.findData("legacy_otsu")
        if index >= 0:
            self.metal_segmentation_strategy_combo.setCurrentIndex(index)
        if hasattr(self, "metal_auto_contrast_step_spin"):
            self.metal_auto_contrast_step_spin.setEnabled(strategy == "auto")
        if hasattr(self, "metal_auto_source_contrast_step_spin"):
            self.metal_auto_source_contrast_step_spin.setEnabled(strategy == "auto")
        if hasattr(self, "metal_auto_directional_gap_bridge_spin"):
            self.metal_auto_directional_gap_bridge_spin.setEnabled(strategy == "auto")
        if hasattr(self, "metal_auto_directional_gap_min_source_spin"):
            self.metal_auto_directional_gap_min_source_spin.setEnabled(strategy == "auto")

    def _metal_strategy_parameters_from_widgets(self) -> dict[str, dict[str, object]]:
        from ...vision.metal_recovery.strategy_registry import MetalStrategyConfigs

        raw: dict[str, dict[str, object]] = {}
        for strategy, controls in getattr(self, "metal_strategy_parameter_widgets", {}).items():
            values: dict[str, object] = {}
            for key, widget in controls.items():
                if isinstance(widget, QCheckBox):
                    values[key] = widget.isChecked()
                elif isinstance(widget, QComboBox):
                    values[key] = str(widget.currentData())
                else:
                    values[key] = widget.value()
            raw[strategy] = values
        return {strategy: dict(values) for strategy, values in MetalStrategyConfigs.from_mapping(raw).to_dict().items()}

    def _set_metal_strategy_parameter_widgets(self, raw: object) -> None:
        from ...vision.metal_recovery.strategy_registry import MetalStrategyConfigs

        mapping = raw if isinstance(raw, dict) else None
        configs = MetalStrategyConfigs.from_mapping(mapping)
        for strategy, controls in getattr(self, "metal_strategy_parameter_widgets", {}).items():
            values = configs.for_strategy(strategy)
            for key, widget in controls.items():
                blocker = QSignalBlocker(widget)
                value = values[key]
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    if index >= 0:
                        widget.setCurrentIndex(index)
                else:
                    widget.setValue(value)
                del blocker
