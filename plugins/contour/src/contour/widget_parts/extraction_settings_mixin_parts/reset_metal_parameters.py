"""extraction_settings_mixin: reset metal parameters."""
from __future__ import annotations
from contour.widget_parts._imports import (
    ContourExtractionSettings,
)


class ResetMetalParametersMixin:
    def _reset_metal_parameters(self, *_args) -> None:
        defaults = ContourExtractionSettings()
        if hasattr(self, "metal_preset_combo"):
            label = "Стандартный" if self._ui_language == "ru" else "Standard"
            ix = self.metal_preset_combo.findText(label)
            if ix >= 0:
                self.metal_preset_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_min_contrast_slider"):
            self.metal_min_contrast_slider.setValue(round(defaults.metal_min_contrast))
        if hasattr(self, "metal_min_object_source_contrast_spin"):
            self.metal_min_object_source_contrast_spin.setValue(
                float(defaults.metal_min_object_source_contrast)
            )
        if hasattr(self, "metal_min_object_rim_contrast_spin"):
            self.metal_min_object_rim_contrast_spin.setValue(
                float(defaults.metal_min_object_rim_contrast)
            )
        if hasattr(self, "metal_min_object_rim_area_fraction_spin"):
            self.metal_min_object_rim_area_fraction_spin.setValue(
                float(defaults.metal_min_object_rim_area_fraction)
            )
        if hasattr(self, "metal_auto_contrast_step_spin"):
            self.metal_auto_contrast_step_spin.setValue(float(defaults.metal_auto_contrast_step))
        if hasattr(self, "metal_auto_source_contrast_step_spin"):
            self.metal_auto_source_contrast_step_spin.setValue(
                float(defaults.metal_auto_source_contrast_step)
            )
        if hasattr(self, "metal_auto_directional_gap_bridge_spin"):
            self.metal_auto_directional_gap_bridge_spin.setValue(
                int(defaults.metal_auto_directional_gap_bridge_px)
            )
        if hasattr(self, "metal_auto_directional_gap_min_source_spin"):
            self.metal_auto_directional_gap_min_source_spin.setValue(
                float(defaults.metal_auto_directional_gap_min_source_intensity)
            )
        if hasattr(self, "metal_gap_bridge_spin"):
            self.metal_gap_bridge_spin.setValue(int(defaults.metal_gap_bridge_px))
        if hasattr(self, "metal_speckle_removal_spin"):
            self.metal_speckle_removal_spin.setValue(int(defaults.metal_speckle_removal_px))
        if hasattr(self, "metal_epsilon_spin"):
            self.metal_epsilon_spin.setValue(float(defaults.epsilon))
        if hasattr(self, "metal_min_width_spin"):
            self.metal_min_width_spin.setValue(float(defaults.metal_min_trace_width_px))
        if hasattr(self, "metal_max_width_spin"):
            mw = defaults.metal_max_trace_width_px
            self.metal_max_width_spin.setValue(0.0 if mw is None else float(mw))
        if hasattr(self, "metal_conductor_size_offset_slider"):
            self.metal_conductor_size_offset_slider.setValue(
                round(float(defaults.metal_conductor_size_offset_px))
            )
        if hasattr(self, "metal_min_area_spin"):
            self.metal_min_area_spin.setValue(float(defaults.metal_min_area))
        if hasattr(self, "metal_max_area_spin"):
            ma = defaults.metal_max_area
            self.metal_max_area_spin.setValue(0.0 if ma is None else float(ma))
        if hasattr(self, "metal_min_perimeter_spin"):
            self.metal_min_perimeter_spin.setValue(float(defaults.metal_min_perimeter))
        if hasattr(self, "metal_max_perimeter_spin"):
            mp = defaults.metal_max_perimeter
            self.metal_max_perimeter_spin.setValue(0.0 if mp is None else float(mp))
        if hasattr(self, "metal_min_length_spin"):
            self.metal_min_length_spin.setValue(float(defaults.metal_min_trace_length_px))
        if hasattr(self, "metal_min_points_spin"):
            self.metal_min_points_spin.setValue(int(defaults.min_points))
        if hasattr(self, "metal_min_angle_spin"):
            self.metal_min_angle_spin.setValue(float(defaults.min_polygon_angle))
        if hasattr(self, "metal_min_hole_source_contrast_spin"):
            self.metal_min_hole_source_contrast_spin.setValue(defaults.metal_min_hole_source_contrast)
        if hasattr(self, "metal_min_hole_source_contrast_fraction_spin"):
            self.metal_min_hole_source_contrast_fraction_spin.setValue(
                defaults.metal_min_hole_source_contrast_fraction
            )
        if hasattr(self, "metal_approximation_checkbox"):
            self.metal_approximation_checkbox.setChecked(bool(defaults.metal_approximation_enabled))
        if hasattr(self, "metal_segmentation_strategy_combo"):
            self._apply_metal_strategy_to_combo(defaults)
        if hasattr(self, "metal_strategy_parameter_widgets"):
            self._set_metal_strategy_parameter_widgets(
                getattr(defaults, "metal_strategy_parameters", None)
            )
        if hasattr(self, "metal_ws_smoothing_spin"):
            self.metal_ws_smoothing_spin.setValue(float(defaults.metal_watershed_smoothing_sigma))
        if hasattr(self, "metal_ws_core_margin_spin"):
            self.metal_ws_core_margin_spin.setValue(float(defaults.metal_watershed_core_margin))
        if hasattr(self, "metal_ws_groove_margin_spin"):
            self.metal_ws_groove_margin_spin.setValue(float(defaults.metal_watershed_groove_margin))
        if hasattr(self, "metal_ws_rim_probe_spin"):
            self.metal_ws_rim_probe_spin.setValue(int(defaults.metal_watershed_rim_probe_px))
        if hasattr(self, "metal_ws_seed_speckle_spin"):
            self.metal_ws_seed_speckle_spin.setValue(int(defaults.metal_watershed_seed_speckle_px))
        if hasattr(self, "metal_ws_valley_span_spin"):
            self.metal_ws_valley_span_spin.setValue(int(defaults.metal_watershed_valley_span_px))
        if hasattr(self, "metal_ws_valley_depth_spin"):
            self.metal_ws_valley_depth_spin.setValue(float(defaults.metal_watershed_valley_depth))
        if hasattr(self, "metal_rw_beta_spin"):
            self.metal_rw_beta_spin.setValue(float(defaults.metal_random_walker_beta))
        if hasattr(self, "metal_rw_iterations_spin"):
            self.metal_rw_iterations_spin.setValue(int(defaults.metal_random_walker_iterations))
        if hasattr(self, "metal_gc_iterations_spin"):
            self.metal_gc_iterations_spin.setValue(int(defaults.metal_graph_cut_iterations))
        if hasattr(self, "metal_recon_erode_spin"):
            self.metal_recon_erode_spin.setValue(int(defaults.metal_reconstruction_erode_px))
        if hasattr(self, "metal_boundary_relief_spin"):
            self.metal_boundary_relief_spin.setValue(float(defaults.metal_boundary_relief))
        if hasattr(self, "metal_boundary_background_spin"):
            self.metal_boundary_background_spin.setValue(
                float(defaults.metal_boundary_background_sigma)
            )
        if hasattr(self, "metal_adaptive_block_spin"):
            self.metal_adaptive_block_spin.setValue(int(defaults.metal_adaptive_block_size))
        if hasattr(self, "metal_adaptive_c_spin"):
            self.metal_adaptive_c_spin.setValue(float(defaults.metal_adaptive_c))
        if hasattr(self, "metal_adaptive_method_combo"):
            ix = self.metal_adaptive_method_combo.findData(defaults.metal_adaptive_method)
            if ix >= 0:
                self.metal_adaptive_method_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_hierarchy_combo"):
            ix = self.metal_hierarchy_combo.findData(defaults.metal_hierarchy_mode)
            if ix >= 0:
                self.metal_hierarchy_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_border_handling_combo"):
            ix = self.metal_border_handling_combo.findData(defaults.metal_border_handling)
            if ix >= 0:
                self.metal_border_handling_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_show_conductors_checkbox"):
            self.metal_show_conductors_checkbox.setChecked(bool(defaults.metal_display_show_conductors))
        if hasattr(self, "metal_show_rejected_checkbox"):
            self.metal_show_rejected_checkbox.setChecked(bool(defaults.metal_display_show_rejected))
        if hasattr(self, "metal_show_suspicious_checkbox"):
            self.metal_show_suspicious_checkbox.setChecked(bool(defaults.metal_display_show_suspicious))
        if hasattr(self, "metal_show_border_checkbox"):
            self.metal_show_border_checkbox.setChecked(bool(defaults.metal_display_show_border_highlight))
        if hasattr(self, "metal_show_mask_checkbox"):
            self.metal_show_mask_checkbox.setChecked(bool(defaults.metal_display_show_mask))
        if hasattr(self, "metal_debug_visual_combo"):
            ix = self.metal_debug_visual_combo.findData(defaults.metal_debug_visual)
            if ix >= 0:
                self.metal_debug_visual_combo.setCurrentIndex(ix)
        if hasattr(self, "metal_overlay_opacity_spin"):
            self.metal_overlay_opacity_spin.setValue(float(defaults.metal_overlay_opacity))
        self._on_extraction_settings_changed()
