"""builders: sync metal strategy panel."""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QFormLayout,
)
from ...vision.metal_recovery.strategy_registry import (
    IMPLEMENTED_NEW_STRATEGIES,
    strategy_spec,
)
from contour.ui.metal_strategy_i18n import (
    strategy_description,
    strategy_name,
)

def _sync_metal_strategy_panel(self) -> None:
    strategy = str(self.metal_segmentation_strategy_combo.currentData() or "legacy_otsu")
    spec = strategy_spec(strategy)
    language = str(getattr(self, "_ui_language", "ru") or "ru")
    self.metal_strategy_description_label.setText(strategy_description(spec, language))
    page_index = self.metal_strategy_parameter_pages.get(strategy)
    self.metal_strategy_parameters_group.setVisible(page_index is not None)
    if page_index is not None:
        if hasattr(self, "metal_advanced_group"):
            self.metal_advanced_group.setChecked(True)
        self.metal_strategy_parameter_stack.setCurrentIndex(page_index)
        title = strategy_name(spec, language)
        self.metal_strategy_parameters_group.setTitle(
            f"Параметры: {title}" if language == "ru" else f"{title} parameters"
        )

    new_strategy = strategy in IMPLEMENTED_NEW_STRATEGIES
    seeded_strategy = strategy in {
        "gradient_watershed",
        "random_walker",
        "graph_cut",
        "reconstruction",
        "closed_boundary",
        "structural_watershed",
        *IMPLEMENTED_NEW_STRATEGIES,
    }
    basic_form = self.metal_basic_group.layout()
    if isinstance(basic_form, QFormLayout):
        basic_form.setRowVisible(self.metal_min_contrast_widget, not seeded_strategy)
        basic_form.setRowVisible(self.metal_gap_bridge_spin, not seeded_strategy)
        basic_form.setRowVisible(self.metal_speckle_removal_spin, not new_strategy)
        for auto_control in (
            self.metal_auto_contrast_step_spin,
            self.metal_auto_source_contrast_step_spin,
            self.metal_auto_directional_gap_bridge_spin,
            self.metal_auto_directional_gap_min_source_spin,
        ):
            basic_form.setRowVisible(auto_control, strategy == "auto")

    self.metal_adaptive_group.setVisible(strategy in {"auto", "local_adaptive"})
    self.metal_watershed_group.setVisible(
        strategy
        in {
            "auto",
            "gradient_watershed",
            "random_walker",
            "graph_cut",
            "reconstruction",
            "closed_boundary",
            "structural_watershed",
        }
    )
    self.metal_random_walker_group.setVisible(strategy == "random_walker")
    self.metal_graph_cut_group.setVisible(strategy == "graph_cut")
    self.metal_reconstruction_group.setVisible(strategy == "reconstruction")
    self.metal_closed_boundary_group.setVisible(strategy == "closed_boundary")
