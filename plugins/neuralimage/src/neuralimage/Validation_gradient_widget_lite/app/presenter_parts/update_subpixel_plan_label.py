"""presenter: update subpixel plan label."""
from __future__ import annotations
from pathlib import Path
from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtGui import QImageReader
from PyQt6.QtWidgets import QListWidgetItem
from ...core.domain import BuildOptions, BuildResult
from ...core.repository import (
    load_grayscale_image,
)
from ...core.subpixel_grid import SubpixelGridSpec, subpixel_spec_from_options
from ...core.tile_grid import TileGridPlan, plan_tile_grid
from ...ui.matrix_view import MatrixLayoutConfig
from ...ui.ui_constants import (
    DEFAULT_BOUNDARY_RADIUS,
    DEFAULT_MASK_THRESHOLD,
    DEFAULT_SUBPIXEL_AGGREGATION,
    DEFAULT_SUBPIXEL_VIEW_MODE,
    DEFAULT_POLYGON_COMPARE_PROFILE,
    DEFAULT_TILE_OVERLAP_MODE,
    FOLDER_CHECKED_ROLE,
    FOLDER_CONFIDENCE_ROLE,
    FOLDER_LABEL_ROLE,
)


class UpdateSubpixelPlanLabelMixin:
    def _selected_polygon_compare_values(self) -> tuple[float, int]:
        values = self._polygon_compare_values_for_profile(self._selected_polygon_compare_profile())
        if values is None:
            values = self._polygon_compare_values_for_profile(DEFAULT_POLYGON_COMPARE_PROFILE)
        if values is None:
            return float(DEFAULT_MASK_THRESHOLD), int(DEFAULT_BOUNDARY_RADIUS)
        return values

    def _selected_tile_mode(self) -> str:
        return "tile" if self._selected_subpixel_view_mode() == "tile" else "pixel"

    def _selected_subpixel_view_mode(self) -> str:
        value = str(self.subpixel_view_mode_combo.currentData() or DEFAULT_SUBPIXEL_VIEW_MODE)
        if value == "tile":
            return "tile"
        return "pixel"

    def _selected_subpixel_rows(self) -> int:
        return int(self.subpixel_rows_spin.value())

    def _selected_subpixel_columns(self) -> int:
        return int(self.subpixel_columns_spin.value())

    def _selected_subpixel_aggregation(self) -> str:
        return str(self.subpixel_aggregation_combo.currentData() or DEFAULT_SUBPIXEL_AGGREGATION)

    def _selected_tile_width(self) -> int:
        return int(self.tile_width_spin.value())

    def _selected_tile_height(self) -> int:
        return int(self.tile_height_spin.value())

    def _selected_tile_overlap_mode(self) -> str:
        return str(self.tile_overlap_mode_combo.currentData() or DEFAULT_TILE_OVERLAP_MODE)

    def _selected_tile_overlap(self) -> int:
        return int(self.tile_overlap_spin.value())

    def _sync_tile_overlap_bounds(self) -> None:
        maximum = max(0, min(self._selected_tile_width(), self._selected_tile_height()) - 1)
        self.tile_overlap_spin.setMaximum(int(maximum))
        if int(self.tile_overlap_spin.value()) > maximum:
            self.tile_overlap_spin.setValue(int(maximum))

    @staticmethod
    def _set_row_enabled(row: object | None, enabled: bool) -> None:
        if row is not None and hasattr(row, "setEnabled"):
            row.setEnabled(bool(enabled))

    @staticmethod
    def _image_shape(path: Path) -> tuple[int, int] | None:
        try:
            reader = QImageReader(str(path))
            size = reader.size()
            if size.isValid() and size.width() > 0 and size.height() > 0:
                return int(size.height()), int(size.width())
        except Exception:
            pass
        try:
            return tuple(int(v) for v in load_grayscale_image(path).shape)
        except Exception:
            return None

    @staticmethod
    def _first_image_shape(folder_path: Path) -> tuple[int, int] | None:
        from neuralimage.Validation_gradient_widget_lite.app.presenter import ValidationGradientExtendPresenter

        allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        try:
            if folder_path.is_file() and folder_path.suffix.lower() in allowed:
                return ValidationGradientExtendPresenter._image_shape(folder_path)
            for image_path in Path(folder_path).rglob("*"):
                if not image_path.is_file() or image_path.suffix.lower() not in allowed:
                    continue
                shape = ValidationGradientExtendPresenter._image_shape(image_path)
                if shape is not None:
                    return shape
        except Exception:
            return None
        return None

    def _tile_source_shape_for_result(self, build_result: BuildResult | None) -> tuple[int, int] | None:
        if build_result is None or not build_result.records:
            return None
        for record in build_result.records:
            candidate_path = record.original_path or record.base_path
            if not candidate_path and record.model_mask_paths:
                candidate_path = next(iter(record.model_mask_paths.values()), None)
            if not candidate_path:
                continue
            shape = self._image_shape(Path(candidate_path))
            if shape is not None:
                return shape
        original_folder = getattr(build_result, "original_folder", None)
        if original_folder is not None:
            shape = self._first_image_shape(Path(original_folder.path))
            if shape is not None:
                return shape
        return None

    def _tile_grid_plan_for_result(self, build_result: BuildResult | None) -> TileGridPlan | None:
        if self._selected_tile_mode() != "tile":
            return None
        source_shape = self._tile_source_shape_for_result(build_result)
        if source_shape is None:
            return None
        return plan_tile_grid(
            source_shape,
            self._selected_tile_width(),
            self._selected_tile_height(),
            self._selected_tile_overlap(),
        )

    def _tile_grid_layout_config_for_result(self, build_result: BuildResult | None) -> MatrixLayoutConfig | None:
        plan = self._tile_grid_plan_for_result(build_result)
        if plan is None or not plan.applied_exact:
            return None
        return MatrixLayoutConfig(
            mode="manual_grid",
            total_frames=max(1, int(plan.rows * plan.columns)),
            frames_per_row=max(1, int(plan.columns)),
            rows=max(1, int(plan.rows)),
            columns=max(1, int(plan.columns)),
        )

    def _subpixel_grid_spec(self, build_result: BuildResult | None = None) -> SubpixelGridSpec | None:
        if self._selected_subpixel_view_mode() != "tile":
            return None
        source_shape = self._tile_source_shape_for_result(build_result)
        options = BuildOptions(
            subpixel_view_mode="tile",
            subpixel_rows=max(1, self._selected_subpixel_rows()),
            subpixel_columns=max(1, self._selected_subpixel_columns()),
            tile_width=self._selected_tile_width(),
            tile_height=self._selected_tile_height(),
            tile_overlap=self._selected_tile_overlap(),
        )
        return subpixel_spec_from_options(options, source_shape)

    def _update_subpixel_plan_label(self, build_result: BuildResult | None) -> None:
        if self._selected_subpixel_view_mode() != "tile":
            self.tile_plan_label.setText(self._t("matrix.subpixel_plan.unavailable"))
            self.tile_plan_label.setToolTip("")
            self.tile_plan_label.setStyleSheet("")
            return
        spec = self._subpixel_grid_spec(build_result)
        if spec is None:
            self.tile_plan_label.setText(self._t("matrix.subpixel_plan.waiting"))
            self.tile_plan_label.setToolTip("")
            self.tile_plan_label.setStyleSheet("")
            return
        aggregation_label = self._t(f"subpixel_aggregation.{self._selected_subpixel_aggregation()}")
        self.tile_plan_label.setText(
            f"{int(spec.rows)}x{int(spec.columns)} | "
            f"{int(spec.tile_width)}x{int(spec.tile_height)} | "
            f"ovl {int(spec.overlap)} | {aggregation_label}"
        )
        self.tile_plan_label.setToolTip(self._t("matrix.subpixel_plan.tooltip"))
        self.tile_plan_label.setStyleSheet("")

    def _apply_polygon_compare_profile(self, profile_key: str | None) -> None:
        profile = str(profile_key or DEFAULT_POLYGON_COMPARE_PROFILE)
        values = self._polygon_compare_values_for_profile(profile)
        if values is None:
            values = self._polygon_compare_values_for_profile(DEFAULT_POLYGON_COMPARE_PROFILE)
        if values is None:
            return
        mask_threshold, boundary_radius = values
        blockers = [
            QSignalBlocker(self.polygon_compare_profile_combo),
        ]
        _ = blockers
        self.mask_threshold_spin.setValue(float(mask_threshold))
        self.boundary_radius_spin.setValue(int(boundary_radius))
        profile_index = self.polygon_compare_profile_combo.findData(profile)
        if profile_index < 0:
            profile_index = self.polygon_compare_profile_combo.findData(DEFAULT_POLYGON_COMPARE_PROFILE)
        self.polygon_compare_profile_combo.setCurrentIndex(profile_index if profile_index >= 0 else 0)

    def _append_folder_item(self, folder_path: Path, *, checked: bool) -> QListWidgetItem:
        folder_path = Path(folder_path)
        for row in range(self.folder_list.count()):
            existing_item = self.folder_list.item(row)
            if Path(existing_item.data(Qt.ItemDataRole.UserRole)) == folder_path:
                existing_item.setData(FOLDER_CHECKED_ROLE, bool(checked))
                if not existing_item.data(FOLDER_LABEL_ROLE):
                    existing_item.setData(FOLDER_LABEL_ROLE, folder_path.name)
                return existing_item
        item = QListWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        item.setData(Qt.ItemDataRole.UserRole, str(folder_path))
        item.setData(FOLDER_CHECKED_ROLE, bool(checked))
        item.setData(FOLDER_LABEL_ROLE, folder_path.name)
        item.setData(FOLDER_CONFIDENCE_ROLE, "")
        item.setToolTip(str(folder_path))
        self.folder_list.addItem(item)
        return item
