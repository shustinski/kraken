"""details_dialog: update tile lod."""
from __future__ import annotations

from neuralimage.Validation_gradient_widget_lite.ui.details_dialog_parts.module_state import DETAIL_PIXEL_VIEW_THRESHOLD


class UpdateTileLodMixin:
    def _update_tile_hint(self) -> None:
        plan = self._tile_plan
        if plan is None and self._subpixel_selection is None:
            self.zoom_hint_label.hide()
            return
        if self._subpixel_selection is not None:
            selection = self._subpixel_selection
            spec = selection.spec or self._default_subpixel_spec()
            hint = self._t(
                "details.subpixel_view.hint",
                parent_row=int(selection.parent_row) + 1,
                parent_column=int(selection.parent_column) + 1,
                sub_row=int(selection.sub_row) + 1,
                sub_column=int(selection.sub_column) + 1,
                rows=int(spec.rows),
                columns=int(spec.columns),
                value=float(selection.subpixel_value),
                parent_value=float(selection.parent_value),
            )
            self.zoom_hint_label.setText(hint)
            self.zoom_hint_label.show()
            return
        scale = max(0.01, float(self.overlay_view.transform().m11()))
        tile_screen_size = min(int(plan.tile_width), int(plan.tile_height)) * scale if plan is not None else 0.0
        mode_text = self._t("details.tile_view.zoomed_in") if tile_screen_size >= DETAIL_PIXEL_VIEW_THRESHOLD else self._t("details.tile_view.zoomed_out")
        hint = self._t(
            "details.tile_view.hint",
            width=int(plan.tile_width) if plan is not None else 0,
            height=int(plan.tile_height) if plan is not None else 0,
            overlap=int(plan.overlap) if plan is not None else 0,
            mode=mode_text,
        )
        if self._tile_selection is not None:
            row, column = self._tile_selection
            hint = f"{hint} | {self._t('details.tile_view.selected')}: r{row + 1}, c{column + 1}"
        self.zoom_hint_label.setText(hint)
        self.zoom_hint_label.show()

    def _update_tile_lod(self) -> None:
        if self._subpixel_selection is not None:
            self.original_item.setVisible(True)
            self.tile_preview_item.setVisible(False)
            self.tile_grid_item.setVisible(False)
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        plan = self._tile_plan
        if plan is None:
            self.tile_preview_item.setVisible(False)
            self.tile_grid_item.setVisible(False)
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        if self._tile_selection is not None:
            self.tile_preview_item.setVisible(False)
            self.tile_grid_item.setVisible(False)
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        base_visible = self.original_item.isVisible()
        if not base_visible:
            self.original_item.setVisible(False)
            self.tile_preview_item.setVisible(False)
            self.tile_grid_item.setVisible(False)
            self.tile_selection_item.setVisible(False)
            self._update_tile_hint()
            return
        scale = max(0.01, float(self.overlay_view.transform().m11()))
        tile_screen_size = min(int(plan.tile_width), int(plan.tile_height)) * scale
        show_pixel_view = tile_screen_size >= DETAIL_PIXEL_VIEW_THRESHOLD
        self.original_item.setVisible(show_pixel_view)
        self.tile_preview_item.setVisible(not show_pixel_view)
        self.tile_preview_item.setOpacity(self.original_item.opacity())
        self.tile_grid_item.setVisible(show_pixel_view)
        self.tile_grid_item.setOpacity(0.85)
        if self._tile_selection is not None and show_pixel_view:
            row, column = self._tile_selection
            rect = self._tile_rect_for_index(row, column, plan)
            self.tile_selection_item.setRect(rect)
            self.tile_selection_item.setVisible(True)
            self.tile_selection_item.setOpacity(1.0)
        else:
            self.tile_selection_item.setVisible(False)
        self._update_tile_hint()
