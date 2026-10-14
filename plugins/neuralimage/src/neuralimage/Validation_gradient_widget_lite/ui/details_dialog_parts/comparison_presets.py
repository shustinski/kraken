"""details_dialog: comparison presets."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QThread
from ...core.subpixel_grid import (
    SubpixelGrid,
    SubpixelGridSpec,
    build_subpixel_grid_from_pair,
    subpixel_spec_from_options,
)
from ...core.repository import (
    compute_comparison_score,
)
from ...core.workers import DetailPayloadWorker
from ...ui.ui_constants import DEFAULT_SUBPIXEL_AGGREGATION


class ComparisonPresetsMixin:
    def _comparison_presets(self, group_key: str | None = None) -> list[tuple[str, str, str, str]]:
        presets: list[tuple[str, str, str, str]] = []
        model_ids = [spec.model_id for spec in self._build_result.model_specs]
        selected_group = group_key or self._current_comparison_group()
        if selected_group == "model_model":
            for first_model_id in model_ids:
                for second_model_id in model_ids:
                    if first_model_id == second_model_id:
                        continue
                    presets.append((
                        f"model_vs_model::{first_model_id}::{second_model_id}",
                        f"{self._model_display_name(first_model_id)} vs {self._model_display_name(second_model_id)}",
                        f"model:{first_model_id}",
                        f"model:{second_model_id}",
                    ))
            return presets
        if selected_group == "model_labeled" and self._has_ground_truth():
            for model_id in model_ids:
                presets.append((
                    f"gt_vs_model::{model_id}",
                    f"{self._t("details.ground_truth")} vs {self._model_display_name(model_id)}",
                    "gt",
                    f"model:{model_id}",
                ))
            return presets
        for first_model_id in model_ids:
            for second_model_id in model_ids:
                if first_model_id == second_model_id:
                    continue
                presets.append((
                    f"model_vs_model::{first_model_id}::{second_model_id}",
                    f"{self._model_display_name(first_model_id)} vs {self._model_display_name(second_model_id)}",
                    f"model:{first_model_id}",
                    f"model:{second_model_id}",
                ))
        return presets

    def _refresh_comparison_presets(self, previous_key: str | None) -> None:
        presets = self._comparison_presets(self._current_comparison_group())
        self.comparison_preset_combo.blockSignals(True)
        self.comparison_preset_combo.clear()
        for key, label, source_a, source_b in presets:
            self.comparison_preset_combo.addItem(label, (key, source_a, source_b))
        target_key = previous_key
        if target_key is None and presets:
            target_key = presets[0][0]
        chosen_index = 0
        for index in range(self.comparison_preset_combo.count()):
            entry = self.comparison_preset_combo.itemData(index)
            if isinstance(entry, tuple) and len(entry) == 3 and entry[0] == target_key:
                chosen_index = index
                break
        if self.comparison_preset_combo.count() > 0:
            self.comparison_preset_combo.setCurrentIndex(chosen_index)
        self.comparison_preset_combo.blockSignals(False)

    def _current_comparison_tuple(self) -> tuple[str, str, str]:
        group_key = self._current_comparison_group()
        if group_key == "model_labeled" and self._has_ground_truth():
            return self._auto_model_labeled_tuple()
        if group_key == "model_model":
            return self._auto_model_model_tuple()
        return self._auto_model_model_tuple()

    def _start_loading_payload(self, *, reset_view: bool, preserve_selection: bool) -> None:
        self._stop_detail_worker()
        self._stop_confidence_worker()
        self._detail_request_generation += 1
        generation = int(self._detail_request_generation)
        self._payload_loading = True
        previous_result_kind = self._selected_result_kind() if preserve_selection else self._sticky_result_kind
        default_model_id = self._preferred_confidence_model_id() or (self._build_result.model_specs[0].model_id if self._build_result.model_specs else None)
        max_side = int(getattr(self._build_result.options, "analysis_max_side", 0) or 0) or None
        self.comparison_score_card.set_payload(
            self._t("details.frame_score"),
            "Loading...",
            self._comparison_score_style(None),
            "Loading frame details...",
        )
        self._refresh_scene(reset_view=reset_view)
        self._refresh_info()
        self._detail_thread = QThread(self)
        self._detail_worker = DetailPayloadWorker(self._record, self._build_result, default_model_id, max_side)
        self._detail_worker.moveToThread(self._detail_thread)
        self._detail_thread.started.connect(self._detail_worker.run)
        self._detail_worker.finished.connect(
            lambda payload, rv=reset_view, ps=preserve_selection, prk=previous_result_kind, g=generation: self._on_payload_loaded(
                payload,
                reset_view=rv,
                preserve_selection=ps,
                previous_result_kind=prk,
                generation=g,
            )
        )
        self._detail_worker.failed.connect(lambda message, g=generation: self._on_detail_worker_failed(message, generation=g, worker_kind="detail"))
        self._detail_worker.finished.connect(self._detail_thread.quit)
        self._detail_worker.failed.connect(self._detail_thread.quit)
        self._detail_thread.finished.connect(self._cleanup_detail_worker)
        self._detail_thread.start()

    def _on_payload_loaded(
        self,
        payload: dict[str, object],
        *,
        reset_view: bool,
        preserve_selection: bool,
        previous_result_kind: str | None,
        generation: int | None = None,
    ) -> None:
        if generation is not None and int(generation) != int(self._detail_request_generation):
            return
        self._payload_loading = False
        self._payload = payload
        self._overlay_cache.clear()
        self._derived_cache.clear()
        self._set_confidence_loading_state(False)
        preferred_kind = previous_result_kind if preserve_selection else (self._sticky_result_kind or self._restored_result_kind)
        self._refresh_result_kind_options(preferred_kind)
        self._restored_result_kind = None
        self._apply_selected_model(self._selected_model_for_current_comparison())
        self._refresh_scene(reset_view=reset_view)
        self._refresh_info()

    def _load_current_payload(self, *, reset_view: bool, preserve_selection: bool) -> None:
        self._start_loading_payload(reset_view=reset_view, preserve_selection=preserve_selection)

    def _default_subpixel_spec(self) -> SubpixelGridSpec:
        source_shape = None
        try:
            base = self._full_base_array()
            if isinstance(base, np.ndarray) and base.ndim == 2 and base.size > 0:
                source_shape = (int(base.shape[0]), int(base.shape[1]))
        except Exception:
            source_shape = None
        return subpixel_spec_from_options(self._build_result.options, source_shape)

    def _subpixel_grid_for_view(self) -> SubpixelGrid | None:
        selection = self._subpixel_selection
        if selection is None:
            return None
        spec = selection.spec or self._default_subpixel_spec()
        mode = self._current_operation_mode()
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        cache_key = (
            "subpixel_grid_view",
            str(self._record.key),
            str(first_key),
            str(second_key),
            str(mode.value),
            int(spec.rows),
            int(spec.columns),
            str(getattr(selection, "aggregation", DEFAULT_SUBPIXEL_AGGREGATION) or DEFAULT_SUBPIXEL_AGGREGATION),
        )
        cached = self._derived_cache.get(cache_key)
        if isinstance(cached, SubpixelGrid):
            return cached
        first, second = self._pairwise_comparison_inputs(first_key, second_key, mode)
        if first.ndim != 2 or second.ndim != 2 or first.size == 0 or second.size == 0 or first.shape != second.shape:
            return None
        grid = build_subpixel_grid_from_pair(
            first,
            second,
            spec,
            score_fn=lambda first_tile, second_tile: compute_comparison_score(first_tile, second_tile, mode),
            aggregation=str(getattr(selection, "aggregation", DEFAULT_SUBPIXEL_AGGREGATION) or DEFAULT_SUBPIXEL_AGGREGATION),
            value_kind="risk",
        )
        self._derived_cache[cache_key] = grid
        return grid
