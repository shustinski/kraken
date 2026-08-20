from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from ..domain import compute_polygon_metrics
from ._imports import *  # noqa: F403


class WidgetDebugMixin:
    if TYPE_CHECKING:
        _workspace: WorkspaceSession
        _show_source_while_middle_held: bool
        _show_source_while_filter_hotkey_held: bool

        control_tabs: QTabWidget
        gradient_overlay_mode_combo: QComboBox
        metal_debug_visual_combo: QComboBox
        metal_gradient_3d_button: QPushButton
        metal_overlay_opacity_spin: QDoubleSpinBox
        metal_show_mask_checkbox: QCheckBox
        pipeline_tab: QWidget
        polygon_editor: Any
        recognition_mode_combo: QComboBox
        _ui_language: str

        def _append_log(self, message: str) -> None: ...
        def _refresh_current_display_image_only(self, *, preserve_view: bool = True) -> None: ...
        def _current_contour_settings(self) -> ContourExtractionSettings: ...
        def _sync_current_state_views(
            self,
            *,
            preserve_view: bool = False,
            sync_neighbors: bool = True,
        ) -> None: ...
        def _tr(self, key: str, default: str = "", **kwargs: object) -> str: ...

    def _debug_parent_widget(self) -> QWidget:
        return cast(QWidget, self)

    def _on_via_debug_requested(self, polygon: PolygonData) -> None:
        current_state = self._workspace.current_state
        candidates = list(current_state.debug_candidates) if current_state is not None else []
        is_via_like = (polygon.shape_hint or "") == "box" or (polygon.category or "") == "via"
        if not is_via_like:
            title = "Свойства контура" if self._ui_language == "ru" else "Contour properties"
            message = "\n".join(self._conductor_property_lines(polygon))
            self._append_log(message.replace("\n", " | "))
            self._show_nonblocking_via_debug_message(title, message)
            return
        title = self._tr("debug.via_title")
        is_manual_via = bool(is_via_like and polygon.recognition_score is None)
        candidate = self._best_debug_candidate_for_polygon(polygon, candidates)
        if candidate is None and is_via_like:
            candidate = self._manual_via_debug_candidate(polygon)
        if candidate is None:
            message = self._tr(
                "debug.no_source_candidate" if candidates or is_manual_via else "debug.no_current_frame_data"
            )
            self._show_nonblocking_via_debug_message(title, message)
            return
        source = self._debug_candidate_source(candidate)
        reason = str(getattr(candidate, "reason", "") or "")
        accepted = bool(getattr(candidate, "accepted", False))
        bbox = getattr(candidate, "bbox", (0, 0, 0, 0))
        template_index = getattr(candidate, "template_index", None)
        is_template_match = is_via_like and template_index is not None
        status = self._tr("debug.status_accepted" if accepted else "debug.status_rejected")
        lines = [
            f"{self._tr('debug.field_status')}: {status}",
            f"{self._tr('debug.field_method')}: {self._debug_method_text(source)}",
            f"{self._tr('debug.field_criterion')}: {self._debug_criterion_text(source, reason, accepted)}",
        ]
        if is_template_match:
            similarity = max(0.0, min(1.0, float(getattr(candidate, "score", 0.0)) / 100.0))
            similarity_text = f"{similarity:.3f}"
            if self._ui_language == "ru":
                similarity_text = similarity_text.replace(".", ",")
            lines += [
                f"{self._tr('debug.field_template_number')}: {int(template_index) + 1}",
                f"{self._tr('debug.field_similarity')}: {similarity_text}",
            ]
        elif is_via_like:
            lines.append(f"{self._tr('debug.field_reason')}: {reason or '-'}")
            metric_lines = self._heuristic_via_diagnostic_lines(candidate)
            if metric_lines:
                lines.extend(metric_lines)
            else:
                lines.append(
                    f"{self._tr('debug.field_score')}: {float(getattr(candidate, 'score', 0.0)):.1f}"
                )
        else:
            lines.append(f"{self._tr('debug.field_reason')}: {reason or '-'}")
            area_v = float(getattr(candidate, "area", 0.0) or 0.0)
            per_v = float(getattr(candidate, "perimeter", 0.0) or 0.0)
            ew = float(getattr(candidate, "effective_width", 0.0) or 0.0)
            wm = str(getattr(candidate, "width_metric", "") or "")
            wline = f"{self._tr('debug.field_width_estimate')}: {ew:.2f} px"
            if wm:
                wline += f" ({wm})"
            lines += [
                f"{self._tr('debug.field_area')}: {area_v:.1f} px²",
                f"{self._tr('debug.field_perimeter')}: {per_v:.1f} px",
                wline,
            ]
        lines += [
            f"{self._tr('debug.field_candidate_size')}: {int(bbox[2])} x {int(bbox[3])} px",
            f"{self._tr('debug.field_position')}: x={int(bbox[0])}, y={int(bbox[1])}",
        ]
        message = "\n".join(lines)
        self._append_log(message.replace("\n", " | "))
        self._show_nonblocking_via_debug_message(title, message)

    def _conductor_property_lines(self, polygon: PolygonData) -> list[str]:
        points = [(float(x_coord), float(y_coord)) for x_coord, y_coord in polygon.points]
        area, perimeter, bbox = compute_polygon_metrics(points)
        contour = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if len(points) >= 3:
            (center_x, center_y), (rect_width, rect_height), raw_angle = cv2.minAreaRect(contour)
            length = max(float(rect_width), float(rect_height))
            width = min(float(rect_width), float(rect_height))
            orientation = float(raw_angle) if rect_width >= rect_height else float(raw_angle) + 90.0
            orientation %= 180.0
            hull_area = abs(float(cv2.contourArea(cv2.convexHull(contour))))
        else:
            center_x = float(np.mean([point[0] for point in points])) if points else 0.0
            center_y = float(np.mean([point[1] for point in points])) if points else 0.0
            length = max(float(bbox[2]), float(bbox[3]))
            width = min(float(bbox[2]), float(bbox[3]))
            orientation = 0.0
            hull_area = 0.0
        aspect_ratio = length / width if width > 1e-9 else 0.0
        circularity = 4.0 * float(np.pi) * area / (perimeter * perimeter) if perimeter > 1e-9 else 0.0
        solidity = area / hull_area if hull_area > 1e-9 else 0.0
        bbox_area = float(bbox[2] * bbox[3])
        extent = area / bbox_area if bbox_area > 1e-9 else 0.0

        def number(value: float, decimals: int = 2) -> str:
            text = f"{float(value):.{decimals}f}"
            return text.replace(".", ",") if self._ui_language == "ru" else text

        if self._ui_language == "ru":
            object_type = "внутреннее отверстие" if polygon.is_hole else "проводник"
            parent = "нет" if polygon.parent_id is None else str(int(polygon.parent_id))
            lines = [
                f"ID: {int(polygon.id)}",
                f"Тип: {object_type}",
                f"Категория: {polygon.category or '-'}",
                f"Количество точек: {len(points)}",
                f"Площадь: {number(area)} px²",
                f"Периметр: {number(perimeter)} px",
                f"Длина: {number(length)} px",
                f"Ширина: {number(width)} px",
                f"Отношение длины к ширине: {number(aspect_ratio, 3)}",
                f"Габариты: {int(bbox[2])} × {int(bbox[3])} px",
                f"Позиция: x={int(bbox[0])}, y={int(bbox[1])}",
                f"Центр: x={number(center_x)}, y={number(center_y)}",
                f"Ориентация: {number(orientation)}°",
                f"Округлость: {number(circularity, 3)}",
                f"Выпуклость: {number(solidity, 3)}",
                f"Заполнение габарита: {number(extent * 100.0, 1)}%",
                f"Представление: {polygon.shape_hint or '-'}",
                f"Родительский контур: {parent}",
            ]
        else:
            object_type = "inner hole" if polygon.is_hole else "conductor"
            parent = "none" if polygon.parent_id is None else str(int(polygon.parent_id))
            lines = [
                f"ID: {int(polygon.id)}",
                f"Type: {object_type}",
                f"Category: {polygon.category or '-'}",
                f"Point count: {len(points)}",
                f"Area: {number(area)} px²",
                f"Perimeter: {number(perimeter)} px",
                f"Length: {number(length)} px",
                f"Width: {number(width)} px",
                f"Length-to-width ratio: {number(aspect_ratio, 3)}",
                f"Bounding box: {int(bbox[2])} × {int(bbox[3])} px",
                f"Position: x={int(bbox[0])}, y={int(bbox[1])}",
                f"Center: x={number(center_x)}, y={number(center_y)}",
                f"Orientation: {number(orientation)}°",
                f"Circularity: {number(circularity, 3)}",
                f"Solidity: {number(solidity, 3)}",
                f"Bounding-box extent: {number(extent * 100.0, 1)}%",
                f"Shape representation: {polygon.shape_hint or '-'}",
                f"Parent contour: {parent}",
            ]
        if polygon.recognition_score is not None:
            score_label = "Оценка распознавания" if self._ui_language == "ru" else "Recognition score"
            lines.append(f"{score_label}: {number(float(polygon.recognition_score), 1)}")
        if str(polygon.reject_reason or "").strip():
            reason_label = "Причина отклонения" if self._ui_language == "ru" else "Rejection reason"
            lines.append(f"{reason_label}: {polygon.reject_reason}")
        return lines

    def _manual_via_debug_candidate(self, polygon: PolygonData) -> ContourDebugCandidate | None:
        image = self._workspace.current_display_image()
        if image is None:
            return None
        polygon_rect = self._polygon_rect(polygon)
        if polygon_rect.isNull():
            return None
        from ..vision.via_detection import analyze_via_at
        from ..vision.via_detection.settings_bridge import heuristic_config_from_settings

        settings = self._current_contour_settings()
        config = heuristic_config_from_settings(settings)
        detection = analyze_via_at(
            np.asarray(image),
            polygon_rect.center().x(),
            polygon_rect.center().y(),
            config,
        )
        if detection is None:
            return None
        accepted = bool(
            detection.reject_reason is None
            and float(detection.score) >= float(config.min_final_score)
        )
        reason = str(
            detection.reject_reason
            or ("accepted:heuristic" if accepted else "below_threshold")
        )
        return ContourDebugCandidate(
            contour_index=-1,
            bbox=tuple(int(value) for value in detection.bbox),
            area=float(detection.bbox[2] * detection.bbox[3]),
            perimeter=float(2 * (detection.bbox[2] + detection.bbox[3])),
            roundness=float(detection.features.get("circularity", 0.0) * 100.0),
            accepted=accepted,
            reason=reason,
            source="heuristic",
            score=float(detection.score),
            metrics=dict(detection.features),
        )

    def _show_nonblocking_via_debug_message(self, title: str, message: str) -> QMessageBox:
        dialog = QMessageBox(self._debug_parent_widget())
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.setWindowTitle(title)
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setText(message)
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setModal(False)

        dialogs = getattr(self, "_open_via_debug_dialogs", None)
        if dialogs is None:
            dialogs = []
            self._open_via_debug_dialogs = dialogs
        dialogs.append(dialog)

        def release_dialog(_result: int, *, finished_dialog: QMessageBox = dialog) -> None:
            open_dialogs = getattr(self, "_open_via_debug_dialogs", [])
            if finished_dialog in open_dialogs:
                open_dialogs.remove(finished_dialog)

        dialog.finished.connect(release_dialog)
        dialog.show()
        offset = 28 * ((len(dialogs) - 1) % 8)
        parent_top_left = self._debug_parent_widget().mapToGlobal(QPoint(24, 48))
        dialog.move(parent_top_left.x() + offset, parent_top_left.y() + offset)
        return dialog

    def _heuristic_via_diagnostic_lines(self, candidate: object) -> list[str]:
        metrics = dict(getattr(candidate, "metrics", {}) or {})
        if not metrics:
            return []
        settings = self._current_contour_settings()

        def number(value: object, decimals: int = 3) -> str:
            text = f"{float(value):.{decimals}f}"
            return text.replace(".", ",") if self._ui_language == "ru" else text

        def setting(name: str, default: float = 0.0) -> float:
            return float(getattr(settings, f"heuristic_{name}", default))

        diameter = max(1e-9, float(metrics.get("diameter", 0.0)))
        drift = float(metrics.get("center_drift", 0.0))
        drift_ratio = drift / diameter
        polarity = str(getattr(settings, "via_heuristic_polarity", "") or "")
        bilateral = bool(getattr(settings, "heuristic_use_bilateral", False))
        score = float(metrics.get("final_score", getattr(candidate, "score", 0.0)))

        if self._ui_language != "ru":
            return [
                "",
                "Measured contact parameters:",
                f"Center brightness: {number(metrics.get('center_brightness', 0.0), 1)} / 255",
                f"Minimum center brightness: {number(setting('min_center_brightness'), 1)} / 255",
                f"Local binarization threshold: {number(metrics.get('binarization_threshold', 0.0), 1)}",
                f"Center contrast: {number(metrics.get('contrast', 0.0), 2)} (minimum {number(setting('min_center_contrast'), 2)})",
                f"Peak prominence: {number(metrics.get('prominence', 0.0), 2)} (minimum {number(setting('min_peak_prominence'), 2)})",
                f"Diameter: {number(diameter, 2)} px; equivalent diameter: {number(metrics.get('equivalent_diameter', 0.0), 2)} px",
                f"Center drift: {number(drift, 2)} px = {number(drift_ratio, 3)} of diameter (maximum {number(setting('max_center_drift_ratio'), 3)})",
                f"Compactness: {number(metrics.get('compactness', 0.0))} (minimum {number(setting('min_compactness'))})",
                f"Circularity: {number(metrics.get('circularity', 0.0))} (minimum {number(setting('min_circularity'))})",
                f"Elongation: {number(metrics.get('aspect', 0.0))} (maximum {number(setting('max_elongation'))})",
                f"Edge direction coherence: {number(metrics.get('line_coherence', 0.0))} (maximum {number(setting('max_line_coherence'))})",
                f"Edge-to-noise ratio: {number(metrics.get('edge_snr', 0.0))} (normalization {number(setting('edge_snr_score_min'))}..{number(setting('edge_snr_score_max'))})",
                f"Edge sharpness: {number(metrics.get('edge_sharpness', 0.0))} (minimum {number(setting('min_edge_sharpness'))})",
                f"Border imbalance: {number(metrics.get('border_imbalance', 0.0))}; line likeness: {number(metrics.get('line_likeness', 0.0))}",
                f"Polarity used: {polarity or '-'}",
                f"Final score: {number(score, 2)}",
                *self._heuristic_score_and_settings_lines(metrics, settings, number, english=True),
            ]

        return [
            "",
            "Измеренные параметры контакта:",
            f"Яркость центра: {number(metrics.get('center_brightness', 0.0), 1)} из 255",
            f"Минимальная яркость центра: {number(setting('min_center_brightness'), 1)} из 255",
            f"Локальный порог бинаризации: {number(metrics.get('binarization_threshold', 0.0), 1)}",
            f"Контраст центра: {number(metrics.get('contrast', 0.0), 2)} (минимум {number(setting('min_center_contrast'), 2)})",
            f"Выраженность пика: {number(metrics.get('prominence', 0.0), 2)} (минимум {number(setting('min_peak_prominence'), 2)})",
            f"Диаметр: {number(diameter, 2)} px; эквивалентный диаметр: {number(metrics.get('equivalent_diameter', 0.0), 2)} px",
            f"Смещение центра: {number(drift, 2)} px = {number(drift_ratio, 3)} диаметра (максимум {number(setting('max_center_drift_ratio'), 3)})",
            f"Компактность: {number(metrics.get('compactness', 0.0))} (минимум {number(setting('min_compactness'))})",
            f"Округлость формы: {number(metrics.get('circularity', 0.0))} (минимум {number(setting('min_circularity'))})",
            f"Вытянутость: {number(metrics.get('aspect', 0.0))} (максимум {number(setting('max_elongation'))})",
            f"Направленность границ: {number(metrics.get('line_coherence', 0.0))} (максимум {number(setting('max_line_coherence'))})",
            f"Отношение края к шуму: {number(metrics.get('edge_snr', 0.0))} (нормализация {number(setting('edge_snr_score_min'))}…{number(setting('edge_snr_score_max'))})",
            f"Резкость края: {number(metrics.get('edge_sharpness', 0.0))} (минимум {number(setting('min_edge_sharpness'))})",
            f"Дисбаланс границы: {number(metrics.get('border_imbalance', 0.0))}; сходство с линией: {number(metrics.get('line_likeness', 0.0))}",
            f"Применённая полярность: {polarity or '-'}",
            f"Итоговая оценка: {number(score, 2)}",
            *self._heuristic_score_and_settings_lines(metrics, settings, number, english=False),
        ]

    @staticmethod
    def _heuristic_score_and_settings_lines(metrics, settings, number, *, english: bool) -> list[str]:
        def setting(name: str, default: float = 0.0) -> float:
            return float(getattr(settings, f"heuristic_{name}", default))

        contributions = (
            ("Contrast", "Контраст", "contribution_contrast", "w_contrast"),
            ("Peak prominence", "Выраженность пика", "contribution_prominence", "w_prominence"),
            ("Size match", "Соответствие размеру", "contribution_size", "w_size"),
            ("Compactness", "Компактность", "contribution_compactness", "w_compact"),
            ("Circularity", "Округлость", "contribution_roundness", "w_round"),
            ("Balance", "Баланс", "contribution_balance", "w_balance"),
        )
        lines = ["", "Score contributions:" if english else "Вклад в итоговую оценку:"]
        for en_label, ru_label, metric_name, weight_name in contributions:
            label = en_label if english else ru_label
            lines.append(
                f"{label}: +{number(metrics.get(metric_name, 0.0), 2)} "
                f"({'weight' if english else 'вес'} {number(setting(weight_name), 1)})"
            )
        lines.extend(
            [
                f"{'Line penalty' if english else 'Штраф линии'}: -{number(metrics.get('penalty_line', 0.0), 2)} "
                f"({'weight' if english else 'вес'} {number(setting('w_line'), 1)}; "
                f"{'scale' if english else 'множитель'} {number(setting('line_penalty_scale'), 2)})",
                f"{'Border penalty' if english else 'Штраф границы'}: -{number(metrics.get('penalty_border', 0.0), 2)} "
                f"({'weight' if english else 'вес'} {number(setting('w_border'), 1)}; "
                f"{'scale' if english else 'множитель'} {number(setting('border_penalty_scale'), 2)})",
                "",
                "Applied candidate-generation settings:" if english else "Применённые настройки генерации кандидатов:",
                f"{'Background correction radius' if english else 'Радиус коррекции фона'}: {number(setting('background_sigma'), 1)} px",
                f"{'Analysis window scale' if english else 'Множитель окна анализа'}: {number(setting('analysis_window_scale'), 2)}",
                f"{'Local binarization percentile' if english else 'Процентиль локальной бинаризации'}: {number(setting('local_binarize_percentile'), 1)} %",
                f"{'Minimum response peak' if english else 'Минимальная яркость пика отклика'}: {number(setting('min_abs_peak'), 1)}",
                f"{'Seed percentiles (low/medium/high)' if english else 'Процентили пиков (низкая/средняя/высокая чувствительность)'}: "
                f"{number(setting('seed_percentile'), 1)} %",
                f"{'Edge quality minimum contribution' if english else 'Минимальный вклад качества края'}: {number(setting('edge_quality_floor'), 3)}",
                f"{'Border imbalance sensitivity' if english else 'Чувствительность к дисбалансу границы'}: {number(setting('border_balance_scale'), 3)}",
                f"{'Size tolerance (range/fixed)' if english else 'Допуск размера (диапазон/фиксированный)'}: "
                f"{number(setting('size_tolerance_range'), 3)} / {number(setting('size_tolerance_fixed'), 3)}",
                f"{'Contrast normalization' if english else 'Нормализация контраста'}: "
                f"{number(setting('contrast_score_min'), 1)}…{number(setting('contrast_score_max'), 1)}",
                f"{'Peak prominence normalization' if english else 'Нормализация выраженности пика'}: "
                f"{number(setting('prominence_score_min'), 1)}…{number(setting('prominence_score_max'), 1)}",
                f"{'Denoising' if english else 'Шумоподавление'}: "
                f"{'bilateral' if bool(getattr(settings, 'heuristic_use_bilateral', False)) else ('median' if english else 'медианное')}",
            ]
        )
        return lines

    def _on_metal_overlay_detail_requested(self, layer_key: str, reason: str) -> None:
        titles = {
            "rejected": "debug.metal_title_rejected",
            "suspicious": "debug.metal_title_suspicious",
            "border": "debug.metal_title_border",
            "wide_pairs_suspicious": "debug.metal_title_wide_pairs_suspicious",
            "wide_pairs_rejected": "debug.metal_title_wide_pairs_rejected",
        }
        title = self._tr(titles.get(layer_key, "debug.metal_title_default"))
        r = (reason or "").strip()
        if not r:
            body = self._tr("debug.metal_no_detailed_reason")
        else:
            body = f"{self._tr('debug.field_reason')}:\n{r}"
        self._append_log(f"{title}: {r or body}")
        QMessageBox.information(self._debug_parent_widget(), title, body)

    def _on_middle_preview_hold_changed(self, active: bool) -> None:
        should_show_source = bool(active and self._is_filters_tab_active())
        if self._show_source_while_middle_held == should_show_source:
            return
        self._show_source_while_middle_held = should_show_source
        self._refresh_current_display_image_only(preserve_view=True)

    def _on_filter_preview_hold_changed(self, active: bool) -> None:
        should_show_source = bool(active)
        if self._show_source_while_filter_hotkey_held == should_show_source:
            return
        self._show_source_while_filter_hotkey_held = should_show_source
        self._refresh_current_display_image_only(preserve_view=True)

    def _is_filters_tab_active(self) -> bool:
        if not hasattr(self, "control_tabs") or not hasattr(self, "pipeline_tab"):
            return False
        return self.control_tabs.currentWidget() is self.pipeline_tab

    def _on_control_tab_changed(self, _index: int) -> None:
        if not self._show_source_while_middle_held:
            return
        if self._is_filters_tab_active():
            return
        self._show_source_while_middle_held = False
        self._refresh_current_display_image_only(preserve_view=True)

    def _compute_gradient_debug_maps_on_demand(self) -> dict[str, object]:
        current_state = self._workspace.current_state
        if current_state is None or current_state.source_image is None:
            return {}
        from ..application.use_cases.processing import build_detection_debug_maps

        settings = self._current_contour_settings()
        preprocessed = current_state.preprocessed_image
        if preprocessed is None:
            preprocessed = current_state.source_image
        maps: dict[str, object] = dict(build_detection_debug_maps(current_state.source_image, preprocessed, settings))
        try:
            current_state.debug_gradient_maps = dict(maps)
        except Exception:  # pragma: no cover - defensive
            pass
        return maps

    def _refresh_gradient_overlay(self) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        rec = (
            str(self.recognition_mode_combo.currentData() or "")
            if hasattr(self, "recognition_mode_combo")
            else ""
        )
        if (
            rec == "conductors"
            and hasattr(self, "metal_show_mask_checkbox")
            and self.metal_show_mask_checkbox.isChecked()
        ):
            _st = self._workspace.current_state
            _maps: dict = getattr(_st, "debug_gradient_maps", None) or {} if _st is not None else {}
            if any(k in _maps for k in ("metal_filtered_mask", "metal_binary_mask", "metal_mask")):
                self._apply_metal_visual_overlay()
                return
        if not hasattr(self, "gradient_overlay_mode_combo"):
            self.polygon_editor.clear_gradient_overlay()
            return
        display_image = self._workspace.current_display_image()
        if display_image is None:
            self.polygon_editor.clear_gradient_overlay()
            return
        try:
            overlay = self._build_gradient_overlay_image(display_image)
        except Exception:  # pragma: no cover - defensive: UI must never crash
            self.polygon_editor.clear_gradient_overlay()
            return
        if overlay is None:
            self.polygon_editor.clear_gradient_overlay()
            return
        self.polygon_editor.set_gradient_overlay(overlay, 1.0)

    def _apply_metal_visual_overlay(self) -> None:
        if not hasattr(self, "polygon_editor"):
            return
        current_state = self._workspace.current_state
        if current_state is None:
            self.polygon_editor.clear_gradient_overlay()
            return
        maps: dict = getattr(current_state, "debug_gradient_maps", None) or {}
        mode = (
            str(self.metal_debug_visual_combo.currentData() or "overlay")
            if hasattr(self, "metal_debug_visual_combo")
            else "overlay"
        )
        op = float(self.metal_overlay_opacity_spin.value()) if hasattr(self, "metal_overlay_opacity_spin") else 0.45
        try:
            if mode == "overlay":
                src = self._workspace.current_display_image()
                if src is None:
                    self.polygon_editor.clear_gradient_overlay()
                    return
                vis = np.asarray(src)
                if vis.ndim == 2:
                    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                m = None
                for _mask_key in ("metal_filtered_mask", "metal_binary_mask", "metal_mask"):
                    _candidate = maps.get(_mask_key)
                    if _candidate is not None:
                        m = _candidate
                        break
                if m is None or np.asarray(m).size == 0:
                    self.polygon_editor.clear_gradient_overlay()
                    return
                binm = (np.asarray(m) > 0).astype(np.uint8)
                vector_clip = self._metal_overlay_vector_clip_mask(
                    binm.shape[:2],
                    list(getattr(current_state, "polygons", []) or []),
                )
                if vector_clip is not None:
                    binm = binm * vector_clip
                tint = np.zeros_like(vis)
                tint[:, :, 1] = binm * 200
                tint[:, :, 0] = binm * 40
                out = cv2.addWeighted(vis, 1.0 - 0.55 * op, tint, 0.55 * op, 0)
                if current_state and getattr(current_state, "polygons", None):
                    for poly in current_state.polygons:
                        if str(getattr(poly, "category", "")) != "metal_wide_gradient":
                            continue
                        if len(poly.points) < 2:
                            continue
                        pts = np.array([(int(x), int(y)) for x, y in poly.points], dtype=np.int32).reshape(
                            -1, 1, 2
                        )
                        cv2.polylines(out, [pts], True, (255, 120, 40), 2)
                self.polygon_editor.set_gradient_overlay(out, 1.0)
                return
            arr = maps.get(mode)
            if arr is None:
                self.polygon_editor.clear_gradient_overlay()
                return
            image = np.asarray(arr)
            if image.dtype != np.uint8:
                image = cv2.convertScaleAbs(image)
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.polygon_editor.set_gradient_overlay(image, min(1.0, max(0.05, op)))
            if mode == "metal_gradient_field":
                gx = maps.get("metal_gradient_x_f32")
                gy = maps.get("metal_gradient_y_f32")
                if gx is not None and gy is not None:
                    self.polygon_editor.set_gradient_field_maps(gx, gy)
        except Exception:  # pragma: no cover
            self.polygon_editor.clear_gradient_overlay()

    def _metal_overlay_vector_clip_mask(self, shape: tuple[int, int], polygons: list[PolygonData]) -> np.ndarray | None:
        if not polygons:
            return None
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        has_fill = False
        for polygon in polygons:
            if str(getattr(polygon, "category", "") or "") == "metal_wide_gradient":
                continue
            points = getattr(polygon, "points", []) or []
            if len(points) < 3:
                continue
            pts = np.array([(int(round(float(x))), int(round(float(y)))) for x, y in points], dtype=np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [pts.reshape((-1, 1, 2))], 0 if bool(getattr(polygon, "is_hole", False)) else 1)
            if not bool(getattr(polygon, "is_hole", False)):
                has_fill = True
        return mask if has_fill else None

    def _build_gradient_overlay_image(self, source_image: np.ndarray) -> np.ndarray | None:
        from ..application.use_cases.processing import (
            _resolve_conductor_edge_method,
            _resolve_via_edge_method,
            _via_grayscale,
        )
        from ..edge_detection import build_gradient_elevation

        settings = self._current_contour_settings()
        if settings.object_type == "via" or settings.output_mode == "box":
            method = _resolve_via_edge_method(settings)
        else:
            method = _resolve_conductor_edge_method(settings)
        gray = _via_grayscale(source_image)
        if gray.size == 0:
            return None
        mode = str(self.gradient_overlay_mode_combo.currentData() or "source")
        if mode == "source":
            current_state = self._workspace.current_state
            original = (
                current_state.source_image
                if current_state is not None and current_state.source_image is not None
                else source_image
            )
            data = np.asarray(original)
            if data.ndim == 2:
                return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
            return np.ascontiguousarray(data[..., :3])
        if mode not in {"heatmap", "threshold", "elevation"}:
            current_state = self._workspace.current_state
            maps = (
                dict(getattr(current_state, "debug_gradient_maps", {}) or {})
                if current_state is not None
                else {}
            )
            if mode not in maps:
                maps = self._compute_gradient_debug_maps_on_demand()
            debug_image = maps.get(mode)
            if debug_image is None:
                return None
            return self._debug_map_overlay_image(np.asarray(debug_image))

        elevation = build_gradient_elevation(gray, method)
        if mode == "elevation":
            return cv2.cvtColor(elevation, cv2.COLOR_GRAY2BGR)
        if mode == "threshold":
            threshold = float(settings.via_min_contrast)
            mask = elevation >= threshold
            overlay = np.zeros((elevation.shape[0], elevation.shape[1], 3), dtype=np.uint8)
            overlay[..., 1] = mask.astype(np.uint8) * 230
            overlay[..., 2] = mask.astype(np.uint8) * 60
            return overlay
        heatmap = cv2.applyColorMap(elevation, cv2.COLORMAP_TURBO)
        threshold = float(settings.via_min_contrast)
        if settings.object_type == "via" or settings.output_mode == "box":
            below = (elevation < max(0.0, threshold)).astype(np.uint8)
            if below.any():
                dimmed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                below3 = below[..., None]
                heatmap = heatmap * (1 - below3) + (dimmed // 3) * below3
                heatmap = heatmap.astype(np.uint8)
        return heatmap

    @staticmethod
    def _debug_map_overlay_image(image: np.ndarray) -> np.ndarray | None:
        data = np.asarray(image)
        if data.size == 0:
            return None
        if data.dtype == bool:
            data = data.astype(np.uint8) * 255
        elif data.dtype != np.uint8:
            values = data.astype(np.float32)
            finite = np.isfinite(values)
            if not finite.any():
                return None
            minimum = float(values[finite].min())
            maximum = float(values[finite].max())
            if minimum >= 0.0 and maximum <= 1.0001:
                data = np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8)
            elif maximum - minimum > 1e-6:
                data = np.clip((values - minimum) / (maximum - minimum) * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                data = np.clip(values, 0.0, 255.0).astype(np.uint8)
        if data.ndim == 2:
            return cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
        if data.ndim == 3 and data.shape[2] == 4:
            return cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
        if data.ndim == 3 and data.shape[2] == 3:
            return np.ascontiguousarray(data)
        return None

    def _gradient_debug_pixmap(self, image: np.ndarray) -> QPixmap | None:
        data = np.asarray(image)
        if data.size == 0:
            return None
        if data.dtype != np.uint8:
            if data.dtype == bool:
                data = data.astype(np.uint8) * 255
            else:
                as_float = data.astype(np.float32)
                max_val = float(as_float.max()) if as_float.size else 0.0
                if max_val <= 1.0001:
                    data = np.clip(as_float * 255.0, 0, 255).astype(np.uint8)
                else:
                    min_val = float(as_float.min())
                    span = max_val - min_val
                    if span <= 1e-6:
                        data = np.clip(as_float, 0, 255).astype(np.uint8)
                    else:
                        data = np.clip((as_float - min_val) / span * 255.0, 0, 255).astype(np.uint8)
        try:
            qimage = cv_to_qimage(data)
        except Exception:  # pragma: no cover - defensive
            return None
        return QPixmap.fromImage(qimage)

    def _best_debug_candidate_for_polygon(self, polygon: PolygonData, candidates: Sequence[object]) -> object | None:
        polygon_rect = self._polygon_rect(polygon)
        if polygon_rect.isNull() or not candidates:
            return None
        polygon_center = polygon_rect.center()
        best_candidate: object | None = None
        best_rank: tuple[int, int, float, float] | None = None
        for index, candidate in enumerate(candidates):
            bbox = getattr(candidate, "bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            candidate_rect = QRectF(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])).normalized()
            if candidate_rect.isNull():
                continue
            overlap = self._rect_overlap_area(polygon_rect, candidate_rect)
            candidate_center = candidate_rect.center()
            dx = polygon_center.x() - candidate_center.x()
            dy = polygon_center.y() - candidate_center.y()
            distance_sq = dx * dx + dy * dy
            max_span = max(
                polygon_rect.width(), polygon_rect.height(), candidate_rect.width(), candidate_rect.height(), 1.0
            )
            if overlap <= 0.0 and distance_sq > (max_span * 1.5) * (max_span * 1.5):
                continue
            accepted_rank = 1 if bool(getattr(candidate, "accepted", False)) else 0
            rank = (accepted_rank, 1 if overlap > 0.0 else 0, overlap, -distance_sq - index * 1e-9)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_candidate = candidate
        return best_candidate

    @staticmethod
    def _polygon_rect(polygon: PolygonData) -> QRectF:
        if polygon.points:
            x_values = [point[0] for point in polygon.points]
            y_values = [point[1] for point in polygon.points]
            return QRectF(
                min(x_values),
                min(y_values),
                max(x_values) - min(x_values),
                max(y_values) - min(y_values),
            ).normalized()
        x_coord, y_coord, width, height = polygon.bbox
        return QRectF(float(x_coord), float(y_coord), float(width), float(height)).normalized()

    @staticmethod
    def _rect_overlap_area(first: QRectF, second: QRectF) -> float:
        overlap = first.intersected(second)
        if overlap.isNull():
            return 0.0
        return max(0.0, overlap.width()) * max(0.0, overlap.height())

    @staticmethod
    def _debug_candidate_source(candidate: object) -> str:
        source = str(getattr(candidate, "source", "") or "")
        reason = str(getattr(candidate, "reason", "") or "")
        if not source and ":" in reason:
            source = reason.split(":", 1)[1]
        return source

    def _debug_method_text(self, source: str) -> str:
        source = source.lower()
        if "template" in source:
            source = "template"
        labels = {
            "range-components": "debug.method.range_components",
            "range-contours": "debug.method.range_contours",
            "gradient": "debug.method.gradient",
            "spot": "debug.method.spot",
            "hough-gray": "debug.method.hough_gray",
            "hough": "debug.method.hough",
            "components": "debug.method.components",
            "contours-response": "debug.method.contours_response",
            "contours": "debug.method.contours",
            "morphology": "debug.method.morphology",
            "template": "debug.method.template",
            "blob": "debug.method.blob",
        }
        for prefix, key in labels.items():
            if source.startswith(prefix):
                return self._tr(key)
        return source or self._tr("debug.method.unknown")

    def _debug_criterion_text(self, source: str, reason: str, accepted: bool) -> str:
        if "template" in source.lower():
            source = "template"
        if not accepted:
            rejection_labels = {
                "duplicate": "debug.rejection.duplicate",
                "component_score": "debug.rejection.component_score",
                "contour_score": "debug.rejection.contour_score",
                "min_via_width": "debug.rejection.min_via_width",
                "max_via_width": "debug.rejection.max_via_width",
                "min_via_height": "debug.rejection.min_via_height",
                "max_via_height": "debug.rejection.max_via_height",
                "min_aspect_ratio": "debug.rejection.min_aspect_ratio",
                "max_aspect_ratio": "debug.rejection.max_aspect_ratio",
                "roundness": "debug.rejection.roundness",
                "empty_geometry": "debug.rejection.empty_geometry",
                "min_polygon_width": "debug.rejection.min_polygon_width",
            }
            key = rejection_labels.get(reason)
            if key is not None:
                return self._tr(key)
            return reason or self._tr("debug.rejection.default")
        source = source.lower()
        accepted_labels = {
            "range-components": "debug.accepted.range_components",
            "range-contours": "debug.accepted.range_contours",
            "gradient": "debug.accepted.gradient",
            "spot": "debug.accepted.spot",
            "hough-gray": "debug.accepted.hough_gray",
            "hough": "debug.accepted.hough",
            "components": "debug.accepted.components",
            "contours-response": "debug.accepted.contours_response",
            "contours": "debug.accepted.contours",
            "morphology": "debug.accepted.morphology",
            "template": "debug.accepted.template",
            "blob": "debug.accepted.blob",
        }
        for prefix, key in accepted_labels.items():
            if source.startswith(prefix):
                return self._tr(key)
        return self._tr("debug.accepted.default")

    def _open_metal_gradient_field_3d(self) -> None:
        gradient_x, gradient_y, intensity = self._gradient_field_3d_sources()
        if gradient_x is None or gradient_y is None:
            QMessageBox.information(
                self._debug_parent_widget(),
                "Градиентное поле — 3D" if self._ui_language == "ru" else "Gradient field — 3D",
                "Нет карт Sobel. Сначала выполните распознавание проводников."
                if self._ui_language == "ru"
                else "Sobel maps are missing. Run conductor recognition first.",
            )
            return
        from ..graphics.gradient_field_3d_window import GradientField3DWindow

        window = getattr(self, "_gradient_field_3d_window", None)
        if window is None:
            window = GradientField3DWindow(self._debug_parent_widget())
            self._gradient_field_3d_window = window
        window.set_field(
            gradient_x,
            gradient_y,
            intensity=intensity,
            language=str(self._ui_language),
        )

    def _gradient_field_3d_sources(self) -> tuple[object | None, object | None, object | None]:
        current_state = self._workspace.current_state
        maps: dict = getattr(current_state, "debug_gradient_maps", None) or {} if current_state is not None else {}
        gradient_x = maps.get("metal_gradient_x_f32")
        gradient_y = maps.get("metal_gradient_y_f32")
        intensity = maps.get("metal_source_gray")
        if gradient_x is not None and gradient_y is not None:
            return gradient_x, gradient_y, intensity
        source = intensity
        if source is None and current_state is not None:
            source = current_state.preprocessed_image
            if source is None:
                source = current_state.source_image
        if source is None:
            return None, None, None
        gray = np.asarray(source)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        from ..vision.metal_recovery.pipeline_stages import axis_gradient_debug_images

        debug = axis_gradient_debug_images(gray)
        return debug.get("metal_gradient_x_f32"), debug.get("metal_gradient_y_f32"), gray


