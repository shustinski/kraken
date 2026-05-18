from __future__ import annotations

from ._imports import *  # noqa: F403


class WidgetDebugMixin:
    def _on_via_debug_requested(self, polygon: PolygonData) -> None:
        current_state = self._workspace.current_state
        candidates = list(current_state.debug_candidates) if current_state is not None else []
        is_via_like = (polygon.shape_hint or "") == "box" or (polygon.category or "") == "via"
        title = self._tr("debug.via_title" if is_via_like else "debug.polygon_title")
        if not candidates:
            message = self._tr("debug.no_current_frame_data")
            QMessageBox.information(self, title, message)
            return
        candidate = self._best_debug_candidate_for_polygon(polygon, candidates)
        if candidate is None:
            message = self._tr("debug.no_source_candidate")
            QMessageBox.information(self, title, message)
            return
        source = self._debug_candidate_source(candidate)
        reason = str(getattr(candidate, "reason", "") or "")
        accepted = bool(getattr(candidate, "accepted", False))
        bbox = getattr(candidate, "bbox", (0, 0, 0, 0))
        status = self._tr("debug.status_accepted" if accepted else "debug.status_rejected")
        lines = [
            f"{self._tr('debug.field_status')}: {status}",
            f"{self._tr('debug.field_method')}: {self._debug_method_text(source)}",
            f"{self._tr('debug.field_criterion')}: {self._debug_criterion_text(source, reason, accepted)}",
            f"{self._tr('debug.field_reason')}: {reason or '-'}",
        ]
        if is_via_like:
            lines += [
                f"{self._tr('debug.field_score')}: {float(getattr(candidate, 'score', 0.0)):.1f}",
                f"{self._tr('debug.field_roundness')}: {float(getattr(candidate, 'roundness', 0.0)):.1f}",
            ]
        else:
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
        QMessageBox.information(self, title, message)

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
        QMessageBox.information(self, title, body)

    def _on_middle_preview_hold_changed(self, active: bool) -> None:
        should_show_source = bool(active and self._is_filters_tab_active())
        if self._show_source_while_middle_held == should_show_source:
            return
        self._show_source_while_middle_held = should_show_source
        self._sync_current_state_views()

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
        self._sync_current_state_views()

    def _show_gradient_debug_window(self) -> None:
        title = self._tr("debug.gradient_title")
        current_state = self._workspace.current_state
        maps: dict[str, object] = {}
        if current_state is not None:
            maps = dict(getattr(current_state, "debug_gradient_maps", {}) or {})
        if not maps:
            try:
                maps = self._compute_gradient_debug_maps_on_demand()
            except Exception as exc:  # pragma: no cover - defensive UI path
                QMessageBox.warning(
                    self,
                    title,
                    self._tr("debug.gradient_build_failed", error=exc),
                )
                return
        if not maps:
            message = self._tr("debug.gradient_no_maps")
            QMessageBox.information(self, title, message)
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(1100, 780)
        layout = QVBoxLayout(dialog)
        tabs = QTabWidget(dialog)
        layout.addWidget(tabs, 1)
        ordering = [
            "source_gray",
            "gradient_elevation",
            "gradient_color",
            "scharr",
            "phase_congruency",
            "structured",
            "ridge",
            "conductor_gradient_elevation",
            "spot_response",
            "spot_response_dark",
            "raw_gray",
            "processed",
            "tophat",
            "dog",
            "tophat_mask",
            "dog_mask",
            "via_mask",
            "candidate_mask",
            "metal_mask",
            "radial_symmetry",
            "edge_likeness",
            "line_likeness",
            "distance_to_edge",
            "final_overlay",
            "mask",
        ]
        pretty_names = {
            "source_gray": self._tr("debug.gradient_map.source_gray"),
            "gradient_elevation": self._tr("debug.gradient_map.gradient_elevation"),
            "gradient_color": self._tr("debug.gradient_map.gradient_color"),
            "scharr": "Scharr",
            "phase_congruency": "Phase congruency",
            "structured": self._tr("debug.gradient_map.structured"),
            "ridge": self._tr("debug.gradient_map.ridge"),
            "conductor_gradient_elevation": self._tr("debug.gradient_map.conductor_gradient_elevation"),
            "spot_response": self._tr("debug.gradient_map.spot_response"),
            "spot_response_dark": self._tr("debug.gradient_map.spot_response_dark"),
            "raw_gray": self._tr("debug.gradient_map.raw_gray"),
            "processed": self._tr("debug.gradient_map.processed"),
            "tophat": self._tr("debug.gradient_map.tophat"),
            "dog": self._tr("debug.gradient_map.dog"),
            "tophat_mask": self._tr("debug.gradient_map.tophat_mask"),
            "dog_mask": self._tr("debug.gradient_map.dog_mask"),
            "via_mask": self._tr("debug.gradient_map.via_mask"),
            "candidate_mask": self._tr("debug.gradient_map.candidate_mask"),
            "metal_mask": self._tr("debug.gradient_map.metal_mask"),
            "radial_symmetry": self._tr("debug.gradient_map.radial_symmetry"),
            "edge_likeness": self._tr("debug.gradient_map.edge_likeness"),
            "line_likeness": self._tr("debug.gradient_map.line_likeness"),
            "distance_to_edge": self._tr("debug.gradient_map.distance_to_edge"),
            "final_overlay": self._tr("debug.gradient_map.final_overlay"),
            "mask": self._tr("debug.gradient_map.mask"),
        }
        seen: set[str] = set()
        for key in ordering + sorted(maps.keys()):
            if key in seen or key not in maps:
                continue
            seen.add(key)
            array = maps.get(key)
            if array is None:
                continue
            try:
                image = np.asarray(array)
            except Exception:  # pragma: no cover - defensive
                continue
            if image.size == 0:
                continue
            pixmap = self._gradient_debug_pixmap(image)
            if pixmap is None:
                continue
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(4, 4, 4, 4)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setPixmap(pixmap)
            scroll.setWidget(label)
            page_layout.addWidget(scroll, 1)
            info = QLabel(f"{image.shape[1]} x {image.shape[0]} px" + (f" · dtype={image.dtype}" if hasattr(image, "dtype") else ""))
            page_layout.addWidget(info)
            tabs.addTab(page, pretty_names.get(key, key))
        close_button = QPushButton(self._tr("debug.close_button"))
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        dialog.exec()

    def _compute_gradient_debug_maps_on_demand(self) -> dict[str, object]:
        current_state = self._workspace.current_state
        if current_state is None or current_state.source_image is None:
            return {}
        from .application.use_cases.processing import build_detection_debug_maps

        settings = self._current_contour_settings()
        preprocessed = current_state.preprocessed_image
        if preprocessed is None:
            preprocessed = current_state.source_image
        maps = build_detection_debug_maps(current_state.source_image, preprocessed, settings)
        try:
            current_state.debug_gradient_maps = dict(maps)
        except Exception:  # pragma: no cover - defensive
            pass
        return maps

    def _on_gradient_overlay_toggled(self, _checked: bool = False) -> None:
        if not self.gradient_overlay_checkbox.isChecked():
            if hasattr(self, "polygon_editor"):
                self.polygon_editor.clear_gradient_overlay()
            return
        self._refresh_gradient_overlay()

    def _on_gradient_overlay_opacity_changed(self, value: float) -> None:
        if hasattr(self, "polygon_editor"):
            self.polygon_editor.set_gradient_overlay_opacity(float(value))

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
        if not hasattr(self, "gradient_overlay_checkbox"):
            self.polygon_editor.clear_gradient_overlay()
            return
        if not self.gradient_overlay_checkbox.isChecked():
            self.polygon_editor.clear_gradient_overlay()
            return
        current_state = self._workspace.current_state
        if current_state is None or current_state.source_image is None:
            self.polygon_editor.clear_gradient_overlay()
            return
        try:
            overlay = self._build_gradient_overlay_image(current_state.source_image)
        except Exception:  # pragma: no cover - defensive: UI must never crash
            self.polygon_editor.clear_gradient_overlay()
            return
        if overlay is None:
            self.polygon_editor.clear_gradient_overlay()
            return
        self.polygon_editor.set_gradient_overlay(overlay, float(self.gradient_overlay_opacity_spin.value()))

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
                src = current_state.source_image
                if src is None:
                    self.polygon_editor.clear_gradient_overlay()
                    return
                vis = np.asarray(src)
                if vis.ndim == 2:
                    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                m = maps.get("metal_filtered_mask") or maps.get("metal_binary_mask") or maps.get("metal_mask")
                if m is None or np.asarray(m).size == 0:
                    self.polygon_editor.clear_gradient_overlay()
                    return
                binm = (np.asarray(m) > 0).astype(np.uint8)
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
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.polygon_editor.set_gradient_overlay(image, min(1.0, max(0.05, op)))
        except Exception:  # pragma: no cover
            self.polygon_editor.clear_gradient_overlay()

    def _build_gradient_overlay_image(self, source_image: np.ndarray) -> np.ndarray | None:
        from .application.use_cases.processing import (
            _resolve_conductor_edge_method,
            _resolve_via_edge_method,
            _via_grayscale,
        )
        from .edge_detection import build_gradient_elevation

        settings = self._current_contour_settings()
        if settings.object_type == "via" or settings.output_mode == "box":
            method = _resolve_via_edge_method(settings)
        else:
            method = _resolve_conductor_edge_method(settings)
        gray = _via_grayscale(source_image)
        if gray.size == 0:
            return None
        elevation = build_gradient_elevation(gray, method)
        mode = str(self.gradient_overlay_mode_combo.currentData() or "heatmap")
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

    def _best_debug_candidate_for_polygon(self, polygon: PolygonData, candidates: list[object]) -> object | None:
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


