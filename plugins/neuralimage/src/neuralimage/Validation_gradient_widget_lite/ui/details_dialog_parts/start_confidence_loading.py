"""details_dialog: start confidence loading."""
from __future__ import annotations
import numpy as np
from PyQt6.QtCore import QThread
from ...core.workers import DetailConfidenceWorker


class StartConfidenceLoadingMixin:
    def _cleanup_detail_worker(self) -> None:
        if self._detail_worker is not None:
            self._detail_worker.deleteLater()
        if self._detail_thread is not None:
            self._detail_thread.deleteLater()
        self._detail_worker = None
        self._detail_thread = None

    def _stop_detail_worker(self) -> None:
        worker = self._detail_worker
        if worker is not None:
            request_cancel = getattr(worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()
        thread = self._detail_thread
        if thread is not None:
            try:
                thread.quit()
            except Exception:
                pass
            if thread.isRunning():
                thread.wait(30000)

    def _cleanup_confidence_worker(
        self,
        worker: DetailConfidenceWorker | None = None,
        thread: QThread | None = None,
        generation: int | None = None,
    ) -> None:
        target_worker = worker or self._confidence_worker
        target_thread = thread or self._confidence_thread
        if target_worker is not None:
            target_worker.deleteLater()
        if target_thread is not None:
            target_thread.deleteLater()
        self._retired_confidence_workers = [
            (retired_worker, retired_thread)
            for retired_worker, retired_thread in self._retired_confidence_workers
            if retired_worker is not target_worker and retired_thread is not target_thread
        ]
        if generation is None or int(generation) == int(self._confidence_request_generation):
            if worker is None or worker is self._confidence_worker:
                self._confidence_worker = None
            if thread is None or thread is self._confidence_thread:
                self._confidence_thread = None
            self._loading_confidence_model_id = None
            self._set_confidence_loading_state(False)

    def _cancel_confidence_worker(self) -> None:
        if self._confidence_worker is not None:
            request_cancel = getattr(self._confidence_worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()

    def _stop_confidence_worker(self) -> None:
        active_pairs: list[tuple[DetailConfidenceWorker | None, QThread | None]] = [
            (self._confidence_worker, self._confidence_thread),
            *list(self._retired_confidence_workers),
        ]
        for worker, _thread in active_pairs:
            if worker is None:
                continue
            request_cancel = getattr(worker, "request_cancel", None)
            if callable(request_cancel):
                request_cancel()
        for _worker, thread in active_pairs:
            if thread is None:
                continue
            try:
                thread.quit()
            except Exception:
                pass
            if thread.isRunning():
                thread.wait(30000)
        self._retired_confidence_workers.clear()
        self._confidence_worker = None
        self._confidence_thread = None
        self._loading_confidence_model_id = None
        self._set_confidence_loading_state(False)

    def _on_detail_worker_failed(self, message: str, *, generation: int | None = None, worker_kind: str = "detail") -> None:
        if generation is not None:
            active_generation = self._confidence_request_generation if worker_kind == "confidence" else self._detail_request_generation
            if int(generation) != int(active_generation):
                return
        self._payload_loading = False
        self._set_confidence_loading_state(False)
        self.comparison_score_card.set_payload(
            self._t("details.frame_score"),
            "-",
            self._comparison_score_style(None),
            str(message),
        )

    def _confidence_payload_ready(self, model_id: str | None) -> bool:
        if model_id is None:
            return False
        confidence_row = self._confidence_metrics_for_model(model_id)
        if confidence_row is None:
            return False
        if self._is_point_geometry():
            return hasattr(confidence_row, 'mean_point_confidence')
        return hasattr(confidence_row, 'mean_object_confidence')

    def _result_kind_requires_confidence(self, kind: str | None = None) -> bool:
        result_kind = str(kind or self._selected_result_kind())
        return result_kind == "confidence" or result_kind.startswith("confidence_")

    def _start_confidence_loading(self, model_id: str | None) -> None:
        if model_id is None or not self._payload:
            return
        if self._confidence_payload_ready(model_id):
            return
        if self._loading_confidence_model_id == model_id and self._confidence_thread is not None:
            return
        previous_worker = self._confidence_worker
        previous_thread = self._confidence_thread
        self._cancel_confidence_worker()
        if previous_worker is not None and previous_thread is not None:
            self._retired_confidence_workers.append((previous_worker, previous_thread))
        self._confidence_request_generation += 1
        generation = int(self._confidence_request_generation)
        max_side = int(getattr(self._build_result.options, "analysis_max_side", 0) or 0) or None
        thread = QThread(self)
        worker = DetailConfidenceWorker(self._record, self._build_result, model_id, max_side, self._payload)
        self._confidence_thread = thread
        self._confidence_worker = worker
        self._loading_confidence_model_id = model_id
        self._set_confidence_loading_state(True, model_id)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(lambda row, mid=model_id, g=generation: self._on_confidence_loaded(mid, row, generation=g))
        worker.failed.connect(lambda message, g=generation: self._on_detail_worker_failed(message, generation=g, worker_kind="confidence"))
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(lambda w=worker, t=thread, g=generation: self._cleanup_confidence_worker(w, t, g))
        thread.start()

    def _on_confidence_loaded(self, model_id: str | None, confidence_row, *, generation: int | None = None) -> None:
        if generation is not None and int(generation) != int(self._confidence_request_generation):
            return
        if self._loading_confidence_model_id is not None and model_id != self._loading_confidence_model_id:
            return
        if model_id is None or confidence_row is None:
            return
        (self._payload.setdefault("model_confidence", {}))[model_id] = confidence_row
        self._overlay_cache.clear()
        self._derived_cache.clear()
        self._set_confidence_loading_state(False)
        self._refresh_scene(reset_view=False)
        self._refresh_info()

    def _ensure_confidence_ready(self) -> None:
        if not self._result_kind_requires_confidence():
            return
        model_id = self._confidence_model_id()
        if model_id is None or self._payload_loading:
            return
        self._start_confidence_loading(model_id)

    def _apply_selected_model(self, model_id: str | None) -> None:
        probabilities = self._payload.get("model_probabilities") or {}
        masks = self._payload.get("model_masks") or {}
        selected_model_id = model_id if model_id in probabilities else (next(iter(probabilities.keys()), None))
        fallback_prob = np.zeros_like(next(iter(probabilities.values()))) if probabilities else np.zeros((1, 1), dtype=np.float32)
        selected_prob = probabilities.get(selected_model_id, fallback_prob)
        selected_mask = masks.get(selected_model_id, np.asarray(selected_prob >= 0.5, dtype=bool))
        self._payload["selected_model_id"] = selected_model_id
        self._payload["selected_prob"] = np.asarray(selected_prob, dtype=np.float32)
        self._payload["selected_mask"] = np.asarray(selected_mask, dtype=bool)

    def _selected_model_for_current_comparison(self) -> str | None:
        _preset_key, first_key, second_key = self._current_comparison_tuple()
        for source_key in (first_key, second_key):
            if isinstance(source_key, str) and source_key.startswith("model:"):
                return source_key.split(":", 1)[1]
        return self._build_result.model_specs[0].model_id if self._build_result.model_specs else None

    def _preferred_confidence_model_id(self) -> str | None:
        metric_key = str(self._preferred_metric_key or "")
        if '::' not in metric_key:
            return None
        family, model_id = metric_key.split('::', 1)
        if family in {"model_confidence", "model_uncertain_fraction", "model_point_contrast"}:
            return model_id
        return None

    def _confidence_model_id(self) -> str | None:
        preferred_model_id = self._preferred_confidence_model_id()
        if preferred_model_id:
            return preferred_model_id
        return self._selected_model_for_current_comparison()
