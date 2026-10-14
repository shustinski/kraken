"""main_window: append metrics."""
from __future__ import annotations
import math


class AppendMetricsMixin:
    def _append_metrics(self, data):
        if not isinstance(data, dict):
            return

        metric_type = data.get("type")
        if metric_type == "train_epoch":
            self.metrics_panel.add_train_epoch_point(int(data.get("epoch", 0)), float(data.get("loss", 0.0)))
            return

        if metric_type == "val_epoch":
            self.metrics_panel.add_val_epoch_point(int(data.get("epoch", 0)), float(data.get("loss", 0.0)))
            iou = data.get("iou")
            dice = data.get("dice")
            f1 = data.get("f1")
            if iou is not None and dice is not None and f1 is not None:
                self.metrics_panel.add_val_quality_point(
                    int(data.get("epoch", 0)),
                    float(iou),
                    float(dice),
                )
                self._last_validation_metrics = (float(iou), float(dice), float(f1))
                self.validation_quality_label.setText(
                    self._format_validation_quality_text(float(iou), float(dice), float(f1))
                )
            return

        if metric_type == "train_batch":
            epoch = int(data.get("epoch", 0))
            batch_index = float(data.get("batch_index", 0.0))
            loss = float(data.get("loss", 0.0))
            epoch_points = self._batch_points_by_epoch.setdefault(epoch, [])
            epoch_points.append((batch_index, loss))
            self.metrics_panel.set_batch_points(epoch, self._sparsify_batch_points(epoch_points))
            return

        if metric_type == "train_epoch_progress":
            current = int(data.get("current", 0))
            total = int(data.get("total", 0))
            self._last_epoch_progress = (current, total)
            self._show_progress_mode("train")
            self._set_progress_bar(self.epoch_progress_bar, current, total)
            return

        if metric_type == "train_batch_progress":
            current = int(data.get("current", 0))
            total = int(data.get("total", 0))
            self._last_batch_progress = (current, total)
            self._show_progress_mode("train")
            self._set_training_batch_progress_bar()
            return

        if metric_type == "recognition_progress":
            current = int(data.get("current", 0))
            total = int(data.get("total", 0))
            self._last_recognition_progress = (current, total)
            self._show_progress_mode("recognition")
            self._update_recognition_speed(current, total)
            self._set_recognition_progress_bar()
            return

        if metric_type == "train_batch_preview":
            image = data.get("image")
            label = data.get("label")
            outputs = data.get("outputs", data.get("output"))
            topograph_overlay = data.get("topograph_overlay")
            sample_name = str(data.get("sample_name", data.get("frame_name", ""))).strip()
            self._set_preview_mode("train")
            self._set_preview_image(self.preview_image_label, image)
            self._set_preview_image(self.preview_label_label, label)
            preview_output = topograph_overlay if topograph_overlay is not None else outputs
            self._set_preview_image(self.preview_output_label, preview_output)
            self._set_preview_frame_name(sample_name)
            return

        if metric_type == "recognition_preview":
            image = data.get("image")
            outputs = data.get("outputs", data.get("output", data.get("result")))
            sample_name = str(data.get("sample_name", data.get("frame_name", ""))).strip()
            self._set_preview_mode("recognition")
            self.preview_label_label.clear()
            self._set_preview_image(self.preview_image_label, image)
            self._set_preview_image(self.preview_output_label, outputs)
            self._set_preview_frame_name(sample_name)
            return

        if metric_type == "system_memory":
            ram_mb = data.get("ram_mb")
            vram_alloc_mb = data.get("vram_allocated_mb")
            vram_reserved_mb = data.get("vram_reserved_mb")
            vram_used_mb = data.get("vram_used_mb")
            vram_total_mb = data.get("vram_total_mb")
            self._ram_mb = float(ram_mb) if ram_mb is not None else None
            self._vram_alloc_mb = float(vram_alloc_mb) if vram_alloc_mb is not None else None
            self._vram_reserved_mb = float(vram_reserved_mb) if vram_reserved_mb is not None else None
            self._vram_used_mb = float(vram_used_mb) if vram_used_mb is not None else None
            self._vram_total_mb = float(vram_total_mb) if vram_total_mb is not None else None
            self._update_memory_runtime_label()
            return

        if metric_type in ("train_perf", "train_perf_epoch"):
            data_wait_ms = float(data.get("data_wait_ms", 0.0))
            augmentation_ms = float(data.get("augmentation_ms", 0.0))
            forward_ms = float(data.get("forward_ms", 0.0))
            backward_ms = float(data.get("backward_ms", 0.0))
            optimizer_ms = float(data.get("optimizer_ms", 0.0))
            total_ms = float(data.get("total_ms", 0.0))
            self._last_performance_metrics = {
                "data_wait_ms": data_wait_ms,
                "augmentation_ms": augmentation_ms,
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "optimizer_ms": optimizer_ms,
                "total_ms": total_ms,
            }
            self.performance_label.setText(
                self._format_performance_text(
                    data_wait_ms,
                    augmentation_ms,
                    forward_ms,
                    backward_ms,
                    optimizer_ms,
                    total_ms,
                )
            )
            if math.isfinite(total_ms) and total_ms > 0.0:
                self._train_speed_batches_per_sec = 1000.0 / total_ms
            else:
                self._train_speed_batches_per_sec = None
            self._set_training_batch_progress_bar()
            self._update_memory_runtime_label()
            return
