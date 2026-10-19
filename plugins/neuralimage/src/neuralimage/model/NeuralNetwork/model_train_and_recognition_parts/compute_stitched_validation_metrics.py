"""model_train_and_recognition: compute stitched validation metrics."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from neuralimage.model.NeuralNetwork.dataset import NoCutDataset
from neuralimage.lib.image_processing import sew_image
from neuralimage.metrics.segmentation import compute_segmentation_metrics


class ComputeStitchedValidationMetricsMixin:
    def _compute_stitched_validation_metrics(
        self,
        export_cache: _ValidationExportCache | None,
        *,
        threshold: float,
    ) -> tuple[dict[str, float], int]:
        if export_cache is None or export_cache.mode != 'no_cut':
            return {}, 0
        base_dataset = self._unwrap_validation_dataset(export_cache.dataset)
        if not isinstance(base_dataset, NoCutDataset):
            return {}, 0

        count_metrics = {
            'connected_component_difference',
            'wire_break_count',
            'false_bridge_count',
            'missed_component_count',
            'spurious_component_count',
            'foreground_component_delta',
            'background_component_delta',
            'topology_violation_count',
        }
        totals: dict[str, float] = {}
        frame_count = 0
        confidence_histogram: dict[str, int] = {}
        boundary_tolerance, include_hd95, confidence_bins = self._advanced_validation_options()
        for frame_cache in (export_cache.frame_predictions or {}).values():
            predicted = self._stack_validation_patches(frame_cache, frame_cache.patches)
            if predicted is None:
                return {}, 0
            probability = np.asarray(
                sew_image(
                    frame_cache.baseim_size,
                    predicted,
                    int(frame_cache.overlap),
                    threshold=None,
                    postprocess_kernel_size=0,
                ),
                dtype=np.float32,
            ) / 255.0
            confidence = None
            confidence_patches = self._stack_validation_patches(
                frame_cache,
                frame_cache.confidence_patches,
            )
            if confidence_patches is not None:
                confidence = np.asarray(
                    sew_image(
                        frame_cache.baseim_size,
                        confidence_patches,
                        int(frame_cache.overlap),
                        threshold=None,
                        postprocess_kernel_size=0,
                    ),
                    dtype=np.float32,
                ) / 255.0

            _image_path, _prepared_image, prepared_label, _rare_mask = (
                base_dataset._prepare_frame_images(int(frame_cache.frame_index))
            )
            target = np.asarray(prepared_label.convert('L'), dtype=np.float32) / 255.0
            frame_metrics = compute_segmentation_metrics(
                probability,
                target,
                threshold=float(threshold),
                confidence=confidence,
                boundary_tolerance=boundary_tolerance,
                include_hd95=include_hd95,
                confidence_bins=confidence_bins,
            ).as_dict()
            for metric_name, metric_value in frame_metrics.items():
                if metric_name == 'confidence_histogram':
                    for bin_name, bin_count in dict(metric_value or {}).items():
                        confidence_histogram[str(bin_name)] = (
                            confidence_histogram.get(str(bin_name), 0) + int(bin_count)
                        )
                elif isinstance(metric_value, (int, float)):
                    totals[metric_name] = totals.get(metric_name, 0.0) + float(metric_value)
            frame_count += 1

        if frame_count <= 0:
            return {}, 0
        for metric_name in tuple(totals):
            if metric_name not in count_metrics:
                totals[metric_name] /= float(frame_count)
        for bin_name, bin_count in confidence_histogram.items():
            totals[f'confidence_histogram/{bin_name}'] = float(bin_count)
        return totals, frame_count

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.model_train_and_recognition_parts.validation_no_cut_frame import _ValidationExportCache
