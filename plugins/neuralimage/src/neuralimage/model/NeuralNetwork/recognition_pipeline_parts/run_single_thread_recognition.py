"""recognition_pipeline: run single thread recognition."""
from __future__ import annotations
from typing import TYPE_CHECKING
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from kraken_core.asset_source import AssetSource
from neuralimage.active_learning.config import ActiveLearningConfig
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.uncertainty.config import UncertaintyConfig

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.cut_image_prepare import cut_image_prepare

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.cut_image_process import _publish_memory_metrics

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.gpu_predict import gpu_predict

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.load_preview_array import _publish_recognition_preview

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.recognition_workload import ProgressReporter

from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.sew import sew

def run_single_thread_recognition(
    *,
    source_files: list[Path],
    result_folder: Path,
    part_size: tuple[int, int],
    overlap: int,
    batch_size: int,
    colors: int,
    model: nn.Module,
    device: torch.device,
    stop_event: Any,
    publish: Publisher,
    collect_memory_metrics: MemoryMetricsCollector,
    jpeg_quality: int = 95,
    binarize_output: bool = True,
    threshold: float | None = None,
    postprocess_enabled: bool = False,
    postprocess_kernel_size: int = 3,
    recognition_tta_enabled: bool = False,
    confidence_tta_enabled: bool = False,
    confidence_save_mode: str = 'off',
    use_context_branch: bool = False,
    use_cross_attention: bool = True,
    context_crop_size: tuple[int, int] | None = None,
    context_input_size: tuple[int, int] | None = None,
    compression_factor: int = 1,
    source_root: Path | None = None,
    preprocessing: PreprocessingConfig | None = None,
    lossless_binary_png: bool = False,
    active_learning: ActiveLearningConfig | None = None,
    active_learning_metadata: dict[str, object] | None = None,
    uncertainty: UncertaintyConfig | None = None,
    asset_source: AssetSource | None = None,
) -> None:
    from neuralimage.model.NeuralNetwork.recognition_pipeline import gpu_predict, sew

    from neuralimage.model.NeuralNetwork.recognition_pipeline import cut_image_prepare

    model.eval()
    model.to(device)
    publish('logging', 'Recognition started in single-thread mode.')

    reporter = ProgressReporter(publish=publish, total_frames=len(source_files))
    reporter.publish_started()
    _publish_memory_metrics(publish=publish, collect_memory_metrics=collect_memory_metrics)
    compile_fallback_checked = False

    shape = (colors, part_size[0], part_size[1])
    for index, image_path in enumerate(source_files, start=1):
        if stop_event.is_set():
            break
        prepared = cut_image_prepare(
            image_path,
            shape,
            overlap,
            use_context_branch=use_context_branch,
            use_cross_attention=use_cross_attention,
            context_crop_size=context_crop_size,
            context_input_size=context_input_size,
            compression_factor=compression_factor,
            source_root=source_root,
            preprocessing=preprocessing,
            asset_source=asset_source,
        )
        predicted = gpu_predict(
            prepared,
            model,
            device,
            batch_size,
            recognition_tta_enabled=recognition_tta_enabled,
            confidence_tta_enabled=confidence_tta_enabled,
            uncertainty=uncertainty,
        )
        # Some torch.compile configurations can produce degenerate outputs on inference.
        if (not compile_fallback_checked) and hasattr(model, '_orig_mod'):
            compile_fallback_checked = True
            compiled_stats = predicted.get('_prediction_stats')
            compiled_max = (
                float(compiled_stats.get('max', 0.0))
                if isinstance(compiled_stats, dict)
                else 0.0
            )
            original_model = getattr(model, '_orig_mod', None)
            if isinstance(original_model, nn.Module) and compiled_max <= 0.05:
                publish(
                    'logging',
                    (
                        'Recognition: low-probability output detected on torch.compile model; '
                        'retrying with uncompiled model.'
                    ),
                )
                original_model.eval()
                original_model.to(device)
                fallback_predicted = gpu_predict(
                    prepared,
                    original_model,
                    device,
                    batch_size,
                    recognition_tta_enabled=recognition_tta_enabled,
                    confidence_tta_enabled=confidence_tta_enabled,
                    uncertainty=uncertainty,
                )
                fallback_stats = fallback_predicted.get('_prediction_stats')
                fallback_max = (
                    float(fallback_stats.get('max', 0.0))
                    if isinstance(fallback_stats, dict)
                    else 0.0
                )
                if fallback_max > (compiled_max + 0.01):
                    model = original_model
                    predicted = fallback_predicted
                    publish(
                        'logging',
                        (
                            'Recognition: switched to uncompiled model '
                            f'(max prob {compiled_max:.6f} -> {fallback_max:.6f}).'
                        ),
                    )
        stats = predicted.get('_prediction_stats')
        if isinstance(stats, dict):
            prob_min = float(stats.get('min', 0.0))
            prob_max = float(stats.get('max', 0.0))
            prob_mean = float(stats.get('mean', 0.0))
            non_finite = int(stats.get('non_finite', 0))
            fp32_retries = int(stats.get('fp32_retries', 0))
            should_log_stats = index == 1 or non_finite > 0 or prob_max <= 0.05
            if should_log_stats:
                publish(
                    'logging',
                    (
                        f'Recognition output stats [{index}/{len(source_files)}]: '
                        f'min={prob_min:.6f}, max={prob_max:.6f}, mean={prob_mean:.6f}, '
                        f'non_finite={non_finite}, fp32_retries={fp32_retries}'
                    ),
                )
            if prob_max <= 0.05:
                publish(
                    'logging',
                    (
                        'Recognition warning: output probabilities are very low '
                        '(max <= 0.05), resulting masks can look black.'
                    ),
                )
        output_path = sew(
            result_folder,
            predicted,
            jpeg_quality=jpeg_quality,
            threshold=(float(threshold) if binarize_output and threshold is not None else None),
            postprocess_kernel_size=(int(postprocess_kernel_size) if postprocess_enabled else 0),
            confidence_save_mode=confidence_save_mode,
            lossless_binary_png=lossless_binary_png,
            active_learning=active_learning,
            active_learning_metadata=active_learning_metadata,
        )
        _publish_recognition_preview(
            publish=publish,
            source_path=image_path,
            output_path=output_path,
            frame_name=image_path.name,
            asset_source=asset_source,
        )
        reporter.publish_frame(index)
        publish('metrics', {
            'type': 'recognition_completed',
            'source_path': str(image_path),
            'output_path': str(output_path),
        })
        _publish_memory_metrics(publish=publish, collect_memory_metrics=collect_memory_metrics)

if TYPE_CHECKING:
    from neuralimage.model.NeuralNetwork.recognition_pipeline_parts.module_state import MemoryMetricsCollector, Publisher
