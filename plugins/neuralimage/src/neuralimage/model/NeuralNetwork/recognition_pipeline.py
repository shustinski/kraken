from __future__ import annotations
import os
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable, Iterable, cast
from contextlib import suppress
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from multiprocessing.synchronize import Event as MpEvent
from PIL import Image
from kraken_core.asset_source import AssetSource
from neuralimage.lib.images import open_image
from neuralimage.active_learning.config import ActiveLearningConfig
from neuralimage.active_learning.export import ActiveLearningExporter
from neuralimage.lib.file_retry import retry_file_read
from neuralimage.lib.image_processing import cut_image, sew_image
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.uncertainty.config import UncertaintyConfig
from neuralimage.preprocessing.pipeline import SemPreprocessingPipeline, image_to_channel_first_float01
from neuralimage.model.NeuralNetwork.model_io import load_model_artifact
from neuralimage.model.NeuralNetwork.context_utils import (
    build_context_batch,
    build_global_context_image,
    build_patch_coordinate_batch,
)
from neuralimage.model.NeuralNetwork.blocks import extract_confidence_output, extract_mask_outputs

from .recognition_pipeline_parts.module_state import (
    Publisher,
    MemoryMetricsCollector,
    _QUEUE_EMPTY,
)
from .recognition_pipeline_parts.run import RunMixin
from .recognition_pipeline_parts.monitor_processing import MonitorProcessingMixin
from .recognition_pipeline_parts.recognition_workload import (
    _resolve_relative_frame_name,
    RecognitionWorkload,
    WorkerCounts,
    RuntimeWorkerConfig,
    PipelineQueues,
    WorkerGroups,
    ProgressReporter,
    _try_get_queue_item,
    _report_worker_exception,
)
from .recognition_pipeline_parts.run_cut_worker import (
    _run_cut_worker,
    _run_predict_worker,
    _run_sew_worker,
    _try_get_worker_error_message,
    run_multiprocessing_recognition,
)
from .recognition_pipeline_parts.run_single_thread_recognition import run_single_thread_recognition
from .recognition_pipeline_parts.cut_image_process import (
    _publish_memory_metrics,
    cut_image_process,
    _apply_preprocessing_to_channel_first,
)
from .recognition_pipeline_parts.cut_image_prepare import cut_image_prepare
from .recognition_pipeline_parts.imgpredict import (
    get_array_from_image,
    _to_channel_first,
    imgpredict,
)
from .recognition_pipeline_parts.gpu_predict import gpu_predict
from .recognition_pipeline_parts.imgsew import (
    create_batches,
    imgsew,
    sew_from_queue,
    _confidence_output_root,
    _save_jpeg_atomic,
    _save_png_atomic,
)
from .recognition_pipeline_parts.sew import sew
from .recognition_pipeline_parts.load_preview_array import (
    _load_preview_array,
    _publish_recognition_preview,
    _store_payload_array_for_multiprocessing,
    _restore_payload_array_from_multiprocessing,
    _should_use_torch_tensor_transport,
)

class MultiprocessingRecognitionRunner(RunMixin, MonitorProcessingMixin):
    pass

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    Publisher,
    MemoryMetricsCollector,
    _QUEUE_EMPTY,
    _resolve_relative_frame_name,
    RecognitionWorkload,
    WorkerCounts,
    RuntimeWorkerConfig,
    PipelineQueues,
    WorkerGroups,
    ProgressReporter,
    _try_get_queue_item,
    _report_worker_exception,
    _run_cut_worker,
    _run_predict_worker,
    _run_sew_worker,
    _try_get_worker_error_message,
    run_multiprocessing_recognition,
    run_single_thread_recognition,
    _publish_memory_metrics,
    cut_image_process,
    _apply_preprocessing_to_channel_first,
    cut_image_prepare,
    get_array_from_image,
    _to_channel_first,
    imgpredict,
    gpu_predict,
    create_batches,
    imgsew,
    sew_from_queue,
    _confidence_output_root,
    _save_jpeg_atomic,
    _save_png_atomic,
    sew,
    _load_preview_array,
    _publish_recognition_preview,
    _store_payload_array_for_multiprocessing,
    _restore_payload_array_from_multiprocessing,
    _should_use_torch_tensor_transport,
)
