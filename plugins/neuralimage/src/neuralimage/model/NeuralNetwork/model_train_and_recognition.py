import os
import datetime
import time
import socket
import sys
import importlib.util
import math
import random
import re
import csv
import json
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty
from collections import deque
from collections.abc import Callable, Mapping, Sized
from contextlib import nullcontext
from typing import Any, ContextManager, Iterator, Protocol, cast
import multiprocessing as mp
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler
from PIL import Image, ImageDraw, ImageFont
from neuralimage.lib import System
from neuralimage.model.NeuralNetwork.dataset import NoCutDataset, index_in_list
from neuralimage.lib.data_interfaces import (
    CutoutParameters,
    RecognitionParameters,
    OptimizerParameters,
    OptimizerName,
    MixupParameters,
    MixedPrecisionMode,
    EarlyStoppingParameters,
    RandomArtifactsParameters,
    SchedulerName,
    SchedulerParameters,
    WarmupParameters,
    HardMiningParameters,
    normalize_multi_gpu_mode,
)
from neuralimage.lib.file_retry import retry_file_read
from neuralimage.lib.file_func import filter_images
from neuralimage.lib.func import get_input_channels
from neuralimage.lib.images import ImagePreparator, SampleCalculator
from kraken_core.asset_source import AssetSource
from neuralimage.lib.image_processing import sew_image
from neuralimage.lib.images import _build_enabled_transform_variants
from neuralimage.lib.loss_config import format_loss_formula, resolve_loss_term_weights, sanitize_loss_term_weights
from neuralimage.lib.message_bus import AbstractMessageBus
from neuralimage.lib.optimizer_availability import MUON_UNAVAILABLE_MESSAGE, resolve_muon_optimizer_class
from neuralimage.lib.random_artifacts import generate_random_artifact_patch
from neuralimage.losses.boundary import compute_boundary_loss, compute_soft_boundary_map
from neuralimage.losses.cldice import compute_cldice_loss, soft_skeletonize
from neuralimage.losses.composite import (
    AuxiliaryHeadLoss,
    HomoscedasticLossWeighter,
    compute_auxiliary_head_loss,
    resolve_auxiliary_head_weights,
)
from neuralimage.losses.distance_boundary import compute_distance_boundary_loss
from neuralimage.losses.topograph import (
    build_topograph_loss,
    compute_topograph_loss_per_sample,
    extract_critical_region_mask,
)
from neuralimage.losses.topograph_viz import render_critical_regions_overlay
from neuralimage.metrics.segmentation import compute_segmentation_metrics
from neuralimage.preprocessing.config import build_preprocessing_config
from neuralimage.model.NeuralNetwork.blocks import extract_confidence_output, extract_mask_outputs
from neuralimage.targets.dataset_hooks import extract_primary_label
from neuralimage.training.batch_transfer import move_batch_to_device
from neuralimage.model.NeuralNetwork.context_utils import normalize_size_pair
from neuralimage.model.NeuralNetwork.early_stopping import (
    CheckpointManager,
    EarlyStoppingConfigCalculator,
    EarlyStoppingPolicy,
    EarlyStoppingRuntimeConfig,
    MetricEvaluator,
)
from neuralimage.model.NeuralNetwork.model_io import load_model_artifact, save_model_artifact
from neuralimage.model.NeuralNetwork.recognition_pipeline import (
    RecognitionWorkload,
    WorkerCounts,
    cut_image_prepare as _cut_image_prepare,
    create_batches as _create_batches,
    cut_image_process as _cut_image_process,
    get_array_from_image as _get_array_from_image,
    gpu_predict as _gpu_predict,
    imgpredict as _imgpredict,
    imgsew as _imgsew,
    run_multiprocessing_recognition,
    run_single_thread_recognition,
    sew as _sew,
    sew_from_queue as _sew_from_queue,
)

from .model_train_and_recognition_parts.module_state import (
    CHECKPOINT_SUFFIX,
    STOP_TOKEN,
    TRAINING_PROFILER_ENABLED,
    RANDOM_PATCH_PROFILING_ENABLED,
    RANDOM_PATCH_PROFILE_BATCHES,
    RANDOM_PATCH_PROFILE_LOG_EVERY,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    FOCAL_TVERSKY_ALPHA,
    FOCAL_TVERSKY_BETA,
    BOUNDARY_LOSS_KERNEL_SIZE,
    CLDICE_SKELETON_ITERATIONS,
    VALIDATION_THRESHOLD_CANDIDATES,
    VALIDATION_CHECK_INTERVAL_BATCHES,
    DEEP_SUPERVISION_DECAY,
    CONFIDENCE_LOSS_WEIGHT,
    RECOGNITION_AUX_WORKERS_PER_GPU,
    VALIDATION_EXPORT_CACHE_MAX_BYTES,
    _ARTIFACT_NAME_SANITIZE_PATTERN,
    _RANDOM_ARTIFACT_BANK_TARGET_PER_BUCKET,
    _RANDOM_ARTIFACT_BANK_BUCKET_GRANULARITY,
    _RANDOM_ARTIFACT_BANK_READY_TIMEOUT_SEC,
    NON_FINITE_GRADIENT_SKIP_LIMIT,
    TRAINING_INSTABILITY_MESSAGE,
    _TORCH_COMPILE_MODES,
    _MAX_AUTOTUNE_MIN_SMS,
)
from .model_train_and_recognition_parts.construction import ConstructionMixin
from .model_train_and_recognition_parts.create_training_process import CreateTrainingProcessMixin
from .model_train_and_recognition_parts.run import RunMixin
from .model_train_and_recognition_parts.stop import StopMixin
from .model_train_and_recognition_parts.construction_2 import Construction2Mixin
from .model_train_and_recognition_parts.resolve_artifact_runtime_metadata import ResolveArtifactRuntimeMetadataMixin
from .model_train_and_recognition_parts.render_chart_image import RenderChartImageMixin
from .model_train_and_recognition_parts.save_chart_csv import SaveChartCsvMixin
from .model_train_and_recognition_parts.save_metric_charts import SaveMetricChartsMixin
from .model_train_and_recognition_parts.build_distributed_loader import BuildDistributedLoaderMixin
from .model_train_and_recognition_parts.save_checkpoint import SaveCheckpointMixin
from .model_train_and_recognition_parts.save_pause_checkpoint import SavePauseCheckpointMixin
from .model_train_and_recognition_parts.load_checkpoint_if_available import LoadCheckpointIfAvailableMixin
from .model_train_and_recognition_parts.create_warmup_scheduler import CreateWarmupSchedulerMixin
from .model_train_and_recognition_parts.create_lr_scheduler import CreateLrSchedulerMixin
from .model_train_and_recognition_parts.step_epoch_scheduler import StepEpochSchedulerMixin
from .model_train_and_recognition_parts.compute_auxiliary_supervision_loss import ComputeAuxiliarySupervisionLossMixin
from .model_train_and_recognition_parts.resolve_random_artifact_prewarm import ResolveRandomArtifactPrewarmMixin
from .model_train_and_recognition_parts.compute_soft_overlap_losses import ComputeSoftOverlapLossesMixin
from .model_train_and_recognition_parts.compute_single_per_sample import ComputeSinglePerSampleMixin
from .model_train_and_recognition_parts.compute_per_sample_loss import ComputePerSampleLossMixin
from .model_train_and_recognition_parts.compute_binary_metrics import ComputeBinaryMetricsMixin
from .model_train_and_recognition_parts.create_validation_export_cache import CreateValidationExportCacheMixin
from .model_train_and_recognition_parts.stack_validation_patches import StackValidationPatchesMixin
from .model_train_and_recognition_parts.compute_stitched_validation_metrics import ComputeStitchedValidationMetricsMixin
from .model_train_and_recognition_parts.collect_validation_export_batch import CollectValidationExportBatchMixin
from .model_train_and_recognition_parts.save_validation_binary_predictions import SaveValidationBinaryPredictionsMixin
from .model_train_and_recognition_parts.save_validation_binary_predictions_2 import SaveValidationBinaryPredictions2Mixin
from .model_train_and_recognition_parts.run_validation_epoch import RunValidationEpochMixin
from .model_train_and_recognition_parts.create_split_muon_adamw import CreateSplitMuonAdamwMixin
from .model_train_and_recognition_parts.try_compile_model import TryCompileModelMixin
from .model_train_and_recognition_parts.resolve_training_profiler_config import ResolveTrainingProfilerConfigMixin
from .model_train_and_recognition_parts.record_random_patch_profile import RecordRandomPatchProfileMixin
from .model_train_and_recognition_parts.save_random_patch_profile import SaveRandomPatchProfileMixin
from .model_train_and_recognition_parts.start_training_profiler import StartTrainingProfilerMixin
from .model_train_and_recognition_parts.log_loss_configuration import LogLossConfigurationMixin
from .model_train_and_recognition_parts.log_scheduler_configuration import LogSchedulerConfigurationMixin
from .model_train_and_recognition_parts.create_run_context import CreateRunContextMixin
from .model_train_and_recognition_parts.restore_training_state import RestoreTrainingStateMixin
from .model_train_and_recognition_parts.filter_uniform_batch_samples import FilterUniformBatchSamplesMixin
from .model_train_and_recognition_parts.apply_mixup_batch import ApplyMixupBatchMixin
from .model_train_and_recognition_parts.apply_random_artifacts_batch import ApplyRandomArtifactsBatchMixin
from .model_train_and_recognition_parts.inject_local_batch_into import InjectLocalBatchIntoMixin
from .model_train_and_recognition_parts.update_loss_aware_sampling import UpdateLossAwareSamplingMixin
from .model_train_and_recognition_parts.log_non_finite_training import LogNonFiniteTrainingMixin
from .model_train_and_recognition_parts.has_non_finite_gradients import HasNonFiniteGradientsMixin
from .model_train_and_recognition_parts.prepare_train_batch import PrepareTrainBatchMixin
from .model_train_and_recognition_parts.run_train_step import RunTrainStepMixin
from .model_train_and_recognition_parts.publish_train_batch_runtime import PublishTrainBatchRuntimeMixin
from .model_train_and_recognition_parts.evaluate_automatic_stop_metric import EvaluateAutomaticStopMetricMixin
from .model_train_and_recognition_parts.run_automatic_early_stopping import RunAutomaticEarlyStoppingMixin
from .model_train_and_recognition_parts.periodic_validation_due import PeriodicValidationDueMixin
from .model_train_and_recognition_parts.run_train_epoch import RunTrainEpochMixin
from .model_train_and_recognition_parts.reduce_epoch_train_stats import ReduceEpochTrainStatsMixin
from .model_train_and_recognition_parts.publish_epoch_load_breakdown import PublishEpochLoadBreakdownMixin
from .model_train_and_recognition_parts.publish_validation_metrics import PublishValidationMetricsMixin
from .model_train_and_recognition_parts.run_single_training_epoch import RunSingleTrainingEpochMixin
from .model_train_and_recognition_parts.should_stop_after_epoch import ShouldStopAfterEpochMixin
from .model_train_and_recognition_parts.run_impl import RunImplMixin
from .model_train_and_recognition_parts.resolve_output_threshold import ResolveOutputThresholdMixin
from .model_train_and_recognition_parts.resolve_context_branch_settings import ResolveContextBranchSettingsMixin
from .model_train_and_recognition_parts.prepare_model import PrepareModelMixin
from .model_train_and_recognition_parts.combined_optimizer import (
    _CombinedOptimizer,
    TrainingInstabilityError,
    _TrainingPauseRequested,
    MuonOptimizerUnavailableError,
    _NoOpQueue,
    _RecordingQueue,
    _drain_process_queue,
    _join_process_with_escalation,
    _get_process_descendants,
)
from .model_train_and_recognition_parts.stop_process_descendants import (
    _stop_process_descendants,
    _ddp_worker_entry,
    _SupportsSetEpoch,
    _SupportsLossAwareSampling,
)
from .model_train_and_recognition_parts.distributed_loss_aware_sampler import _DistributedLossAwareSampler
from .model_train_and_recognition_parts.epoch_stats import (
    _TrainLoopStrides,
    _EpochStats,
    _EarlyStoppingState,
    _EarlyStoppingConfig,
    _RunContext,
    _TrainingProfilerConfig,
    _ActiveTrainProfiler,
    _PreparedTrainBatch,
    _TrainStepResult,
    _TrainingRuntimeState,
)
from .model_train_and_recognition_parts.random_artifact_bank import _RandomArtifactBank
from .model_train_and_recognition_parts.validation_no_cut_frame import (
    _ValidationNoCutFrameExportCache,
    _ValidationExportCache,
    _release_torch_memory,
)
from .model_train_and_recognition_parts.collect_memory_metrics import _collect_memory_metrics
from .model_train_and_recognition_parts.recognizer_process import (
    _resolve_memory_root_process,
    _is_module_available,
    _get_torch_compile_unavailable_reason,
    _is_debugger_attached,
    _resolve_torch_compile_mode,
    RecognitionRuntimePlan,
    _QueueMessageBus,
    RecognizerProcess,
)
from .model_train_and_recognition_parts.model_recognizer import ModelRecognizer

class ModelTrainer(ConstructionMixin, CreateTrainingProcessMixin, RunMixin, StopMixin, threading.Thread):
    pass

class TrainerProcess(
    Construction2Mixin,
    ResolveArtifactRuntimeMetadataMixin,
    RenderChartImageMixin,
    SaveChartCsvMixin,
    SaveMetricChartsMixin,
    BuildDistributedLoaderMixin,
    SaveCheckpointMixin,
    SavePauseCheckpointMixin,
    LoadCheckpointIfAvailableMixin,
    CreateWarmupSchedulerMixin,
    CreateLrSchedulerMixin,
    StepEpochSchedulerMixin,
    ComputeAuxiliarySupervisionLossMixin,
    ResolveRandomArtifactPrewarmMixin,
    ComputeSoftOverlapLossesMixin,
    ComputeSinglePerSampleMixin,
    ComputePerSampleLossMixin,
    ComputeBinaryMetricsMixin,
    CreateValidationExportCacheMixin,
    StackValidationPatchesMixin,
    ComputeStitchedValidationMetricsMixin,
    CollectValidationExportBatchMixin,
    SaveValidationBinaryPredictionsMixin,
    SaveValidationBinaryPredictions2Mixin,
    RunValidationEpochMixin,
    CreateSplitMuonAdamwMixin,
    TryCompileModelMixin,
    ResolveTrainingProfilerConfigMixin,
    RecordRandomPatchProfileMixin,
    SaveRandomPatchProfileMixin,
    StartTrainingProfilerMixin,
    LogLossConfigurationMixin,
    LogSchedulerConfigurationMixin,
    CreateRunContextMixin,
    RestoreTrainingStateMixin,
    FilterUniformBatchSamplesMixin,
    ApplyMixupBatchMixin,
    ApplyRandomArtifactsBatchMixin,
    InjectLocalBatchIntoMixin,
    UpdateLossAwareSamplingMixin,
    LogNonFiniteTrainingMixin,
    HasNonFiniteGradientsMixin,
    PrepareTrainBatchMixin,
    RunTrainStepMixin,
    PublishTrainBatchRuntimeMixin,
    EvaluateAutomaticStopMetricMixin,
    RunAutomaticEarlyStoppingMixin,
    PeriodicValidationDueMixin,
    RunTrainEpochMixin,
    ReduceEpochTrainStatsMixin,
    PublishEpochLoadBreakdownMixin,
    PublishValidationMetricsMixin,
    RunSingleTrainingEpochMixin,
    ShouldStopAfterEpochMixin,
    RunImplMixin,
    mp.Process,
):
    pass

class NeuralRecognizer(
    ResolveOutputThresholdMixin,
    ResolveContextBranchSettingsMixin,
    PrepareModelMixin,
    threading.Thread,
):
    pass

from .model_train_and_recognition_parts.cut_image_process import (
    NeuralRecognitioner,
    cut_image_process,
    cut_image_prepare,
    get_array_from_image,
    imgpredict,
    gpu_predict,
    create_batches,
    imgsew,
    sew_from_queue,
    sew,
)

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    CHECKPOINT_SUFFIX,
    STOP_TOKEN,
    TRAINING_PROFILER_ENABLED,
    RANDOM_PATCH_PROFILING_ENABLED,
    RANDOM_PATCH_PROFILE_BATCHES,
    RANDOM_PATCH_PROFILE_LOG_EVERY,
    FOCAL_LOSS_ALPHA,
    FOCAL_LOSS_GAMMA,
    FOCAL_TVERSKY_ALPHA,
    FOCAL_TVERSKY_BETA,
    BOUNDARY_LOSS_KERNEL_SIZE,
    CLDICE_SKELETON_ITERATIONS,
    VALIDATION_THRESHOLD_CANDIDATES,
    VALIDATION_CHECK_INTERVAL_BATCHES,
    DEEP_SUPERVISION_DECAY,
    CONFIDENCE_LOSS_WEIGHT,
    RECOGNITION_AUX_WORKERS_PER_GPU,
    VALIDATION_EXPORT_CACHE_MAX_BYTES,
    _ARTIFACT_NAME_SANITIZE_PATTERN,
    _RANDOM_ARTIFACT_BANK_TARGET_PER_BUCKET,
    _RANDOM_ARTIFACT_BANK_BUCKET_GRANULARITY,
    _RANDOM_ARTIFACT_BANK_READY_TIMEOUT_SEC,
    NON_FINITE_GRADIENT_SKIP_LIMIT,
    TRAINING_INSTABILITY_MESSAGE,
    _TORCH_COMPILE_MODES,
    _MAX_AUTOTUNE_MIN_SMS,
    _CombinedOptimizer,
    TrainingInstabilityError,
    _TrainingPauseRequested,
    MuonOptimizerUnavailableError,
    _NoOpQueue,
    _RecordingQueue,
    _drain_process_queue,
    _join_process_with_escalation,
    _get_process_descendants,
    _stop_process_descendants,
    _ddp_worker_entry,
    _SupportsSetEpoch,
    _SupportsLossAwareSampling,
    _DistributedLossAwareSampler,
    _TrainLoopStrides,
    _EpochStats,
    _EarlyStoppingState,
    _EarlyStoppingConfig,
    _RunContext,
    _TrainingProfilerConfig,
    _ActiveTrainProfiler,
    _PreparedTrainBatch,
    _TrainStepResult,
    _TrainingRuntimeState,
    _RandomArtifactBank,
    _ValidationNoCutFrameExportCache,
    _ValidationExportCache,
    _release_torch_memory,
    _collect_memory_metrics,
    _resolve_memory_root_process,
    _is_module_available,
    _get_torch_compile_unavailable_reason,
    _is_debugger_attached,
    _resolve_torch_compile_mode,
    RecognitionRuntimePlan,
    _QueueMessageBus,
    RecognizerProcess,
    ModelRecognizer,
    NeuralRecognitioner,
    cut_image_process,
    cut_image_prepare,
    get_array_from_image,
    imgpredict,
    gpu_predict,
    create_batches,
    imgsew,
    sew_from_queue,
    sew,
)
