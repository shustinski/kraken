import shutil
import threading
import os
import sys
import hashlib
import math
import random
import time
import json
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler, Subset
from neuralimage.lib.data_interfaces import (
    WorkMode,
    RecognitionParameters,
    TrainingParameters,
    SampleCutMode,
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
    build_tech_augmentation_config,
    normalize_multi_gpu_mode,
    normalize_validation_source,
)
from neuralimage.lib.file_func import filter_files, filter_images
from neuralimage.lib.func import get_input_channels
from neuralimage.lib.images import ImagePreparator, open_image
from kraken_core.asset_source import AssetSource
from neuralimage.infrastructure.dataset_source import DatasetStreamSource
from neuralimage.lib.message_bus import AbstractMessageBus
from neuralimage.model.NeuralNetwork import create_model, model_supports_init_kwarg
from neuralimage.model.NeuralNetwork.context_utils import normalize_size_pair
from neuralimage.model.NeuralNetwork.dataset import (
    NoCutDataset,
    PatchSampleRequest,
    SyntheticDefectDataset,
)
from neuralimage.model.NeuralNetwork.model_io import load_model_artifact
from neuralimage.model.NeuralNetwork.model_train_and_recognition import ModelRecognizer, ModelTrainer
from neuralimage.model.image_workers import ConvertCifThread
from neuralimage.preprocessing.config import PreprocessingConfig, build_preprocessing_config
from neuralimage.preprocessing.pipeline import image_to_channel_first_float01
from neuralimage.preprocessing.statistics import compute_dataset_statistics
from neuralimage.training.hard_mining import compute_geometry_difficulty_score

from .general_neural_handler_parts.module_state import (
    _VALIDATION_SPLIT_SEED,
    _VALIDATION_FOREGROUND_BUCKETS,
)
from .general_neural_handler_parts.construction import ConstructionMixin
from .general_neural_handler_parts.resolve_model_creation_kwargs import ResolveModelCreationKwargsMixin
from .general_neural_handler_parts.resolve_training_model import ResolveTrainingModelMixin
from .general_neural_handler_parts.create_dataset import CreateDatasetMixin
from .general_neural_handler_parts.resolve_training_preprocessing import ResolveTrainingPreprocessingMixin
from .general_neural_handler_parts.create_dataloader import CreateDataloaderMixin
from .general_neural_handler_parts.resolve_dataloader_workers import ResolveDataloaderWorkersMixin
from .general_neural_handler_parts.collect_matched_samples import CollectMatchedSamplesMixin
from .general_neural_handler_parts.declare_model_name import DeclareModelNameMixin
from .general_neural_handler_parts.start_training import StartTrainingMixin
from .general_neural_handler_parts.wait_current_thread import WaitCurrentThreadMixin
from .general_neural_handler_parts.estimate_label_foreground_ratio import (
    _validation_grid_step,
    _is_debugger_attached,
    _stable_sample_sort_key,
    _estimate_label_foreground_ratio,
    _label_ratio_bucket,
)
from .general_neural_handler_parts.deterministic_validation_split import _deterministic_validation_split
from .general_neural_handler_parts.indexed_dataset import IndexedDataset
from .general_neural_handler_parts.composite_dataset import CompositeDataset
from .general_neural_handler_parts.hard_frame_sampler import HardFrameSampler
from .general_neural_handler_parts.random_patch_batch_sampler import (
    RandomPatchBatchSampler,
    FrameBatchSampler,
)

class GeneralNeuralHandler(
    ConstructionMixin,
    ResolveModelCreationKwargsMixin,
    ResolveTrainingModelMixin,
    CreateDatasetMixin,
    ResolveTrainingPreprocessingMixin,
    CreateDataloaderMixin,
    ResolveDataloaderWorkersMixin,
    CollectMatchedSamplesMixin,
    DeclareModelNameMixin,
    StartTrainingMixin,
    WaitCurrentThreadMixin,
):
    dataset_source: DatasetStreamSource | None = None
    EXISTING_FOLDER_DEFAULT_ANSWER = False
    EXISTING_FOLDER_TIMEOUT_SECONDS = 15
    STOP_JOIN_POLL_SECONDS = 0.2
    STOP_JOIN_GRACE_SECONDS = 10.0

# Imported so existing callers and tests can patch or import these names
# from this module after the implementation moved into focused units.
_FACADE_EXPORTS = (
    _VALIDATION_SPLIT_SEED,
    _VALIDATION_FOREGROUND_BUCKETS,
    _validation_grid_step,
    _is_debugger_attached,
    _stable_sample_sort_key,
    _estimate_label_foreground_ratio,
    _label_ratio_bucket,
    _deterministic_validation_split,
    IndexedDataset,
    CompositeDataset,
    HardFrameSampler,
    RandomPatchBatchSampler,
    FrameBatchSampler,
)
