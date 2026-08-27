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
from PIL import Image

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
from neuralimage.preprocessing.config import PreprocessingConfig
from neuralimage.preprocessing.pipeline import image_to_channel_first_float01
from neuralimage.preprocessing.statistics import compute_dataset_statistics
from neuralimage.training.hard_mining import compute_geometry_difficulty_score


_VALIDATION_SPLIT_SEED = 1337
_VALIDATION_FOREGROUND_BUCKETS: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.2, 1.0)


def _validation_grid_step(segment_size: tuple[int, int]) -> int:
    """Choose a deterministic covering stride without dense training overlap."""
    return max(1, min(int(segment_size[0]), int(segment_size[1])))


def _is_debugger_attached() -> bool:
    gettrace = getattr(sys, 'gettrace', None)
    if not callable(gettrace):
        return False
    try:
        return bool(gettrace())
    except Exception:
        return False


def _stable_sample_sort_key(sample: tuple[Path, Path], *, seed: int = _VALIDATION_SPLIT_SEED) -> str:
    image_path, label_path = sample
    payload = f'{seed}:{image_path.stem}:{label_path.stem}'.encode('utf-8', errors='ignore')
    return hashlib.sha1(payload).hexdigest()


def _estimate_label_foreground_ratio(label_path: Path) -> float:
    with Image.open(label_path) as image:
        grayscale = image.convert('L')
        histogram = grayscale.histogram()
    if not histogram:
        return 0.0
    total_pixels = int(sum(histogram))
    if total_pixels <= 0:
        return 0.0
    foreground_pixels = int(sum(histogram[1:]))
    return min(max(float(foreground_pixels) / float(total_pixels), 0.0), 1.0)


def _label_ratio_bucket(ratio: float) -> int:
    normalized = min(max(float(ratio), 0.0), 1.0)
    for bucket_index, upper_bound in enumerate(_VALIDATION_FOREGROUND_BUCKETS):
        if normalized <= upper_bound:
            return bucket_index
    return len(_VALIDATION_FOREGROUND_BUCKETS)


def _deterministic_validation_split(
    samples: list[tuple[Path, Path]],
    *,
    val_count: int,
    seed: int = _VALIDATION_SPLIT_SEED,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    ordered_samples = sorted(samples, key=lambda sample: _stable_sample_sort_key(sample, seed=seed))
    total_count = len(ordered_samples)
    resolved_val_count = max(0, min(int(val_count), max(0, total_count - 1)))
    if resolved_val_count <= 0:
        return ordered_samples, []

    try:
        bucket_to_samples: dict[int, list[tuple[Path, Path]]] = {}
        for sample in ordered_samples:
            ratio = _estimate_label_foreground_ratio(sample[1])
            bucket = _label_ratio_bucket(ratio)
            bucket_to_samples.setdefault(bucket, []).append(sample)
    except Exception:
        return ordered_samples[resolved_val_count:], ordered_samples[:resolved_val_count]

    bucket_counts = {bucket: len(bucket_samples) for bucket, bucket_samples in bucket_to_samples.items()}
    assigned_val_counts = {bucket: 0 for bucket in bucket_to_samples}
    remainders: list[tuple[float, int]] = []
    for bucket, bucket_size in bucket_counts.items():
        ideal = (bucket_size * resolved_val_count) / max(1, total_count)
        assigned = min(bucket_size, int(ideal))
        assigned_val_counts[bucket] = assigned
        remainders.append((ideal - assigned, bucket))

    remaining_slots = resolved_val_count - sum(assigned_val_counts.values())
    for _remainder, bucket in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining_slots <= 0:
            break
        if assigned_val_counts[bucket] >= bucket_counts[bucket]:
            continue
        assigned_val_counts[bucket] += 1
        remaining_slots -= 1

    validation_samples: list[tuple[Path, Path]] = []
    training_candidates: list[tuple[Path, Path]] = []
    for bucket, bucket_samples in sorted(bucket_to_samples.items()):
        take = min(assigned_val_counts[bucket], len(bucket_samples))
        validation_samples.extend(bucket_samples[:take])
        training_candidates.extend(bucket_samples[take:])

    if len(validation_samples) < resolved_val_count:
        missing = resolved_val_count - len(validation_samples)
        validation_samples.extend(training_candidates[:missing])
        training_candidates = training_candidates[missing:]

    validation_keys = {_stable_sample_sort_key(sample, seed=seed) for sample in validation_samples}
    training_samples = [sample for sample in ordered_samples if _stable_sample_sort_key(sample, seed=seed) not in validation_keys]
    validation_samples = [sample for sample in ordered_samples if _stable_sample_sort_key(sample, seed=seed) in validation_keys]
    return training_samples, validation_samples


class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset):
        self._base_dataset = base_dataset

    def __len__(self) -> int:
        return int(len(self._base_dataset))

    def __getitem__(self, index: int):
        resolved_index = int(index.index) if isinstance(index, PatchSampleRequest) else int(index)
        image, label = self._base_dataset[index]
        return image, label, resolved_index

    def describe_sample(self, index: int) -> str:
        describe_fn = getattr(self._base_dataset, 'describe_sample', None)
        if callable(describe_fn):
            return str(describe_fn(int(index)))
        return f'sample_{int(index):06d}'

    def set_epoch(self) -> None:
        set_epoch_fn = getattr(self._base_dataset, 'set_epoch', None)
        if callable(set_epoch_fn):
            set_epoch_fn()

    def frame_key(self, index: int) -> str:
        resolver = getattr(self._base_dataset, 'frame_key', None)
        return str(resolver(int(index))) if callable(resolver) else self.describe_sample(int(index))

    def sample_key(self, index: int) -> str:
        resolver = getattr(self._base_dataset, 'sample_key', None)
        return str(resolver(int(index))) if callable(resolver) else self.describe_sample(int(index))

    def geometry_score(self, index: int) -> float:
        resolver = getattr(self._base_dataset, 'geometry_score', None)
        if callable(resolver):
            return float(resolver(int(index)))
        _image, label = self._base_dataset[int(index)]
        if isinstance(label, dict):
            label = label['mask']
        if torch.is_tensor(label):
            label = label.detach().cpu().numpy()
        return compute_geometry_difficulty_score(label)


class CompositeDataset(Dataset):
    def __init__(self, *datasets: Dataset):
        self._datasets = [dataset for dataset in datasets if dataset is not None]
        self._lengths = [int(len(dataset)) for dataset in self._datasets]
        self._offsets: list[int] = []
        total = 0
        for length in self._lengths:
            self._offsets.append(total)
            total += int(length)
        self._total_length = total

    def __len__(self) -> int:
        return self._total_length

    def __getitem__(self, index: int):
        requested_size: tuple[int, int] | None = None
        if isinstance(index, PatchSampleRequest):
            requested_size = tuple(index.size_xy)
            index = int(index.index)
        if index < 0 or index >= self._total_length:
            raise IndexError('dataset index out of range')
        for dataset, offset, length in zip(self._datasets, self._offsets, self._lengths):
            if index < offset + length:
                local_index = int(index - offset)
                if requested_size is not None and isinstance(dataset, (NoCutDataset, SyntheticDefectDataset)):
                    return dataset[PatchSampleRequest(local_index, requested_size)]
                return dataset[local_index]
        raise IndexError('dataset index out of range')

    def describe_sample(self, index: int) -> str:
        if index < 0 or index >= self._total_length:
            raise IndexError('dataset index out of range')
        for dataset, offset, length in zip(self._datasets, self._offsets, self._lengths):
            if index >= offset + length:
                continue
            describe_fn = getattr(dataset, 'describe_sample', None)
            if callable(describe_fn):
                return str(describe_fn(index - offset))
            return f'sample_{int(index):06d}'
        return f'sample_{int(index):06d}'

    def set_epoch(self) -> None:
        for dataset in self._datasets:
            set_epoch_fn = getattr(dataset, 'set_epoch', None)
            if callable(set_epoch_fn):
                set_epoch_fn()

    def _resolve_dataset_index(self, index: int) -> tuple[Dataset, int, int]:
        if index < 0 or index >= self._total_length:
            raise IndexError('dataset index out of range')
        for dataset_number, (dataset, offset, length) in enumerate(zip(self._datasets, self._offsets, self._lengths)):
            if index < offset + length:
                return dataset, int(index - offset), int(dataset_number)
        raise IndexError('dataset index out of range')

    def frame_key(self, index: int) -> str:
        dataset, local_index, dataset_number = self._resolve_dataset_index(int(index))
        resolver = getattr(dataset, 'frame_key', None)
        value = resolver(local_index) if callable(resolver) else self.describe_sample(int(index))
        return f'{dataset_number}::{value}'

    def sample_key(self, index: int) -> str:
        dataset, local_index, dataset_number = self._resolve_dataset_index(int(index))
        resolver = getattr(dataset, 'sample_key', None)
        value = resolver(local_index) if callable(resolver) else self.describe_sample(int(index))
        return f'{dataset_number}::{value}'

    def geometry_score(self, index: int) -> float:
        dataset, local_index, _dataset_number = self._resolve_dataset_index(int(index))
        resolver = getattr(dataset, 'geometry_score', None)
        if callable(resolver):
            return float(resolver(local_index))
        _image, label = dataset[local_index]
        if isinstance(label, dict):
            label = label['mask']
        if torch.is_tensor(label):
            label = label.detach().cpu().numpy()
        return compute_geometry_difficulty_score(label)


class HardFrameSampler(Sampler[int]):
    """Deterministic online sampler combining geometry and per-sample EMA loss."""

    def __init__(self, dataset: Dataset, *, shuffle: bool = True, parameters: Any = None) -> None:
        self.dataset = dataset
        self.size = max(0, int(len(dataset)))
        self.shuffle = bool(shuffle)
        self._hard_frame_keys: set[str] = set()
        self._epoch_losses: dict[str, tuple[str, float]] = {}
        self.last_frame_losses: dict[str, float] = {}
        self.last_sigma: float = 0.0
        self._parameters = parameters
        self._ema_losses: dict[str, float] = {}
        self._geometry_scores: dict[str, float] = {}
        self._epoch = 0

    def set_geometry_scores(self, scores: dict[str, float]) -> None:
        self._geometry_scores = {str(key): max(0.0, float(value)) for key, value in scores.items()}

    @staticmethod
    def _normalize(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        low, high = min(values.values()), max(values.values())
        if high <= low:
            return {key: 0.0 for key in values}
        return {key: (value - low) / (high - low) for key, value in values.items()}

    def __iter__(self):
        base_indices = list(range(self.size))
        rng = random.Random(1337 + self._epoch)
        if self.shuffle:
            rng.shuffle(base_indices)
        if self._parameters is None:
            extra_indices = [
                index for index in range(self.size)
                if self._frame_key(index) in self._hard_frame_keys
            ]
            if self.shuffle:
                rng.shuffle(extra_indices)
            return iter(base_indices + extra_indices)
        weights = self._sampling_weights()
        exploration = min(max(float(getattr(self._parameters, 'exploration_floor', 0.1)), 0.0), 1.0)
        extra_count = sum(1 for weight in weights if weight > exploration)
        extra_indices = rng.choices(range(self.size), weights=weights, k=extra_count) if extra_count else []
        return iter(base_indices + extra_indices)

    def _sampling_weights(self) -> list[float]:
        geometry = self._normalize(self._geometry_scores)
        losses = self._normalize(self._ema_losses)
        geometry_weight = max(0.0, float(getattr(self._parameters, 'geometry_weight', 0.5)))
        loss_weight = max(0.0, float(getattr(self._parameters, 'loss_weight', 0.5)))
        exploration = min(max(float(getattr(self._parameters, 'exploration_floor', 0.1)), 0.0), 1.0)
        score_clip = max(1e-6, float(getattr(self._parameters, 'score_clip', 5.0)))
        weights = []
        for index in range(self.size):
            key = self._sample_key(index)
            score = geometry_weight * geometry.get(key, 0.0) + loss_weight * losses.get(key, 0.0)
            weights.append(min(score_clip, max(exploration, score)))
        return weights

    def __len__(self) -> int:
        if self._parameters is not None:
            exploration = min(max(float(getattr(self._parameters, 'exploration_floor', 0.1)), 0.0), 1.0)
            return self.size + sum(1 for weight in self._sampling_weights() if weight > exploration)
        return self.size + sum(
            1 for index in range(self.size)
            if self._frame_key(index) in self._hard_frame_keys
        )

    def resize(self, size: int, *, reset: bool = False) -> None:
        self.size = max(0, int(size))
        if reset:
            self._hard_frame_keys.clear()

    def start_epoch(self) -> None:
        self._epoch_losses.clear()
        self._epoch += 1

    def _frame_key(self, index: int) -> str:
        resolver = getattr(self.dataset, 'frame_key', None)
        return str(resolver(int(index))) if callable(resolver) else str(int(index))

    def _sample_key(self, index: int) -> str:
        resolver = getattr(self.dataset, 'sample_key', None)
        return str(resolver(int(index))) if callable(resolver) else str(int(index))

    def update_batch_losses(self, sample_indices: torch.Tensor, sample_losses: torch.Tensor) -> None:
        indices = sample_indices.detach().to(device='cpu', dtype=torch.long).flatten().tolist()
        losses = sample_losses.detach().to(device='cpu', dtype=torch.float64).flatten().tolist()
        for index, loss in zip(indices, losses):
            if index < 0 or index >= self.size or not math.isfinite(float(loss)):
                continue
            sample_key = self._sample_key(int(index))
            if sample_key in self._epoch_losses:
                continue
            self._epoch_losses[sample_key] = (self._frame_key(int(index)), float(loss))
            alpha = min(max(float(getattr(self._parameters, 'ema_alpha', 0.1)), 0.0), 1.0)
            previous = self._ema_losses.get(sample_key, float(loss))
            self._ema_losses[sample_key] = previous * (1.0 - alpha) + float(loss) * alpha

    def finalize_epoch(self) -> set[str]:
        squared_by_frame: dict[str, list[float]] = {}
        for frame_key, loss in self._epoch_losses.values():
            squared_by_frame.setdefault(frame_key, []).append(float(loss) ** 2)
        frame_losses = {
            frame_key: math.sqrt(sum(values) / len(values))
            for frame_key, values in squared_by_frame.items()
            if values
        }
        self.last_frame_losses = frame_losses
        if not frame_losses:
            self.last_sigma = 0.0
            self._hard_frame_keys.clear()
            return set()
        values = list(frame_losses.values())
        mean_value = sum(values) / len(values)
        sigma = math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))
        self.last_sigma = float(sigma)
        self._hard_frame_keys = {
            frame_key for frame_key, value in frame_losses.items()
            if value >= mean_value + sigma
        }
        return set(self._hard_frame_keys)

    def state_dict(self) -> dict[str, object]:
        return {
            'size': self.size,
            'hard_frame_keys': sorted(self._hard_frame_keys),
            'epoch_losses': dict(self._epoch_losses),
            'last_frame_losses': dict(self.last_frame_losses),
            'last_sigma': float(self.last_sigma),
            'ema_losses': dict(self._ema_losses),
            'geometry_scores': dict(self._geometry_scores),
            'epoch': self._epoch,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.size = max(0, int(state.get('size', self.size)))
        self._hard_frame_keys = {str(value) for value in state.get('hard_frame_keys', ())}
        raw_epoch_losses = state.get('epoch_losses', {})
        self._epoch_losses = {
            str(sample_key): (str(value[0]), float(value[1]))
            for sample_key, value in raw_epoch_losses.items()
            if isinstance(value, (tuple, list)) and len(value) == 2
        } if isinstance(raw_epoch_losses, dict) else {}
        raw_frame_losses = state.get('last_frame_losses', {})
        self.last_frame_losses = {
            str(frame_key): float(value)
            for frame_key, value in raw_frame_losses.items()
        } if isinstance(raw_frame_losses, dict) else {}
        self.last_sigma = float(state.get('last_sigma', 0.0))
        self._ema_losses = {str(key): float(value) for key, value in dict(state.get('ema_losses', {})).items()}
        self._geometry_scores = {str(key): float(value) for key, value in dict(state.get('geometry_scores', {})).items()}
        self._epoch = max(0, int(state.get('epoch', 0)))


class RandomPatchBatchSampler(Sampler[list[PatchSampleRequest]]):
    variable_patch_sizes = True

    def __init__(
        self,
        sampler: Sampler[int],
        *,
        batch_size: int,
        min_size: tuple[int, int],
        max_size: tuple[int, int],
        drop_last: bool = False,
    ) -> None:
        self.sampler = sampler
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self._widths = self._aligned_values(int(min_size[0]), int(max_size[0]))
        self._heights = self._aligned_values(int(min_size[1]), int(max_size[1]))

    @staticmethod
    def _aligned_values(lower: int, upper: int) -> tuple[int, ...]:
        first = int(math.ceil(int(lower) / 32.0) * 32)
        values = tuple(range(first, int(upper) + 1, 32))
        if not values:
            raise ValueError('random patch range must contain a multiple of 32')
        return values

    def __iter__(self):
        batch: list[int] = []
        for index in self.sampler:
            batch.append(int(index))
            if len(batch) >= self.batch_size:
                yield self._requests(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._requests(batch)

    def _requests(self, indices: list[int]) -> list[PatchSampleRequest]:
        size = (random.choice(self._widths), random.choice(self._heights))
        return [PatchSampleRequest(index=index, size_xy=size) for index in indices]

    def __len__(self) -> int:
        size = len(self.sampler)
        if self.drop_last:
            return size // self.batch_size
        return math.ceil(size / self.batch_size)


class FrameBatchSampler(Sampler[list[int]]):
    """Yield sequential batches without mixing patches from different frames."""

    def __init__(self, frame_lengths: Iterable[int], *, batch_size: int) -> None:
        self.frame_lengths = tuple(max(0, int(length)) for length in frame_lengths)
        self.batch_size = max(1, int(batch_size))

    def __iter__(self):
        offset = 0
        for frame_length in self.frame_lengths:
            frame_end = offset + frame_length
            for batch_start in range(offset, frame_end, self.batch_size):
                yield list(range(batch_start, min(batch_start + self.batch_size, frame_end)))
            offset = frame_end

    def __len__(self) -> int:
        return sum(math.ceil(length / self.batch_size) for length in self.frame_lengths if length > 0)


class GeneralNeuralHandler:
    EXISTING_FOLDER_DEFAULT_ANSWER = False
    EXISTING_FOLDER_TIMEOUT_SECONDS = 15
    STOP_JOIN_POLL_SECONDS = 0.2
    STOP_JOIN_GRACE_SECONDS = 10.0

    def __init__(
        self,
        work_mode: WorkMode,
        question_module: Callable[..., bool],
        message_bus: AbstractMessageBus,
        recogniton_parameters: RecognitionParameters | None = None,
        tranining_parameters: TrainingParameters | None = None,
        callback: Callable[..., None] | None = None,
    ):
        self.work_mode = work_mode
        self.callback = callback
        self.recognition_parameters = recogniton_parameters
        self.tranining_parameters = tranining_parameters
        self.question = question_module
        self.message_bus = message_bus
        self.message_bus.publish('logging', 'Инициализация основных функций')

        self.current_thread: threading.Thread | None = None
        self._need_stop = False
        self._need_pause = False
        self._training_failed = False
        self._hard_mining_active = False
        self.train_loader = None
        self.val_loader = None
        self.control_loader = None
        self._train_control_dataset = None

    @staticmethod
    def _release_torch_memory() -> None:
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            # Memory release is best-effort and must not break the workflow.
            pass

    def _drop_runtime_references(self) -> None:
        self.current_thread = None
        self.train_loader = None
        self.val_loader = None
        self._release_torch_memory()

    def start(self):
        self._training_failed = False
        if self.work_mode == WorkMode.recognition_only:
            self._start_recognition()
            return

        self._prepare_training_pipeline()
        if self._need_stop:
            return

        model, model_save_path = self._resolve_training_model()
        if self._need_stop:
            return
        self._start_training(model, model_save_path)

        if self._need_stop or self._training_failed or self.work_mode in (WorkMode.train_only, WorkMode.continue_training):
            return

        if isinstance(self.recognition_parameters.model, str):
            self.recognition_parameters.model = model_save_path
        self._start_recognition()

    def _prepare_training_pipeline(self):
        if self.tranining_parameters.cut_mode != SampleCutMode.online:
            raise ValueError(
                'Disk-cut datasets are no longer supported. Use online patch generation.'
            )
        dataset_image, dataset_label = self._prepare_dataset_folders()
        validation_image = None
        validation_label = None
        if self._uses_external_validation():
            validation_image, validation_label = self._prepare_external_validation_folders()
            if self._need_stop:
                return
        train_dataset, val_dataset = self._create_dataset(
            dataset_image,
            dataset_label,
            validation_image_folder=validation_image,
            validation_label_folder=validation_label,
        )
        self._create_dataloader(train_dataset, val_dataset)

    def _uses_external_validation(self) -> bool:
        if not bool(getattr(self.tranining_parameters, 'validation', False)):
            return False
        validation_source = normalize_validation_source(
            getattr(self.tranining_parameters, 'validation_source', 'split')
        )
        return validation_source == 'external'

    def _prepare_dataset_folders(self) -> tuple[Path, Path]:
        return self._prepare_dataset_pair(
            self.tranining_parameters.image_path,
            self.tranining_parameters.label_path,
            purpose='train',
        )

    def _prepare_external_validation_folders(self) -> tuple[Path, Path]:
        validation_image_path = getattr(self.tranining_parameters, 'validation_image_path', None)
        validation_label_path = getattr(self.tranining_parameters, 'validation_label_path', None)
        if validation_image_path is None or validation_label_path is None:
            self.message_bus.publish('error', 'External validation requires both image and label folders.')
            self._need_stop = True
            return Path(), Path()
        return self._prepare_dataset_pair(
            Path(validation_image_path),
            Path(validation_label_path),
            purpose='validation',
        )

    def _prepare_dataset_pair(self, dataset_image: Path, dataset_label: Path, *, purpose: str) -> tuple[Path, Path]:
        normalized_purpose = str(purpose or 'train').strip().lower()
        is_training_data = normalized_purpose == 'train'

        recursive = bool(getattr(self.tranining_parameters, 'recursive_file_search', False))
        if filter_files(dataset_label, ('.cif',), recursive=recursive):
            binary_dir_name = 'binary_cif' if is_training_data else f'binary_cif_{normalized_purpose}'
            binary_labels = dataset_label.parent / binary_dir_name
            self._start_cif_conversion(dataset_label, binary_labels)
            dataset_label = binary_labels

        return dataset_image, dataset_label

    def _resolve_model_creation_kwargs(self, model_name: str) -> dict[str, Any]:
        resolved_name = str(model_name)
        model_kwargs: dict[str, Any] = {}
        if resolved_name == 'Transformer':
            patch_size = self.tranining_parameters.generation.segment_size
            img_size = max(1, int(patch_size[0]))
            if int(patch_size[0]) != int(patch_size[1]):
                self.message_bus.publish(
                    'logging',
                    (
                        'Transformer expects square inputs. '
                        f'Using img_size={img_size} derived from patch size {tuple(patch_size)}.'
                    ),
                )
            model_kwargs['img_size'] = img_size
            return model_kwargs

        if resolved_name in {'quasi_dual_scale_unet', 'FrameUnet', 'UNetWithContextBranch'}:
            requested_context_branch = getattr(self.tranining_parameters, 'use_context_branch', None)
            if requested_context_branch is None:
                requested_context_branch = True
            if bool(requested_context_branch) and self.tranining_parameters.cut_mode != SampleCutMode.online:
                raise ValueError(
                    'Context branch requires online patch generation (cut_mode=online) because '
                    'the dataset must build context crops from the full prepared frame.'
                )

            local_crop_size = normalize_size_pair(
                getattr(self.tranining_parameters, 'local_crop_size', None),
                fallback=tuple(self.tranining_parameters.generation.segment_size),
            )
            context_crop_size = normalize_size_pair(
                getattr(self.tranining_parameters, 'context_crop_size', None),
                fallback=(local_crop_size[0] * 2, local_crop_size[1] * 2),
            )
            context_input_size = normalize_size_pair(
                getattr(self.tranining_parameters, 'context_input_size', None),
                fallback=local_crop_size,
            )
            context_branch_channels = tuple(
                int(channel)
                for channel in getattr(self.tranining_parameters, 'context_branch_channels', (16, 32, 64, 128))
            )
            model_kwargs.update(
                {
                    'local_crop_size': local_crop_size,
                    'context_crop_size': context_crop_size,
                    'context_input_size': context_input_size,
                    'context_branch_channels': context_branch_channels,
                    'fusion_type': str(getattr(self.tranining_parameters, 'fusion_type', 'concat')),
                    'use_context_branch': bool(requested_context_branch),
                    'use_cross_attention': bool(getattr(self.tranining_parameters, 'use_cross_attention', True)),
                    'attention_dim': int(getattr(self.tranining_parameters, 'attention_dim', 128)),
                    'attention_heads': int(getattr(self.tranining_parameters, 'attention_heads', 4)),
                    'attention_max_global_tokens': int(
                        getattr(self.tranining_parameters, 'attention_max_global_tokens', 1024)
                    ),
                }
            )

        if model_supports_init_kwarg(resolved_name, 'deep_supervision'):
            model_kwargs['deep_supervision'] = bool(getattr(self.tranining_parameters, 'deep_supervision', True))

        supervision_targets = getattr(self.tranining_parameters, 'supervision_targets', None)
        if supervision_targets is not None and getattr(supervision_targets, 'any_enabled', lambda: False)():
            enabled_heads = tuple(supervision_targets.enabled_targets())
            if not model_supports_init_kwarg(resolved_name, 'supervision_heads'):
                raise ValueError(
                    f'Model {resolved_name!r} does not support auxiliary supervision heads: '
                    f'{", ".join(enabled_heads)}.'
                )
            model_kwargs['supervision_heads'] = enabled_heads

        return model_kwargs

    def _resolve_training_model(self):
        artifact_dir = self._resolve_training_artifact_dir()
        if self.work_mode in (WorkMode.train_and_recognition, WorkMode.train_only):
            model_name = str(self.recognition_parameters.model)
            model_kwargs = self._resolve_model_creation_kwargs(model_name)
            model = create_model(model_name, self.tranining_parameters.colors, **model_kwargs)
            setattr(model, '_neuralimage_model_name', model_name)
            setattr(model, '_neuralimage_input_channels', int(self.tranining_parameters.colors))
            setattr(model, '_neuralimage_model_kwargs', dict(model_kwargs))
            model_save_path = artifact_dir / self._declare_model_name()
        else:
            model = load_model_artifact(self.recognition_parameters.model, map_location='cpu')
            self._validate_loaded_model_input_channels(model)
            if self.work_mode in (WorkMode.further_training, WorkMode.continue_training) and hasattr(model, 'deep_supervision'):
                deep_supervision_enabled = bool(getattr(self.tranining_parameters, 'deep_supervision', True))
                setattr(model, 'deep_supervision', deep_supervision_enabled)
                model_kwargs = getattr(model, '_neuralimage_model_kwargs', {})
                if not isinstance(model_kwargs, dict):
                    model_kwargs = {}
                model_kwargs = dict(model_kwargs)
                model_kwargs['deep_supervision'] = deep_supervision_enabled
                setattr(model, '_neuralimage_model_kwargs', model_kwargs)
            model_save_path = artifact_dir / Path(self.recognition_parameters.model).name
        return model, model_save_path

    def _resolve_loaded_model_input_channels(self, model: Any) -> int:
        declared_channels = getattr(model, '_neuralimage_input_channels', None)
        if declared_channels is not None:
            try:
                return int(declared_channels)
            except (TypeError, ValueError):
                pass
        return int(get_input_channels(model))

    def _validate_loaded_model_input_channels(self, model: Any) -> bool:
        expected_channels = max(1, int(getattr(self.tranining_parameters, 'colors', 1)))
        actual_channels = self._resolve_loaded_model_input_channels(model)
        if actual_channels == expected_channels:
            return True

        selected_mode = 'RGB' if expected_channels == 3 else 'grayscale'
        self.message_bus.publish(
            'error',
            (
                'Training input channels mismatch: '
                f'selected {selected_mode} mode ({expected_channels} channels), '
                f'but the loaded model expects {actual_channels} channel(s). '
                'Choose a matching checkpoint or change the color mode.'
            ),
        )
        self._need_stop = True
        return False

    def _resolve_training_artifact_dir(self) -> Path:
        artifact_dir = getattr(self.tranining_parameters, 'artifact_dir', None)
        if artifact_dir is not None:
            resolved = Path(artifact_dir)
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved
        if self.work_mode in (WorkMode.further_training, WorkMode.continue_training) and str(getattr(self.recognition_parameters, 'model', '')).strip():
            return Path(self.recognition_parameters.model).parent
        return self.tranining_parameters.image_path.parent

    def _start_cif_conversion(self, source: Path, result: Path):
        if self._check_folder_existance(result):
            return
        self.current_thread = ConvertCifThread(
            source,
            result,
            message_bus=self.message_bus,
            recursive=bool(getattr(self.tranining_parameters, 'recursive_file_search', False)),
        )
        self.current_thread.start()
        self._wait_for_current_thread('cif conversion')

    def _create_dataset(
        self,
        image_folder: Path,
        label_folder: Path,
        *,
        validation_image_folder: Path | None = None,
        validation_label_folder: Path | None = None,
    ):
        if self._need_stop:
            return None, None
        train_samples = self._collect_matched_samples(image_folder, label_folder)
        if self._need_stop or train_samples is None:
            return None, None

        val_samples: list[tuple[Path, Path]] | None = None
        if self._uses_external_validation():
            if validation_image_folder is None or validation_label_folder is None:
                self.message_bus.publish('error', 'External validation folders are not configured.')
                self._need_stop = True
                return None, None
            val_samples = self._collect_matched_samples(validation_image_folder, validation_label_folder)
            if self._need_stop or val_samples is None:
                return None, None
            self.message_bus.publish(
                'logging',
                (
                    'Validation source: external dataset '
                    f'(train={len(train_samples)}, val={len(val_samples)}).'
                ),
            )
        elif self.tranining_parameters.validation:
            train_samples, val_samples = self._split_validation_samples(train_samples)
            if self._need_stop:
                return None, None

        self._resolve_training_preprocessing(train_samples)

        training_without_tech_aug = replace(
            self.tranining_parameters,
            generation=replace(
                self.tranining_parameters.generation,
                tech_aug=build_tech_augmentation_config(None),
            ),
            pcb_defects=build_pcb_defect_parameters(None),
        )
        evaluation_settings = replace(
            training_without_tech_aug,
            shuffle=False,
            generation=replace(
                training_without_tech_aug.generation,
                # Validation evaluates a deterministic covering grid. Reusing
                # the small training shift here can multiply work by hundreds
                # (for example, 2000px frames, 256px patches, step=12).
                step=_validation_grid_step(training_without_tech_aug.generation.segment_size),
                vertical_rotation=False,
                horizontal_rotation=False,
                flip_x=False,
                flip_y=False,
                additional_augmentation=False,
                random_crop=False,
                scale_augmentation=False,
            ),
            cutout=replace(training_without_tech_aug.cutout, enabled=False),
            random_artifacts=replace(training_without_tech_aug.random_artifacts, enabled=False),
            mixup=replace(training_without_tech_aug.mixup, enabled=False),
        )
        synthetic_generator = build_synthetic_defect_generator_parameters(
            getattr(self.tranining_parameters, 'synthetic_defect_generator', None)
        )

        train_dataset = NoCutDataset(
            train_samples,
            training_without_tech_aug,
            apply_train_only_transforms=True,
        )
        val_dataset = (
            NoCutDataset(
                val_samples,
                evaluation_settings,
                apply_train_only_transforms=False,
            )
            if val_samples
            else None
        )

        self._train_control_dataset = None
        if bool(getattr(self.tranining_parameters.early_stopping, 'enabled', False)) and not val_samples:
            self._train_control_dataset = NoCutDataset(
                train_samples,
                evaluation_settings,
                apply_train_only_transforms=False,
            )
            self.message_bus.publish(
                'logging',
                'Early stopping without validation uses a fixed train-control set. '
                'This mode cannot objectively determine overfitting.',
            )

        if (
            synthetic_generator.enabled
            and float(synthetic_generator.epoch_size_factor) > 0.0
            and train_dataset is not None
            and bool(len(train_samples))
        ):
            synthetic_frame_count = max(
                1,
                int(round(len(train_samples) * float(synthetic_generator.epoch_size_factor))),
            )
            synthetic_settings = replace(
                training_without_tech_aug,
                synthetic_defect_generator=synthetic_generator,
            )
            synthetic_dataset = SyntheticDefectDataset(
                synthetic_frame_count,
                synthetic_settings,
                apply_train_only_transforms=True,
            )
            train_dataset = CompositeDataset(train_dataset, synthetic_dataset)
            self.message_bus.publish(
                'logging',
                (
                    'Synthetic defect dataset generator enabled '
                    f'(real_frames={len(train_samples)}, synthetic_frames={synthetic_frame_count}, '
                    f'synthetic_samples={len(synthetic_dataset)}).'
                ),
            )

        return train_dataset, val_dataset

    def _resolve_training_preprocessing(self, train_samples: list[tuple[Path, Path]]) -> None:
        config = getattr(self.tranining_parameters, 'preprocessing', None) or PreprocessingConfig()
        if config.mode != 'dataset_zscore':
            return
        if config.has_dataset_statistics():
            return
        channels = int(self.tranining_parameters.generation.channels)

        def training_images() -> Iterable[Any]:
            for image_path, _label_path in train_samples:
                image = ImagePreparator(image_path, self.tranining_parameters.prepare).image
                yield image_to_channel_first_float01(image, channels)

        statistics = compute_dataset_statistics(training_images())
        resolved = replace(
            config,
            dataset_mean=statistics.mean,
            dataset_std=statistics.std,
        )
        self.tranining_parameters.preprocessing = resolved
        self.message_bus.publish(
            'logging',
            (
                'Dataset z-score statistics calculated from train split only: '
                f'mean={statistics.mean:.8f}, std={statistics.std:.8f}, '
                f'pixels={statistics.pixel_count}.'
            ),
        )

    def _split_validation_samples(
        self,
        samples: list[tuple[Path, Path]],
    ) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
        total_count = len(samples)
        val_count = int(total_count * self.tranining_parameters.validation_percent / 100)
        if self.tranining_parameters.validation_percent > 0 and total_count > 1 and val_count == 0:
            val_count = 1
        if val_count >= total_count:
            val_count = max(total_count - 1, 0)
        train_samples, val_samples = _deterministic_validation_split(
            samples,
            val_count=val_count,
        )
        self.message_bus.publish(
            'logging',
            (
                'Validation split: deterministic stratified split by label coverage '
                f'(train={len(train_samples)}, val={len(val_samples)}).'
            ),
        )
        return train_samples, val_samples

    def _create_dataloader(self, train_dataset, val_dataset):
        self._hard_mining_active = False
        if self._need_stop or train_dataset is None:
            return

        shuffle = False
        workers = self._resolve_dataloader_workers()
        pin_memory = bool(torch.cuda.is_available())
        train_persistent_workers = workers > 0 and not self._dataset_requires_worker_restart(train_dataset)
        val_persistent_workers = workers > 0 and not self._dataset_requires_worker_restart(val_dataset)
        train_loader_kwargs = {
            'batch_size': self.tranining_parameters.batch_size,
            'num_workers': workers,
            'pin_memory': pin_memory,
            'persistent_workers': train_persistent_workers,
        }
        val_loader_kwargs = {
            'batch_size': self.tranining_parameters.batch_size,
            'shuffle': False,
            'num_workers': workers,
            'pin_memory': pin_memory,
            'persistent_workers': val_persistent_workers,
        }
        if workers > 0:
            # Each prefetched Windows batch is transferred through a named
            # shared-memory file mapping.  Keeping only one batch queued per
            # explicitly requested worker substantially reduces mapping/pagefile
            # pressure while preserving the opt-in multiprocessing path.
            prefetch_factor = 1 if sys.platform.startswith('win') else 2
            train_loader_kwargs['prefetch_factor'] = prefetch_factor
            val_loader_kwargs['prefetch_factor'] = prefetch_factor

        hard_mining = self.tranining_parameters.hard_mining
        hard_mining_enabled = bool(hard_mining.enabled)
        self._hard_mining_active = hard_mining_enabled
        if hard_mining_enabled:
            train_dataset = IndexedDataset(train_dataset)
            hard_sampler = HardFrameSampler(
                train_dataset,
                shuffle=bool(shuffle),
                parameters=hard_mining,
            )
            geometry_resolver = getattr(train_dataset, 'geometry_score', None)
            if callable(geometry_resolver) and float(getattr(hard_mining, 'geometry_weight', 0.0)) > 0.0:
                geometry_scores = {
                    hard_sampler._sample_key(index): float(geometry_resolver(index))
                    for index in range(len(train_dataset))
                }
                hard_sampler.set_geometry_scores(geometry_scores)
            train_loader_kwargs['sampler'] = hard_sampler
            train_loader_kwargs['shuffle'] = False
            self.message_bus.publish(
                'logging',
                'Hard-frame sampling enabled: frames with RMS loss above population sigma '
                'will be repeated once in the next epoch.',
            )
        else:
            train_loader_kwargs['shuffle'] = shuffle
        random_patch_size = getattr(self.tranining_parameters, 'random_patch_size', None)
        random_patch_enabled = bool(
            random_patch_size is not None
            and getattr(random_patch_size, 'enabled', False)
            and self.tranining_parameters.cut_mode == SampleCutMode.online
        )
        if random_patch_enabled:
            base_sampler = train_loader_kwargs.pop('sampler', None)
            if base_sampler is None:
                base_sampler = RandomSampler(train_dataset) if shuffle else SequentialSampler(train_dataset)
            train_loader_kwargs.pop('shuffle', None)
            train_loader_kwargs.pop('batch_size', None)
            train_loader_kwargs['batch_sampler'] = RandomPatchBatchSampler(
                base_sampler,
                batch_size=self.tranining_parameters.batch_size,
                min_size=tuple(random_patch_size.min_size),
                max_size=tuple(random_patch_size.max_size),
            )
            self.message_bus.publish(
                'logging',
                (
                    'Random physical patch size enabled for online cutting: '
                    f'min={tuple(random_patch_size.min_size)}, max={tuple(random_patch_size.max_size)}, '
                    'alignment=32, one size per batch.'
                ),
            )
        if val_dataset is not None:
            val_dataset = IndexedDataset(val_dataset)
            base_val_dataset = getattr(val_dataset, '_base_dataset', None)
            frame_lengths_fn = getattr(base_val_dataset, 'validation_frame_lengths', None)
            if callable(frame_lengths_fn):
                val_loader_kwargs.pop('batch_size', None)
                val_loader_kwargs.pop('shuffle', None)
                val_loader_kwargs['batch_sampler'] = FrameBatchSampler(
                    frame_lengths_fn(),
                    batch_size=self.tranining_parameters.batch_size,
                )
        try:
            self.train_loader = DataLoader(
                train_dataset,
                **train_loader_kwargs,
            )
            self.val_loader = (
                DataLoader(
                    val_dataset,
                    **val_loader_kwargs,
                )
                if val_dataset
                else None
            )
            if self.val_loader is not None:
                try:
                    validation_batches = len(self.val_loader)
                except TypeError:
                    validation_batches = 'unknown'
                self.message_bus.publish(
                    'logging',
                    (
                        'Validation loader ready: '
                        f'frames={len(getattr(getattr(val_dataset, "_base_dataset", None), "samples", ()))}, '
                        f'patches={len(val_dataset)}, batches={validation_batches}, '
                        f'batch_size={self.tranining_parameters.batch_size}.'
                    ),
                )
            self.control_loader = self._create_train_control_loader()
        except Exception as error:
            self.message_bus.publish(
                'logging',
                f'Ошибка DataLoader (workers={workers}, pin_memory={pin_memory}): {error}. '
                f'Используется безопасный fallback workers=0.',
            )
            fallback_train_kwargs = {'num_workers': 0, 'pin_memory': False}
            if 'batch_sampler' in train_loader_kwargs:
                fallback_train_kwargs['batch_sampler'] = train_loader_kwargs['batch_sampler']
            elif 'sampler' in train_loader_kwargs:
                fallback_train_kwargs['batch_size'] = self.tranining_parameters.batch_size
                fallback_train_kwargs['sampler'] = train_loader_kwargs['sampler']
                fallback_train_kwargs['shuffle'] = False
            else:
                fallback_train_kwargs['batch_size'] = self.tranining_parameters.batch_size
                fallback_train_kwargs['shuffle'] = shuffle
            self.train_loader = DataLoader(train_dataset, **fallback_train_kwargs)
            fallback_val_kwargs = {'num_workers': 0, 'pin_memory': False}
            if 'batch_sampler' in val_loader_kwargs:
                fallback_val_kwargs['batch_sampler'] = val_loader_kwargs['batch_sampler']
            else:
                fallback_val_kwargs['batch_size'] = self.tranining_parameters.batch_size
                fallback_val_kwargs['shuffle'] = False
            self.val_loader = DataLoader(val_dataset, **fallback_val_kwargs) if val_dataset else None
            self.control_loader = self._create_train_control_loader()

    def _create_train_control_loader(self):
        dataset = getattr(self, '_train_control_dataset', None)
        if dataset is None:
            return None
        train_size = int(len(dataset))
        if train_size <= 0:
            return None
        control_size = min(train_size, 1024, max(32, math.ceil(0.05 * train_size)))
        generator = torch.Generator().manual_seed(_VALIDATION_SPLIT_SEED)
        indices = torch.randperm(train_size, generator=generator)[:control_size].tolist()
        subset = Subset(dataset, [int(index) for index in indices])
        self.message_bus.publish(
            'logging',
            f'Fixed train-control set created: seed={_VALIDATION_SPLIT_SEED}, size={control_size}/{train_size}.',
        )
        return DataLoader(
            subset,
            batch_size=self.tranining_parameters.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=bool(torch.cuda.is_available()),
        )

    def _resolve_dataloader_workers(self) -> int:
        if _is_debugger_attached():
            return 0
        try:
            configured_workers = int(getattr(self.tranining_parameters, 'dataloader_num_workers', -1))
        except (TypeError, ValueError):
            configured_workers = -1
        if configured_workers >= 0:
            return configured_workers
        # PyTorch's spawn-based Windows workers transfer collated tensors via
        # named shared-memory file mappings.  Large image batches can exhaust
        # those mappings only while iterating (after DataLoader construction),
        # so the constructor fallback above cannot recover.  Make automatic
        # mode reliable on Windows; users can still explicitly opt into workers.
        if sys.platform.startswith('win'):
            return 0
        cpu_count = os.cpu_count() or 1
        max_workers = 16
        workers = max(0, min(max_workers, cpu_count - 1))
        if self.tranining_parameters.cut_mode == SampleCutMode.online:
            workers = min(workers, 8)
        if self.tranining_parameters.batch_size <= 4:
            workers = min(workers, 2)
        return workers

    @staticmethod
    def _dataset_requires_worker_restart(dataset) -> bool:
        if dataset is None:
            return False
        return callable(getattr(dataset, 'set_epoch', None))

    def _collect_matched_samples(
        self,
        image_folder: Path,
        label_folder: Path,
    ) -> list[tuple[Path, Path]] | None:
        recursive = bool(getattr(self.tranining_parameters, 'recursive_file_search', False))
        image_files = sorted(filter_images(image_folder, recursive=recursive))
        label_files = sorted(filter_images(label_folder, recursive=recursive))

        def _sample_key(file: Path, root: Path) -> str:
            if not recursive:
                return file.stem
            try:
                relative = file.relative_to(root)
            except ValueError:
                return file.stem
            return relative.with_suffix('').as_posix()

        def _build_file_map(files: list[Path], kind: str, root: Path) -> dict[str, Path]:
            result: dict[str, Path] = {}
            duplicates: list[str] = []
            for file in files:
                stem = _sample_key(file, root)
                if stem in result:
                    duplicates.append(stem)
                    continue
                result[stem] = file
            if duplicates:
                unique_duplicates = ', '.join(sorted(set(duplicates))[:10])
                self.message_bus.publish(
                    'error',
                    f'Duplicate {kind} stems detected: {unique_duplicates}. '
                    'Ensure unique base file names before training.',
                )
                self._need_stop = True
            return result

        image_map = _build_file_map(image_files, 'image', image_folder)
        label_map = _build_file_map(label_files, 'label', label_folder)
        if self._need_stop:
            return None

        image_stems = set(image_map.keys())
        label_stems = set(label_map.keys())
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        common_stems = sorted(image_stems & label_stems)
        if (missing_labels or missing_images) and common_stems:
            missing_labels_preview = ', '.join(missing_labels[:10]) if missing_labels else '-'
            missing_images_preview = ', '.join(missing_images[:10]) if missing_images else '-'
            pair_word = 'pair' if len(common_stems) == 1 else 'pairs'
            self.message_bus.publish(
                'warning',
                (
                    'Image/label mismatch detected. '
                    f'Missing labels for images: {missing_labels_preview}. '
                    f'Missing images for labels: {missing_images_preview}. '
                    f'Training will continue with {len(common_stems)} matched {pair_word}.'
                ),
            )

        zipped_images = [(image_map[stem], label_map[stem]) for stem in common_stems]
        if not zipped_images:
            self.message_bus.publish('error', 'No matched image/label pairs found in the selected dataset.')
            self._need_stop = True
            return None
        return zipped_images

    def _get_zipped_samples(self, image_folder, label_folder) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]] | tuple[list[tuple[Path, Path]], None]:
        matched_samples = self._collect_matched_samples(Path(image_folder), Path(label_folder))
        if self._need_stop or matched_samples is None:
            return [], None
        if self.tranining_parameters.validation and not self._uses_external_validation():
            return self._split_validation_samples(matched_samples)
        return matched_samples, None

    def _declare_model_name(self) -> str:
        generation = self.tranining_parameters.generation

        model_name = str(self.recognition_parameters.model)
        model_name += f'_shift_{generation.step}_'
        model_name += 'r90_' if generation.horizontal_rotation else ''
        model_name += 'r180_' if generation.vertical_rotation else ''
        model_name += 'fx_' if bool(getattr(generation, 'flip_x', False)) else ''
        model_name += 'fy_' if bool(getattr(generation, 'flip_y', False)) else ''
        model_name += f'epoch{self.tranining_parameters.epochs}'
        model_name += '.pth'

        return model_name

    def _start_training(self, model, model_save_path: Path):
        self.message_bus.publish('metrics', {'type': 'workflow_phase', 'phase': 'training'})
        self._write_training_manifest(model_save_path)
        resume_from_checkpoint = bool(
            self.work_mode in (WorkMode.further_training, WorkMode.continue_training)
            or getattr(self.tranining_parameters, 'resume_from_checkpoint', False)
        )
        multi_gpu_mode = normalize_multi_gpu_mode(
            getattr(self.tranining_parameters, 'multi_gpu_mode', ''),
            use_multi_gpu_fallback=bool(getattr(self.tranining_parameters, 'use_multi_gpu', False)),
        )
        self.current_thread = ModelTrainer(
            self.train_loader,
            self.val_loader,
            model,
            model_save_path,
            self.tranining_parameters.epochs,
            message_bus=self.message_bus,
            callback=self._stop_training_callback,
            optimizer_params=self.tranining_parameters.optimizer,
            mixed_precision=self.tranining_parameters.mixed_precision,
            loss_function=self.tranining_parameters.loss_function,
            loss_term_weights=getattr(self.tranining_parameters, 'loss_term_weights', {}),
            dice_loss_weight=self.tranining_parameters.dice_loss_weight,
            iou_loss_weight=self.tranining_parameters.iou_loss_weight,
            hard_mining_params=self.tranining_parameters.hard_mining,
            cutout_params=getattr(self.tranining_parameters, 'cutout', None),
            random_artifacts_params=getattr(self.tranining_parameters, 'random_artifacts', None),
            mixup_params=getattr(self.tranining_parameters, 'mixup', None),
            early_stopping_params=self.tranining_parameters.early_stopping,
            warmup_params=self.tranining_parameters.warmup,
            scheduler_params=getattr(self.tranining_parameters, 'scheduler', None),
            skip_uniform_labels=self.tranining_parameters.skip_uniform_labels,
            resume_from_checkpoint=resume_from_checkpoint,
            extend_epochs_on_resume=self.work_mode in (WorkMode.further_training, WorkMode.continue_training),
            use_multi_gpu=multi_gpu_mode != 'off',
            multi_gpu_mode=multi_gpu_mode,
            show_batch_preview=self.tranining_parameters.show_batch_preview,
            log_update_frequency=self.tranining_parameters.log_update_frequency,
            save_validation_binary_images=bool(
                getattr(self.tranining_parameters, 'save_validation_binary_images', False)
            ),
            control_dataloader=getattr(self, 'control_loader', None),
            loss_weighting_strategy=str(
                getattr(self.tranining_parameters, 'loss_weighting_strategy', 'static')
            ),
            mask_loss_weight_floor=float(
                getattr(self.tranining_parameters, 'mask_loss_weight_floor', 0.25)
            ),
        )
        self.current_thread.daemon = False
        self.current_thread.start()
        self._wait_for_current_thread('training')
        if not getattr(self.current_thread, 'succeeded', False) and not getattr(self, '_need_stop', False):
            self._training_failed = True
            if getattr(self.current_thread, 'error_message', None) is None:
                self.message_bus.publish('error', 'Обучение завершилось с ошибкой.')
        elif not getattr(self, '_need_stop', False):
            self.message_bus.publish('metrics', {
                'type': 'training_completed',
                'model_path': str(model_save_path),
                'checkpoint_path': str(model_save_path.with_suffix('.ckpt')),
            })
        self.current_thread = None
        # The process/thread lifecycle is over; drop heavy references eagerly.
        model = None
        self._release_torch_memory()

    def _write_training_manifest(self, model_save_path: Path) -> None:
        def normalize(value: Any) -> Any:
            if is_dataclass(value):
                return normalize(asdict(value))
            if isinstance(value, dict):
                return {str(key): normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            if isinstance(value, Path):
                return str(value)
            if hasattr(value, 'value'):
                return normalize(value.value)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return str(value)

        payload = {
            'schema': 'neuralimage_training_run',
            'version': 1,
            'model_path': str(model_save_path),
            'training': normalize(self.tranining_parameters),
        }
        manifest_path = model_save_path.parent / 'training_manifest.json'
        temporary_path = manifest_path.with_suffix('.json.tmp')
        temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary_path, manifest_path)

    def _stop_training_callback(self):
        if self._need_stop:
            return
        self.message_bus.publish('logging', 'Обучение завершено')

    def _start_recognition(self):
        self.message_bus.publish('metrics', {'type': 'workflow_phase', 'phase': 'recognition'})
        self.current_thread = ModelRecognizer(
            self.recognition_parameters,
            message_bus=self.message_bus,
            callback=None,
            multithreading=bool(getattr(self.recognition_parameters, 'recognition_multiprocessing_enabled', True)),
        )
        self.current_thread.daemon = False
        self.current_thread.start()
        self._wait_for_current_thread('recognition')
        if not getattr(self.current_thread, 'succeeded', False):
            if (not self._need_stop) and getattr(self.current_thread, 'error_message', None) is None:
                self.message_bus.publish('error', 'Распознавание завершилось с ошибкой.')
        self._stop_recognition_callback()

    def _stop_recognition_callback(self):
        self._drop_runtime_references()
        if self.callback is not None:
            self.callback()

    def stop_execution(self):
        self._need_stop = True
        if self.current_thread is None:
            self._drop_runtime_references()
            return
        if hasattr(self.current_thread, 'stop'):
            self.current_thread.stop()

    def pause_execution(self):
        self._need_pause = True
        self._need_stop = True
        if self.current_thread is None:
            self._drop_runtime_references()
            return
        pause = getattr(self.current_thread, 'pause', None)
        if callable(pause):
            pause()
        elif hasattr(self.current_thread, 'stop'):
            self.current_thread.stop()

    def _wait_for_current_thread(self, operation_name: str) -> None:
        thread = self.current_thread
        if thread is None:
            return
        join_fn = getattr(thread, 'join', None)
        is_alive_fn = getattr(thread, 'is_alive', None)
        if not callable(join_fn):
            return
        if not callable(is_alive_fn):
            join_fn()
            return
        stop_wait_started_at: float | None = None
        while is_alive_fn():
            join_fn(timeout=self.STOP_JOIN_POLL_SECONDS)
            if not self._need_stop:
                continue
            if stop_wait_started_at is None:
                stop_wait_started_at = time.monotonic()
                continue
            waited_after_stop = time.monotonic() - stop_wait_started_at
            if waited_after_stop < self.STOP_JOIN_GRACE_SECONDS:
                continue
            self.message_bus.publish(
                'error',
                (
                    f'Не удалось корректно завершить {operation_name} в течение '
                    f'{int(self.STOP_JOIN_GRACE_SECONDS)} сек. Операция переведена в аварийное завершение.'
                ),
            )
            break

    def _check_folder_existance(self, folder: Path):
        answer = False
        if folder.exists():
            answer = self.question(
                f'Папка {folder.name} существует, использовать данные из неё?',
                'Папка существует',
                default_answer=self.EXISTING_FOLDER_DEFAULT_ANSWER,
                timeout_seconds=self.EXISTING_FOLDER_TIMEOUT_SECONDS,
            )
            if not answer:
                shutil.rmtree(folder, ignore_errors=False)
        return answer
