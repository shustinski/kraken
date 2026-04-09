import shutil
import threading
import os
import sys
import hashlib
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image

from lib.data_interfaces import (
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
from lib.file_func import filter_files, filter_images
from lib.func import get_input_channels
from lib.message_bus import AbstractMessageBus
from model.NeuralNetwork import create_model, model_supports_init_kwarg
from model.NeuralNetwork.context_utils import normalize_size_pair
from model.NeuralNetwork.dataset import CustomDataset, NoCutDataset, SyntheticDefectDataset
from model.NeuralNetwork.model_io import load_model_artifact
from model.NeuralNetwork.model_train_and_recognition import ModelRecognizer, ModelTrainer
from model.image_workers import ConvertCifThread, CutImageThread


_VALIDATION_SPLIT_SEED = 1337
_VALIDATION_FOREGROUND_BUCKETS: tuple[float, ...] = (0.0, 0.001, 0.01, 0.05, 0.2, 1.0)
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
        image, label = self._base_dataset[index]
        return image, label, int(index)

    def describe_sample(self, index: int) -> str:
        describe_fn = getattr(self._base_dataset, 'describe_sample', None)
        if callable(describe_fn):
            return str(describe_fn(int(index)))
        return f'sample_{int(index):06d}'

    def set_epoch(self) -> None:
        set_epoch_fn = getattr(self._base_dataset, 'set_epoch', None)
        if callable(set_epoch_fn):
            set_epoch_fn()


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
        if index < 0 or index >= self._total_length:
            raise IndexError('dataset index out of range')
        for dataset, offset, length in zip(self._datasets, self._offsets, self._lengths):
            if index < offset + length:
                return dataset[index - offset]
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


class LossAwareSampler(Sampler[int]):
    MULTINOMIAL_MAX_CATEGORIES = 1 << 24

    def __init__(self, size: int, strength: float = 2.0, ema_alpha: float = 0.2, replacement: bool = True):
        self.size = max(0, int(size))
        self.strength = max(0.0, float(strength))
        self.ema_alpha = float(min(max(ema_alpha, 0.0), 1.0))
        self.replacement = bool(replacement)
        self._difficulty = torch.ones(self.size, dtype=torch.float32)
        self._weights = torch.ones(self.size, dtype=torch.float32)
        self._eps = 1e-8

    def __iter__(self):
        if self.size <= 0:
            return iter([])
        if self.size > self.MULTINOMIAL_MAX_CATEGORIES:
            # torch.multinomial cannot handle category counts above 2^24.
            if self.replacement:
                indices = torch.randint(0, self.size, (self.size,), dtype=torch.long)
            else:
                indices = torch.randperm(self.size, dtype=torch.long)
            return iter(indices.tolist())
        indices = torch.multinomial(self._weights, self.size, replacement=self.replacement)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.size

    def resize(self, size: int, *, reset: bool = False) -> None:
        resolved_size = max(0, int(size))
        if resolved_size == self.size and not reset:
            return

        self.size = resolved_size
        if reset or resolved_size <= 0:
            self._difficulty = torch.ones(self.size, dtype=torch.float32)
            self._weights = torch.ones(self.size, dtype=torch.float32)
            return

        new_difficulty = torch.ones(self.size, dtype=torch.float32)
        new_weights = torch.ones(self.size, dtype=torch.float32)
        shared = min(len(self._difficulty), self.size)
        if shared > 0:
            new_difficulty[:shared] = self._difficulty[:shared]
            new_weights[:shared] = self._weights[:shared]
        self._difficulty = new_difficulty
        self._weights = new_weights

    def update_batch_losses(self, sample_indices: torch.Tensor, sample_losses: torch.Tensor) -> None:
        if self.size <= 0:
            return
        if sample_indices.numel() == 0 or sample_losses.numel() == 0:
            return

        idx = sample_indices.to(dtype=torch.long).flatten()
        losses = sample_losses.to(dtype=torch.float32).flatten()
        valid_mask = (idx >= 0) & (idx < self.size)
        if not bool(valid_mask.any()):
            return

        idx = idx[valid_mask]
        losses = losses[valid_mask]
        batch_mean = float(losses.mean().item())
        if batch_mean <= self._eps:
            normalized = torch.ones_like(losses)
        else:
            normalized = losses / (batch_mean + self._eps)

        old_scores = self._difficulty[idx]
        updated_scores = old_scores * (1.0 - self.ema_alpha) + normalized * self.ema_alpha
        self._difficulty[idx] = updated_scores
        self._weights[idx] = 1.0 + self.strength * torch.clamp(updated_scores - 1.0, min=0.0)


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
        self._training_failed = False
        self._hard_mining_active = False
        self.train_loader = None
        self.val_loader = None

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

        if self._need_stop or self._training_failed or self.work_mode == WorkMode.train_only:
            return

        if isinstance(self.recognition_parameters.model, str):
            self.recognition_parameters.model = model_save_path
        self._start_recognition()

    def _prepare_training_pipeline(self):
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

        if filter_files(dataset_label, ('.cif',)):
            binary_dir_name = 'binary_cif' if is_training_data else f'binary_cif_{normalized_purpose}'
            binary_labels = dataset_label.parent / binary_dir_name
            self._start_cif_conversion(dataset_label, binary_labels)
            dataset_label = binary_labels

        if self.tranining_parameters.cut_mode == SampleCutMode.disk:
            image_dir_name = 'input_dir' if is_training_data else f'input_dir_{normalized_purpose}'
            label_dir_name = 'label_dir' if is_training_data else f'label_dir_{normalized_purpose}'
            cut_images = dataset_image.parent / image_dir_name
            cut_labels = dataset_label.parent / label_dir_name
            self._start_cut(dataset_image, cut_images)
            self._start_cut(dataset_label, cut_labels)
            dataset_image = cut_images
            dataset_label = cut_labels

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
                }
            )

        if model_supports_init_kwarg(resolved_name, 'deep_supervision'):
            model_kwargs['deep_supervision'] = bool(getattr(self.tranining_parameters, 'deep_supervision', True))
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
            if self.work_mode == WorkMode.further_training and hasattr(model, 'deep_supervision'):
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
        if self.work_mode == WorkMode.further_training and str(getattr(self.recognition_parameters, 'model', '')).strip():
            return Path(self.recognition_parameters.model).parent
        return self.tranining_parameters.image_path.parent

    def _start_cif_conversion(self, source: Path, result: Path):
        if self._check_folder_existance(result):
            return
        self.current_thread = ConvertCifThread(source, result, message_bus=self.message_bus)
        self.current_thread.start()
        self._wait_for_current_thread('cif conversion')

    def _start_cut(self, source: Path, result: Path):
        if self._check_folder_existance(result):
            return
        self.current_thread = CutImageThread(
            source,
            result,
            self.tranining_parameters.generation,
            message_bus=self.message_bus,
        )
        self.current_thread.daemon = False
        self.current_thread.start()
        self._wait_for_current_thread('dataset cutting')

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
        if (
            bool(getattr(self.tranining_parameters, 'use_context_branch', False))
            and self.tranining_parameters.cut_mode == SampleCutMode.disk
        ):
            raise ValueError(
                'Context branch is supported only with online patch generation (cut_mode=online).'
            )

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

        training_without_tech_aug = replace(
            self.tranining_parameters,
            generation=replace(
                self.tranining_parameters.generation,
                tech_aug=build_tech_augmentation_config(None),
            ),
            pcb_defects=build_pcb_defect_parameters(None),
        )
        synthetic_generator = build_synthetic_defect_generator_parameters(
            getattr(self.tranining_parameters, 'synthetic_defect_generator', None)
        )

        if self.tranining_parameters.cut_mode == SampleCutMode.disk:
            train_dataset = CustomDataset(
                train_samples,
                self.tranining_parameters.generation.channels,
                pcb_defects=None,
                tech_aug=None,
                apply_train_only_transforms=True,
            )
            val_dataset = (
                CustomDataset(
                    val_samples,
                    self.tranining_parameters.generation.channels,
                    pcb_defects=None,
                    tech_aug=None,
                    apply_train_only_transforms=False,
                )
                if val_samples
                else None
            )
        else:
            train_dataset = NoCutDataset(
                train_samples,
                training_without_tech_aug,
                apply_train_only_transforms=True,
            )
            val_dataset = (
                NoCutDataset(
                    val_samples,
                    training_without_tech_aug,
                    apply_train_only_transforms=False,
                )
                if val_samples
                else None
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

        shuffle = self.tranining_parameters.shuffle if self.tranining_parameters.cut_mode == SampleCutMode.disk else False
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
            train_loader_kwargs['prefetch_factor'] = 2
            val_loader_kwargs['prefetch_factor'] = 2

        hard_mining = self.tranining_parameters.hard_mining
        train_dataset_size = len(train_dataset)
        hard_mining_enabled = bool(hard_mining.enabled)
        if hard_mining_enabled and train_dataset_size > LossAwareSampler.MULTINOMIAL_MAX_CATEGORIES:
            hard_mining_enabled = False
            self.message_bus.publish(
                'logging',
                (
                    'Hard mining отключен: размер train dataset '
                    f'({train_dataset_size}) превышает лимит torch.multinomial '
                    f'({LossAwareSampler.MULTINOMIAL_MAX_CATEGORIES}).'
                ),
            )
        self._hard_mining_active = hard_mining_enabled
        if hard_mining_enabled:
            train_dataset = IndexedDataset(train_dataset)
            train_loader_kwargs['sampler'] = LossAwareSampler(
                size=len(train_dataset),
                strength=hard_mining.strength,
                ema_alpha=hard_mining.ema_alpha,
            )
            train_loader_kwargs['shuffle'] = False
            self.message_bus.publish(
                'logging',
                (
                    f'Hard mining включен: strength={float(hard_mining.strength):.2f}, '
                    f'ema_alpha={float(hard_mining.ema_alpha):.2f}.'
                ),
            )
        else:
            train_loader_kwargs['shuffle'] = shuffle
        if val_dataset is not None:
            val_dataset = IndexedDataset(val_dataset)
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
        except Exception as error:
            self.message_bus.publish(
                'logging',
                f'Ошибка DataLoader (workers={workers}, pin_memory={pin_memory}): {error}. '
                f'Используется безопасный fallback workers=0.',
            )
            fallback_train_kwargs = {
                'batch_size': self.tranining_parameters.batch_size,
                'num_workers': 0,
                'pin_memory': False,
            }
            if 'sampler' in train_loader_kwargs:
                fallback_train_kwargs['sampler'] = train_loader_kwargs['sampler']
                fallback_train_kwargs['shuffle'] = False
            else:
                fallback_train_kwargs['shuffle'] = shuffle
            self.train_loader = DataLoader(train_dataset, **fallback_train_kwargs)
            self.val_loader = (
                DataLoader(
                val_dataset,
                batch_size=self.tranining_parameters.batch_size,
                shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )
                if val_dataset
                else None
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
        cpu_count = os.cpu_count() or 1
        max_workers = 8 if sys.platform.startswith('win') else 16
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
        image_files = sorted(filter_images(image_folder))
        label_files = sorted(filter_images(label_folder))

        def _build_file_map(files: list[Path], kind: str) -> dict[str, Path]:
            result: dict[str, Path] = {}
            duplicates: list[str] = []
            for file in files:
                stem = file.stem
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

        image_map = _build_file_map(image_files, 'image')
        label_map = _build_file_map(label_files, 'label')
        if self._need_stop:
            return None

        image_stems = set(image_map.keys())
        label_stems = set(label_map.keys())
        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)
        if missing_labels or missing_images:
            missing_labels_preview = ', '.join(missing_labels[:10]) if missing_labels else '-'
            missing_images_preview = ', '.join(missing_images[:10]) if missing_images else '-'
            self.message_bus.publish(
                'error',
                (
                    'Image/label mismatch detected. '
                    f'Missing labels for images: {missing_labels_preview}. '
                    f'Missing images for labels: {missing_images_preview}.'
                ),
            )
            self._need_stop = True
            return None

        common_stems = sorted(image_stems)
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
        resume_from_checkpoint = self.work_mode == WorkMode.further_training
        multi_gpu_mode = normalize_multi_gpu_mode(
            getattr(self.tranining_parameters, 'multi_gpu_mode', ''),
            use_multi_gpu_fallback=bool(getattr(self.tranining_parameters, 'use_multi_gpu', False)),
        )
        if self._hard_mining_active and multi_gpu_mode == 'distributeddataparallel':
            multi_gpu_mode = 'dataparallel'
            self.message_bus.publish(
                'logging',
                'Hard mining включен: DistributedDataParallel заменен на nn.DataParallel для этого запуска.',
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
            use_multi_gpu=multi_gpu_mode != 'off',
            multi_gpu_mode=multi_gpu_mode,
            show_batch_preview=self.tranining_parameters.show_batch_preview,
            log_update_frequency=self.tranining_parameters.log_update_frequency,
            save_validation_binary_images=bool(
                getattr(self.tranining_parameters, 'save_validation_binary_images', False)
            ),
        )
        self.current_thread.daemon = False
        self.current_thread.start()
        self._wait_for_current_thread('training')
        if not getattr(self.current_thread, 'succeeded', False):
            self._training_failed = True
            if getattr(self.current_thread, 'error_message', None) is None:
                self.message_bus.publish('error', 'Обучение завершилось с ошибкой.')
        self.current_thread = None
        # The process/thread lifecycle is over; drop heavy references eagerly.
        model = None
        self._release_torch_memory()

    def _stop_training_callback(self):
        if self._need_stop:
            return
        self.message_bus.publish('logging', 'Обучение завершено')

    def _start_recognition(self):
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
