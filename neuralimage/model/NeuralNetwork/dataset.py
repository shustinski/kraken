import random
from collections import OrderedDict
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision.transforms import ToTensor

from augmentations import (
    ICDefectAugmentor,
    PCBDefectAugmentor,
    SyntheticTopologyGenerator,
    SyntheticTopologyParameters,
    TechVariationAugmentor,
)
from lib.data_interfaces import TrainingParameters, SampleGenerationSettings, SamplePrepareSettings
from lib.data_interfaces import (
    build_pcb_defect_parameters,
    build_synthetic_defect_generator_parameters,
    build_tech_augmentation_config,
)
from lib.file_retry import retry_file_read
from lib.images import ImagePreparator, SampleCalculator, SampleFastCutter
from lib.rare_patch_masks import resolve_rare_patch_mask_path
from model.NeuralNetwork.context_utils import PatchWindow, extract_centered_crop, normalize_size_pair, resize_chw_image


_REPLAY_RARE_PATCH_DIR = Path('replay_buffer') / 'rare_patch_masks'


def _resolve_artifact_replay_mask_path(artifact_dir: Path | None, sample_name: str | Path) -> Path | None:
    if artifact_dir is None:
        return None
    return Path(artifact_dir) / _REPLAY_RARE_PATCH_DIR / f'{Path(str(sample_name)).stem}.png'


def _unwrap_tech_augmented_mask(result: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if isinstance(result, tuple):
        return np.asarray(result[1])
    return np.asarray(result)


def _extract_binary_single_channel(image: np.ndarray, *, binary_tolerance: float) -> np.ndarray | None:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        if not _is_binary_like_plane(array, tolerance=binary_tolerance):
            return None
        return array[None, :, :].astype(np.float32, copy=False)
    if array.ndim != 3 or array.shape[0] <= 0:
        return None
    single_channel = array[:1]
    if array.shape[0] > 1 and not np.allclose(
        array,
        np.repeat(single_channel, array.shape[0], axis=0),
        atol=1e-6,
    ):
        return None
    if not _is_binary_like_plane(single_channel[0], tolerance=binary_tolerance):
        return None
    return single_channel.astype(np.float32, copy=False)


def _broadcast_augmented_mask(template: np.ndarray, augmented_mask: np.ndarray) -> np.ndarray:
    template_array = np.asarray(template, dtype=np.float32)
    augmented = np.asarray(augmented_mask, dtype=np.float32)
    if augmented.ndim == 2:
        augmented = augmented[None, :, :]
    if template_array.ndim == 2:
        return augmented[0].astype(np.float32, copy=False)
    if template_array.ndim != 3:
        return augmented.astype(np.float32, copy=False)
    if template_array.shape[0] <= 1:
        return augmented[:1].astype(np.float32, copy=False)
    return np.repeat(augmented[:1], template_array.shape[0], axis=0).astype(np.float32, copy=False)


def _is_binary_like_plane(plane: np.ndarray, tolerance: float) -> bool:
    if plane.size == 0:
        return False
    distances = np.minimum(np.abs(plane), np.abs(plane - 1.0))
    return bool(np.mean(distances <= float(tolerance)) >= 0.98)


def _apply_binary_tech_augmentation(
    image: np.ndarray,
    augmentor: TechVariationAugmentor | None,
    *,
    binary_tolerance: float,
) -> np.ndarray:
    if augmentor is None:
        return image.astype(np.float32, copy=False)
    if image.ndim != 3 or image.shape[0] <= 0:
        return image.astype(np.float32, copy=False)

    single_channel = image[:1]
    if image.shape[0] > 1 and not np.allclose(image, np.repeat(single_channel, image.shape[0], axis=0), atol=1e-6):
        return image.astype(np.float32, copy=False)
    if not _is_binary_like_plane(single_channel[0], tolerance=binary_tolerance):
        return image.astype(np.float32, copy=False)

    augmented = _unwrap_tech_augmented_mask(augmentor(single_channel.astype(np.float32, copy=False)))
    if augmented.ndim == 2:
        augmented = augmented[None, :, :]
    augmented = augmented.astype(np.float32, copy=False)
    if image.shape[0] == 1:
        return augmented
    return np.repeat(augmented[:1], image.shape[0], axis=0).astype(np.float32, copy=False)


def _apply_binary_tech_augmentation_to_pair(
    image: np.ndarray,
    label: np.ndarray,
    augmentor: TechVariationAugmentor | None,
    *,
    binary_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    image_array = np.asarray(image, dtype=np.float32)
    label_array = np.asarray(label, dtype=np.float32)
    if augmentor is None:
        return image_array.astype(np.float32, copy=False), label_array.astype(np.float32, copy=False)

    source_mask = _extract_binary_single_channel(label_array, binary_tolerance=binary_tolerance)
    if source_mask is None:
        source_mask = _extract_binary_single_channel(image_array, binary_tolerance=binary_tolerance)
    if source_mask is None:
        return image_array.astype(np.float32, copy=False), label_array.astype(np.float32, copy=False)

    augmented_mask = _unwrap_tech_augmented_mask(augmentor(source_mask.astype(np.float32, copy=False)))
    augmented_label = _broadcast_augmented_mask(label_array, augmented_mask)
    if _extract_binary_single_channel(image_array, binary_tolerance=binary_tolerance) is None:
        return image_array.astype(np.float32, copy=False), augmented_label
    augmented_image = _broadcast_augmented_mask(image_array, augmented_mask)
    return augmented_image, augmented_label


def _apply_binary_tech_augmentation_with_seed(
    image: np.ndarray,
    augmentor: TechVariationAugmentor | None,
    *,
    binary_tolerance: float,
    seed: int,
) -> np.ndarray:
    if augmentor is None:
        return image.astype(np.float32, copy=False)
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        return _apply_binary_tech_augmentation(
            image,
            augmentor,
            binary_tolerance=binary_tolerance,
        )
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)


def _apply_binary_tech_augmentation_to_pair_with_seed(
    image: np.ndarray,
    label: np.ndarray,
    augmentor: TechVariationAugmentor | None,
    *,
    binary_tolerance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if augmentor is None:
        return (
            np.asarray(image, dtype=np.float32).astype(np.float32, copy=False),
            np.asarray(label, dtype=np.float32).astype(np.float32, copy=False),
        )
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        return _apply_binary_tech_augmentation_to_pair(
            image,
            label,
            augmentor,
            binary_tolerance=binary_tolerance,
        )
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)


class NoCutDataset(Dataset):
    def __init__(self, samples, settings: TrainingParameters, *, apply_train_only_transforms: bool = True):
        self.samples = samples
        self._sample_folder = Path(settings.image_path)
        self.colors = settings.colors
        self.shuffle_frames = bool(getattr(settings, 'shuffle', True))
        # Backward-compatible alias used by old code/tests.
        self.shuffle = self.shuffle_frames
        self._prep_settings: SamplePrepareSettings = settings.prepare
        self._cut_settings: SampleGenerationSettings = settings.generation
        self._skip_uniform_labels = bool(getattr(settings, 'skip_uniform_labels', False))
        self._rare_patch_oversampling_enabled = bool(
            getattr(settings, 'rare_patch_oversampling_enabled', False)
        )
        self._rare_patch_oversampling_factor = max(
            1,
            int(getattr(settings, 'rare_patch_oversampling_factor', 2)),
        )
        self._use_context_branch = bool(getattr(settings, 'use_context_branch', False))
        self._local_crop_size = normalize_size_pair(
            getattr(settings, 'local_crop_size', None),
            fallback=tuple(self._cut_settings.segment_size),
        )
        self._context_crop_size = normalize_size_pair(
            getattr(settings, 'context_crop_size', None),
            fallback=(self._local_crop_size[0] * 2, self._local_crop_size[1] * 2),
        )
        self._context_input_size = normalize_size_pair(
            getattr(settings, 'context_input_size', None),
            fallback=self._local_crop_size,
        )
        self._tech_aug_config = build_tech_augmentation_config(getattr(self._cut_settings, 'tech_aug', None))
        self._tech_aug_binary_tolerance = float(
            getattr(self._tech_aug_config, 'binary_tolerance', 0.15)
        )
        self._pcb_defects = build_pcb_defect_parameters(getattr(settings, 'pcb_defects', None))
        self._apply_train_only_transforms = bool(apply_train_only_transforms)
        self._tech_augmentor = (
            TechVariationAugmentor(self._tech_aug_config)
            if self._apply_train_only_transforms and self._tech_aug_config.enabled
            else None
        )
        self._defect_augmentor = (
            PCBDefectAugmentor(self._pcb_defects)
            if self._apply_train_only_transforms and self._pcb_defects.enabled
            else None
        )
        self._use_defect_mask_as_label = bool(self._pcb_defects.use_defect_mask_as_label)
        self.shuffle_patches_in_frame = bool(
            getattr(self._cut_settings, 'shuffle_patches_in_frame', self.shuffle_frames)
        )
        self._dynamic_frame_lengths = self._resolve_dynamic_frame_lengths()
        self._samples_amount: int = 0
        self._frame_lengths: list[int] = []
        self._lookup_len_list: list[int] = []
        self._epoch_index: int = 0

        self._current_frame_index: int | None = None
        self._current_image_cutter: SampleFastCutter | None = None
        self._frame_cache: OrderedDict[tuple[int, int, bool], SampleFastCutter] = OrderedDict()
        self._frame_cache_limit = self._resolve_frame_cache_limit()
        artifact_dir = getattr(settings, 'artifact_dir', None)
        self._artifact_replay_dir = Path(artifact_dir) if artifact_dir is not None else None

        self._create_files_list()
        self._calculate_len()

    def set_epoch(self):
        self._epoch_index += 1
        if self.shuffle_frames:
            self._shuffle_samples_and_lengths()
        if self._dynamic_frame_lengths:
            self._refresh_frame_lengths()
        elif self.shuffle_frames:
            self._rebuild_lookup()
        self._current_frame_index = None
        self._current_image_cutter = None
        self._frame_cache.clear()

    def __getitem__(self, index):
        if index < 0 or index >= self._samples_amount:
            raise IndexError('dataset index out of range')
        frame, part = index_in_list(index, self._lookup_len_list)
        if self._current_frame_index != frame:
            self._current_frame_index = frame
            self._current_image_cutter = self._get_frame_cutter(
                frame,
                shuffle=self.shuffle_patches_in_frame,
            )
        image, label = self._current_image_cutter[part]
        image, label, augmented_local_image = self._apply_pcb_defects(
            image,
            label,
            frame=frame,
            part=part,
        )
        if not self._use_context_branch:
            return image, label
        context_image = self._build_context_crop(self._current_image_cutter, part)
        if augmented_local_image is not None:
            left, top, right, bottom = self._current_image_cutter.resolve_part_coordinates(part)
            context_image = self._inject_local_patch_into_context(
                context_image,
                augmented_local_image,
                source_crop_size_xy=(max(1, int(right - left)), max(1, int(bottom - top))),
            )
        return {'local_image': image, 'context_image': context_image}, label

    def __len__(self):
        return self._samples_amount

    def describe_sample(self, index: int) -> str:
        frame, part = index_in_list(int(index), self._lookup_len_list)
        image_path, _label_path = self.samples[frame]
        return f'{image_path.stem}__part_{int(part):06d}'

    def _create_files_list(self):
        if self.shuffle_frames:
            random.shuffle(self.samples)

    def _calculate_len(self):
        len_list: list[int] = []
        for frame_index in range(len(self.samples)):
            len_list.append(self._calculate_frame_len(frame_index))

        self._frame_lengths = len_list
        self._rebuild_lookup()

    def _resolve_dynamic_frame_lengths(self) -> bool:
        if self._rare_patch_oversampling_enabled:
            return True
        if not self._skip_uniform_labels:
            return False
        return bool(
            getattr(self._cut_settings, 'random_crop', False)
            or getattr(self._cut_settings, 'scale_augmentation', False)
        )

    def _refresh_frame_lengths(self) -> None:
        self._frame_lengths = [
            self._calculate_frame_len(frame_index)
            for frame_index in range(len(self.samples))
        ]
        self._rebuild_lookup()

    def _rebuild_lookup(self):
        self._lookup_len_list = summarise_list(self._frame_lengths.copy())
        self._samples_amount = sum(self._frame_lengths)

    def _shuffle_samples_and_lengths(self):
        if not self.samples:
            return
        order = list(range(len(self.samples)))
        random.shuffle(order)
        self.samples = [self.samples[index] for index in order]
        if self._frame_lengths:
            self._frame_lengths = [self._frame_lengths[index] for index in order]

    def _prepare_frame_images(self, frame_index: int):
        image_path, label_path = self.samples[frame_index]
        prepared_image = ImagePreparator(image_path, self._prep_settings).image
        prepared_label = ImagePreparator(label_path, self._prep_settings).image

        if self.colors == 1:
            prepared_image = prepared_image.convert('L')
        else:
            prepared_image = prepared_image.convert('RGB')

        prepared_label = prepared_label.convert('L')
        prepared_rare_mask = self._load_prepared_rare_mask(image_path, prepared_image.size)
        return image_path, prepared_image, prepared_label, prepared_rare_mask

    def _frame_seed(self, frame_index: int) -> int:
        return hash((self._epoch_index, frame_index, len(self.samples))) & 0xFFFFFFFF

    def _build_frame_cutter(self, frame_index: int, *, shuffle: bool) -> SampleFastCutter:
        _image_path, prepared_image, prepared_label, prepared_rare_mask = self._prepare_frame_images(frame_index)
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        random.seed(self._frame_seed(frame_index))
        try:
            np.random.seed(self._frame_seed(frame_index))
            image_matrix = SampleFastCutter.get_matrix_from_image(prepared_image, self.colors)
            label_matrix = SampleFastCutter.get_matrix_from_image(prepared_label, 1)
            image_matrix, label_matrix = _apply_binary_tech_augmentation_to_pair(
                image_matrix,
                label_matrix,
                self._tech_augmentor,
                binary_tolerance=self._tech_aug_binary_tolerance,
            )
            rare_mask_matrix = None
            if prepared_rare_mask is not None:
                rare_mask_matrix = SampleFastCutter.get_matrix_from_image(prepared_rare_mask, 1)
            return SampleFastCutter(
                (image_matrix, label_matrix),
                self._cut_settings,
                shuffle=shuffle,
                skip_uniform_labels=self._skip_uniform_labels,
                rare_mask_matrix=rare_mask_matrix,
                rare_patch_oversampling_factor=self._rare_patch_oversampling_factor,
            )
        finally:
            random.setstate(random_state)
            np.random.set_state(np_random_state)

    def _calculate_frame_len(self, frame_index: int) -> int:
        if self._skip_uniform_labels or self._rare_patch_oversampling_enabled:
            return len(self._build_frame_cutter(frame_index, shuffle=False))
        image_path, _label_path = self.samples[frame_index]
        prepared_size = ImagePreparator(image_path, self._prep_settings).size
        return len(SampleCalculator((prepared_size[1], prepared_size[0]), self._cut_settings))

    @staticmethod
    def _resolve_frame_cache_limit() -> int:
        worker_info = get_worker_info()
        if worker_info is None:
            return 2
        return 1

    def _get_frame_cutter(self, frame_index: int, *, shuffle: bool) -> SampleFastCutter:
        cache_limit = max(1, int(self._frame_cache_limit))
        cache_key = (self._epoch_index, int(frame_index), bool(shuffle))
        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            self._frame_cache.move_to_end(cache_key)
            return cached

        cutter = self._build_frame_cutter(frame_index, shuffle=shuffle)
        self._frame_cache[cache_key] = cutter
        self._frame_cache.move_to_end(cache_key)
        while len(self._frame_cache) > cache_limit:
            self._frame_cache.popitem(last=False)
        return cutter

    def _load_prepared_rare_mask(self, image_path: Path, image_size: tuple[int, int]) -> Image.Image | None:
        if not self._rare_patch_oversampling_enabled or self._rare_patch_oversampling_factor <= 1:
            return None
        candidate_paths = [
            resolve_rare_patch_mask_path(self._sample_folder, image_path.stem),
            _resolve_artifact_replay_mask_path(self._artifact_replay_dir, image_path.stem),
        ]
        combined_mask: np.ndarray | None = None
        for rare_mask_path in candidate_paths:
            if rare_mask_path is None or not rare_mask_path.exists():
                continue
            rare_mask = ImagePreparator(rare_mask_path, self._prep_settings).image.convert('L')
            if rare_mask.size != image_size:
                rare_mask = rare_mask.resize(image_size, resample=Image.Resampling.NEAREST)
            rare_mask_array = np.asarray(rare_mask, dtype=np.uint8)
            if combined_mask is None:
                combined_mask = rare_mask_array
            else:
                combined_mask = np.maximum(combined_mask, rare_mask_array)
        if combined_mask is None or int(np.count_nonzero(combined_mask)) <= 0:
            return None
        return Image.fromarray(combined_mask.astype(np.uint8, copy=False), mode='L')

    def _build_context_crop(self, cutter: SampleFastCutter, part: int):
        left, top, right, bottom = cutter.resolve_part_coordinates(part)
        window = PatchWindow(
            left=int(left),
            top=int(top),
            width=max(1, int(right - left)),
            height=max(1, int(bottom - top)),
        )
        context_image = extract_centered_crop(
            cutter.image_matrix,
            center_x=window.center_x,
            center_y=window.center_y,
            crop_size_xy=self._context_crop_size,
            output_size_xy=self._context_crop_size,
            interpolation_mode='bilinear',
        )
        context_image = cutter.transform_patch_for_part(context_image, part)
        return resize_chw_image(
            context_image,
            output_size_xy=self._context_input_size,
            interpolation_mode='bilinear',
        )

    def _apply_pcb_defects(
        self,
        image: np.ndarray,
        label: np.ndarray,
        *,
        frame: int,
        part: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self._defect_augmentor is None:
            return image, label, None
        seed = self._sample_seed(frame, part)
        augmented_image, defect_mask, augmented_mask = self._defect_augmentor(
            image,
            label,
            seed=seed,
            return_augmented_mask=True,
        )
        updated_label = defect_mask if self._use_defect_mask_as_label else augmented_mask
        return augmented_image, updated_label, augmented_image

    def _sample_seed(self, frame: int, part: int) -> int:
        return hash((self._epoch_index, int(frame), int(part), len(self.samples), 9157)) & 0xFFFFFFFF

    def _inject_local_patch_into_context(
        self,
        context_image: np.ndarray,
        local_image: np.ndarray,
        *,
        source_crop_size_xy: tuple[int, int] | None = None,
    ) -> np.ndarray:
        if context_image.ndim != 3 or local_image.ndim != 3:
            return context_image
        context_copy = context_image.copy()
        source_crop_size = source_crop_size_xy or self._local_crop_size
        target_w = max(
            1,
            int(round((source_crop_size[0] / max(1, self._context_crop_size[0])) * self._context_input_size[0])),
        )
        target_h = max(
            1,
            int(round((source_crop_size[1] / max(1, self._context_crop_size[1])) * self._context_input_size[1])),
        )
        resized_local = SampleFastCutter._resize_patch_tensor(
            local_image,
            (target_w, target_h),
            resample=Image.Resampling.BILINEAR,
        )
        top = max(0, (context_copy.shape[1] - target_h) // 2)
        left = max(0, (context_copy.shape[2] - target_w) // 2)
        bottom = min(context_copy.shape[1], top + target_h)
        right = min(context_copy.shape[2], left + target_w)
        context_copy[:, top:bottom, left:right] = resized_local[:, :bottom - top, :right - left]
        return context_copy


class SyntheticDefectDataset(Dataset):
    """Generate full synthetic frames and cut them into patches like real images."""

    def __init__(
        self,
        sample_count: int,
        settings: TrainingParameters,
        *,
        apply_train_only_transforms: bool = True,
    ) -> None:
        self.colors = int(settings.colors)
        self._generation = settings.generation
        self._config = build_synthetic_defect_generator_parameters(
            getattr(settings, 'synthetic_defect_generator', None)
        )
        self._apply_train_only_transforms = bool(apply_train_only_transforms)
        self.shuffle_frames = bool(getattr(settings, 'shuffle', True))
        self.shuffle_patches_in_frame = bool(
            getattr(self._generation, 'shuffle_patches_in_frame', self.shuffle_frames)
        )
        self._use_context_branch = bool(getattr(settings, 'use_context_branch', False))
        self._local_crop_size = normalize_size_pair(
            getattr(settings, 'local_crop_size', None),
            fallback=tuple(settings.generation.segment_size),
        )
        self._context_crop_size = normalize_size_pair(
            getattr(settings, 'context_crop_size', None),
            fallback=(self._local_crop_size[0] * 2, self._local_crop_size[1] * 2),
        )
        self._context_input_size = normalize_size_pair(
            getattr(settings, 'context_input_size', None),
            fallback=self._local_crop_size,
        )
        self._frame_size_xy = (
            max(int(self._local_crop_size[0]), int(self._config.image_size_xy[0])),
            max(int(self._local_crop_size[1]), int(self._config.image_size_xy[1])),
        )
        self._frame_count = max(0, int(sample_count))
        self._parts_per_frame = max(
            0,
            int(
                len(
                    SampleCalculator(
                        (int(self._frame_size_xy[1]), int(self._frame_size_xy[0])),
                        self._generation,
                    )
                )
            ),
        )
        self._sample_count = int(self._frame_count * self._parts_per_frame)
        self._epoch_index = 0
        self._frame_order = list(range(self._frame_count))
        self._current_frame_index: int | None = None
        self._current_frame_cutter: SampleFastCutter | None = None
        self._frame_cache: OrderedDict[tuple[int, int, bool], SampleFastCutter] = OrderedDict()
        self._frame_cache_limit = 2
        defect_augmentor_cls = ICDefectAugmentor if self._config.topology_domain == 'ic' else PCBDefectAugmentor
        self._defect_augmentor = (
            defect_augmentor_cls(self._config.defects)
            if self._apply_train_only_transforms and self._config.enabled and self._config.defects.enabled
            else None
        )

    def __len__(self) -> int:
        return self._sample_count

    def set_epoch(self) -> None:
        self._epoch_index += 1
        if self.shuffle_frames:
            random.shuffle(self._frame_order)
        self._current_frame_index = None
        self._current_frame_cutter = None
        self._frame_cache.clear()

    def describe_sample(self, idx: int) -> str:
        frame, part = self._resolve_index(int(idx))
        return f'synthetic_frame_{int(frame):06d}__part_{int(part):06d}'

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._sample_count:
            raise IndexError('dataset index out of range')
        frame, part = self._resolve_index(int(idx))
        if self._current_frame_index != frame:
            self._current_frame_index = frame
            self._current_frame_cutter = self._get_frame_cutter(
                frame,
                shuffle=self.shuffle_patches_in_frame,
            )
        if self._current_frame_cutter is None:
            raise RuntimeError('Synthetic frame cutter is not initialized')
        image, label = self._current_frame_cutter[part]
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        label_tensor = torch.from_numpy(np.ascontiguousarray(label)).float()

        if not self._use_context_branch:
            return image_tensor, label_tensor

        context_image = self._build_context_crop(self._current_frame_cutter, part)
        return {
            'local_image': image_tensor,
            'context_image': torch.from_numpy(np.ascontiguousarray(context_image)).float(),
        }, label_tensor

    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if self._parts_per_frame <= 0:
            raise IndexError('synthetic dataset contains no patches')
        frame = int(idx) // int(self._parts_per_frame)
        part = int(idx) % int(self._parts_per_frame)
        return frame, part

    def _resolve_frame_id(self, frame_index: int) -> int:
        if frame_index < 0 or frame_index >= len(self._frame_order):
            raise IndexError('synthetic frame index out of range')
        return int(self._frame_order[frame_index])

    def _build_frame_cutter(self, frame_index: int, *, shuffle: bool) -> SampleFastCutter:
        resolved_frame = self._resolve_frame_id(int(frame_index))
        topology_generator = SyntheticTopologyGenerator(self._sample_topology_parameters(resolved_frame))
        base_image, base_label = topology_generator.generate(
            size_hw=(int(self._frame_size_xy[1]), int(self._frame_size_xy[0])),
            channels=self.colors,
            seed=self._sample_seed(resolved_frame, salt=0),
        )
        augmented_image, augmented_label = self._apply_defects(
            base_image,
            base_label,
            seed=self._sample_seed(resolved_frame, salt=1),
        )
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        frame_seed = self._sample_seed(resolved_frame, salt=2)
        random.seed(frame_seed)
        np.random.seed(frame_seed)
        try:
            return SampleFastCutter(
                (
                    np.asarray(augmented_image, dtype=np.float32).astype(np.float32, copy=False),
                    np.asarray(augmented_label, dtype=np.float32).astype(np.float32, copy=False),
                ),
                self._generation,
                shuffle=shuffle,
            )
        finally:
            random.setstate(random_state)
            np.random.set_state(np_random_state)

    def _get_frame_cutter(self, frame_index: int, *, shuffle: bool) -> SampleFastCutter:
        cache_key = (self._epoch_index, int(frame_index), bool(shuffle))
        cached = self._frame_cache.get(cache_key)
        if cached is not None:
            self._frame_cache.move_to_end(cache_key)
            return cached
        cutter = self._build_frame_cutter(frame_index, shuffle=shuffle)
        self._frame_cache[cache_key] = cutter
        self._frame_cache.move_to_end(cache_key)
        while len(self._frame_cache) > self._frame_cache_limit:
            self._frame_cache.popitem(last=False)
        return cutter

    def _build_context_crop(self, cutter: SampleFastCutter, part: int) -> np.ndarray:
        left, top, right, bottom = cutter.resolve_part_coordinates(part)
        window = PatchWindow(
            left=int(left),
            top=int(top),
            width=max(1, int(right - left)),
            height=max(1, int(bottom - top)),
        )
        context_image = extract_centered_crop(
            cutter.image_matrix,
            center_x=window.center_x,
            center_y=window.center_y,
            crop_size_xy=self._context_crop_size,
            output_size_xy=self._context_crop_size,
            interpolation_mode='bilinear',
        )
        context_image = cutter.transform_patch_for_part(context_image, part)
        return resize_chw_image(
            context_image,
            output_size_xy=self._context_input_size,
            interpolation_mode='bilinear',
        )

    def _apply_defects(
        self,
        image: np.ndarray,
        label: np.ndarray,
        *,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        image_array = np.asarray(image, dtype=np.float32).astype(np.float32, copy=False)
        label_array = np.asarray(label, dtype=np.float32).astype(np.float32, copy=False)
        if self._defect_augmentor is None:
            return image_array, label_array
        augmented_image, _defect_mask, _augmented_mask = self._defect_augmentor(
            image_array,
            label_array,
            seed=seed,
            return_augmented_mask=True,
        )
        return (
            np.asarray(augmented_image, dtype=np.float32).astype(np.float32, copy=False),
            label_array,
        )

    def _sample_topology_parameters(self, idx: int) -> SyntheticTopologyParameters:
        rng = np.random.default_rng(int(self._sample_seed(idx, salt=17)))
        curriculum_progress = min(1.0, max(0.0, float(self._epoch_index) / 6.0))
        topology_domain = str(self._config.topology_domain or 'pcb').strip().lower()
        topology_family = str(self._config.topology_family or 'pcb_mixed').strip().lower()

        density_scale = 1.15 if topology_domain == 'ic' else 0.95
        width_scale = 0.9 if topology_domain == 'ic' else 1.1
        noise_scale = 0.7 if topology_domain == 'ic' else 1.0
        if 'cell_array' in topology_family:
            density_scale *= 1.2
            width_scale *= 0.9
        elif 'tree' in topology_family:
            density_scale *= 0.9
            noise_scale *= 0.9
        elif 'parallel' in topology_family:
            density_scale *= 1.05

        def _blend_int_range(values: tuple[int, int], *, scale: float = 1.0) -> tuple[int, int]:
            low, high = int(values[0]), int(values[1])
            scaled_low = max(1, int(round(low * scale)))
            scaled_high = max(scaled_low, int(round(high * scale)))
            start_high = max(scaled_low, int(round(scaled_low + ((scaled_high - scaled_low) * 0.35))))
            current_high = int(round(start_high + ((scaled_high - start_high) * curriculum_progress)))
            return scaled_low, max(scaled_low, current_high)

        def _blend_float_range(values: tuple[float, float], *, scale: float = 1.0) -> tuple[float, float]:
            low, high = float(values[0]) * float(scale), float(values[1]) * float(scale)
            start_high = low + ((high - low) * 0.35)
            current_high = start_high + ((high - start_high) * curriculum_progress)
            return float(low), float(max(low, current_high))

        trace_count_range = _blend_int_range(tuple(self._config.trace_count_range), scale=density_scale)
        segment_count_range = _blend_int_range(tuple(self._config.segment_count_range))
        trace_half_width_range = _blend_int_range(tuple(self._config.trace_half_width_range), scale=width_scale)
        background_noise_sigma_range = _blend_float_range(tuple(self._config.background_noise_sigma_range), scale=noise_scale)
        trace_noise_sigma_range = _blend_float_range(tuple(self._config.trace_noise_sigma_range), scale=noise_scale)
        current_margin = int(
            round((float(self._config.margin) * (1.25 if topology_domain == 'pcb' else 1.1)) * (1.0 - curriculum_progress))
            + round(float(self._config.margin) * curriculum_progress)
        )
        return SyntheticTopologyParameters(
            trace_count=int(rng.integers(int(trace_count_range[0]), int(trace_count_range[1]) + 1)),
            segment_count_range=segment_count_range,
            trace_half_width_range=trace_half_width_range,
            margin=max(4, int(current_margin)),
            topology_domain=str(self._config.topology_domain),
            topology_family=str(self._config.topology_family),
            via_count_range=(1, max(1, min(6, int(round(float(trace_count_range[1]) / 3.0))))),
            background_noise_sigma=float(
                rng.uniform(float(background_noise_sigma_range[0]), float(background_noise_sigma_range[1]))
            ),
            trace_noise_sigma=float(
                rng.uniform(float(trace_noise_sigma_range[0]), float(trace_noise_sigma_range[1]))
            ),
        )

    def _sample_seed(self, idx: int, *, salt: int) -> int:
        return hash((self._epoch_index, int(idx), self._frame_count, int(salt), 18679)) & 0xFFFFFFFF


def summarise_list(datalist: list[int]):
    for i in range(len(datalist)):
        if i == 0:
            continue
        datalist[i] += datalist[i - 1]
    return datalist


def index_in_list(index: int, datalist: list[int]):
    """Map a global sample index to (frame_index, local_part_index).

    ``datalist`` is expected to be a cumulative-length lookup list.
    """
    if not datalist:
        raise ValueError('datalist must not be empty')
    if index < 0:
        raise IndexError('dataset index out of range')

    frame = bisect_right(datalist, index)
    if frame == 0:
        return 0, index

    if frame < len(datalist):
        return frame, index - datalist[frame - 1]

    raise IndexError('dataset index out of range')


class CustomDataset(Dataset):
    def __init__(
        self,
        samples,
        channels: int,
        transform=None,
        *,
        pcb_defects=None,
        apply_train_only_transforms: bool = True,
        tech_aug=None,
    ):
        self.samples: list[tuple[Path, Path]] = samples
        self.channels = channels
        self.transform = transform
        self._epoch_index: int = 0
        self._tech_aug_config = build_tech_augmentation_config(tech_aug)
        self._tech_aug_binary_tolerance = float(
            getattr(self._tech_aug_config, 'binary_tolerance', 0.15)
        )
        self._tech_augmentor = (
            TechVariationAugmentor(self._tech_aug_config)
            if bool(apply_train_only_transforms) and self._tech_aug_config.enabled
            else None
        )
        self._pcb_defects = build_pcb_defect_parameters(pcb_defects)
        self._defect_augmentor = (
            PCBDefectAugmentor(self._pcb_defects)
            if bool(apply_train_only_transforms) and self._pcb_defects.enabled
            else None
        )
        self._use_defect_mask_as_label = bool(self._pcb_defects.use_defect_mask_as_label)

    def __len__(self):
        return len(self.samples)

    def describe_sample(self, idx: int) -> str:
        image_path, _label_path = self.samples[int(idx)]
        return str(image_path.stem)

    def set_epoch(self) -> None:
        self._epoch_index += 1

    def __getitem__(self, idx):
        if self._defect_augmentor is not None:
            image_path = self.samples[idx][0]
            label_path = self.samples[idx][1]
            image_pil = retry_file_read(
                lambda: Image.open(image_path).convert("RGB" if self.channels == 3 else "L"),
                path=image_path,
            )
            label_pil = retry_file_read(
                lambda: Image.open(label_path).convert("L"),
                path=label_path,
            )
            image_array = np.asarray(image_pil, dtype=np.float32)
            if self.channels == 1:
                image_array = (image_array / 255.0)[None, :, :]
            else:
                image_array = np.transpose(image_array, (2, 0, 1)) / 255.0
            label_array = (np.asarray(label_pil, dtype=np.float32) / 255.0)[None, :, :]
            image_array, label_array = _apply_binary_tech_augmentation_to_pair_with_seed(
                image_array,
                label_array,
                self._tech_augmentor,
                binary_tolerance=self._tech_aug_binary_tolerance,
                seed=self._sample_seed(idx),
            )
            augmented_image, defect_mask, augmented_mask = self._defect_augmentor(
                image_array,
                label_array,
                seed=self._sample_seed(idx),
                return_augmented_mask=True,
            )
            target_array = defect_mask if self._use_defect_mask_as_label else augmented_mask
            return (
                torch.from_numpy(np.asarray(augmented_image, dtype=np.float32)).float(),
                torch.from_numpy(np.asarray(target_array, dtype=np.float32)).float(),
            )

        image_path = self.samples[idx][0]
        label_path = self.samples[idx][1]
        image = retry_file_read(
            lambda: Image.open(image_path).convert("RGB" if self.channels == 3 else "L"),
            path=image_path,
        )

        # Convert to tensor and normalize to [0., 1.]
        image_tensor = ToTensor()(image)

        # If needed, ensure the data type is float32
        image_tensor = image_tensor.float()

        label = retry_file_read(
            lambda: Image.open(label_path).convert("L"),
            path=label_path,
        )

        # Convert to tensor and normalize to [0., 1.]
        label_tensor = ToTensor()(label)

        # If needed, ensure the data type is float32
        label_tensor = label_tensor.float()
        if self._tech_augmentor is not None:
            augmented_image, augmented_label = _apply_binary_tech_augmentation_to_pair_with_seed(
                image_tensor.numpy(),
                label_tensor.numpy(),
                self._tech_augmentor,
                binary_tolerance=self._tech_aug_binary_tolerance,
                seed=self._sample_seed(idx),
            )
            image_tensor = torch.from_numpy(np.ascontiguousarray(augmented_image)).float()
            label_tensor = torch.from_numpy(np.ascontiguousarray(augmented_label)).float()

        # if self.transform:
        #     image = self.transform(image)
        #     label = self.transform(label)

        return image_tensor, label_tensor

    def _sample_seed(self, idx: int) -> int:
        return hash((self._epoch_index, int(idx), len(self.samples), 2741)) & 0xFFFFFFFF
