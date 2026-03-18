import random
from collections import OrderedDict
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from torchvision.transforms import ToTensor

from augmentations import PCBDefectAugmentor, TechVariationAugmentor
from lib.data_interfaces import TrainingParameters, SampleGenerationSettings, SamplePrepareSettings
from lib.data_interfaces import build_pcb_defect_parameters, build_tech_augmentation_config
from lib.images import ImagePreparator, SampleCalculator, SampleFastCutter
from lib.rare_patch_masks import resolve_rare_patch_mask_path
from model.NeuralNetwork.context_utils import PatchWindow, extract_centered_crop, normalize_size_pair


def _unwrap_tech_augmented_mask(result: np.ndarray | tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    if isinstance(result, tuple):
        return np.asarray(result[1])
    return np.asarray(result)


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
        self._samples_amount: int = 0
        self._frame_lengths: list[int] = []
        self._lookup_len_list: list[int] = []
        self._epoch_index: int = 0

        self._current_frame_index: int | None = None
        self._current_image_cutter: SampleFastCutter | None = None
        self._frame_cache: OrderedDict[tuple[int, int, bool], SampleFastCutter] = OrderedDict()
        self._frame_cache_limit = self._resolve_frame_cache_limit()

        self._create_files_list()
        self._calculate_len()

    def set_epoch(self):
        self._epoch_index += 1
        if self.shuffle_frames:
            self._shuffle_samples_and_lengths()
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
            context_image = self._inject_local_patch_into_context(
                context_image,
                augmented_local_image,
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
            image_matrix = _apply_binary_tech_augmentation(
                image_matrix,
                self._tech_augmentor,
                binary_tolerance=self._tech_aug_binary_tolerance,
            )
            label_matrix = SampleFastCutter.get_matrix_from_image(prepared_label, 1)
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
            return 4
        worker_count = max(1, int(worker_info.num_workers))
        if worker_count >= 6:
            return 1
        if worker_count >= 3:
            return 2
        return 4

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
        rare_mask_path = resolve_rare_patch_mask_path(self._sample_folder, image_path.stem)
        if not rare_mask_path.exists():
            return None
        rare_mask = ImagePreparator(rare_mask_path, self._prep_settings).image.convert('L')
        if rare_mask.size != image_size:
            rare_mask = rare_mask.resize(image_size, resample=Image.Resampling.NEAREST)
        if rare_mask.getbbox() is None:
            return None
        return rare_mask

    def _build_context_crop(self, cutter: SampleFastCutter, part: int):
        left, top, right, bottom = cutter.resolve_part_coordinates(part)
        window = PatchWindow(
            left=int(left),
            top=int(top),
            width=max(1, int(right - left)),
            height=max(1, int(bottom - top)),
        )
        return extract_centered_crop(
            cutter.image_matrix,
            center_x=window.center_x,
            center_y=window.center_y,
            crop_size_xy=self._context_crop_size,
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
        augmented_image, defect_mask = self._defect_augmentor(
            image,
            label,
            seed=seed,
        )
        updated_label = defect_mask if self._use_defect_mask_as_label else label
        return augmented_image, updated_label, augmented_image

    def _sample_seed(self, frame: int, part: int) -> int:
        return hash((self._epoch_index, int(frame), int(part), len(self.samples), 9157)) & 0xFFFFFFFF

    def _inject_local_patch_into_context(
        self,
        context_image: np.ndarray,
        local_image: np.ndarray,
    ) -> np.ndarray:
        if context_image.ndim != 3 or local_image.ndim != 3:
            return context_image
        context_copy = context_image.copy()
        target_w = max(
            1,
            int(round((self._local_crop_size[0] / max(1, self._context_crop_size[0])) * self._context_input_size[0])),
        )
        target_h = max(
            1,
            int(round((self._local_crop_size[1] / max(1, self._context_crop_size[1])) * self._context_input_size[1])),
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
            image_pil = Image.open(self.samples[idx][0]).convert("RGB" if self.channels == 3 else "L")
            label_pil = Image.open(self.samples[idx][1]).convert("L")
            image_array = np.asarray(image_pil, dtype=np.float32)
            if self.channels == 1:
                image_array = (image_array / 255.0)[None, :, :]
            else:
                image_array = np.transpose(image_array, (2, 0, 1)) / 255.0
            image_array = _apply_binary_tech_augmentation_with_seed(
                image_array,
                self._tech_augmentor,
                binary_tolerance=self._tech_aug_binary_tolerance,
                seed=self._sample_seed(idx),
            )
            label_array = (np.asarray(label_pil, dtype=np.float32) / 255.0)[None, :, :]
            augmented_image, defect_mask = self._defect_augmentor(
                image_array,
                label_array,
                seed=self._sample_seed(idx),
            )
            target_array = defect_mask if self._use_defect_mask_as_label else label_array
            return (
                torch.from_numpy(np.asarray(augmented_image, dtype=np.float32)).float(),
                torch.from_numpy(np.asarray(target_array, dtype=np.float32)).float(),
            )

        image = Image.open(self.samples[idx][0]).convert("RGB" if self.channels == 3 else "L")

        # Convert to tensor and normalize to [0., 1.]
        image_tensor = ToTensor()(image)

        # If needed, ensure the data type is float32
        image_tensor = image_tensor.float()
        if self._tech_augmentor is not None:
            augmented_image = _apply_binary_tech_augmentation_with_seed(
                image_tensor.numpy(),
                self._tech_augmentor,
                binary_tolerance=self._tech_aug_binary_tolerance,
                seed=self._sample_seed(idx),
            )
            image_tensor = torch.from_numpy(np.ascontiguousarray(augmented_image)).float()

        label = Image.open(self.samples[idx][1]).convert("L")

        # Convert to tensor and normalize to [0., 1.]
        label_tensor = ToTensor()(label)

        # If needed, ensure the data type is float32
        label_tensor = label_tensor.float()

        # if self.transform:
        #     image = self.transform(image)
        #     label = self.transform(label)

        return image_tensor, label_tensor

    def _sample_seed(self, idx: int) -> int:
        return hash((self._epoch_index, int(idx), len(self.samples), 2741)) & 0xFFFFFFFF
