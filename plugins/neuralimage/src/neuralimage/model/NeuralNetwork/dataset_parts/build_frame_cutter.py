"""dataset: build frame cutter."""
from __future__ import annotations
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import get_worker_info
from neuralimage.lib.images import ImagePreparator, SampleCalculator, SampleFastCutter
from neuralimage.lib.rare_patch_masks import resolve_rare_patch_mask_path
from neuralimage.model.NeuralNetwork.context_utils import (
    PatchWindow,
    extract_centered_crop,
)
from neuralimage.preprocessing.pipeline import image_to_channel_first_float01
from neuralimage.targets.dataset_hooks import (
    apply_dataset_preprocessing,
    apply_dataset_sem_augmentation,
)

from neuralimage.model.NeuralNetwork.dataset_parts.apply_binary_tech_augmentation import _apply_binary_tech_augmentation_to_pair

from neuralimage.model.NeuralNetwork.dataset_parts.index_in_list import (
    index_in_list,
    summarise_list,
)


class BuildFrameCutterMixin:
    def __len__(self):
        return self._samples_amount

    def validation_frame_lengths(self) -> tuple[int, ...]:
        """Return stable patch counts for sequential frame-wise evaluation."""
        return tuple(max(0, int(length)) for length in self._frame_lengths)

    def describe_sample(self, index: int) -> str:
        frame, part = index_in_list(int(index), self._lookup_len_list)
        image_path, _label_path = self.samples[frame]
        return f'{image_path.stem}__part_{int(part):06d}'

    def frame_key(self, index: int) -> str:
        frame, _part = index_in_list(int(index), self._lookup_len_list)
        image_path, _label_path = self.samples[frame]
        return str(image_path.resolve())

    def sample_key(self, index: int) -> str:
        frame, part = index_in_list(int(index), self._lookup_len_list)
        return f'{self.frame_key(index)}::{int(part)}'

    def _create_files_list(self):
        if self.shuffle_frames:
            random.shuffle(self.samples)

    def _calculate_len(self):
        frame_count = len(self.samples)
        if frame_count <= 0:
            self._frame_lengths = []
            self._rebuild_lookup()
            return

        use_parallel = (
            frame_count >= 128
            and not self._skip_uniform_labels
            and not self._rare_patch_oversampling_enabled
        )
        if use_parallel:
            max_workers = min(8, max(2, (os.cpu_count() or 2)))
            try:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    len_list = list(executor.map(self._calculate_frame_len, range(frame_count)))
            except Exception:
                len_list = [self._calculate_frame_len(frame_index) for frame_index in range(frame_count)]
        else:
            len_list = [self._calculate_frame_len(frame_index) for frame_index in range(frame_count)]

        self._frame_lengths = len_list
        self._rebuild_lookup()

    def _resolve_dynamic_frame_lengths(self) -> bool:
        if not (self._skip_uniform_labels or self._rare_patch_oversampling_enabled):
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
        prepared_image = ImagePreparator(image_path, self._prep_settings, asset_source=self._asset_source).image
        prepared_label = ImagePreparator(label_path, self._prep_settings, asset_source=self._asset_source).image

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
            image_matrix = image_to_channel_first_float01(prepared_image, self.colors)
            label_matrix = SampleFastCutter.get_matrix_from_image(prepared_label, 1)
            image_matrix, label_matrix = _apply_binary_tech_augmentation_to_pair(
                image_matrix,
                label_matrix,
                self._tech_augmentor,
                binary_tolerance=self._tech_aug_binary_tolerance,
            )
            if self._apply_train_only_transforms:
                image_matrix, label_matrix = apply_dataset_sem_augmentation(
                    image_matrix,
                    label_matrix,
                    self._sem_augmentation,
                )
            # Deterministic SEM preprocessing is frame-scoped so local patches
            # and every context view are sampled from the identical signal.
            image_matrix = apply_dataset_preprocessing(image_matrix, self._preprocessing)
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
        prepared_size = ImagePreparator(image_path, self._prep_settings, asset_source=self._asset_source).size
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
        rare_mask_path = resolve_rare_patch_mask_path(self._sample_folder, image_path.stem)
        if not rare_mask_path.exists():
            return None
        rare_mask = ImagePreparator(rare_mask_path, self._prep_settings).image.convert('L')
        if rare_mask.size != image_size:
            rare_mask = rare_mask.resize(image_size, resample=Image.Resampling.NEAREST)
        if rare_mask.getbbox() is None:
            return None
        return rare_mask

    def _build_context_crop(
        self,
        cutter: SampleFastCutter,
        part: int,
        *,
        coords_px: tuple[int, int, int, int] | None = None,
    ):
        geometry = cutter.resolve_part_geometry(part)
        left, top, right, bottom = coords_px or tuple(int(value) for value in geometry['coords_px'])
        window = PatchWindow(
            left=int(left),
            top=int(top),
            width=max(1, int(right - left)),
            height=max(1, int(bottom - top)),
        )
        return extract_centered_crop(
            cutter.resolve_transformed_image_matrix(part),
            center_x=window.center_x,
            center_y=window.center_y,
            crop_size_xy=self._context_crop_size,
            output_size_xy=self._context_input_size,
            interpolation_mode='bilinear',
        )
