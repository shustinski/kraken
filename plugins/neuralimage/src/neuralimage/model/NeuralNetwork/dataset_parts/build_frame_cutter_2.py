"""dataset: build frame cutter 2."""
from __future__ import annotations
import random
import numpy as np
from neuralimage.augmentations import (
    SyntheticTopologyGenerator,
)
from neuralimage.lib.images import SampleFastCutter
from neuralimage.model.NeuralNetwork.context_utils import (
    PatchWindow,
    build_global_context_image,
    extract_centered_crop,
    normalize_patch_coords,
)
from neuralimage.targets.dataset_hooks import (
    apply_dataset_preprocessing,
    apply_dataset_sem_augmentation,
)

from neuralimage.model.NeuralNetwork.dataset_parts.index_in_list import (
    index_in_list,
    summarise_list,
)


class BuildFrameCutter2Mixin:
    def _resolve_index(self, idx: int) -> tuple[int, int]:
        if self._samples_amount <= 0 or not self._lookup_len_list:
            raise IndexError('synthetic dataset contains no patches')
        return index_in_list(int(idx), self._lookup_len_list)

    def _calculate_len(self) -> None:
        self._frame_lengths = [
            self._calculate_frame_len(frame_index)
            for frame_index in range(self._frame_count)
        ]
        self._rebuild_lookup()

    def _refresh_frame_lengths(self) -> None:
        self._frame_lengths = [
            self._calculate_frame_len(frame_index)
            for frame_index in range(self._frame_count)
        ]
        self._rebuild_lookup()

    def _rebuild_lookup(self) -> None:
        self._lookup_len_list = summarise_list(self._frame_lengths.copy())
        self._samples_amount = sum(self._frame_lengths)

    def _calculate_frame_len(self, frame_index: int) -> int:
        if self._skip_uniform_labels:
            return len(self._build_frame_cutter(frame_index, shuffle=False))
        return int(self._static_parts_per_frame)

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
        if self._apply_train_only_transforms:
            augmented_image, augmented_label = apply_dataset_sem_augmentation(
                augmented_image,
                augmented_label,
                self._sem_augmentation,
            )
        augmented_image = apply_dataset_preprocessing(augmented_image, self._preprocessing)
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
                skip_uniform_labels=self._skip_uniform_labels,
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

    def _build_context_crop(
        self,
        cutter: SampleFastCutter,
        part: int,
        *,
        coords_px: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
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

    def _build_patch_metadata(
        self,
        cutter: SampleFastCutter,
        part: int,
        *,
        coords_px: tuple[int, int, int, int] | None = None,
    ) -> dict[str, np.ndarray | str]:
        geometry = cutter.resolve_part_geometry(part)
        resolved_coords = np.asarray(coords_px or geometry['coords_px'], dtype=np.float32)
        source_size_hw_tuple = tuple(int(value) for value in geometry['source_size_hw'])
        return {
            'patch_coords_px': resolved_coords,
            'patch_coords_norm': normalize_patch_coords(resolved_coords, source_size_hw_tuple),
            'source_size_hw': np.asarray(source_size_hw_tuple, dtype=np.float32),
            'transform_variant': str(geometry['transform_variant']),
        }

    def _build_global_context_image(
        self,
        cutter: SampleFastCutter,
        part: int,
        *,
        patch_metadata: dict[str, np.ndarray | str],
    ) -> np.ndarray:
        transform_variant = str(patch_metadata['transform_variant'])
        cache_key = (id(cutter), transform_variant, tuple(self._context_input_size))
        cached = self._global_context_cache.get(cache_key)
        if cached is not None:
            self._global_context_cache.move_to_end(cache_key)
            return cached

        global_image = build_global_context_image(
            cutter.resolve_transformed_image_matrix(part),
            output_size_xy=self._context_input_size,
            interpolation_mode='bilinear',
        )
        self._global_context_cache[cache_key] = global_image
        self._global_context_cache.move_to_end(cache_key)
        while len(self._global_context_cache) > max(2, self._frame_cache_limit * 4):
            self._global_context_cache.popitem(last=False)
        return global_image

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
