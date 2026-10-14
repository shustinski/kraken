"""dataset: build global context image."""
from __future__ import annotations
import numpy as np
from PIL import Image
from neuralimage.lib.images import SampleFastCutter
from neuralimage.model.NeuralNetwork.context_utils import (
    build_global_context_image,
    normalize_patch_coords,
)


class BuildGlobalContextImageMixin:
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
        augmented_local_image: np.ndarray | None = None,
    ) -> np.ndarray:
        transform_variant = str(patch_metadata['transform_variant'])
        cache_key = (id(cutter), transform_variant, tuple(self._context_input_size))
        if augmented_local_image is None:
            cached = self._global_context_cache.get(cache_key)
            if cached is not None:
                self._global_context_cache.move_to_end(cache_key)
                return cached

        source = cutter.resolve_transformed_image_matrix(part)
        if augmented_local_image is not None:
            source = self._inject_local_patch_into_global(
                source,
                augmented_local_image,
                coords_px=np.asarray(patch_metadata['patch_coords_px'], dtype=np.float32),
            )
        global_image = build_global_context_image(
            source,
            output_size_xy=self._context_input_size,
            interpolation_mode='bilinear',
        )
        if augmented_local_image is None:
            self._global_context_cache[cache_key] = global_image
            self._global_context_cache.move_to_end(cache_key)
            while len(self._global_context_cache) > max(2, self._frame_cache_limit * 4):
                self._global_context_cache.popitem(last=False)
        return global_image

    def _inject_local_patch_into_global(
        self,
        global_image: np.ndarray,
        local_image: np.ndarray,
        *,
        coords_px: np.ndarray,
    ) -> np.ndarray:
        if global_image.ndim != 3 or local_image.ndim != 3:
            return global_image
        left, top, right, bottom = [int(round(float(value))) for value in coords_px.tolist()]
        height, width = int(global_image.shape[1]), int(global_image.shape[2])
        left = max(0, min(width, left))
        right = max(0, min(width, right))
        top = max(0, min(height, top))
        bottom = max(0, min(height, bottom))
        if right <= left or bottom <= top:
            return global_image
        global_copy = global_image.copy()
        resized_local = SampleFastCutter._resize_patch_tensor(
            local_image,
            (right - left, bottom - top),
            resample=Image.Resampling.BILINEAR,
        )
        global_copy[:, top:bottom, left:right] = resized_local[:, :bottom - top, :right - left]
        return global_copy

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
