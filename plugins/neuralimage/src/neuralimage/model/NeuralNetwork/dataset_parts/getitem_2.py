"""dataset: getitem 2."""
from __future__ import annotations
import numpy as np
import torch
from neuralimage.targets.dataset_hooks import (
    maybe_build_supervision_target,
)

from neuralimage.model.NeuralNetwork.dataset_parts.extract_binary_single_channel import PatchSampleRequest


class Getitem2Mixin:
    def __getitem__(self, idx: int):
        requested_size: tuple[int, int] | None = None
        if isinstance(idx, PatchSampleRequest):
            requested_size = tuple(int(value) for value in idx.size_xy)
            idx = int(idx.index)
        if idx < 0 or idx >= self._samples_amount:
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
        if requested_size is None:
            image, label = self._current_frame_cutter[part]
            requested_coords = None
        else:
            image, label, requested_coords = self._current_frame_cutter.get_sized_item(
                part,
                size_xy=requested_size,
            )
        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        base_label_tensor = torch.from_numpy(np.ascontiguousarray(label)).float()
        label_tensor = maybe_build_supervision_target(base_label_tensor, self._supervision_targets)

        if not self._use_context_branch:
            return image_tensor, label_tensor

        patch_metadata = self._build_patch_metadata(
            self._current_frame_cutter,
            part,
            coords_px=requested_coords,
        )
        if self._use_cross_attention:
            global_image = self._build_global_context_image(
                self._current_frame_cutter,
                part,
                patch_metadata=patch_metadata,
            )
            return {
                'local_image': image_tensor,
                'global_image': torch.from_numpy(np.ascontiguousarray(global_image)).float(),
                'context_image': torch.from_numpy(np.ascontiguousarray(global_image)).float(),
                'patch_coords_norm': torch.from_numpy(np.ascontiguousarray(patch_metadata['patch_coords_norm'])).float(),
                'patch_coords_px': torch.from_numpy(np.ascontiguousarray(patch_metadata['patch_coords_px'])).float(),
                'source_size_hw': torch.from_numpy(np.ascontiguousarray(patch_metadata['source_size_hw'])).float(),
            }, label_tensor

        context_image = self._build_context_crop(
            self._current_frame_cutter,
            part,
            coords_px=requested_coords,
        )
        return {
            'local_image': image_tensor,
            'context_image': torch.from_numpy(np.ascontiguousarray(context_image)).float(),
            'patch_coords_norm': torch.from_numpy(np.ascontiguousarray(patch_metadata['patch_coords_norm'])).float(),
            'patch_coords_px': torch.from_numpy(np.ascontiguousarray(patch_metadata['patch_coords_px'])).float(),
            'source_size_hw': torch.from_numpy(np.ascontiguousarray(patch_metadata['source_size_hw'])).float(),
        }, label_tensor
