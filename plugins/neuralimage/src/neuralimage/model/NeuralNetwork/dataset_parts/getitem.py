"""dataset: getitem."""
from __future__ import annotations
from neuralimage.targets.dataset_hooks import (
    maybe_build_supervision_target,
)

from neuralimage.model.NeuralNetwork.dataset_parts.extract_binary_single_channel import PatchSampleRequest

from neuralimage.model.NeuralNetwork.dataset_parts.index_in_list import index_in_list


class GetitemMixin:
    def __getitem__(self, index):
        requested_size: tuple[int, int] | None = None
        if isinstance(index, PatchSampleRequest):
            requested_size = tuple(int(value) for value in index.size_xy)
            index = int(index.index)
        if index < 0 or index >= self._samples_amount:
            raise IndexError('dataset index out of range')
        frame, part = index_in_list(index, self._lookup_len_list)
        if self._current_frame_index != frame:
            self._current_frame_index = frame
            self._current_image_cutter = self._get_frame_cutter(
                frame,
                shuffle=self.shuffle_patches_in_frame,
            )
        if requested_size is None:
            image, label = self._current_image_cutter[part]
            requested_coords = None
        else:
            image, label, requested_coords = self._current_image_cutter.get_sized_item(
                part,
                size_xy=requested_size,
            )
        image, label, augmented_local_image = self._apply_pcb_defects(
            image,
            label,
            frame=frame,
            part=part,
        )
        label = maybe_build_supervision_target(label, self._supervision_targets)
        if not self._use_context_branch:
            return image, label
        patch_metadata = self._build_patch_metadata(
            self._current_image_cutter,
            part,
            coords_px=requested_coords,
        )
        if self._use_cross_attention:
            global_image = self._build_global_context_image(
                self._current_image_cutter,
                part,
                patch_metadata=patch_metadata,
                augmented_local_image=augmented_local_image,
            )
            return {
                'local_image': image,
                'global_image': global_image,
                'context_image': global_image,
                'patch_coords_norm': patch_metadata['patch_coords_norm'],
                'patch_coords_px': patch_metadata['patch_coords_px'],
                'source_size_hw': patch_metadata['source_size_hw'],
            }, label

        context_image = self._build_context_crop(
            self._current_image_cutter,
            part,
            coords_px=requested_coords,
        )
        if augmented_local_image is not None:
            context_image = self._inject_local_patch_into_context(
                context_image,
                augmented_local_image,
            )
        return {
            'local_image': image,
            'context_image': context_image,
            'patch_coords_norm': patch_metadata['patch_coords_norm'],
            'patch_coords_px': patch_metadata['patch_coords_px'],
            'source_size_hw': patch_metadata['source_size_hw'],
        }, label
