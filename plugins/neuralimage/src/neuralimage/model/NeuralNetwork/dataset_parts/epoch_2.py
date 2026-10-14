"""dataset: epoch 2."""
from __future__ import annotations
import random


class Epoch2Mixin:
    def __len__(self) -> int:
        return self._samples_amount

    def set_epoch(self) -> None:
        self._epoch_index += 1
        if self.shuffle_frames:
            random.shuffle(self._frame_order)
        if self._dynamic_frame_lengths:
            self._refresh_frame_lengths()
        elif self.shuffle_frames:
            self._rebuild_lookup()
        self._current_frame_index = None
        self._current_frame_cutter = None
        self._frame_cache.clear()
        self._global_context_cache.clear()

    def describe_sample(self, idx: int) -> str:
        frame, part = self._resolve_index(int(idx))
        return f'synthetic_frame_{int(frame):06d}__part_{int(part):06d}'

    def frame_key(self, idx: int) -> str:
        frame, _part = self._resolve_index(int(idx))
        return f'synthetic::{int(self._resolve_frame_id(frame))}'

    def sample_key(self, idx: int) -> str:
        frame, part = self._resolve_index(int(idx))
        return f'{self.frame_key(idx)}::{int(part)}'
