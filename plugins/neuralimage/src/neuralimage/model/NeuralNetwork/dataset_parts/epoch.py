"""dataset: epoch."""
from __future__ import annotations


class EpochMixin:
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
        self._global_context_cache.clear()
