"""synthetic_topology: channel first."""
from __future__ import annotations
import numpy as np


class ChannelFirstMixin:
    @staticmethod
    def _to_channel_first(image: np.ndarray, channels: int) -> np.ndarray:
        array = np.asarray(image, dtype=np.float32)
        resolved_channels = max(1, int(channels))
        if array.ndim == 2:
            base = array[None, :, :]
            if resolved_channels == 1:
                return base.astype(np.float32, copy=False)
            return np.repeat(base, resolved_channels, axis=0).astype(np.float32, copy=False)
        if array.ndim != 3 or array.shape[2] <= 0:
            raise ValueError('SyntheticTopologyGenerator expects 2D grayscale or HWC RGB image.')
        if resolved_channels == 1:
            grayscale = (
                (array[..., 0] * 0.299)
                + (array[..., 1] * 0.587)
                + (array[..., 2] * 0.114)
            )
            return grayscale[None, :, :].astype(np.float32, copy=False)
        chw = np.transpose(array, (2, 0, 1)).astype(np.float32, copy=False)
        if resolved_channels == chw.shape[0]:
            return chw
        if resolved_channels < chw.shape[0]:
            return chw[:resolved_channels].astype(np.float32, copy=False)
        repeats = [chw]
        while sum(item.shape[0] for item in repeats) < resolved_channels:
            repeats.append(chw[: min(chw.shape[0], resolved_channels - sum(item.shape[0] for item in repeats))])
        return np.concatenate(repeats, axis=0).astype(np.float32, copy=False)
