"""dataset: sample topology parameters."""
from __future__ import annotations
import numpy as np
from neuralimage.augmentations import (
    SyntheticTopologyParameters,
)


class SampleTopologyParametersMixin:
    def _sample_topology_parameters(self, idx: int) -> SyntheticTopologyParameters:
        rng = np.random.default_rng(int(self._sample_seed(idx, salt=17)))
        trace_count_range = tuple(self._config.trace_count_range)
        background_noise_sigma_range = tuple(self._config.background_noise_sigma_range)
        trace_noise_sigma_range = tuple(self._config.trace_noise_sigma_range)
        return SyntheticTopologyParameters(
            trace_count=int(rng.integers(int(trace_count_range[0]), int(trace_count_range[1]) + 1)),
            segment_count_range=tuple(self._config.segment_count_range),
            trace_half_width_range=tuple(self._config.trace_half_width_range),
            margin=int(self._config.margin),
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
