"""synthetic_topology: generate."""
from __future__ import annotations
import random
import numpy as np

from neuralimage.augmentations.synthetic_topology_parts.synthetic_topology_parameters import SyntheticTopologyParameters


class GenerateMixin:
    def __init__(self, params: SyntheticTopologyParameters | None = None) -> None:
        self.params = params or SyntheticTopologyParameters()

    def generate(
        self,
        *,
        size_hw: tuple[int, int],
        channels: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        height = max(32, int(size_hw[0]))
        width = max(32, int(size_hw[1]))
        rng = random.Random(int(seed))
        np_rng = np.random.default_rng(int(seed))

        family = self._resolve_family(rng)
        mask_u8, anchors = self._build_family_mask(family, height, width, rng)
        mask_u8 = self._decorate_terminals(mask_u8, anchors, family, rng)

        if not np.any(mask_u8):
            mask_u8 = self._build_fallback_mask(height, width)

        image_array = self._render_image(mask_u8, np_rng)
        image_chw = self._to_channel_first(image_array, channels)
        label_chw = (mask_u8.astype(np.float32) / 255.0)[None, :, :]
        return image_chw.astype(np.float32, copy=False), label_chw.astype(np.float32, copy=False)

    def _resolve_family(self, rng: random.Random) -> str:
        domain = self._domain()
        family = str(getattr(self.params, 'topology_family', 'independent') or 'independent').strip().lower()
        if domain == 'ic' and family in ('ic_mixed', 'mixed'):
            return 'ic_mixed'
        if family in ('pcb_mixed', 'mixed'):
            mixed_population = self._PCB_MIXED_FAMILIES if domain != 'ic' else self._IC_MIXED_FAMILIES
            population = [item[0] for item in mixed_population]
            weights = [item[1] for item in mixed_population]
            return str(rng.choices(population, weights=weights, k=1)[0])
        if family in self._PCB_FAMILIES:
            return self._PCB_FAMILIES[family]
        if family in self._IC_FAMILIES:
            return self._IC_FAMILIES[family]
        if family:
            return family
        population = [item[0] for item in self._MIXED_FAMILIES]
        weights = [item[1] for item in self._MIXED_FAMILIES]
        return str(rng.choices(population, weights=weights, k=1)[0])

    def _build_family_mask(
        self,
        family: str,
        height: int,
        width: int,
        rng: random.Random,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        domain = self._domain()
        if domain == 'ic':
            if family == 'ic_mixed':
                return self._build_ic_mixed_family_mask(height, width, rng)
            if family == 'cell_array':
                return self._build_ic_cell_array_family_mask(height, width, rng)
            if family == 'tree':
                return self._build_ic_tree_family_mask(height, width, rng)
            return self._build_ic_channel_family_mask(height, width, rng)
        if family == 'parallel':
            return self._build_pcb_parallel_family_mask(height, width, rng)
        if family == 'tree':
            return self._build_pcb_tree_family_mask(height, width, rng)
        if family == 'cell_array':
            return self._build_pcb_cell_array_family_mask(height, width, rng)
        return self._build_pcb_independent_family_mask(height, width, rng)

    def _domain(self) -> str:
        domain = str(getattr(self.params, 'topology_domain', 'pcb') or 'pcb').strip().lower()
        return 'ic' if domain == 'ic' else 'pcb'
