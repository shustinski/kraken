"""general_neural_handler: random patch batch sampler."""
from __future__ import annotations
import math
import random
from typing import Iterable
from torch.utils.data import Sampler
from neuralimage.model.NeuralNetwork.dataset import (
    PatchSampleRequest,
)

class RandomPatchBatchSampler(Sampler[list[PatchSampleRequest]]):
    variable_patch_sizes = True

    def __init__(
        self,
        sampler: Sampler[int],
        *,
        batch_size: int,
        min_size: tuple[int, int],
        max_size: tuple[int, int],
        drop_last: bool = False,
    ) -> None:
        self.sampler = sampler
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self._widths = self._aligned_values(int(min_size[0]), int(max_size[0]))
        self._heights = self._aligned_values(int(min_size[1]), int(max_size[1]))

    @staticmethod
    def _aligned_values(lower: int, upper: int) -> tuple[int, ...]:
        first = int(math.ceil(int(lower) / 32.0) * 32)
        values = tuple(range(first, int(upper) + 1, 32))
        if not values:
            raise ValueError('random patch range must contain a multiple of 32')
        return values

    def __iter__(self):
        batch: list[int] = []
        for index in self.sampler:
            batch.append(int(index))
            if len(batch) >= self.batch_size:
                yield self._requests(batch)
                batch = []
        if batch and not self.drop_last:
            yield self._requests(batch)

    def _requests(self, indices: list[int]) -> list[PatchSampleRequest]:
        size = (random.choice(self._widths), random.choice(self._heights))
        return [PatchSampleRequest(index=index, size_xy=size) for index in indices]

    def __len__(self) -> int:
        size = len(self.sampler)
        if self.drop_last:
            return size // self.batch_size
        return math.ceil(size / self.batch_size)

class FrameBatchSampler(Sampler[list[int]]):
    """Yield sequential batches without mixing patches from different frames."""

    def __init__(self, frame_lengths: Iterable[int], *, batch_size: int) -> None:
        self.frame_lengths = tuple(max(0, int(length)) for length in frame_lengths)
        self.batch_size = max(1, int(batch_size))

    def __iter__(self):
        offset = 0
        for frame_length in self.frame_lengths:
            frame_end = offset + frame_length
            for batch_start in range(offset, frame_end, self.batch_size):
                yield list(range(batch_start, min(batch_start + self.batch_size, frame_end)))
            offset = frame_end

    def __len__(self) -> int:
        return sum(math.ceil(length / self.batch_size) for length in self.frame_lengths if length > 0)
