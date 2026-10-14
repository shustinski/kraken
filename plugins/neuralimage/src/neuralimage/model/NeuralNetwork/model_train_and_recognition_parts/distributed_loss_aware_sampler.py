"""model_train_and_recognition: distributed loss aware sampler."""
from __future__ import annotations
import math
from typing import Iterator
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

class _DistributedLossAwareSampler(Sampler[int]):
    """Shard one deterministic loss-aware sampling plan and synchronize its epoch state."""

    def __init__(self, sampler: Sampler[int], *, rank: int, world_size: int, drop_last: bool) -> None:
        self.sampler = sampler
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        indices = [int(index) for index in self.sampler]
        if self.drop_last:
            usable = len(indices) - (len(indices) % self.world_size)
            indices = indices[:usable]
        elif indices:
            padding = (-len(indices)) % self.world_size
            indices.extend(indices[:padding])
        return iter(indices[self.rank::self.world_size])

    def __len__(self) -> int:
        size = int(len(self.sampler))
        return size // self.world_size if self.drop_last else math.ceil(size / self.world_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = max(0, int(epoch))
        if hasattr(self.sampler, '_epoch'):
            setattr(self.sampler, '_epoch', self.epoch)

    def resize(self, size: int, *, reset: bool = False) -> None:
        resize = getattr(self.sampler, 'resize', None)
        if callable(resize):
            resize(size, reset=reset)

    def start_epoch(self) -> None:
        start = getattr(self.sampler, 'start_epoch', None)
        if callable(start):
            start()

    def update_batch_losses(self, sample_indices: torch.Tensor, sample_losses: torch.Tensor) -> None:
        update = getattr(self.sampler, 'update_batch_losses', None)
        if callable(update):
            update(sample_indices, sample_losses)

    def finalize_epoch(self) -> set[str]:
        state_fn = getattr(self.sampler, 'state_dict', None)
        load_fn = getattr(self.sampler, 'load_state_dict', None)
        if dist.is_available() and dist.is_initialized() and callable(state_fn) and callable(load_fn):
            local_state = state_fn()
            gathered: list[dict[str, object] | None] = [None] * self.world_size
            dist.all_gather_object(gathered, local_state)
            merged_state = dict(local_state)
            merged_epoch_losses: dict[str, object] = {}
            merged_ema = dict(local_state.get('ema_losses', {}))
            for state in gathered:
                if not isinstance(state, dict):
                    continue
                epoch_losses = state.get('epoch_losses', {})
                if not isinstance(epoch_losses, dict):
                    continue
                merged_epoch_losses.update(epoch_losses)
                state_ema = state.get('ema_losses', {})
                if isinstance(state_ema, dict):
                    for sample_key in epoch_losses:
                        if sample_key in state_ema:
                            merged_ema[str(sample_key)] = state_ema[sample_key]
            merged_state['epoch_losses'] = merged_epoch_losses
            merged_state['ema_losses'] = merged_ema
            load_fn(merged_state)
        finalize = getattr(self.sampler, 'finalize_epoch', None)
        return set(finalize()) if callable(finalize) else set()

    def state_dict(self) -> dict[str, object]:
        state_fn = getattr(self.sampler, 'state_dict', None)
        return {
            'epoch': self.epoch,
            'sampler': state_fn() if callable(state_fn) else {},
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.epoch = max(0, int(state.get('epoch', 0)))
        load_fn = getattr(self.sampler, 'load_state_dict', None)
        sampler_state = state.get('sampler')
        if callable(load_fn) and isinstance(sampler_state, dict):
            load_fn(sampler_state)

    @property
    def last_frame_losses(self) -> dict[str, float]:
        return dict(getattr(self.sampler, 'last_frame_losses', {}))

    @property
    def last_sigma(self) -> float:
        return float(getattr(self.sampler, 'last_sigma', 0.0))
