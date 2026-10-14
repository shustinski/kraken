"""general_neural_handler: hard frame sampler."""
from __future__ import annotations
import math
import random
from typing import Any
import torch
from torch.utils.data import Dataset, Sampler

class HardFrameSampler(Sampler[int]):
    """Deterministic online sampler combining geometry and per-sample EMA loss."""

    def __init__(self, dataset: Dataset, *, shuffle: bool = True, parameters: Any = None) -> None:
        self.dataset = dataset
        self.size = max(0, int(len(dataset)))
        self.shuffle = bool(shuffle)
        self._hard_frame_keys: set[str] = set()
        self._epoch_losses: dict[str, tuple[str, float]] = {}
        self.last_frame_losses: dict[str, float] = {}
        self.last_sigma: float = 0.0
        self._parameters = parameters
        self._ema_losses: dict[str, float] = {}
        self._geometry_scores: dict[str, float] = {}
        self._epoch = 0

    def set_geometry_scores(self, scores: dict[str, float]) -> None:
        self._geometry_scores = {str(key): max(0.0, float(value)) for key, value in scores.items()}

    @staticmethod
    def _normalize(values: dict[str, float]) -> dict[str, float]:
        if not values:
            return {}
        low, high = min(values.values()), max(values.values())
        if high <= low:
            return {key: 0.0 for key in values}
        return {key: (value - low) / (high - low) for key, value in values.items()}

    def __iter__(self):
        base_indices = list(range(self.size))
        rng = random.Random(1337 + self._epoch)
        if self.shuffle:
            rng.shuffle(base_indices)
        if self._parameters is None:
            extra_indices = [
                index for index in range(self.size)
                if self._frame_key(index) in self._hard_frame_keys
            ]
            if self.shuffle:
                rng.shuffle(extra_indices)
            return iter(base_indices + extra_indices)
        weights = self._sampling_weights()
        exploration = min(max(float(getattr(self._parameters, 'exploration_floor', 0.1)), 0.0), 1.0)
        extra_count = sum(1 for weight in weights if weight > exploration)
        extra_indices = rng.choices(range(self.size), weights=weights, k=extra_count) if extra_count else []
        return iter(base_indices + extra_indices)

    def _sampling_weights(self) -> list[float]:
        geometry = self._normalize(self._geometry_scores)
        losses = self._normalize(self._ema_losses)
        geometry_weight = max(0.0, float(getattr(self._parameters, 'geometry_weight', 0.5)))
        loss_weight = max(0.0, float(getattr(self._parameters, 'loss_weight', 0.5)))
        exploration = min(max(float(getattr(self._parameters, 'exploration_floor', 0.1)), 0.0), 1.0)
        score_clip = max(1e-6, float(getattr(self._parameters, 'score_clip', 5.0)))
        weights = []
        for index in range(self.size):
            key = self._sample_key(index)
            score = geometry_weight * geometry.get(key, 0.0) + loss_weight * losses.get(key, 0.0)
            weights.append(min(score_clip, max(exploration, score)))
        return weights

    def __len__(self) -> int:
        if self._parameters is not None:
            exploration = min(max(float(getattr(self._parameters, 'exploration_floor', 0.1)), 0.0), 1.0)
            return self.size + sum(1 for weight in self._sampling_weights() if weight > exploration)
        return self.size + sum(
            1 for index in range(self.size)
            if self._frame_key(index) in self._hard_frame_keys
        )

    def resize(self, size: int, *, reset: bool = False) -> None:
        self.size = max(0, int(size))
        if reset:
            self._hard_frame_keys.clear()

    def start_epoch(self) -> None:
        self._epoch_losses.clear()
        self._epoch += 1

    def _frame_key(self, index: int) -> str:
        resolver = getattr(self.dataset, 'frame_key', None)
        return str(resolver(int(index))) if callable(resolver) else str(int(index))

    def _sample_key(self, index: int) -> str:
        resolver = getattr(self.dataset, 'sample_key', None)
        return str(resolver(int(index))) if callable(resolver) else str(int(index))

    def update_batch_losses(self, sample_indices: torch.Tensor, sample_losses: torch.Tensor) -> None:
        indices = sample_indices.detach().to(device='cpu', dtype=torch.long).flatten().tolist()
        losses = sample_losses.detach().to(device='cpu', dtype=torch.float64).flatten().tolist()
        for index, loss in zip(indices, losses):
            if index < 0 or index >= self.size or not math.isfinite(float(loss)):
                continue
            sample_key = self._sample_key(int(index))
            if sample_key in self._epoch_losses:
                continue
            self._epoch_losses[sample_key] = (self._frame_key(int(index)), float(loss))
            alpha = min(max(float(getattr(self._parameters, 'ema_alpha', 0.1)), 0.0), 1.0)
            previous = self._ema_losses.get(sample_key, float(loss))
            self._ema_losses[sample_key] = previous * (1.0 - alpha) + float(loss) * alpha

    def finalize_epoch(self) -> set[str]:
        squared_by_frame: dict[str, list[float]] = {}
        for frame_key, loss in self._epoch_losses.values():
            squared_by_frame.setdefault(frame_key, []).append(float(loss) ** 2)
        frame_losses = {
            frame_key: math.sqrt(sum(values) / len(values))
            for frame_key, values in squared_by_frame.items()
            if values
        }
        self.last_frame_losses = frame_losses
        if not frame_losses:
            self.last_sigma = 0.0
            self._hard_frame_keys.clear()
            return set()
        values = list(frame_losses.values())
        mean_value = sum(values) / len(values)
        sigma = math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))
        self.last_sigma = float(sigma)
        self._hard_frame_keys = {
            frame_key for frame_key, value in frame_losses.items()
            if value >= mean_value + sigma
        }
        return set(self._hard_frame_keys)

    def state_dict(self) -> dict[str, object]:
        return {
            'size': self.size,
            'hard_frame_keys': sorted(self._hard_frame_keys),
            'epoch_losses': dict(self._epoch_losses),
            'last_frame_losses': dict(self.last_frame_losses),
            'last_sigma': float(self.last_sigma),
            'ema_losses': dict(self._ema_losses),
            'geometry_scores': dict(self._geometry_scores),
            'epoch': self._epoch,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.size = max(0, int(state.get('size', self.size)))
        self._hard_frame_keys = {str(value) for value in state.get('hard_frame_keys', ())}
        raw_epoch_losses = state.get('epoch_losses', {})
        self._epoch_losses = {
            str(sample_key): (str(value[0]), float(value[1]))
            for sample_key, value in raw_epoch_losses.items()
            if isinstance(value, (tuple, list)) and len(value) == 2
        } if isinstance(raw_epoch_losses, dict) else {}
        raw_frame_losses = state.get('last_frame_losses', {})
        self.last_frame_losses = {
            str(frame_key): float(value)
            for frame_key, value in raw_frame_losses.items()
        } if isinstance(raw_frame_losses, dict) else {}
        self.last_sigma = float(state.get('last_sigma', 0.0))
        self._ema_losses = {str(key): float(value) for key, value in dict(state.get('ema_losses', {})).items()}
        self._geometry_scores = {str(key): float(value) for key, value in dict(state.get('geometry_scores', {})).items()}
        self._epoch = max(0, int(state.get('epoch', 0)))
