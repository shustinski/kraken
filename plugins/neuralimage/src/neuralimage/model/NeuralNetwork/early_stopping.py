from __future__ import annotations

import logging
import math
import os
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterable, Mapping

import torch
from torch import nn


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EarlyStoppingRuntimeConfig:
    train_size: int
    batches_per_epoch: int
    control_size: int
    check_every_batches: int
    checks_per_epoch: int
    patience_checks: int
    warmup_batches: int
    trend_window: int
    max_epochs: int
    single_check_min_improvement: float = 0.005
    trend_min_improvement: float = 0.01


class EarlyStoppingConfigCalculator:
    """Calculate all non-user-tunable early-stopping parameters."""

    @staticmethod
    def calculate(*, train_size: int, batches_per_epoch: int) -> EarlyStoppingRuntimeConfig:
        if not isinstance(train_size, int) or isinstance(train_size, bool) or train_size <= 0:
            raise ValueError('train_size must be a positive integer')
        if not isinstance(batches_per_epoch, int) or isinstance(batches_per_epoch, bool) or batches_per_epoch <= 0:
            raise ValueError('batches_per_epoch must be a positive integer')

        control_size = min(
            train_size,
            1024,
            max(32, math.ceil(0.05 * train_size)),
        )
        if batches_per_epoch <= 20:
            check_every_batches = batches_per_epoch
        elif batches_per_epoch <= 200:
            check_every_batches = math.ceil(batches_per_epoch / 2)
        else:
            check_every_batches = min(5000, math.ceil(batches_per_epoch / 4))

        checks_per_epoch = math.ceil(batches_per_epoch / check_every_batches)
        patience_checks = max(3, 2 * checks_per_epoch)
        warmup_batches = 3 * batches_per_epoch
        trend_window = 2 * checks_per_epoch
        if train_size < 1_000:
            max_epochs = 200
        elif train_size < 10_000:
            max_epochs = 100
        else:
            max_epochs = 60

        config = EarlyStoppingRuntimeConfig(
            train_size=train_size,
            batches_per_epoch=batches_per_epoch,
            control_size=control_size,
            check_every_batches=check_every_batches,
            checks_per_epoch=checks_per_epoch,
            patience_checks=patience_checks,
            warmup_batches=warmup_batches,
            trend_window=trend_window,
            max_epochs=max_epochs,
        )
        LOGGER.info('Calculated early-stopping parameters: %s', config)
        return config


@dataclass(frozen=True)
class EarlyStoppingDecision:
    should_stop: bool
    actual_best_improved: bool
    significant_improvement: bool
    bad_checks: int
    trend_improvement: float | None
    reason: str | None = None


class EarlyStoppingPolicy:
    """State machine for automatic early stopping.

    The actual minimum and the reference used to reset patience are deliberately
    tracked separately. This guarantees that every factual minimum is saved even
    when its relative improvement is below 0.5%.
    """

    def __init__(self, config: EarlyStoppingRuntimeConfig) -> None:
        if not isinstance(config, EarlyStoppingRuntimeConfig):
            raise TypeError('config must be EarlyStoppingRuntimeConfig')
        self.config = config
        self.best_actual_loss: float | None = None
        self.best_actual_batch: int | None = None
        self.patience_reference_loss: float | None = None
        self.bad_checks = 0
        self._history: deque[float] = deque(maxlen=max(1, int(config.trend_window)))

    @property
    def history(self) -> tuple[float, ...]:
        return tuple(self._history)

    def state_dict(self) -> dict[str, Any]:
        return {
            'best_actual_loss': self.best_actual_loss,
            'best_actual_batch': self.best_actual_batch,
            'patience_reference_loss': self.patience_reference_loss,
            'bad_checks': int(self.bad_checks),
            'history': list(self._history),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError('early-stopping state must be a mapping')
        self.best_actual_loss = self._optional_finite_float(state.get('best_actual_loss'))
        best_batch = state.get('best_actual_batch')
        self.best_actual_batch = None if best_batch is None else max(0, int(best_batch))
        self.patience_reference_loss = self._optional_finite_float(state.get('patience_reference_loss'))
        self.bad_checks = max(0, int(state.get('bad_checks', 0)))
        self._history.clear()
        for value in state.get('history', ()):  # type: ignore[assignment]
            resolved = self._optional_finite_float(value)
            if resolved is not None:
                self._history.append(resolved)

    @staticmethod
    def _optional_finite_float(value: Any) -> float | None:
        if value is None:
            return None
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError('loss values must be finite')
        return resolved

    def update(self, *, loss: float, global_batch: int) -> EarlyStoppingDecision:
        current_loss = float(loss)
        if not math.isfinite(current_loss):
            raise ValueError('loss must be finite')
        if current_loss < 0.0:
            raise ValueError('loss must be non-negative')
        if not isinstance(global_batch, int) or isinstance(global_batch, bool) or global_batch < 0:
            raise ValueError('global_batch must be a non-negative integer')

        actual_best_improved = self.best_actual_loss is None or current_loss < self.best_actual_loss
        if actual_best_improved:
            self.best_actual_loss = current_loss
            self.best_actual_batch = int(global_batch)

        significant_improvement = False
        if self.patience_reference_loss is None:
            significant_improvement = True
        else:
            reference = float(self.patience_reference_loss)
            relative_improvement = (reference - current_loss) / max(abs(reference), 1e-12)
            significant_improvement = relative_improvement >= self.config.single_check_min_improvement

        if significant_improvement:
            self.patience_reference_loss = current_loss
            self.bad_checks = 0
        else:
            self.bad_checks += 1

        self._history.append(current_loss)
        trend_improvement: float | None = None
        if len(self._history) >= self.config.trend_window:
            window_start = float(self._history[0])
            trend_improvement = (window_start - current_loss) / max(abs(window_start), 1e-12)

        should_stop = bool(
            global_batch >= self.config.warmup_batches
            and self.bad_checks >= self.config.patience_checks
            and trend_improvement is not None
            and trend_improvement < self.config.trend_min_improvement
        )
        reason = None
        if should_stop:
            reason = (
                f'no significant improvement for {self.bad_checks} checks and '
                f'trend improvement {float(trend_improvement):.4%} is below '
                f'{self.config.trend_min_improvement:.2%}'
            )
        return EarlyStoppingDecision(
            should_stop=should_stop,
            actual_best_improved=actual_best_improved,
            significant_improvement=significant_improvement,
            bad_checks=int(self.bad_checks),
            trend_improvement=trend_improvement,
            reason=reason,
        )


BatchAdapter = Callable[[Any, torch.device], tuple[Any, torch.Tensor]]
ForwardFunction = Callable[[nn.Module, Any], Any]
PerSampleLossFunction = Callable[[Any, torch.Tensor], torch.Tensor]


class MetricEvaluator:
    """Evaluate a loss without augmentations and average it per object."""

    def evaluate(
        self,
        *,
        model: nn.Module,
        dataloader: Iterable[Any],
        device: torch.device,
        batch_adapter: BatchAdapter,
        forward_fn: ForwardFunction,
        per_sample_loss_fn: PerSampleLossFunction,
        autocast_ctx: Callable[[], ContextManager[Any]] | None = None,
    ) -> float:
        if model is None:
            raise ValueError('model is required')
        if dataloader is None:
            raise ValueError('dataloader is required')
        was_training = bool(model.training)
        loss_sum = 0.0
        object_count = 0
        model.eval()
        context_factory = autocast_ctx or nullcontext
        try:
            with torch.no_grad():
                for batch in dataloader:
                    inputs, target = batch_adapter(batch, device)
                    with context_factory():
                        outputs = forward_fn(model, inputs)
                        losses = per_sample_loss_fn(outputs, target)
                    loss_tensor = torch.as_tensor(losses).detach().to(dtype=torch.float64)
                    if loss_tensor.ndim == 0:
                        batch_objects = int(target.shape[0]) if target.ndim > 0 else 1
                        loss_sum += float(loss_tensor.item()) * batch_objects
                        object_count += batch_objects
                    else:
                        flattened = loss_tensor.reshape(-1)
                        finite = torch.isfinite(flattened)
                        loss_sum += float(flattened[finite].sum().item())
                        object_count += int(finite.sum().item())
        finally:
            model.train(was_training)
        if object_count <= 0:
            raise ValueError('metric evaluator received no finite objects')
        return loss_sum / object_count


class CheckpointManager:
    """Own best/regular checkpoint files and restore the factual best state."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        logger: Callable[[str], None] | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.best_path = self.checkpoint_path.with_name(f'{self.checkpoint_path.stem}.best{self.checkpoint_path.suffix}')
        self._logger = logger or LOGGER.info
        self.best_loss: float | None = None
        self.best_global_batch: int | None = None

    @staticmethod
    def _atomic_save(payload: Mapping[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f'{path.suffix}.tmp')
        torch.save(dict(payload), temporary_path)
        os.replace(temporary_path, path)

    def save_regular(self, payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError('checkpoint payload must be a mapping')
        self._atomic_save(payload, self.checkpoint_path)

    def save_best(
        self,
        *,
        model: nn.Module,
        loss: float,
        global_batch: int,
        extra: Mapping[str, Any] | None = None,
    ) -> bool:
        resolved_loss = float(loss)
        if not math.isfinite(resolved_loss):
            raise ValueError('best checkpoint loss must be finite')
        if self.best_loss is not None and resolved_loss >= self.best_loss:
            return False
        payload: dict[str, Any] = {
            'model_state_dict': {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            'loss': resolved_loss,
            'global_batch': int(global_batch),
        }
        if extra:
            payload.update(dict(extra))
        self._atomic_save(payload, self.best_path)
        self.best_loss = resolved_loss
        self.best_global_batch = int(global_batch)
        self._logger(
            f'Best checkpoint saved at global batch {int(global_batch)} '
            f'(loss={resolved_loss:.8f}).'
        )
        return True

    def restore_best(self, model: nn.Module, *, map_location: Any = 'cpu') -> Mapping[str, Any] | None:
        if not self.best_path.exists():
            return None
        payload = torch.load(self.best_path, map_location=map_location, weights_only=False)
        if not isinstance(payload, Mapping) or 'model_state_dict' not in payload:
            raise ValueError(f'Invalid best checkpoint: {self.best_path}')
        model.load_state_dict(payload['model_state_dict'])
        self.best_loss = float(payload.get('loss'))
        self.best_global_batch = int(payload.get('global_batch', 0))
        self._logger(
            f'Best checkpoint restored from global batch {self.best_global_batch} '
            f'(loss={self.best_loss:.8f}).'
        )
        return payload
